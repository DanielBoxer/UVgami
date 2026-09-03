import collections

from .cuts import CREASED_RELIEF, cut_path, path_cost
from .mesh import face_keys, find, norm, pair, turn_angle
from .regions import CREASE_ANGLE

# far enough to snap to a crease beside it, never across a region
REROUTE_RINGS = 2
# a closed loop has no junctions to hold it, a dull one would drift
LOOP_SHARE = 0.5


# a tooth is a face with every edge but one on the boundary to the same neighbour
def flatten_teeth(weighted, faces, edges, label, angle=CREASE_ANGLE, forced=None):
    turns = {}

    def sharp(key):
        owners = edges[key]
        if len(owners) != 2:
            return True  # a mesh rim hides a seam as well as a crease
        if key not in turns:
            turns[key] = turn_angle(weighted, owners)
        return turns[key] >= angle

    label = dict(label)
    queue = collections.deque(label)
    queued = set(queue)
    while queue:
        f = queue.popleft()
        queued.discard(f)
        keys = face_keys(faces[f])
        lost = collections.defaultdict(list)
        for key in keys:
            owners = edges[key]
            if len(owners) == 2:
                g = owners[owners[0] == f]
                if label[g] != label[f]:
                    lost[label[g]].append(key)
        if not lost:
            continue
        target, gone = max(lost.items(), key=lambda kv: len(kv[1]))
        if len(gone) != len(keys) - 1:
            continue
        kept = next(k for k in keys if k not in gone)
        if forced and not forced.isdisjoint(gone):
            continue
        if not sharp(kept) and any(sharp(k) for k in gone):
            continue
        label[f] = target
        for key in keys:
            for o in edges[key]:
                if o != f and o not in queued:
                    queued.add(o)
                    queue.append(o)
    return label


# the merges settle face by face and leave boundaries staircased
def reroute_boundaries(verts, faces, areas, edges, label, relief, forced=None):
    label = dict(label)

    # a reroute reads its two regions instead of every face in the mesh
    region_faces = collections.defaultdict(set)
    for f, r in label.items():
        region_faces[r].add(f)

    vert_faces = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for v in face:
            vert_faces[v].append(fi)

    # no path may pass through an anchored vertex, its own run's excepted
    pair_keys = collections.defaultdict(list)
    anchored = set()
    vert_pairs = collections.defaultdict(set)
    for key, owners in edges.items():
        if len(owners) != 2:
            anchored.update(key)
            continue
        ra, rb = label[owners[0]], label[owners[1]]
        if ra == rb:
            continue
        pair_keys[pair(ra, rb)].append(key)
        vert_pairs[key[0]].add(pair(ra, rb))
        vert_pairs[key[1]].add(pair(ra, rb))
    blocked = anchored | set(vert_pairs)

    def loop_arcs(loop, cycle):
        n = len(loop)
        creased = sum(1 for k in loop if relief.get(k, 1.0) < CREASED_RELIEF)
        if creased < LOOP_SHARE * n:
            return
        # a vertex is as sharp as the duller of its two loop edges
        sharp = [
            1.0 - max(relief.get(loop[i - 1], 1.0), relief.get(loop[i], 1.0))
            for i in range(n)
        ]
        anchors = []
        gap = max(1, n // 4)
        for i in sorted(range(n), key=lambda i: sharp[i], reverse=True):
            if all(min((i - a) % n, (a - i) % n) >= gap for a in anchors):
                anchors.append(i)
            if len(anchors) == 3:
                break
        if len(anchors) < 2:
            return
        anchors.sort()
        for a, b in zip(anchors, anchors[1:] + anchors[:1]):
            arc = loop[a:b] if a < b else loop[a:] + loop[:b]
            yield arc, cycle[a], cycle[b]

    def chains(keys):
        deg = collections.Counter()
        incident = collections.defaultdict(list)
        for key in keys:
            for v in key:
                deg[v] += 1
                incident[v].append(key)
        junctions = {
            v for v in deg if deg[v] != 2 or v in anchored or len(vert_pairs[v]) > 1
        }
        seen = set()
        for j in junctions:
            for start in incident[j]:
                if start in seen:
                    continue
                seen.add(start)
                run = [start]
                v = start[start[0] == j]
                while v not in junctions:
                    key = next(k for k in incident[v] if k != run[-1])
                    seen.add(key)
                    run.append(key)
                    v = key[key[0] == v]
                if v != j:
                    yield run, j, v
        left = set(keys) - seen
        while left:
            start = left.pop()
            loop = [start]
            cycle = [start[0], start[1]]
            v = start[1]
            while True:
                key = next((k for k in incident[v] if k in left), None)
                if key is None:
                    break
                left.discard(key)
                loop.append(key)
                v = key[key[0] == v]
                cycle.append(v)
            # anything under three arcs of a few edges each is too small to move
            if cycle[-1] != cycle[0] or len(loop) < 9:
                continue
            cycle.pop()
            yield from loop_arcs(loop, cycle)

    def ec(group):
        ks = {key for f in group for key in face_keys(faces[f])}
        vs = {v for key in ks for v in key}
        return len(vs) - len(ks) + len(group)

    # relief squared, a mild discount loses to a dull shortcut
    pull = {k: r * r for k, r in relief.items()}

    def run_seq(run, j0):
        seq = [j0]
        for key in run:
            seq.append(key[1] if key[0] == seq[-1] else key[0])
        return seq

    def creased_share(keys):
        total = creased = 0.0
        for a, b in keys:
            length = norm([verts[a][i] - verts[b][i] for i in range(3)])
            total += length
            if relief.get(pair(a, b), 1.0) < CREASED_RELIEF:
                creased += length
        return creased / total if total else 0.0

    def reroute(p, run, j0, j1):
        ra, rb = p
        for key in run:
            owners = edges[key]
            if {label[owners[0]], label[owners[1]]} != {ra, rb}:
                return  # an earlier reroute moved this stretch
        if forced and not forced.isdisjoint(run):
            return
        run_set = set(run)
        run_verts = {v for key in run for v in key}

        corridor = set()
        ring = run_verts
        for _ in range(REROUTE_RINGS):
            grown = {
                f
                for v in ring
                for f in vert_faces[v]
                if label[f] in p and f not in corridor
            }
            corridor |= grown
            ring = {v for f in grown for v in faces[f]}
        allowed = {v for f in corridor for v in faces[f]} - blocked | run_verts
        adjacent = collections.defaultdict(set)
        for f in corridor:
            for key in face_keys(faces[f]):
                a, b = key
                if a not in allowed or b not in allowed:
                    continue
                owners = edges[key]
                if len(owners) == 2 and {label[o] for o in owners} <= set(p):
                    adjacent[a].add(b)
                    adjacent[b].add(a)

        path = cut_path(verts, adjacent, {j0}, {j1}, relief=pull)
        if not path:
            return
        new = {pair(a, b) for a, b in zip(path, path[1:])}
        if new == run_set or path_cost(verts, path, relief=pull) >= path_cost(
            verts, run_seq(run, j0), relief=pull
        ):
            return
        # a reroute never trades crease for shortcut
        if creased_share(new) < creased_share(run_set):
            return

        # split the union along the new path, the old run no longer divides
        union = region_faces[ra] | region_faces[rb]
        parent = {f: f for f in union}

        for f in union:
            for key in face_keys(faces[f]):
                if key in new:
                    continue
                owners = edges[key]
                if len(owners) != 2:
                    continue
                g = owners[owners[0] == f]
                if g not in parent:
                    continue
                if label[f] != label[g] and key not in run_set:
                    continue
                fa, fb = find(parent, f), find(parent, g)
                if fa != fb:
                    parent[fa] = fb
        comps = collections.defaultdict(list)
        for f in union:
            comps[find(parent, f)].append(f)
        if len(comps) != 2:
            return
        one, two = comps.values()

        def lean(group):
            return sum(areas[f] if label[f] == ra else -areas[f] for f in group)

        if lean(one) == lean(two):
            return
        if lean(two) > lean(one):
            one, two = two, one
        floor = min(
            ec([f for f in union if label[f] == ra]),
            ec([f for f in union if label[f] == rb]),
        )
        if max(ec(one), ec(two)) > 1 or min(ec(one), ec(two)) < floor:
            return
        for f in one:
            label[f] = ra
        for f in two:
            label[f] = rb
        region_faces[ra] = set(one)
        region_faces[rb] = set(two)
        blocked.update(path)

    for p, keys in pair_keys.items():
        for run, j0, j1 in chains(keys):
            reroute(p, run, j0, j1)
    return label


# edges between two regions, as sorted vertex index pairs
def boundary_edges(edges, label):
    return {
        pair for pair, owners in edges.items() if len({label[o] for o in owners}) > 1
    }
