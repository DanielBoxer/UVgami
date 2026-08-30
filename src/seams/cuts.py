"""Where a seam path goes once one is needed.

Costs first: crease relief makes sharp edges cheaper, concave more
than convex, painted restrictions make edges longer, and turning
between dull edges costs extra, so a cut across featureless area
comes out a line, the way an artist cuts. cut_path is the search over
those costs, and disk_cuts uses it to open every multi-loop region
into a disk."""

import collections
import heapq
import math

from .mesh import LOW_ANGLE, find, norm, pair, turn_angle

# what a fully painted vertex multiplies an edge's length by. bounded on
# purpose: an infinite cost would drop a cut instead of moving it, so a path
# that must cross paint crosses at its narrowest
RESTRICT_COST = 9.0
# free cuts prefer sharp edges: a crease edge counts shorter, sliding from
# full length at LOW_ANGLE to a floor at RELIEF_FULL_ANGLE. concave gets the
# deeper discount, a groove hides a seam best, and the floors are mild so
# only a short detour to reach a crease is worth its cost
CONCAVE_RELIEF = 0.5
CONVEX_RELIEF = 0.3
RELIEF_FULL_ANGLE = 45
# relief below this counts an edge as creased
CREASED_RELIEF = 0.9
# each step turning between two dull edges costs up to this fraction of its
# length extra, so among near-equal paths the straight one wins. a turn with
# both edges creased is exempt, since a seam follows a crease around any
# corner. one creased edge is not enough, bevel rows read as creases and a
# path would zigzag between parallel rows for free
TURN_COST = 1.0


def boundary_components(edges, label, forced=None):
    """Per-region boundary vertices, grouped into connected components.

    Two loops meeting at a vertex count as one component: the cut between them
    would be a point, not a path, and the region is already joined there.
    forced edges are seams already, so inside a region they count as boundary:
    a marked slit joining two rims means the region is open there."""
    parent = {}

    for (v0, v1), owners in edges.items():
        regions = {label[o] for o in owners}
        if len(owners) == 2 and len(regions) == 1:
            if not forced or (v0, v1) not in forced:
                continue
        for r in regions:
            a, b = (r, v0), (r, v1)
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb:
                parent[ra] = rb

    grouped = collections.defaultdict(lambda: collections.defaultdict(set))
    for region, vert in parent:
        grouped[region][find(parent, (region, vert))].add(vert)
    return {r: list(comps.values()) for r, comps in grouped.items()}


def crease_relief(verts, faces, weighted, edges):
    """Per edge, a length factor under 1 where the surface creases, so free
    cuts prefer sharp edges over wandering across flat triangles. The sign
    is read off the neighbour's centroid against the face plane: risen means
    concave, a groove that hides a seam. Flat edges are left out."""
    centroids = []
    for face in faces:
        x = y = z = 0.0
        for v in face:
            px, py, pz = verts[v]
            x += px
            y += py
            z += pz
        centroids.append([x / len(face), y / len(face), z / len(face)])
    relief = {}
    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        angle = turn_angle(weighted, owners)
        if angle <= LOW_ANGLE:
            continue
        na = weighted[owners[0]]
        base = verts[key[0]]
        lift = sum(na[i] * (centroids[owners[1]][i] - base[i]) for i in range(3))
        depth = min((angle - LOW_ANGLE) / (RELIEF_FULL_ANGLE - LOW_ANGLE), 1.0)
        relief[key] = 1 - (CONCAVE_RELIEF if lift > 0 else CONVEX_RELIEF) * depth
    return relief


def edge_cost(verts, weights, a, b, relief=None):
    """An edge's length, longer where a painted restriction repels cuts and
    shorter along a crease, so cuts land on clean lines."""
    length = norm([verts[a][i] - verts[b][i] for i in range(3)])
    if relief:
        length *= relief.get(pair(a, b), 1.0)
    if not weights:
        return length
    paint = (weights.get(a, 0.0) + weights.get(b, 0.0)) / 2
    return length * (1 + RESTRICT_COST * paint)


def turn_cost(verts, u, v, w, relief):
    """Extra cost fraction for the step v->w after arriving from u: zero along
    a crease or a straight continuation, up to TURN_COST on a reversal.
    """
    if (
        relief.get(pair(u, v), 1.0) < CREASED_RELIEF
        and relief.get(pair(v, w), 1.0) < CREASED_RELIEF
    ):
        return 0.0
    a = [verts[v][i] - verts[u][i] for i in range(3)]
    b = [verts[w][i] - verts[v][i] for i in range(3)]
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    cos = sum(a[i] * b[i] for i in range(3)) / (na * nb)
    return TURN_COST * (1.0 - cos) / 2.0


def path_cost(verts, seq, weights=None, relief=None):
    """Cost of an ordered vertex path, turn penalties included, so a
    comparison values straightness the same way cut_path searches for it."""
    total = 0.0
    for i in range(1, len(seq)):
        step = edge_cost(verts, weights, seq[i - 1], seq[i], relief)
        if relief is not None and i >= 2:
            step *= 1.0 + turn_cost(verts, seq[i - 2], seq[i - 1], seq[i], relief)
        total += step
    return total


def cut_path(verts, adjacent, sources, targets, weights=None, relief=None):
    """Cheapest path from any source vertex to any target, over adjacent.

    Painted restrictions count as extra length. With relief the state
    carries the incoming direction and turning between dull edges costs
    extra, so a cut across featureless area comes out a line instead of the
    staircase that happens to be shortest.
    """
    if relief is None:
        # without relief an edge costs at least its length, so with one target
        # the straight line distance left never overestimates
        goal = verts[next(iter(targets))] if len(targets) == 1 else None

        def remaining(v):
            if goal is None:
                return 0.0
            return norm([verts[v][i] - goal[i] for i in range(3)])

        dist = dict.fromkeys(sources, 0.0)
        prev = {}
        heap = [(remaining(v), 0.0, v) for v in sources]
        heapq.heapify(heap)
        while heap:
            _, d, v = heapq.heappop(heap)
            if d != dist[v]:
                continue
            if v in targets:
                path = [v]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return path
            for w in adjacent[v]:
                step = d + edge_cost(verts, weights, v, w, relief)
                if step < dist.get(w, math.inf):
                    dist[w] = step
                    prev[w] = v
                    heapq.heappush(heap, (step + remaining(w), step, w))
        return []

    # state is (vertex, arrived-from), -1 for a start with no direction yet
    dist = {(v, -1): 0.0 for v in sources}
    prev = {}
    heap = [(0.0, v, -1) for v in sources]
    heapq.heapify(heap)
    while heap:
        d, v, u = heapq.heappop(heap)
        if d != dist[v, u]:
            continue
        if v in targets:
            path = [v]
            node = (v, u)
            while node in prev:
                node = prev[node]
                path.append(node[0])
            return path
        for w in adjacent[v]:
            step = edge_cost(verts, weights, v, w, relief)
            if u >= 0:
                step *= 1.0 + turn_cost(verts, u, v, w, relief)
            step += d
            if step < dist.get((w, v), math.inf):
                dist[w, v] = step
                prev[w, v] = (v, u)
                heapq.heappush(heap, (step, w, v))
    return []


def part_labels(adjacent):
    """A loose part id per vertex of the graph."""
    label = {}
    for start in adjacent:
        if start in label:
            continue
        label[start] = start
        stack = [start]
        while stack:
            v = stack.pop()
            for w in adjacent[v]:
                if w not in label:
                    label[w] = start
                    stack.append(w)
    return label


def snap_paths(verts, adjacent, mapped, cuts):
    """Redraw another mesh's cut network on this one, edge by edge.

    Each cut edge becomes the shortest path between the vertices its ends
    map to, and since every vertex maps to one vertex here, segments that
    met still meet and loops stay closed. A segment collapsing to a point
    or with no path is dropped, leaving the rest intact. Ends on two loose
    parts are dropped before searching, a failed search walks the whole part.
    """
    part = part_labels(adjacent)
    paths = set()
    for a, b in cuts:
        va, vb = mapped[a], mapped[b]
        if va == vb or part.get(va) != part.get(vb):
            continue
        path = cut_path(verts, adjacent, {va}, {vb})
        for x, y in zip(path, path[1:]):
            paths.add(pair(x, y))
    return paths


def connect_loops(verts, adjacent, comps, weights=None, relief=None):
    """Cut paths joining every boundary component to the first, one path per
    extra loop, each the shortest available at the time."""
    cuts = set()
    sources = set(comps[0])
    targets = {v: i for i, comp in enumerate(comps[1:], 1) for v in comp}
    while targets:
        path = cut_path(verts, adjacent, sources, targets, weights, relief)
        if not path:
            break  # disconnected, leave the rest to the engine
        for a, b in zip(path, path[1:]):
            cuts.add(pair(a, b))
        reached = comps[targets[path[0]]]
        sources.update(path)
        sources.update(reached)
        for v in reached:
            del targets[v]
    return cuts


def disk_cuts(verts, edges, label, weights=None, relief=None, forced=None):
    """Seam paths that open every multi-loop region into a disk.

    A tube wall is an annulus straight out of the partition and no merge
    can fix that, so cut it: a path joining two boundary loops opens the
    region without splitting it, one cut per extra loop. Genus is left to
    the engine, a handle needs a loop cut. forced edges already cut, so a
    wall a marked seam opens needs no second slit.
    """
    needs = {
        r: c for r, c in boundary_components(edges, label, forced).items() if len(c) > 1
    }
    if not needs:
        return set()

    adjacent = {r: collections.defaultdict(set) for r in needs}
    for (v0, v1), owners in edges.items():
        for r in {label[o] for o in owners} & needs.keys():
            adjacent[r][v0].add(v1)
            adjacent[r][v1].add(v0)

    cuts = set()
    for region, comps in needs.items():
        cuts |= connect_loops(verts, adjacent[region], comps, weights, relief)
    return cuts
