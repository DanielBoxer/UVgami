import collections
import math

from .cuts import connect_loops
from .islands import uv_topology
from .mesh import face_edges, find, island_groups, pair, uv_island_groups


# a dropped face with no route to a kept one could never mirror its seams back
def half_faces(verts, faces, axes, mirrors):
    by_verts = {tuple(sorted(face)): fi for fi, face in enumerate(faces)}

    def image(face, m):
        if any(v not in m for v in face):
            return None
        return by_verts.get(tuple(sorted(m[v] for v in face)))

    images = [[image(face, m) for m in mirrors] for face in faces]
    dropped = set()
    for mi, axis in enumerate(axes):
        for fi, face in enumerate(faces):
            gi = images[fi][mi]
            if gi is None or gi == fi:
                continue
            mine = sum(verts[v][axis] for v in face) / len(face)
            theirs = sum(verts[v][axis] for v in faces[gi]) / len(faces[gi])
            if (mine, fi) < (theirs, gi):
                dropped.add(fi)

    reached = set(range(len(faces))) - dropped
    queue = collections.deque(reached)
    while queue:
        fi = queue.popleft()
        for gi in images[fi]:
            if gi is not None and gi not in reached:
                reached.add(gi)
                queue.append(gi)
    return dropped & reached


# where the halves glue back together on the whole mesh
def interface_edges(faces, dropped, edges):
    return {
        key
        for key, owners in edges.items()
        if len(owners) == 2 and (owners[0] in dropped) != (owners[1] in dropped)
    }


def _interface_arcs(group, faces, edges, seams, interface):
    inside = set(group)
    keys = []
    for key, owners in edges.items():
        if key in interface and key not in seams and set(owners) <= inside:
            keys.append(key)
    parent = {}
    for a, b in keys:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb
    arcs = collections.defaultdict(set)
    for key in keys:
        arcs[find(parent, key[0])].add(key)
    return list(arcs.values())


def _arc_length(verts, arc):
    total = 0.0
    for a, b in arc:
        total += math.dist(verts[a], verts[b])
    return total


# glued along more than one interface arc the merge is a ring, not a disk
def open_merged(verts, faces, edges, seams, interface):
    result = set(seams)
    queue = collections.deque(island_groups(faces, result, edges))
    while queue:
        group = queue.popleft()
        if len(group) < 2:
            continue
        ec, loops = uv_topology(group, faces, edges, result)
        if ec == 1:
            continue
        arcs = _interface_arcs(group, faces, edges, result, interface)
        if arcs:
            result |= min(arcs, key=lambda arc: _arc_length(verts, arc))
        elif len(loops) > 1:
            adjacent = collections.defaultdict(set)
            for f in group:
                face = faces[f]
                n = len(face)
                for i in range(n):
                    key = pair(face[i], face[(i + 1) % n])
                    if len(edges[key]) == 2 and key not in result:
                        adjacent[key[0]].add(key[1])
                        adjacent[key[1]].add(key[0])
            cuts = connect_loops(verts, adjacent, loops)
            if not cuts:
                continue
            result |= cuts
        else:
            continue
        parent = list(range(len(group)))
        index_of = {f: i for i, f in enumerate(group)}
        for f in group:
            face = faces[f]
            n = len(face)
            for i in range(n):
                key = pair(face[i], face[(i + 1) % n])
                if key in result:
                    continue
                owners = edges[key]
                if len(owners) == 2 and owners[0] in index_of and owners[1] in index_of:
                    ra = find(parent, index_of[owners[0]])
                    rb = find(parent, index_of[owners[1]])
                    if ra != rb:
                        parent[ra] = rb
        pieces = collections.defaultdict(list)
        for i, f in enumerate(group):
            pieces[find(parent, i)].append(f)
        queue.extend(pieces.values())
    return result


# the maps may be partial, a seam outside their coverage stays as it is
def mirror_seams(seams, mirrors, edges):
    result = set(seams)
    queue = list(result)
    while queue:
        a, b = queue.pop()
        for m in mirrors:
            if a not in m or b not in m:
                continue
            key = pair(m[a], m[b])
            if key in edges and key not in result:
                result.add(key)
                queue.append(key)
    return result


# an island a map sends onto itself straddles the plane and stays put
def stack_mirrored(faces, uvs, mirrors):
    groups = uv_island_groups(faces, uvs, face_edges(faces))
    group_of = {}
    for gi, group in enumerate(groups):
        for fi in group:
            group_of[fi] = gi
    by_verts = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        by_verts[tuple(sorted(face))].append(fi)

    # the one island m sends this island onto whole, or None
    def image(gi, m):
        targets = set()
        for fi in groups[gi]:
            candidates = by_verts.get(tuple(sorted(m.get(v, -1) for v in faces[fi])))
            if not candidates:
                return None
            targets.update(group_of[c] for c in candidates)
        if len(targets) == 1:
            target = targets.pop()
            if len(groups[target]) == len(groups[gi]):
                return target
        return None

    links = collections.defaultdict(list)
    for gi in range(len(groups)):
        for m in mirrors:
            hi = image(gi, m)
            if hi is not None and hi != gi:
                links[gi].append((hi, m))

    assignments = []
    seen = set()
    for gi in range(len(groups)):
        if gi in seen:
            continue
        component = {gi}
        queue = [gi]
        while queue:
            for hi, _ in links[queue.pop()]:
                if hi not in component:
                    component.add(hi)
                    queue.append(hi)
        seen |= component
        if len(component) == 1:
            continue
        rep = min(component, key=lambda g: min(groups[g]))
        # composed[g] maps each rep-island vertex to its counterpart in g
        composed = {rep: {v: v for fi in groups[rep] for v in faces[fi]}}
        queue = [rep]
        while queue:
            current = queue.pop()
            for hi, m in links[current]:
                if hi not in composed:
                    composed[hi] = {v: m[w] for v, w in composed[current].items()}
                    queue.append(hi)
        for twin, vmap in composed.items():
            if twin == rep:
                continue
            for fi in groups[rep]:
                key = tuple(sorted(vmap[v] for v in faces[fi]))
                target = next(c for c in by_verts[key] if group_of[c] == twin)
                for corner, v in enumerate(faces[fi]):
                    assignments.append(
                        (target, faces[target].index(vmap[v]), fi, corner)
                    )
    return assignments
