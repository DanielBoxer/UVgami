import collections
import heapq

import numpy

from .cancel import check_cancelled
from .cuts import snap_paths
from .mesh import face_edges

# dense vertices looked up between two progress reports
LOOKUP_CHUNK = 20000
# each stage's measured share of the transfer, the straightening is the rest
LOOKUP_SHARE = 0.08
TEAR_SHARE = 0.10
REDRAW_SHARE = 0.11
WELD_SHARE = 0.13
ABSORB_SHARE = 0.29

# a vertex's corners closer than this share of its mean uv edge are one uv
WELD_FRACTION = 0.5


# (low, high) vertex pairs. a boundary edge has one face and never counts
def uv_tears(faces, corner_uvs, tolerance=0.0):
    sizes = [len(face) for face in faces]
    corners = numpy.fromiter(
        (v for face in faces for v in face), dtype=numpy.int64, count=sum(sizes)
    )
    uvs = numpy.array(
        [uv for face in corner_uvs for uv in face], dtype=numpy.float64
    ).reshape(-1, 2)
    following = following_corners(sizes)
    return _edge_tears(corners, corners[following], uvs, uvs[following], tolerance)


def edge_adjacency(edges):
    adjacent = collections.defaultdict(set)
    for a, b in numpy.asarray(edges).reshape(-1, 2).tolist():
        adjacent[a].add(b)
        adjacent[b].add(a)
    return adjacent


def snap_cuts(verts, edges, mapped, cuts):
    return snap_paths(verts, edge_adjacency(edges), mapped, cuts)


# dense positions and normals in the proxy's own space
def _proxy_space(dense, proxy):
    matrix = numpy.linalg.inv(proxy["matrix"]) @ numpy.asarray(
        dense["matrix"], dtype=numpy.float64
    )
    positions = numpy.asarray(dense["positions"], dtype=numpy.float64).reshape(-1, 3)
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
    normals = numpy.asarray(dense["normals"], dtype=numpy.float64).reshape(-1, 3)
    normals = normals @ numpy.linalg.inv(matrix[:3, :3])
    return positions, normals


# the proxy is engine output, so its faces are triangles
class AffineMaps:
    def __init__(self, positions, faces, corner_uvs):
        positions = numpy.asarray(positions, dtype=numpy.float64).reshape(-1, 3)
        first = numpy.array([face[:3] for face in faces], dtype=numpy.int64)
        uvs = numpy.array([uv[:3] for uv in corner_uvs], dtype=numpy.float64)
        corners = positions[first]
        self.origin = corners[:, 0]
        self.origin_uv = uvs[:, 0]
        edges = numpy.stack(
            [corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], axis=2
        )
        uv_edges = numpy.stack([uvs[:, 1] - uvs[:, 0], uvs[:, 2] - uvs[:, 0]], axis=2)
        self.jacobian = uv_edges @ numpy.linalg.pinv(edges)

    # face and positions paired
    def uv(self, face, positions):
        offset = numpy.asarray(positions) - self.origin[face]
        return self.origin_uv[face] + numpy.einsum(
            "nij,nj->ni", self.jacobian[face], offset
        )


# two faces either side of a cut give their shared vertex different uvs
def proxy_links(faces, corner_uvs):
    at_uv = collections.defaultdict(list)
    for f, (face, uvs) in enumerate(zip(faces, corner_uvs)):
        for v, uv in zip(face, uvs):
            at_uv[(v, tuple(uv))].append(f)
    pairs = set()
    for group in at_uv.values():
        for f in group:
            for g in group:
                pairs.add((f << 32) | g)
    return numpy.array(sorted(pairs), dtype=numpy.int64)


# two faces around one vertex give it different uvs
def proxy_splits(faces, corner_uvs):
    at_vertex = collections.defaultdict(list)
    for f, (face, uvs) in enumerate(zip(faces, corner_uvs)):
        for v, uv in zip(face, uvs):
            at_vertex[v].append((f, tuple(uv)))
    pairs = set()
    for group in at_vertex.values():
        for f, uv_f in group:
            for g, uv_g in group:
                if uv_f != uv_g:
                    pairs.add((f << 32) | g)
    return numpy.array(sorted(pairs), dtype=numpy.int64)


# their maps agree all along that edge, so the difference is a crack, not a cut
def proxy_edge_links(faces, corner_uvs):
    pairs = set()
    for (u, v), owners in face_edges(faces).items():
        if len(owners) != 2:
            continue
        f, g = owners
        same_u = corner_uvs[f][faces[f].index(u)] == corner_uvs[g][faces[g].index(u)]
        same_v = corner_uvs[f][faces[f].index(v)] == corner_uvs[g][faces[g].index(v)]
        if same_u and same_v:
            pairs.add((f << 32) | g)
            pairs.add((g << 32) | f)
    return numpy.array(sorted(pairs), dtype=numpy.int64)


def _linked(links, f, g):
    packed = (numpy.asarray(f, dtype=numpy.int64) << 32) | numpy.asarray(
        g, dtype=numpy.int64
    )
    if not len(links):
        return numpy.zeros(numpy.shape(packed), dtype=bool)
    at = numpy.searchsorted(links, packed)
    return (at < len(links)) & (links[numpy.minimum(at, len(links) - 1)] == packed)


# cuts touching a proxy face's vertices kept for the crossing test
CUTS_PER_FACE = 16
# dense edges tested for cut crossings at once
CROSSING_CHUNK = 100000
# a point this close to a cut line, as a share of the cut's length, is on it
ON_LINE = 1e-6


# judged in the first point's proxy face plane, against the cuts at its vertices
class CutCrossings:
    def __init__(self, positions, faces, corner_uvs):
        positions = numpy.asarray(positions, dtype=numpy.float64).reshape(-1, 3)
        cuts = numpy.array(
            sorted(uv_tears(faces, corner_uvs)), dtype=numpy.int64
        ).reshape(-1, 2)
        at_vertex = collections.defaultdict(list)
        for i, (a, b) in enumerate(cuts.tolist()):
            at_vertex[a].append(i)
            at_vertex[b].append(i)
        per_face = numpy.full((len(faces), CUTS_PER_FACE), -1, dtype=numpy.int64)
        for f, face in enumerate(faces):
            nearby = sorted({i for v in face for i in at_vertex[v]})[:CUTS_PER_FACE]
            per_face[f, : len(nearby)] = nearby
        self.per_face = per_face
        self.cut_ends = positions[cuts] if len(cuts) else numpy.empty((0, 2, 3))
        edges = face_edges(faces)
        self.owners = numpy.array(
            [edges[(a, b)][:2] for a, b in cuts.tolist()], dtype=numpy.int64
        ).reshape(-1, 2)
        # the crossing test misses a fold that puts two uv sides in one plane
        self.pairs = proxy_splits(faces, corner_uvs)
        first = numpy.array([face[:3] for face in faces], dtype=numpy.int64)
        corners = positions[first]
        self.origin = corners[:, 0]
        self.middle = corners.mean(axis=1)
        along = corners[:, 1] - corners[:, 0]
        normal = numpy.cross(along, corners[:, 2] - corners[:, 0])
        along /= numpy.maximum(numpy.linalg.norm(along, axis=1), 1e-30)[:, None]
        normal /= numpy.maximum(numpy.linalg.norm(normal, axis=1), 1e-30)[:, None]
        self.axes = numpy.stack([along, numpy.cross(normal, along)], axis=1)
        # most proxy faces touch no cut
        width = int(numpy.max((per_face >= 0).sum(axis=1), initial=0))
        candidates = per_face[:, :width]
        self.valid = candidates >= 0
        self.touches_cut = self.valid.any(axis=1)
        ends = self.cut_ends[numpy.maximum(candidates, 0)]
        face = numpy.repeat(numpy.arange(len(faces)), width)
        shape = (len(faces), width, 2)
        self.cut_p = self._flat(face, ends[:, :, 0].reshape(-1, 3)).reshape(shape)
        self.cut_q = self._flat(face, ends[:, :, 1].reshape(-1, 3)).reshape(shape)
        self.flat_middle = self._flat(numpy.arange(len(faces)), self.middle)

    def _flat(self, faces, points):
        return numpy.einsum("nij,nj->ni", self.axes[faces], points - self.origin[faces])

    # a point on the cut line takes the side its own face's middle is on
    def crosses(self, faces_a, faces_b, points_a, points_b):
        crossed = numpy.zeros(len(faces_a), dtype=bool)
        if not len(self.cut_ends):
            return crossed
        active = numpy.flatnonzero(self.touches_cut[faces_a])
        for start in range(0, len(active), CROSSING_CHUNK):
            rows = active[start : start + CROSSING_CHUNK]
            fa, fb = faces_a[rows], faces_b[rows]
            valid = self.valid[fa]
            p = self.cut_p[fa]
            q = self.cut_q[fa]
            a = self._flat(fa, points_a[rows])[:, None, :]
            b = self._flat(fa, points_b[rows])[:, None, :]
            middle_a = self.flat_middle[fa][:, None, :]
            middle_b = self._flat(fa, self.middle[fb])[:, None, :]

            def orient(u, v, w):
                return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (
                    v[..., 1] - u[..., 1]
                ) * (w[..., 0] - u[..., 0])

            def side(x, middle):
                own = orient(p, q, x)
                on_line = numpy.abs(own) <= ON_LINE * numpy.sum((q - p) ** 2, axis=-1)
                return numpy.where(on_line, orient(p, q, middle), own)

            apart = side(a, middle_a) * side(b, middle_b) < 0
            # a segment through a cut's end vertex spans it whichever way rounding falls
            touch = ON_LINE * numpy.sum((b - a) ** 2, axis=-1)
            spans = (
                (orient(a, b, p) * orient(a, b, q) <= 0)
                | (numpy.abs(orient(a, b, p)) <= touch)
                | (numpy.abs(orient(a, b, q)) <= touch)
            )
            crossed[rows] = numpy.any(apart & spans & valid, axis=1)
        return crossed

    def nearest(self, faces, points):
        candidates = self.per_face[faces]
        valid = candidates >= 0
        if not valid.any():
            return numpy.full(len(faces), -1, dtype=numpy.int64)
        ends = self.cut_ends[numpy.maximum(candidates, 0)]
        p, q = ends[:, :, 0], ends[:, :, 1]
        along = q - p
        offset = points[:, None, :] - p
        t = numpy.einsum("nkj,nkj->nk", offset, along) / numpy.maximum(
            numpy.einsum("nkj,nkj->nk", along, along), 1e-30
        )
        closest = p + numpy.clip(t, 0, 1)[:, :, None] * along
        distance = numpy.linalg.norm(closest - points[:, None, :], axis=2)
        distance[~valid] = numpy.inf
        best = candidates[numpy.arange(len(faces)), distance.argmin(axis=1)]
        return numpy.where(valid.any(axis=1), best, -1)


class ProxyMap:
    def __init__(self, proxy):
        faces, corner_uvs = proxy["faces"], proxy["corner_uvs"]
        self.maps = AffineMaps(proxy["positions"], faces, corner_uvs)
        self.links = proxy_links(faces, corner_uvs)
        self.edge_links = proxy_edge_links(faces, corner_uvs)
        self.crossings = CutCrossings(proxy["positions"], faces, corner_uvs)


# faces are laid out one after another
def following_corners(face_sizes):
    sizes = numpy.asarray(face_sizes, dtype=numpy.int64)
    starts = numpy.cumsum(sizes) - sizes
    face_of = numpy.repeat(numpy.arange(len(sizes)), sizes)
    local = numpy.arange(len(face_of)) - starts[face_of]
    return starts[face_of] + (local + 1) % sizes[face_of]


# each corner's face, the next corner around it, and the twin across the edge
class DenseMesh:
    def __init__(self, corners, face_sizes, positions):
        self.corners = numpy.asarray(corners, dtype=numpy.int64)
        self.sizes = numpy.asarray(face_sizes, dtype=numpy.int64)
        self.positions = positions
        self.starts = numpy.cumsum(self.sizes) - self.sizes
        self.face_of = numpy.repeat(numpy.arange(len(self.sizes)), self.sizes)
        self.following = following_corners(self.sizes)
        self.preceding = numpy.empty_like(self.following)
        self.preceding[self.following] = numpy.arange(len(self.following))
        self.twin = _twins(self.corners, self.following)
        self.lengths = numpy.linalg.norm(
            positions[self.corners] - positions[self.corners[self.following]], axis=1
        )
        self._ring_order = numpy.argsort(self.corners, kind="stable")
        self._ring_sorted = self.corners[self._ring_order]
        self._one_corner = numpy.zeros(len(positions), dtype=numpy.int64)
        self._one_corner[self.corners] = numpy.arange(len(self.corners))

    def face(self, f):
        start = int(self.starts[f])
        return range(start, start + int(self.sizes[f]))

    # an end vertex has corners that disagree
    def suspect(self, corner_uvs):
        split = numpy.zeros(len(self._one_corner), dtype=bool)
        differs = numpy.any(
            corner_uvs != corner_uvs[self._one_corner[self.corners]], axis=1
        )
        split[self.corners[differs]] = True
        return split[self.corners] | split[self.corners[self.following]]

    def corners_of(self, faces):
        faces = numpy.asarray(faces, dtype=numpy.int64)
        return _ranges(self.starts[faces], self.sizes[faces])

    # for each corner, the position in vertices of the vertex it is of
    def rings(self, vertices):
        vertices = numpy.asarray(vertices, dtype=numpy.int64)
        lo = numpy.searchsorted(self._ring_sorted, vertices)
        sizes = numpy.searchsorted(self._ring_sorted, vertices + 1) - lo
        of = numpy.repeat(numpy.arange(len(vertices)), sizes)
        return self._ring_order[_ranges(lo, sizes)], of


# the ranges starts[i] to starts[i] + sizes[i], one after another
def _ranges(starts, sizes):
    offsets = numpy.cumsum(sizes) - sizes
    return numpy.repeat(starts - offsets, sizes) + numpy.arange(int(sizes.sum()))


def _edge_keys(tail, head):
    return (numpy.minimum(tail, head) << 32) | numpy.maximum(tail, head)


# -1 on a boundary or non-manifold edge
def _twins(corners, following):
    keys = _edge_keys(corners, corners[following])
    order = numpy.argsort(keys, kind="stable")
    _, first, counts = numpy.unique(keys[order], return_index=True, return_counts=True)
    paired = first[counts == 2]
    twin = numpy.full(len(corners), -1, dtype=numpy.int64)
    a, b = order[paired], order[paired + 1]
    twin[a] = b
    twin[b] = a
    return twin


# on a ridge the ends land inside both faces of the crease and no plane shows it
def _torn(crossings, face_of_vertex, positions, tail, head):
    faces_tail, faces_head = face_of_vertex[tail], face_of_vertex[head]
    return (
        _linked(crossings.pairs, faces_tail, faces_head)
        | crossings.crosses(faces_tail, faces_head, positions[tail], positions[head])
        | crossings.crosses(faces_head, faces_tail, positions[head], positions[tail])
    )


# the corner agreeing with the most others keeps its map
def _redraw_torn_faces(proxy_map, face_of_vertex, surface, uvs, mesh, torn):
    corners = mesh.corners
    corner_uvs = uvs[corners].copy()
    drawn_by = face_of_vertex[corners].copy()
    torn_faces = numpy.flatnonzero(torn)
    if not len(torn_faces):
        return corner_uvs, drawn_by
    crossings = proxy_map.crossings
    # every ordered corner pair of every torn face, crossing tested at once
    pair_sizes = mesh.sizes[torn_faces] ** 2
    pair_starts = numpy.cumsum(pair_sizes) - pair_sizes
    first = numpy.concatenate(
        [numpy.repeat(corners[mesh.face(f)], len(mesh.face(f))) for f in torn_faces]
    )
    second = numpy.concatenate(
        [numpy.tile(corners[mesh.face(f)], len(mesh.face(f))) for f in torn_faces]
    )
    crossed = crossings.crosses(
        face_of_vertex[first], face_of_vertex[second], surface[first], surface[second]
    )
    for f, pair_start in zip(torn_faces.tolist(), pair_starts.tolist()):
        ring = mesh.face(f)
        size = len(ring)
        verts = corners[ring]
        faces = face_of_vertex[verts]
        cut_between = crossed[pair_start : pair_start + size * size].reshape(size, size)
        cut_between |= cut_between.T
        agree = ~cut_between & ~_linked(crossings.pairs, faces[:, None], faces[None, :])
        reference = int(agree.sum(axis=1).argmax())
        moved = numpy.flatnonzero(~agree[reference])
        corner_uvs[ring.start + moved] = proxy_map.maps.uv(
            numpy.full(len(moved), faces[reference]), surface[verts[moved]]
        )
        drawn_by[ring.start + moved] = faces[reference]
    return corner_uvs, drawn_by


# either side of one uncut edge, or within weld distance, becomes one value
def _weld_vertices(corner_uvs, drawn_by, proxy_map, mesh, vertices):
    corners, of = mesh.rings(vertices)
    if not len(corners):
        return
    uvs = corner_uvs[corners]
    edge_lengths = numpy.linalg.norm(uvs - corner_uvs[mesh.following[corners]], axis=1)
    sizes = numpy.bincount(of, minlength=len(vertices))
    tolerance = WELD_FRACTION * numpy.bincount(of, edge_lengths, len(vertices)) / sizes
    # every ordered corner pair within one ring
    pair_counts = sizes**2
    pair_of = numpy.repeat(numpy.arange(len(vertices)), pair_counts)
    pair_starts = numpy.cumsum(pair_counts) - pair_counts
    local = numpy.arange(len(pair_of)) - pair_starts[pair_of]
    starts = numpy.cumsum(sizes) - sizes
    a = starts[pair_of] + local // sizes[pair_of]
    b = starts[pair_of] + local % sizes[pair_of]
    joins = _linked(proxy_map.edge_links, drawn_by[corners[a]], drawn_by[corners[b]])
    joins |= numpy.linalg.norm(uvs[a] - uvs[b], axis=1) <= tolerance[pair_of]
    cluster = _components(len(corners), a[joins], b[joins])
    sums = numpy.zeros((len(corners), 2))
    numpy.add.at(sums, cluster, uvs)
    counts = numpy.bincount(cluster, minlength=len(corners))
    corner_uvs[corners] = (sums / numpy.maximum(counts, 1)[:, None])[cluster]


# the (a, b) pairs hold both orders of every pair
def _components(count, a, b):
    label = numpy.arange(count)
    while True:
        grown = label.copy()
        numpy.minimum.at(grown, a, label[b])
        if numpy.array_equal(grown, label):
            return label
        label = grown


# the rest share their vertex uv already and have no gap to close
def _weld(corner_uvs, drawn_by, native, proxy_map, mesh):
    moved = numpy.flatnonzero(numpy.any(corner_uvs != native[mesh.corners], axis=1))
    _weld_vertices(
        corner_uvs, drawn_by, proxy_map, mesh, numpy.unique(mesh.corners[moved])
    )
    return corner_uvs


# it carries the uv gradient the mesh has there, a proxy face's can be steeper
def _face_map(corner_uvs, mesh, g):
    start = int(mesh.starts[g])
    p = mesh.positions[mesh.corners[start : start + 3]]
    u = corner_uvs[start : start + 3]
    edge_1, edge_2 = (p[1] - p[0]).tolist(), (p[2] - p[0]).tolist()
    normal = _cross(edge_1, edge_2)
    area = sum(a * a for a in normal)
    # the dual basis of the two edges in the face's plane
    scale = 1 / area if area else 0.0
    dual_1 = [a * scale for a in _cross(edge_2, normal)]
    dual_2 = [a * scale for a in _cross(normal, edge_1)]
    jacobian = numpy.outer(u[1] - u[0], dual_1) + numpy.outer(u[2] - u[0], dual_2)

    def at(points):
        return u[0] + (numpy.asarray(points) - p[0]) @ jacobian.T

    return at


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


# g's own uv where they share a vertex, g's map elsewhere
def _draw_like_neighbour(corner_uvs, drawn_by, mesh, f, g):
    at = _face_map(corner_uvs, mesh, g)
    known = {int(mesh.corners[c]): corner_uvs[c] for c in mesh.face(g)}
    for c in mesh.face(f):
        v = int(mesh.corners[c])
        corner_uvs[c] = known[v] if v in known else at(mesh.positions[[v]])[0]
        drawn_by[c] = drawn_by[int(mesh.starts[g])]


# where a cut runs down a triangle strip the sides alternate and the seam zigzags
def _absorb_stray_faces(corner_uvs, drawn_by, proxy_map, mesh):
    corners, following, twin = mesh.corners, mesh.following, mesh.twin
    paired = numpy.flatnonzero((twin >= 0) & mesh.suspect(corner_uvs))
    torn = paired[_torn_at_twin(corner_uvs, mesh, paired)]
    seam = numpy.zeros(len(corners), dtype=bool)
    seam[torn] = True

    def across(c):
        t = twin[c]
        if corners[t] == corners[c]:
            return corner_uvs[t], corner_uvs[following[t]], drawn_by[t]
        return corner_uvs[following[t]], corner_uvs[t], drawn_by[t]

    # an edge joins once it shares a corner with the side and agrees there
    def side_of(anchor, seams):
        tail_uv, head_uv, face = across(anchor)
        tolerance = WELD_FRACTION * numpy.linalg.norm(tail_uv - head_uv)
        target = {anchor: tail_uv, int(following[anchor]): head_uv}
        joined = [anchor]
        grew = True
        while grew:
            grew = False
            for c in seams:
                if c in joined:
                    continue
                other_tail, other_head, _ = across(c)
                pairs = ((c, other_tail), (int(following[c]), other_head))
                shared = [(corner, uv) for corner, uv in pairs if corner in target]
                if not shared:
                    continue
                agrees = all(
                    numpy.linalg.norm(target[corner] - uv) <= tolerance
                    for corner, uv in shared
                )
                if agrees:
                    for corner, uv in pairs:
                        target.setdefault(corner, uv)
                    joined.append(c)
                    grew = True
        return target, joined, face

    def move(f):
        ring = mesh.face(f)
        seams = [c for c in ring if seam[c]]
        if not seams:
            return False
        plain = [c for c in ring if twin[c] >= 0 and c not in seams]
        before = mesh.lengths[seams].sum()
        best = None
        for anchor in seams:
            target, joined, face = side_of(anchor, seams)
            after = (
                mesh.lengths[plain].sum()
                + mesh.lengths[[c for c in seams if c not in joined]].sum()
            )
            if after < before and (best is None or after < best[0]):
                best = (after, target, face, anchor)
        if best is None:
            return False
        _, target, face, anchor = best
        at = _face_map(corner_uvs, mesh, int(mesh.face_of[twin[anchor]]))
        for c in ring:
            if c not in target:
                target[c] = at(mesh.positions[corners[[c]]])[0]
        for corner, uv in target.items():
            corner_uvs[corner] = uv
            drawn_by[corner] = face
        _weld_vertices(corner_uvs, drawn_by, proxy_map, mesh, corners[ring])
        # the weld can change every edge at the face's vertices
        ring_corners, _ = mesh.rings(corners[ring])
        touched = numpy.concatenate([ring_corners, mesh.preceding[ring_corners]])
        touched = touched[twin[touched] >= 0]
        seam[touched] = _torn_at_twin(corner_uvs, mesh, touched)
        return True

    # no re-queueing, a moved face's neighbours erode small islands corner by corner
    for f in numpy.unique(mesh.face_of[torn]).tolist():
        move(f)
    return corner_uvs


# faces either side of a seam run the straightened path may move through
BAND_RINGS = 3
# rings past the band the side floods extend, clear of any face the new path encloses
SEED_RINGS = 2
# seam runs shorter than this stay as they are
STRAIGHTEN_MIN_EDGES = 3


# a seam means the face across the edge gives either end of it another uv
def _torn_at_twin(corner_uvs, mesh, paired):
    corners, following, twins = mesh.corners, mesh.following, mesh.twin[paired]
    same_direction = (corners[twins] == corners[paired])[:, None]
    across_tail = numpy.where(
        same_direction, corner_uvs[twins], corner_uvs[following[twins]]
    )
    across_head = numpy.where(
        same_direction, corner_uvs[following[twins]], corner_uvs[twins]
    )
    return numpy.any(across_tail != corner_uvs[paired], axis=1) | numpy.any(
        across_head != corner_uvs[following[paired]], axis=1
    )


def _seam_corners(corner_uvs, mesh):
    twin = mesh.twin
    paired = numpy.flatnonzero(
        (twin >= 0) & (numpy.arange(len(twin)) < twin) & mesh.suspect(corner_uvs)
    )
    return paired[_torn_at_twin(corner_uvs, mesh, paired)]


# a short stretch on another cut is the nearest cut flipping at a proxy vertex
def _seam_runs(seam_corners, cut_of, mesh):
    corners, following = mesh.corners, mesh.following
    adjacent = collections.defaultdict(list)
    for i, c in enumerate(seam_corners.tolist()):
        a, b = int(corners[c]), int(corners[following[c]])
        adjacent[a].append((b, i))
        adjacent[b].append((a, i))
    used = numpy.zeros(len(seam_corners), dtype=bool)

    # vertices past v until a junction, a loose end or a used edge
    def extend(v, i):
        path = []
        while len(adjacent[v]) == 2:
            (w, j) = next((w, j) for w, j in adjacent[v] if j != i)
            if used[j]:
                break
            used[j] = True
            path.append((w, j))
            v, i = w, j
        return path

    runs = []
    for i, c in enumerate(seam_corners.tolist()):
        if used[i]:
            continue
        used[i] = True
        a, b = int(corners[c]), int(corners[following[c]])
        before = extend(a, i)
        after = extend(b, i)
        vertices = [w for w, _ in reversed(before)] + [a, b] + [w for w, _ in after]
        edges = [j for _, j in reversed(before)] + [i] + [j for _, j in after]
        ids = [int(cut_of[j]) for j in edges]
        stretches = []
        for k, cut in enumerate(ids):
            if stretches and stretches[-1][0] == cut:
                stretches[-1][1].append(k)
            else:
                stretches.append([cut, [k]])
        for k, (cut, members) in enumerate(stretches):
            if len(members) < STRAIGHTEN_MIN_EDGES and k > 0:
                stretches[k][0] = stretches[k - 1][0]
        for cut, members in stretches:
            for k in members:
                ids[k] = cut
        start = 0
        for k in range(1, len(ids) + 1):
            if k == len(ids) or ids[k] != ids[start]:
                if ids[start] >= 0:
                    runs.append(
                        (
                            vertices[start : k + 1],
                            [int(seam_corners[edges[m]]) for m in range(start, k)],
                            ids[start],
                        )
                    )
                start = k
    return runs


def _band(faces, mesh, rings_out):
    band = frontier = numpy.unique(faces)
    for _ in range(rings_out):
        twin = mesh.twin[mesh.corners_of(frontier)]
        grown = numpy.unique(mesh.face_of[twin[twin >= 0]])
        frontier = numpy.setdiff1d(grown, band, assume_unique=True)
        band = numpy.union1d(band, grown)
    return band


def _shortest_path(band, start, end, mesh):
    band_corners = mesh.corners_of(band)
    tails = mesh.corners[band_corners].tolist()
    heads = mesh.corners[mesh.following[band_corners]].tolist()
    lengths = mesh.lengths[band_corners].tolist()
    adjacent = collections.defaultdict(list)
    for a, b, length in zip(tails, heads, lengths):
        adjacent[a].append((b, length))
        adjacent[b].append((a, length))
    best = {start: 0.0}
    came_from = {}
    queue = [(0.0, start)]
    while queue:
        cost, v = heapq.heappop(queue)
        if v == end:
            break
        if cost > best[v]:
            continue
        for w, length in adjacent[v]:
            total = cost + length
            if total < best.get(w, numpy.inf):
                best[w] = total
                came_from[w] = v
                heapq.heappush(queue, (total, w))
    if end not in best:
        return None
    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return path[::-1]


# -1 when the face across the edge is outside the band
class _Band:
    def __init__(self, faces, mesh):
        self.faces = faces
        self.corners = mesh.corners_of(faces)
        self.sizes = mesh.sizes[faces]
        self.starts = numpy.cumsum(self.sizes) - self.sizes
        self.face = numpy.repeat(numpy.arange(len(faces)), self.sizes)
        twin = mesh.twin[self.corners]
        self.has_twin = twin >= 0
        twin = numpy.maximum(twin, 0)
        across = mesh.face_of[twin]
        local = numpy.searchsorted(faces, across)
        inside = self.has_twin & (local < len(faces))
        inside &= faces[numpy.minimum(local, len(faces) - 1)] == across
        self.across = numpy.where(inside, local, -1)
        self.twin_at = numpy.where(inside, numpy.searchsorted(self.corners, twin), -1)

    # -1 where no seed reaches, None when both sides reach one face
    def sides(self, seeds, seed_sides, blocked):
        passable = (self.across >= 0) & ~blocked
        passable[passable] &= ~blocked[self.twin_at[passable]]
        component = _components(
            len(self.faces), self.face[passable], self.across[passable]
        )
        low = numpy.full(len(self.faces), 2)
        numpy.minimum.at(low, component[seeds], seed_sides)
        high = numpy.full(len(self.faces), -1)
        numpy.maximum.at(high, component[seeds], seed_sides)
        if numpy.any(low < high):
            return None
        return high[component]

    def neighbours(self, f):
        start = self.starts[f]
        return self.across[start : start + self.sizes[f]]


# the side labelling puts a seam wherever the nearest proxy face changes
def _straighten_seams(corner_uvs, drawn_by, proxy_map, mesh):
    corners, following, twin = mesh.corners, mesh.following, mesh.twin
    face_of, positions = mesh.face_of, mesh.positions
    cuts = proxy_map.crossings
    seam_corners = _seam_corners(corner_uvs, mesh)
    seam = numpy.zeros(len(corners), dtype=bool)
    seam[seam_corners] = True
    seam[twin[seam_corners]] = True
    midpoints = (
        positions[corners[seam_corners]] + positions[corners[following[seam_corners]]]
    ) / 2
    cut_of = cuts.nearest(drawn_by[seam_corners], midpoints)
    on_run = numpy.zeros(len(corners), dtype=bool)

    # walking the run in order
    def side_faces(vertices, run_corners):
        left, right = [], []
        for v, c in zip(vertices, run_corners):
            t = int(twin[c])
            mine, other = int(face_of[c]), int(face_of[t])
            if corners[c] == v:
                left.append(mine)
                right.append(other)
            else:
                left.append(other)
                right.append(mine)
        return left, right

    # the owner of the cut on the side those faces are drawn on
    def rim_face(cut, faces):
        drawn = numpy.unique(drawn_by[[int(mesh.starts[f]) for f in faces]])
        owners = cuts.owners[cut]
        matches = [
            int(owner)
            for owner in owners
            if numpy.any(_linked(proxy_map.links, numpy.full(len(drawn), owner), drawn))
        ]
        return matches[0] if len(matches) == 1 else None

    for vertices, run_corners, cut in _seam_runs(seam_corners, cut_of, mesh):
        if len(run_corners) < STRAIGHTEN_MIN_EDGES or vertices[0] == vertices[-1]:
            continue
        left, right = side_faces(vertices, run_corners)
        inner = _band(left + right, mesh, BAND_RINGS)
        path = _shortest_path(inner, vertices[0], vertices[-1], mesh)
        if path is None or path == vertices:
            continue
        band = _Band(_band(inner, mesh, SEED_RINGS), mesh)
        seeds = numpy.searchsorted(band.faces, left + right)
        seed_sides = numpy.repeat([0, 1], [len(left), len(right)])
        was = band.sides(seeds, seed_sides, seam[band.corners])
        if was is None:
            continue
        path = numpy.array(path, dtype=numpy.int64)
        on_path = numpy.isin(
            _edge_keys(corners[band.corners], corners[following[band.corners]]),
            _edge_keys(path[:-1], path[1:]),
        )
        on_run[run_corners] = on_run[twin[run_corners]] = True
        blocked = (seam[band.corners] & ~on_run[band.corners]) | on_path
        on_run[run_corners] = on_run[twin[run_corners]] = False
        rim = numpy.zeros(len(band.faces), dtype=bool)
        rim[band.face[band.has_twin & (band.across < 0)]] = True
        outer = numpy.flatnonzero(rim & (was >= 0))
        now = band.sides(outer, was[outer], blocked)
        if now is None:
            continue
        moved = numpy.flatnonzero((was >= 0) & (now >= 0) & (was != now)).tolist()
        if not moved:
            continue
        rims = [rim_face(cut, left), rim_face(cut, right)]
        if rims[0] is None or rims[1] is None or rims[0] == rims[1]:
            continue
        # drawn like a neighbour already on the new side, outermost first
        same_side = {
            f: [g for g in band.neighbours(f).tolist() if g >= 0 and now[g] == now[f]]
            for f in moved
        }
        waiting = set(moved)
        while waiting:
            progressed = False
            for f in sorted(waiting):
                for g in same_side[f]:
                    if g not in waiting:
                        _draw_like_neighbour(
                            corner_uvs,
                            drawn_by,
                            mesh,
                            int(band.faces[f]),
                            int(band.faces[g]),
                        )
                        waiting.discard(f)
                        progressed = True
                        break
            if not progressed:
                for f in waiting:
                    face = rims[now[f]]
                    ring = mesh.face(int(band.faces[f]))
                    corner_uvs[ring] = proxy_map.maps.uv(
                        numpy.full(len(ring), face), positions[corners[ring]]
                    )
                    drawn_by[ring] = face
                waiting.clear()
        moved_corners = mesh.corners_of(band.faces[moved])
        _weld_vertices(
            corner_uvs, drawn_by, proxy_map, mesh, numpy.unique(corners[moved_corners])
        )
        # the band's edges are the only ones whose seam state can have changed
        paired = band.corners[band.has_twin]
        seam[paired] = _torn_at_twin(corner_uvs, mesh, paired)
    return corner_uvs


# edges given as corner rows, the vertex and uv at each end
def _edge_tears(tail, head, tail_uv, head_uv, tolerance):
    low_first = (tail < head)[:, None]
    at_low = numpy.where(low_first, tail_uv, head_uv)
    at_high = numpy.where(low_first, head_uv, tail_uv)
    keys = _edge_keys(tail, head)
    unique, first, group = numpy.unique(keys, return_index=True, return_inverse=True)
    agrees = numpy.all(numpy.abs(at_low - at_low[first][group]) <= tolerance, axis=1)
    agrees &= numpy.all(numpy.abs(at_high - at_high[first][group]) <= tolerance, axis=1)
    torn = unique[numpy.unique(group[~agrees])]
    return {(int(key >> 32), int(key & 0xFFFFFFFF)) for key in torn.tolist()}


# vertices renumbered from zero. also returns each new vertex's old index
def dense_subset(dense, faces):
    sizes = numpy.asarray(dense["face_sizes"], dtype=numpy.int64)
    starts = numpy.cumsum(sizes) - sizes
    faces = numpy.asarray(faces, dtype=numpy.int64)
    kept_sizes = sizes[faces]
    kept_starts = numpy.cumsum(kept_sizes) - kept_sizes
    offsets = numpy.arange(int(kept_sizes.sum())) - numpy.repeat(
        kept_starts, kept_sizes
    )
    corners = numpy.asarray(dense["corners"], dtype=numpy.int64)
    corners = corners[numpy.repeat(starts[faces], kept_sizes) + offsets]
    used, local = numpy.unique(corners, return_inverse=True)
    subset = {
        "positions": numpy.asarray(dense["positions"], dtype=numpy.float64)[used],
        "normals": numpy.asarray(dense["normals"], dtype=numpy.float64)[used],
        "matrix": dense["matrix"],
        "corners": local,
        "face_sizes": kept_sizes,
    }
    return subset, used


# nearest_faces gives each dense vertex a proxy face and the point on it
def transfer_projected(dense, proxy, nearest_faces, progress=None, cancelled=None):
    done = 0.0

    def report(fraction):
        if progress is not None:
            progress(fraction)

    def finished(share):
        nonlocal done
        done += share
        report(done)

    positions, normals = _proxy_space(dense, proxy)
    proxy_map = ProxyMap(proxy)

    face_of_vertex = numpy.empty(len(positions), dtype=numpy.int64)
    surface = numpy.empty((len(positions), 3))
    for start in range(0, len(positions), LOOKUP_CHUNK):
        check_cancelled(cancelled)
        stop = start + LOOKUP_CHUNK
        face_of_vertex[start:stop], surface[start:stop] = nearest_faces(
            positions[start:stop], normals[start:stop]
        )
        report(LOOKUP_SHARE * min(stop, len(positions)) / max(len(positions), 1))
    finished(LOOKUP_SHARE)
    # a map continued off its face diverges at a crease
    uvs = proxy_map.maps.uv(face_of_vertex, surface)

    mesh = DenseMesh(dense["corners"], dense["face_sizes"], positions)
    torn_corner = _torn(
        proxy_map.crossings,
        face_of_vertex,
        surface,
        mesh.corners,
        mesh.corners[mesh.following],
    )
    torn_face = numpy.zeros(len(mesh.sizes), dtype=bool)
    torn_face[mesh.face_of[torn_corner]] = True
    finished(TEAR_SHARE)

    check_cancelled(cancelled)
    corner_uvs, drawn_by = _redraw_torn_faces(
        proxy_map, face_of_vertex, surface, uvs, mesh, torn_face
    )
    finished(REDRAW_SHARE)
    corner_uvs = _weld(corner_uvs, drawn_by, uvs, proxy_map, mesh)
    finished(WELD_SHARE)
    check_cancelled(cancelled)
    corner_uvs = _absorb_stray_faces(corner_uvs, drawn_by, proxy_map, mesh)
    finished(ABSORB_SHARE)
    check_cancelled(cancelled)
    corner_uvs = _straighten_seams(corner_uvs, drawn_by, proxy_map, mesh)
    suspect = numpy.flatnonzero(mesh.suspect(corner_uvs))
    seams = _edge_tears(
        mesh.corners[suspect],
        mesh.corners[mesh.following[suspect]],
        corner_uvs[suspect],
        corner_uvs[mesh.following[suspect]],
        0.0,
    )
    report(1.0)
    return seams, corner_uvs
