"""Finish a proxy unwrap on plain data.

The proxy's uv map is read onto the dense mesh instead of unwrapping it
again. Each dense vertex takes the uv its nearest proxy face's affine map
gives its position, so the map is exact inside a proxy triangle and continues
linearly past its edges. A dense face straddling a proxy cut gets corners from
both sides of the cut, so it is redrawn through one side's map, and the seams
are then the edges the uvs are torn across. proxy.py adapts a Blender mesh
onto these calls."""

import collections
import heapq

import numpy

from .cancel import check_cancelled
from .cuts import snap_paths
from .mesh import face_edges

# dense vertices looked up between two progress reports
LOOKUP_CHUNK = 20000
LOOKUP_PROGRESS = 0.9

# a vertex's corners closer than this share of its mean uv edge are one uv
WELD_FRACTION = 0.5


def uv_tears(faces, corner_uvs, tolerance=0.0):
    """Edges whose faces put a shared corner at different uvs, as (low, high)
    vertex pairs. Boundary edges have one face and never count."""
    sizes = [len(face) for face in faces]
    corners = numpy.fromiter(
        (v for face in faces for v in face), dtype=numpy.int64, count=sum(sizes)
    )
    uvs = numpy.array(
        [uv for face in corner_uvs for uv in face], dtype=numpy.float64
    ).reshape(-1, 2)
    return _corner_tears(corners, following_corners(sizes), uvs, tolerance)


def edge_adjacency(edges):
    adjacent = collections.defaultdict(set)
    for a, b in numpy.asarray(edges).reshape(-1, 2).tolist():
        adjacent[a].add(b)
        adjacent[b].add(a)
    return adjacent


def snap_cuts(verts, edges, mapped, cuts):
    """Another mesh's cut network redrawn along this mesh's own edges."""
    return snap_paths(verts, edge_adjacency(edges), mapped, cuts)


def _proxy_space(dense, proxy):
    """Dense positions and normals in the proxy's own space."""
    matrix = numpy.linalg.inv(proxy["matrix"]) @ numpy.asarray(
        dense["matrix"], dtype=numpy.float64
    )
    positions = numpy.asarray(dense["positions"], dtype=numpy.float64).reshape(-1, 3)
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
    normals = numpy.asarray(dense["normals"], dtype=numpy.float64).reshape(-1, 3)
    normals = normals @ numpy.linalg.inv(matrix[:3, :3])
    return positions, normals


class AffineMaps:
    """Each proxy face's uv map as an affine function of position: exact on
    the face's plane and continued linearly off it. The proxy is engine
    output, so its faces are triangles and an ngon's map is its first three
    corners'."""

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

    def uv(self, face, positions):
        """Face face's map at each position, face and positions paired."""
        offset = numpy.asarray(positions) - self.origin[face]
        return self.origin_uv[face] + numpy.einsum(
            "nij,nj->ni", self.jacobian[face], offset
        )


def proxy_links(faces, corner_uvs):
    """Sorted packed (f, g) pairs of proxy faces that share a vertex at the
    same uv, which is every pair the proxy's map is continuous across. Two
    faces either side of a cut give their shared vertex different uvs."""
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


def proxy_edge_links(faces, corner_uvs):
    """Sorted packed (f, g) pairs of proxy faces either side of an edge that
    is not a cut. Their maps agree all along that edge, so a dense corner
    drawn through one and another through the other differ by a crack,
    never a cut."""
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


class CutCrossings:
    """Whether a straight segment between two dense points crosses one of
    the proxy's cuts. Judged in the first point's proxy face's plane against
    the cuts touching that face's vertices, which is every cut a segment
    inside one dense face can reach from there."""

    def __init__(self, positions, faces, cuts):
        positions = numpy.asarray(positions, dtype=numpy.float64).reshape(-1, 3)
        cuts = numpy.array(sorted(cuts), dtype=numpy.int64).reshape(-1, 2)
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
        # faces either side of one cut, packed like proxy_links
        either_side = numpy.concatenate([self.owners, self.owners[:, ::-1]])
        self.pairs = numpy.unique((either_side[:, 0] << 32) | either_side[:, 1])
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

    def crosses(self, faces_a, faces_b, points_a, points_b):
        """Whether each segment crosses a cut, judged in the first point's
        face's plane. A point lying on the cut line, common where the proxy
        kept an original edge and the dense vertices along it, takes the side
        its own face's middle is on."""
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

    def nearest(self, face, point):
        """The cut touching face nearest to point, -1 when none does."""
        candidates = self.per_face[face]
        candidates = candidates[candidates >= 0]
        if not len(candidates):
            return -1
        p = self.cut_ends[candidates, 0]
        q = self.cut_ends[candidates, 1]
        along = q - p
        t = numpy.einsum("ij,ij->i", point - p, along) / numpy.maximum(
            numpy.einsum("ij,ij->i", along, along), 1e-30
        )
        closest = p + numpy.clip(t, 0, 1)[:, None] * along
        return int(candidates[numpy.linalg.norm(closest - point, axis=1).argmin()])


class ProxyMap:
    """The proxy's uv map read as geometry: each face's affine map, the face
    pairs the map is continuous across, and the cuts it is torn along."""

    def __init__(self, proxy):
        faces, corner_uvs = proxy["faces"], proxy["corner_uvs"]
        self.maps = AffineMaps(proxy["positions"], faces, corner_uvs)
        self.links = proxy_links(faces, corner_uvs)
        self.edge_links = proxy_edge_links(faces, corner_uvs)
        self.crossings = CutCrossings(
            proxy["positions"], faces, uv_tears(faces, corner_uvs)
        )


def following_corners(face_sizes):
    """Each corner's next corner around its face, faces laid out one after
    another."""
    sizes = numpy.asarray(face_sizes, dtype=numpy.int64)
    starts = numpy.cumsum(sizes) - sizes
    face_of = numpy.repeat(numpy.arange(len(sizes)), sizes)
    local = numpy.arange(len(face_of)) - starts[face_of]
    return starts[face_of] + (local + 1) % sizes[face_of]


class DenseMesh:
    """The dense mesh as a corner table, positions in the proxy's space, with
    the topology every pass reads: each corner's face, the next corner
    around that face, and the twin corner on the face across the edge."""

    def __init__(self, corners, face_sizes, positions):
        self.corners = numpy.asarray(corners, dtype=numpy.int64)
        self.sizes = numpy.asarray(face_sizes, dtype=numpy.int64)
        self.positions = positions
        self.starts = numpy.cumsum(self.sizes) - self.sizes
        self.face_of = numpy.repeat(numpy.arange(len(self.sizes)), self.sizes)
        self.following = following_corners(self.sizes)
        self.twin = _twins(self.corners, self.following)
        self.lengths = numpy.linalg.norm(
            positions[self.corners] - positions[self.corners[self.following]], axis=1
        )
        self._ring_order = numpy.argsort(self.corners, kind="stable")
        self._ring_sorted = self.corners[self._ring_order]

    def face(self, f):
        """Face f's corners."""
        start = int(self.starts[f])
        return range(start, start + int(self.sizes[f]))

    def ring(self, v):
        """Vertex v's corners."""
        lo, hi = numpy.searchsorted(self._ring_sorted, [v, v + 1])
        return self._ring_order[lo:hi]


def _edge_keys(corners, following):
    tail, head = corners, corners[following]
    return (numpy.minimum(tail, head) << 32) | numpy.maximum(tail, head)


def _twins(corners, following):
    """Each corner's twin, the corner of the one other face on its edge, -1
    on a boundary or non-manifold edge."""
    keys = _edge_keys(corners, following)
    order = numpy.argsort(keys, kind="stable")
    _, first, counts = numpy.unique(keys[order], return_index=True, return_counts=True)
    paired = first[counts == 2]
    twin = numpy.full(len(corners), -1, dtype=numpy.int64)
    a, b = order[paired], order[paired + 1]
    twin[a] = b
    twin[b] = a
    return twin


def _torn(crossings, face_of_vertex, positions, tail, head):
    """Whether the dense edge from tail to head is torn: its ends are drawn
    by faces either side of one cut, or it crosses a cut seen from either
    end's face. On a ridge the ends land inside both faces of the crease and
    no plane shows a crossing, past it one plane can show it when the other
    does not."""
    faces_tail, faces_head = face_of_vertex[tail], face_of_vertex[head]
    return (
        _linked(crossings.pairs, faces_tail, faces_head)
        | crossings.crosses(faces_tail, faces_head, positions[tail], positions[head])
        | crossings.crosses(faces_head, faces_tail, positions[head], positions[tail])
    )


def _redraw_torn_faces(proxy_map, face_of_vertex, surface, uvs, mesh, torn):
    """Corner uvs with each torn face drawn through one side's map: the
    corner agreeing with the most others keeps its map, and every corner
    disagreeing with it is moved through that map. Also returns the proxy
    face each corner was drawn through."""
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


def _weld_ring(corner_uvs, drawn_by, proxy_map, mesh, v):
    """Corners of vertex v drawn through proxy faces either side of one
    uncut edge, or sitting within its weld distance of each other, set to
    one value."""
    ring = mesh.ring(v)
    edge_lengths = numpy.linalg.norm(
        corner_uvs[ring] - corner_uvs[mesh.following[ring]], axis=1
    )
    tolerance = WELD_FRACTION * edge_lengths.mean()
    uvs = corner_uvs[ring]
    faces = drawn_by[ring]
    joins = _linked(proxy_map.edge_links, faces[:, None], faces[None, :])
    joins |= numpy.linalg.norm(uvs[:, None] - uvs[None, :], axis=2) <= tolerance
    for cluster in _connected(joins):
        corner_uvs[ring[cluster]] = uvs[cluster].mean(axis=0)


def _connected(joins):
    """Index groups connected through a symmetric boolean matrix."""
    unassigned = numpy.ones(len(joins), dtype=bool)
    while unassigned.any():
        members = numpy.zeros(len(joins), dtype=bool)
        members[numpy.flatnonzero(unassigned)[-1]] = True
        grown = members | numpy.any(joins[members], axis=0)
        while grown.sum() > members.sum():
            members = grown
            grown = members | numpy.any(joins[members], axis=0)
        unassigned &= ~members
        yield numpy.flatnonzero(members)


def _weld(corner_uvs, drawn_by, native, proxy_map, mesh):
    """Every vertex with a moved corner welded. The rest share their vertex
    uv already and have no gap to close."""
    moved = numpy.flatnonzero(numpy.any(corner_uvs != native[mesh.corners], axis=1))
    for v in numpy.unique(mesh.corners[moved]).tolist():
        _weld_ring(corner_uvs, drawn_by, proxy_map, mesh, v)
    return corner_uvs


def _face_map(corner_uvs, mesh, g):
    """The uv map of dense face g as an affine function of position, from
    its first three corners. It carries the uv gradient the mesh has right
    there, where a proxy face's map can be far steeper."""
    ring = numpy.arange(int(mesh.starts[g]), int(mesh.starts[g]) + 3)
    p = mesh.positions[mesh.corners[ring]]
    u = corner_uvs[ring]
    jacobian = (u[1:] - u[0]).T @ numpy.linalg.pinv((p[1:] - p[0]).T)

    def at(points):
        return u[0] + (numpy.asarray(points) - p[0]) @ jacobian.T

    return at


def _draw_like_neighbour(corner_uvs, drawn_by, mesh, f, g):
    """Face f's corners set from neighbour g: g's own uv where they share a
    vertex, g's map elsewhere."""
    at = _face_map(corner_uvs, mesh, g)
    known = {int(mesh.corners[c]): corner_uvs[c] for c in mesh.face(g)}
    for c in mesh.face(f):
        v = int(mesh.corners[c])
        corner_uvs[c] = known[v] if v in known else at(mesh.positions[[v]])[0]
        drawn_by[c] = drawn_by[int(mesh.starts[g])]


def _absorb_stray_faces(corner_uvs, drawn_by, proxy_map, mesh):
    """A face goes over to the side across some of its seam edges when that
    leaves less seam, measured along the mesh. Where a cut runs down the
    middle of a strip of triangles they alternate sides and the seam zigzags
    along the diagonals, this pulls them onto one edge row, and a face that
    ended up as an island of its own is taken back. The face's corners take
    the uvs its new neighbours have there, a corner none of them has is
    drawn through their map, and the vertices are welded again."""
    corners, following, twin = mesh.corners, mesh.following, mesh.twin

    def across(c):
        """The twin face's uvs and map for corner c's tail and head."""
        t = twin[c]
        if corners[t] == corners[c]:
            return corner_uvs[t], corner_uvs[following[t]], drawn_by[t]
        return corner_uvs[following[t]], corner_uvs[t], drawn_by[t]

    def seam(c):
        if twin[c] < 0:
            return False
        tail_uv, head_uv, _ = across(c)
        return bool(
            numpy.any(tail_uv != corner_uvs[c])
            or numpy.any(head_uv != corner_uvs[following[c]])
        )

    def side_of(anchor, seams):
        """The corner uvs the face gets on its anchor edge's far side, with
        the seam edges whose far side is the same, and the map drawn by. An
        edge joins once it shares a corner with the side and agrees there,
        so an edge across the face is judged through the ones between."""
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
        seams = [c for c in ring if seam(c)]
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
        for c in ring:
            _weld_ring(corner_uvs, drawn_by, proxy_map, mesh, int(corners[c]))
        return True

    paired = numpy.flatnonzero(twin >= 0)
    torn = paired[_torn_at_twin(corner_uvs, mesh, paired)]
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


def _torn_at_twin(corner_uvs, mesh, paired):
    """Whether each paired corner's edge is a seam: the face across it gives
    either end of the edge another uv."""
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
    """Corners whose edge is a seam, one per edge."""
    twin = mesh.twin
    paired = numpy.flatnonzero((twin >= 0) & (numpy.arange(len(twin)) < twin))
    return paired[_torn_at_twin(corner_uvs, mesh, paired)]


def _seam_runs(seam_corners, cut_of, mesh):
    """Vertex paths of seam edges that follow one proxy cut, ending at a
    junction, a loose end or a change of cut. Each with its corners and its
    cut. A stretch of fewer than STRAIGHTEN_MIN_EDGES edges assigned to
    another cut mid-run is noise from the nearest cut flipping at a proxy
    vertex, and takes the cut of the stretch before it."""
    corners, following = mesh.corners, mesh.following
    adjacent = collections.defaultdict(list)
    for i, c in enumerate(seam_corners.tolist()):
        a, b = int(corners[c]), int(corners[following[c]])
        adjacent[a].append((b, i))
        adjacent[b].append((a, i))
    used = numpy.zeros(len(seam_corners), dtype=bool)

    def extend(v, i):
        """Vertices past v until a junction, a loose end or a used edge."""
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


def _band(run_faces, mesh, rings_out):
    """Faces within rings_out rings of the run's faces."""
    band = set(run_faces)
    frontier = set(run_faces)
    for _ in range(rings_out):
        grown = set()
        for f in frontier:
            for c in mesh.face(f):
                if mesh.twin[c] >= 0:
                    grown.add(int(mesh.face_of[mesh.twin[c]]))
        frontier = grown - band
        band |= grown
    return band


def _shortest_path(band, start, end, mesh):
    """Vertex path from start to end along the band's edges, or None."""
    adjacent = collections.defaultdict(list)
    for f in band:
        for c in mesh.face(f):
            a, b = int(mesh.corners[c]), int(mesh.corners[mesh.following[c]])
            adjacent[a].append((b, float(mesh.lengths[c])))
            adjacent[b].append((a, float(mesh.lengths[c])))
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


def _flood(band, seeds, blocked, mesh):
    """Each band face's label spread from the seeds across edges that are
    not blocked, None on a face two labels reach."""
    labels = dict(seeds)
    frontier = list(seeds)
    clash = False
    while frontier:
        f, label = frontier.pop()
        for c in mesh.face(f):
            t = mesh.twin[c]
            if t < 0 or c in blocked or int(t) in blocked:
                continue
            g = int(mesh.face_of[t])
            if g not in band:
                continue
            if g in labels:
                clash |= labels[g] != label
                continue
            labels[g] = label
            frontier.append((g, label))
    return None if clash else labels


def _straighten_seams(corner_uvs, drawn_by, proxy_map, mesh):
    """Each seam run redrawn as the shortest path along the mesh between its
    ends. The side labelling puts a seam wherever the nearest proxy face
    changes, a staircase along the projected cut, and a shortest path is
    the straightest line of edges there is. The faces between the old and
    new path go over to the other side, drawn like a neighbour there, or
    through that side's rim face when none is drawn yet."""
    corners, following, twin = mesh.corners, mesh.following, mesh.twin
    face_of, positions = mesh.face_of, mesh.positions
    cuts = proxy_map.crossings
    seam_corners = _seam_corners(corner_uvs, mesh)
    seam_set = set(seam_corners.tolist()) | set(twin[seam_corners].tolist())
    midpoints = (
        positions[corners[seam_corners]] + positions[corners[following[seam_corners]]]
    ) / 2
    cut_of = numpy.array(
        [
            cuts.nearest(int(drawn_by[c]), mid)
            for c, mid in zip(seam_corners.tolist(), midpoints)
        ],
        dtype=numpy.int64,
    )

    def side_faces(vertices, run_corners):
        """The faces left and right of the run, walking it in order."""
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

    def rim_face(cut, faces):
        """The owner of the cut on the side those faces are drawn on."""
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
        band = _band(inner, mesh, SEED_RINGS)
        seeds = [(f, "left") for f in left] + [(f, "right") for f in right]
        was = _flood(band, seeds, seam_set, mesh)
        if was is None:
            continue
        path_edges = {(a, b) for a, b in zip(path, path[1:])}
        path_edges |= {(b, a) for a, b in path_edges}
        run_set = set(run_corners) | set(twin[run_corners].tolist())
        blocked = set()
        for f in band:
            for c in mesh.face(f):
                edge = (int(corners[c]), int(corners[following[c]]))
                if (c in seam_set and c not in run_set) or edge in path_edges:
                    blocked.add(c)
        outer = [
            (f, was[f])
            for f in band
            if f in was
            and any(
                twin[c] >= 0 and int(face_of[twin[c]]) not in band for c in mesh.face(f)
            )
        ]
        now = _flood(band, outer, blocked, mesh)
        if now is None:
            continue
        moved = [f for f in band if f in was and f in now and was[f] != now[f]]
        if not moved:
            continue
        rims = {"left": rim_face(cut, left), "right": rim_face(cut, right)}
        if (
            rims["left"] is None
            or rims["right"] is None
            or rims["left"] == rims["right"]
        ):
            continue
        # drawn like a neighbour already on the new side, outermost first
        waiting = set(moved)
        while waiting:
            progressed = False
            for f in sorted(waiting):
                for c in mesh.face(f):
                    t = twin[c]
                    g = int(face_of[t]) if t >= 0 else -1
                    if g >= 0 and g in now and now[g] == now[f] and g not in waiting:
                        _draw_like_neighbour(corner_uvs, drawn_by, mesh, f, g)
                        waiting.discard(f)
                        progressed = True
                        break
            if not progressed:
                for f in waiting:
                    face = rims[now[f]]
                    ring = mesh.face(f)
                    corner_uvs[ring] = proxy_map.maps.uv(
                        numpy.full(len(ring), face), positions[corners[ring]]
                    )
                    drawn_by[ring] = face
                waiting.clear()
        for f in moved:
            for c in mesh.face(f):
                _weld_ring(corner_uvs, drawn_by, proxy_map, mesh, int(corners[c]))
        # the band's edges are the only ones whose seam state can have changed
        band_corners = numpy.array(
            [c for f in band for c in mesh.face(f)], dtype=numpy.int64
        )
        paired = band_corners[twin[band_corners] >= 0]
        torn = _torn_at_twin(corner_uvs, mesh, paired)
        seam_set -= set(paired.tolist())
        seam_set |= set(paired[torn].tolist())
    return corner_uvs


def _corner_tears(corners, following, corner_uvs, tolerance):
    """uv_tears on a corner table, corner_uvs one row per corner."""
    tail = corners
    head = corners[following]
    low_first = (tail < head)[:, None]
    at_low = numpy.where(low_first, corner_uvs, corner_uvs[following])
    at_high = numpy.where(low_first, corner_uvs[following], corner_uvs)
    keys = _edge_keys(corners, following)
    unique, first, group = numpy.unique(keys, return_index=True, return_inverse=True)
    agrees = numpy.all(numpy.abs(at_low - at_low[first][group]) <= tolerance, axis=1)
    agrees &= numpy.all(numpy.abs(at_high - at_high[first][group]) <= tolerance, axis=1)
    torn = unique[numpy.unique(group[~agrees])]
    return {(int(key >> 32), int(key & 0xFFFFFFFF)) for key in torn.tolist()}


def dense_subset(dense, faces):
    """The dense arrays cut down to these faces, their vertices renumbered
    from zero. Returns the subset and each new vertex's old index."""
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


def transfer_projected(dense, proxy, nearest_faces, progress=None, cancelled=None):
    """(seams, uvs) for the dense mesh, uvs one row per corner.

    nearest_faces(positions, normals) gives each dense vertex the proxy face
    it stands over and the point on it, in the proxy's space. cancelled is
    polled between lookup chunks."""

    def report(fraction):
        if progress is not None:
            progress(fraction)

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
        report(LOOKUP_PROGRESS * min(stop, len(positions)) / max(len(positions), 1))
    # read at the point under the vertex, a map continued off its face
    # diverges at a crease
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

    check_cancelled(cancelled)
    corner_uvs, drawn_by = _redraw_torn_faces(
        proxy_map, face_of_vertex, surface, uvs, mesh, torn_face
    )
    corner_uvs = _weld(corner_uvs, drawn_by, uvs, proxy_map, mesh)
    check_cancelled(cancelled)
    corner_uvs = _absorb_stray_faces(corner_uvs, drawn_by, proxy_map, mesh)
    check_cancelled(cancelled)
    corner_uvs = _straighten_seams(corner_uvs, drawn_by, proxy_map, mesh)
    seams = _corner_tears(mesh.corners, mesh.following, corner_uvs, 0.0)
    report(1.0)
    return seams, corner_uvs
