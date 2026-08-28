import math
from dataclasses import dataclass, field

import numpy as np

from .proxy_transfer import uv_tears

# planner is bpy-free: it turns plain mesh data into a complete uv transfer plan
# or a structured failure. loops are numbered consecutively in polygon order,
# matching blender's poly.loop_start layout.


@dataclass
class TransferPlan:
    loop_uvs: dict  # input loop index -> (u, v), for faces kept whole
    split_faces: dict  # input face index -> [(verts, uvs)] replacing that face
    seam_edges: set  # sorted (input v0, input v1) tuples
    # input faces the output never reached, they keep their uvs and seams
    untouched_faces: set = field(default_factory=set)
    ok: bool = True


# the only transfer failure a cut or symmetry can't cause
AMBIGUOUS_GEOMETRY = "ambiguous_geometry"


@dataclass
class TransferFailure:
    reason: str  # machine string: vertex_match, ambiguous_geometry, ...
    detail: str
    ok: bool = False


def _default_tol(positions):
    # the engine writes 7 significant digits, so error grows with distance from origin
    if len(positions) == 0:
        return 1e-5
    return max(float(np.abs(positions).max()) * 2e-6, 1e-5)


def _grid_key(p, inv):
    return (
        int(math.floor(p[0] * inv)),
        int(math.floor(p[1] * inv)),
        int(math.floor(p[2] * inv)),
    )


def _build_grid(positions, inv):
    grid = {}
    for i in range(len(positions)):
        grid.setdefault(_grid_key(positions[i], inv), []).append(i)
    return grid


def _query_grid(grid, inv, positions, p, tol2):
    base = _grid_key(p, inv)
    found = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                bucket = grid.get((base[0] + dx, base[1] + dy, base[2] + dz))
                if not bucket:
                    continue
                for i in bucket:
                    d = positions[i] - p
                    if float(d @ d) <= tol2:
                        found.append(i)
    return found


def _order_part(local, assign, uvs):
    """Put a piece of a split face in the input face's own winding order."""
    order = sorted(range(len(assign)), key=lambda c: local[assign[c]])
    return [assign[c] for c in order], [uvs[c] for c in order]


def _uv_area(uvs):
    area = 0.0
    n = len(uvs)
    for i in range(n):
        x1, y1 = uvs[i]
        x2, y2 = uvs[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2


def _flatten_part(assign, positions):
    """The part's corners flattened in its own plane as complex numbers,
    counterclockwise, or None when it has no area."""
    rel = positions[assign] - positions[assign[0]]
    normal = np.zeros(3)
    for i in range(1, len(rel) - 1):
        normal += np.cross(rel[i], rel[i + 1])
    normal_length = np.linalg.norm(normal)
    axis_u = rel[1]
    axis_u_length = np.linalg.norm(axis_u)
    if normal_length == 0 or axis_u_length == 0:
        return None
    axis_u = axis_u / axis_u_length
    axis_v = np.cross(normal / normal_length, axis_u)
    return [complex(float(r @ axis_u), float(r @ axis_v)) for r in rel]


def _uv_map_of(flat, uvs):
    """The map z -> p*z + q*conj(z) + r taking a part's flattened corners onto
    its uvs, as (p, q, r). The conjugate term carries the shear and mirroring
    a similarity can't."""
    d1, d2 = flat[1] - flat[0], flat[2] - flat[0]
    u0, u1, u2 = (complex(*uv) for uv in uvs[:3])
    e1, e2 = u1 - u0, u2 - u0
    det = d1 * d2.conjugate() - d2 * d1.conjugate()
    if det == 0:
        return None
    p = (e1 * d2.conjugate() - e2 * d1.conjugate()) / det
    q = (d1 * e2 - d2 * e1) / det
    return p, q, u0 - p * flat[0] - q * flat[0].conjugate()


def _weld_parts(parts, positions):
    """Glue the pieces of a face a uv cut runs through back into one patch.
    The pieces unfold about their shared edges into the largest piece's plane
    and the whole patch is drawn by that piece's uv map, so the face comes out
    at one density instead of either chart's scale. Returns (input vertex ->
    uv, anchor part index), or None when it can't glue and the caller should
    split instead."""
    anchor = max(range(len(parts)), key=lambda i: abs(_uv_area(parts[i][1])))
    anchor_assign, anchor_uvs = parts[anchor]
    anchor_flat = _flatten_part(anchor_assign, positions)
    if anchor_flat is None:
        return None
    uv_map = _uv_map_of(anchor_flat, anchor_uvs)
    if uv_map is None:
        return None
    # the unfold works in 3d units, so the slack scales with the face
    anchor_area = abs(_uv_area([(z.real, z.imag) for z in anchor_flat]))
    if anchor_area == 0:
        return None
    tol = 0.01 * math.sqrt(anchor_area)

    placed = dict(zip(anchor_assign, anchor_flat))
    pending = [i for i in range(len(parts)) if i != anchor]
    while pending:
        progress = False
        for i in list(pending):
            assign, _ = parts[i]
            shared = [v for v in assign if v in placed]
            if len(shared) < 2:
                continue
            flat = _flatten_part(assign, positions)
            if flat is None:
                return None
            a, b = shared[:2]
            za, zb = flat[assign.index(a)], flat[assign.index(b)]
            if za == zb:
                return None
            # both frames hold the true 3d shape wound the same way, so this
            # comes out a rotation and the piece hinges onto the placed edge
            s = (placed[a] - placed[b]) / (za - zb)
            t = placed[a] - s * za

            moved = [(v, s * z + t) for v, z in zip(assign, flat)]
            if any(v in placed and abs(placed[v] - z) > tol for v, z in moved):
                return None
            for v, z in moved:
                placed.setdefault(v, z)
            pending.remove(i)
            progress = True
        if not progress:
            return None

    p, q, r = uv_map
    corner_uvs = {v: uv for v, uv in zip(anchor_assign, anchor_uvs)}
    for v, z in placed.items():
        if v not in corner_uvs:
            w = p * z + q * z.conjugate() + r
            corner_uvs[v] = (w.real, w.imag)
    return corner_uvs, anchor


def _is_ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _is_collinear(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1]) == 0


def _do_overlap(a, b, c, d):
    if _is_collinear(a, c, d) or _is_collinear(b, c, d):
        return False

    return _is_ccw(a, c, d) != _is_ccw(b, c, d) and _is_ccw(a, b, c) != _is_ccw(a, b, d)


class _OutputLayout:
    """The output uv layout indexed for the weld overlap check: face edges by
    corner on a uniform grid, plus the island each output face belongs to."""

    def __init__(self, output_polygons, output_uvs):
        self.parent = list(range(len(output_polygons)))
        corner_face = {}
        self.edges = {}
        total = 0.0
        count = 0
        for fo, (verts, uvs) in enumerate(zip(output_polygons, output_uvs)):
            corners = [(float(uv[0]), float(uv[1])) for uv in uvs]
            # faces sharing a vertex at the same uv are one island
            for v, p in zip(verts, corners):
                other = corner_face.setdefault((v, p), fo)
                if other != fo:
                    self.parent[self.find(fo)] = self.find(other)
            n = len(corners)
            for i in range(n):
                p, q = corners[i], corners[(i + 1) % n]
                self.edges.setdefault(p, []).append((fo, q))
                self.edges.setdefault(q, []).append((fo, p))
                total += math.hypot(q[0] - p[0], q[1] - p[1])
                count += 1
        self.cell = total / count if total > 0 else 1.0
        self.grid = {}
        for p in self.edges:
            key = (math.floor(p[0] / self.cell), math.floor(p[1] / self.cell))
            self.grid.setdefault(key, []).append(p)

    def find(self, fo):
        while self.parent[fo] != fo:
            self.parent[fo] = self.parent[self.parent[fo]]
            fo = self.parent[fo]
        return fo

    def points_near(self, low_u, low_v, high_u, high_v):
        found = []
        for cu in range(
            math.floor(low_u / self.cell), math.floor(high_u / self.cell) + 1
        ):
            for cv in range(
                math.floor(low_v / self.cell), math.floor(high_v / self.cell) + 1
            ):
                found.extend(self.grid.get((cu, cv), ()))
        return found


def _weld_overlaps(parts, part_out_faces, anchor, welded_uvs, layout, repack):
    """True when a welded piece lands on uv space another face already uses.
    With a pack to come only the piece's own island counts, since the pack
    pulls the other islands clear anyway."""
    own = set(part_out_faces)
    island = layout.find(part_out_faces[anchor])

    moved = []
    for i, (assign, _) in enumerate(parts):
        if i == anchor:
            continue
        corners = [welded_uvs[v] for v in assign]
        n = len(corners)
        moved.extend((corners[j], corners[(j + 1) % n]) for j in range(n))

    us = [p[0] for edge in moved for p in edge]
    vs = [p[1] for edge in moved for p in edge]
    # a cell is the mean edge length, so a crossing edge has an end this close
    for p in layout.points_near(
        min(us) - layout.cell,
        min(vs) - layout.cell,
        max(us) + layout.cell,
        max(vs) + layout.cell,
    ):
        for fo, q in layout.edges[p]:
            if fo in own or (repack and layout.find(fo) != island):
                continue
            for m1, m2 in moved:
                if _do_overlap(p, q, m1, m2):
                    return True
    return False


# how far apart two faces may put a shared corner and still count as joined
SEAM_TOLERANCE = 1e-6


def transfer_exact(
    input_positions,
    input_polygons,
    output_positions,
    output_polygons,
    output_uvs,
    tol=None,
    repack=True,
    partial=False,
):
    """Plan the output's uvs onto an input with the same vertex positions,
    each output face matched to the input face on its vertices.

    repack says a pack runs on the result, which sets how strict the weld
    overlap check has to be. partial lets input faces the output doesn't reach
    keep their uvs, for an output missing whole pieces."""
    in_pos = np.asarray(input_positions, dtype=float)
    out_pos = np.asarray(output_positions, dtype=float)

    if tol is None:
        tol = _default_tol(in_pos)
    tol2 = tol * tol
    inv = 1.0 / tol

    in_grid = _build_grid(in_pos, inv)

    faces_by_vertex = [[] for _ in range(len(in_pos))]
    input_face_vsets = []
    input_vertex_local = []
    input_loop_start = []
    loop_cursor = 0
    for fi, poly in enumerate(input_polygons):
        input_loop_start.append(loop_cursor)
        loop_cursor += len(poly)
        input_face_vsets.append(set(poly))
        local = {}
        for corner, v in enumerate(poly):
            faces_by_vertex[v].append(fi)
            local[v] = corner
        input_vertex_local.append(local)

    # optcuts and xatlas keep the input vertex indices and append cut copies,
    # partuv merges doubles and shifts them, so check before trusting an index
    n_in = len(in_pos)
    keeps_indices = 0 < n_in <= len(out_pos) and bool(
        (((out_pos[:n_in] - in_pos) ** 2).sum(axis=1) <= tol2).all()
    )
    candidate_cache = {}

    def candidates(out_v):
        if keeps_indices and out_v < n_in:
            return [out_v]
        cached = candidate_cache.get(out_v)
        if cached is None:
            cached = _query_grid(in_grid, inv, in_pos, out_pos[out_v], tol2)
            candidate_cache[out_v] = cached
        return cached

    # output pieces landing on each input face, gathered before any uv is written
    # so a face that can't hold one uv set can be split instead of failing
    face_parts = [[] for _ in input_polygons]
    part_out_faces = [[] for _ in input_polygons]

    for fo, out_verts in enumerate(output_polygons):
        corner_candidates = []
        for out_v in out_verts:
            cands = candidates(out_v)
            if not cands:
                return TransferFailure(
                    "vertex_match",
                    f"output vertex {out_v} has no matching input vertex",
                )
            corner_candidates.append(cands)

        # input faces incident to a candidate of every output corner
        cand_faces = set()
        for v in corner_candidates[0]:
            cand_faces.update(faces_by_vertex[v])
        for corner in corner_candidates[1:]:
            allowed = set()
            for v in corner:
                allowed.update(faces_by_vertex[v])
            cand_faces &= allowed
            if not cand_faces:
                break
        if not cand_faces:
            return TransferFailure(
                "face_match", f"output face {fo} maps to no input face"
            )

        # each surviving face must give one input vertex per output corner
        valid = []
        for f in cand_faces:
            fvs = input_face_vsets[f]
            assign = []
            for corner in corner_candidates:
                matches = [v for v in corner if v in fvs]
                if len(matches) > 1:
                    return TransferFailure(
                        AMBIGUOUS_GEOMETRY,
                        f"output face {fo} corner matches multiple vertices"
                        " of one input face",
                    )
                assign.append(matches[0])
            valid.append((f, assign))

        if len(valid) > 1:
            return TransferFailure(
                AMBIGUOUS_GEOMETRY, f"output face {fo} matches multiple input faces"
            )
        f, assign = valid[0]
        face_uvs = output_uvs[fo]
        face_parts[f].append(
            (assign, [(float(uv[0]), float(uv[1])) for uv in face_uvs])
        )
        part_out_faces[f].append(fo)

    layout = None
    loop_uvs = {}
    split_faces = {}
    untouched = set()
    for fi, poly in enumerate(input_polygons):
        parts = face_parts[fi]
        if partial and not parts:
            untouched.add(fi)
            continue
        covered = {v for assign, _ in parts for v in assign}
        if covered != set(poly):
            return TransferFailure(
                "incomplete_coverage", f"input face {fi} has corners with no uv"
            )

        corner_uvs = {}
        conflict = False
        for assign, uvs in parts:
            for in_v, uv in zip(assign, uvs):
                prev = corner_uvs.get(in_v)
                if prev is None:
                    corner_uvs[in_v] = uv
                elif abs(prev[0] - uv[0]) > 1e-6 or abs(prev[1] - uv[1]) > 1e-6:
                    conflict = True

        local = input_vertex_local[fi]
        if not conflict:
            base = input_loop_start[fi]
            for in_v, uv in corner_uvs.items():
                loop_uvs[base + local[in_v]] = uv
            continue

        # a uv cut runs through this face, so one face can't hold its uvs
        if len({frozenset(assign) for assign, _ in parts}) != len(parts):
            return TransferFailure(
                AMBIGUOUS_GEOMETRY,
                f"input face {fi} maps to overlapping output faces",
            )

        welded = _weld_parts(parts, in_pos)
        if welded is not None:
            welded_uvs, anchor = welded
            if layout is None:
                layout = _OutputLayout(output_polygons, output_uvs)
            if not _weld_overlaps(
                parts, part_out_faces[fi], anchor, welded_uvs, layout, repack
            ):
                base = input_loop_start[fi]
                for in_v, uv in welded_uvs.items():
                    loop_uvs[base + local[in_v]] = uv
                continue

        split_faces[fi] = [_order_part(local, assign, uvs) for assign, uvs in parts]

    if len(untouched) == len(input_polygons):
        return TransferFailure("incomplete_coverage", "no input face got uvs")

    # a weld moves a cut onto the input face's outer edges, off the engine's
    # seam list
    faces = []
    for fi, poly in enumerate(input_polygons):
        if fi in untouched:
            continue
        parts = split_faces.get(fi)
        if parts is None:
            base = input_loop_start[fi]
            faces.append((poly, [loop_uvs[base + c] for c in range(len(poly))]))
        else:
            faces.extend(parts)
    seams = uv_tears(
        [verts for verts, _ in faces], [uvs for _, uvs in faces], SEAM_TOLERANCE
    )
    return TransferPlan(loop_uvs, split_faces, seams, untouched)
