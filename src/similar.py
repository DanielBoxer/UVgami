import itertools

import numpy

# rms fit error under this fraction of the piece's bounding box diagonal
# counts as the same shape
MATCH_TOLERANCE = 1e-4


def rigid_fit(source, target):
    """Best rigid transform (rotation or reflection, then translation) taking
    source points onto target points matched by index, as a 4x4 matrix, with
    the rms error of the fit."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = numpy.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    moved = source @ rotation.T + translation
    error = numpy.sqrt(numpy.mean(numpy.sum((moved - target) ** 2, axis=1)))
    matrix = numpy.identity(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix, error


def _world_positions(obj):
    mesh = obj.data
    positions = numpy.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", positions)
    matrix = numpy.array(obj.matrix_world)
    return positions.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def _topology(mesh):
    corners = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    return totals, corners


# duplicates land on each other to float precision after pca alignment, so a
# cell this big (fraction of the diagonal) separates distinct vertices while
# keeping a boundary miss rare
MATCH_CELL = 1e-4

# pca eigenvalue mismatch above this fraction of the largest one means a
# different shape, skip the alignment attempts
SPREAD_TOLERANCE = 1e-3

# an eigenvalue pair closer than this fraction of the largest makes the pca
# axes in that plane arbitrary, so alignment needs the rotation search
DEGENERATE_GAP = 1e-2


class _Piece:
    def __init__(self, obj):
        mesh = obj.data
        self.obj = obj
        self.signature = (len(mesh.vertices), len(mesh.loops), len(mesh.polygons))
        self.totals, self.corners = _topology(mesh)
        self.positions = _world_positions(obj)
        self.diagonal = numpy.linalg.norm(
            self.positions.max(axis=0) - self.positions.min(axis=0)
        )
        centered = self.positions - self.positions.mean(axis=0)
        self.spread, axes = numpy.linalg.eigh(centered.T @ centered)
        self.local = centered @ axes
        self.grid = None


def _grid(local, cell):
    lookup = {}
    keys = numpy.round(local / cell).astype(numpy.int64)
    for index, key in enumerate(map(tuple, keys.tolist())):
        lookup.setdefault(key, []).append(index)
    return lookup


_NEIGHBOR_CELLS = [
    offsets
    for offsets in itertools.product((0, -1, 1), repeat=3)
    if offsets != (0, 0, 0)
]


def _match_vertices(rep_local, grid, aligned, cell):
    """Permutation sending each aligned candidate vertex index to the rep
    vertex at the same position, or None when some vertex has no
    counterpart."""
    permutation = numpy.empty(len(aligned), dtype=numpy.int64)
    used = numpy.zeros(len(rep_local), dtype=bool)
    keys = numpy.round(aligned / cell).astype(numpy.int64).tolist()
    for index, (x, y, z) in enumerate(keys):
        options = [i for i in grid.get((x, y, z), ()) if not used[i]]
        if not options:
            for dx, dy, dz in _NEIGHBOR_CELLS:
                options += [
                    i for i in grid.get((x + dx, y + dy, z + dz), ()) if not used[i]
                ]
        if not options:
            return None
        distances = numpy.linalg.norm(rep_local[options] - aligned[index], axis=1)
        nearest = int(numpy.argmin(distances))
        if distances[nearest] > cell:
            return None
        permutation[index] = options[nearest]
        used[options[nearest]] = True
    return permutation


def _plane_alignments(rep, piece, plane, unique, cell):
    """Aligned copies of piece's local coords for a rotationally symmetric
    rep: every in-plane rotation, mirror, and axis flip taking a candidate
    vertex onto the rep vertex farthest from the symmetry axis."""
    rep_radius = numpy.hypot(rep.local[:, plane[0]], rep.local[:, plane[1]])
    anchor = int(rep_radius.argmax())
    anchor_angle = numpy.arctan2(
        rep.local[anchor, plane[1]], rep.local[anchor, plane[0]]
    )
    anchor_height = rep.local[anchor, unique]
    candidate_radius = numpy.hypot(piece.local[:, plane[0]], piece.local[:, plane[1]])
    for axis_sign in (1.0, -1.0):
        heights = piece.local[:, unique] * axis_sign
        anchors = numpy.nonzero(
            (numpy.abs(candidate_radius - rep_radius[anchor]) <= cell)
            & (numpy.abs(heights - anchor_height) <= cell)
        )[0]
        for mirror in (1.0, -1.0):
            for vertex in anchors.tolist():
                angle = numpy.arctan2(
                    piece.local[vertex, plane[1]] * mirror,
                    piece.local[vertex, plane[0]],
                )
                turn = anchor_angle - angle
                cosine, sine = numpy.cos(turn), numpy.sin(turn)
                first = piece.local[:, plane[0]]
                second = piece.local[:, plane[1]] * mirror
                aligned = piece.local.copy()
                aligned[:, unique] = heights
                aligned[:, plane[0]] = first * cosine - second * sine
                aligned[:, plane[1]] = first * sine + second * cosine
                yield aligned


def _alignments(rep, piece, cell):
    """Aligned copies of piece's local coords to try against rep's: the axis
    sign flips when the pca frame is stable, the rotation search when two
    eigenvalues coincide, the axis permutations and sign flips when all three
    do, which covers a box but not a freely rotated ball-like shape."""
    scale = rep.spread[2]
    low_gap = (rep.spread[1] - rep.spread[0]) / scale
    high_gap = (rep.spread[2] - rep.spread[1]) / scale
    if low_gap >= DEGENERATE_GAP and high_gap >= DEGENERATE_GAP:
        for signs in itertools.product((1.0, -1.0), repeat=3):
            yield piece.local * signs
    elif low_gap >= DEGENERATE_GAP or high_gap >= DEGENERATE_GAP:
        plane = (1, 2) if high_gap < DEGENERATE_GAP else (0, 1)
        unique = 0 if high_gap < DEGENERATE_GAP else 2
        yield from _plane_alignments(rep, piece, plane, unique, cell)
    else:
        for order in itertools.permutations((0, 1, 2)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                yield piece.local[:, order] * signs


def _face_keys(totals, corners):
    """Faces as order-free keys: each cycle rotated, and reversed if that is
    smaller, to its smallest tuple, the whole list sorted."""
    keys = []
    face_corners = corners.tolist()
    start = 0
    for total in totals.tolist():
        cycle = face_corners[start : start + total]
        keys.append(
            min(
                tuple(direction[shift:] + direction[:shift])
                for direction in (cycle, cycle[::-1])
                for shift in range(total)
            )
        )
        start += total
    return sorted(keys)


def _twin_matrix(rep, piece):
    """(matrix taking rep's world positions onto piece's, exact) when the two
    are congruent duplicates, else None. Tries the cheap index-for-index fit
    first (exact True), then pca alignment to find the vertex permutation of
    a reordered copy (exact False)."""
    tolerance = MATCH_TOLERANCE * piece.diagonal
    same_order = numpy.array_equal(rep.totals, piece.totals) and numpy.array_equal(
        rep.corners, piece.corners
    )
    if same_order:
        matrix, error = rigid_fit(rep.positions, piece.positions)
        if error <= tolerance:
            return matrix, True
    spread_gap = numpy.abs(rep.spread - piece.spread).max()
    if spread_gap > SPREAD_TOLERANCE * piece.spread.max():
        return None
    if not numpy.array_equal(numpy.sort(rep.totals), numpy.sort(piece.totals)):
        return None
    cell = MATCH_CELL * rep.diagonal
    if rep.grid is None:
        rep.grid = _grid(rep.local, cell)
    face_keys = _face_keys(piece.totals, piece.corners)
    for aligned in _alignments(rep, piece, cell):
        permutation = _match_vertices(rep.local, rep.grid, aligned, cell)
        if permutation is None:
            continue
        inverse = numpy.empty_like(permutation)
        inverse[permutation] = numpy.arange(len(permutation))
        if _face_keys(rep.totals, inverse[rep.corners]) != face_keys:
            continue
        matrix, error = rigid_fit(rep.positions[permutation], piece.positions)
        if error <= tolerance:
            return matrix, False
    return None


def find_twins(objects):
    """Group congruent pieces: same shape under a rigid transform with the
    same topology in any vertex order, so only pieces made by duplication
    qualify. Returns twin object -> (representative object, matrix taking the
    representative's world positions onto the twin's, exact), exact meaning
    the vertex order matches index for index."""
    representatives = {}
    twins = {}
    for obj in objects:
        piece = _Piece(obj)
        candidates = representatives.setdefault(piece.signature, [])
        for rep in candidates:
            match = _twin_matrix(rep, piece)
            if match is not None:
                matrix, exact = match
                twins[obj] = (rep.obj, matrix, exact)
                break
        else:
            candidates.append(piece)
    return twins


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _partial_match(positions, grid, reflected, cell):
    """Index of the vertex at each reflected position, -1 where none is
    close enough. Matched pairs stay one to one."""
    permutation = numpy.full(len(reflected), -1, dtype=numpy.int64)
    used = numpy.zeros(len(positions), dtype=bool)
    keys = numpy.round(reflected / cell).astype(numpy.int64).tolist()
    for index, (x, y, z) in enumerate(keys):
        options = [i for i in grid.get((x, y, z), ()) if not used[i]]
        if not options:
            for dx, dy, dz in _NEIGHBOR_CELLS:
                options += [
                    i for i in grid.get((x + dx, y + dy, z + dz), ()) if not used[i]
                ]
        if not options:
            continue
        distances = numpy.linalg.norm(positions[options] - reflected[index], axis=1)
        nearest = int(numpy.argmin(distances))
        if distances[nearest] > cell:
            continue
        permutation[index] = options[nearest]
        used[options[nearest]] = True
    return permutation


def _canonical_cycle(cycle):
    return min(
        tuple(direction[shift:] + direction[:shift])
        for direction in (cycle, cycle[::-1])
        for shift in range(len(cycle))
    )


def _mesh_positions(mesh):
    positions = numpy.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", positions)
    return positions.reshape(-1, 3)


# looser than MATCH_CELL: seam mirroring wants the nearest counterpart under
# small modeling drift, and mirror_seams drops any pair that is not a mesh edge
SEAM_MATCH_CELL = 1e-3


def mirror_matches(mesh, center, axes):
    """Per-axis partial vertex maps sending each vertex to the nearest vertex
    across the axis plane through center, within tolerance. No topology
    requirement: a vertex with no counterpart is absent from its map, so an
    asymmetric region simply drops out. None when nothing matches."""
    if len(mesh.vertices) == 0:
        return None
    positions = _mesh_positions(mesh)
    diagonal = numpy.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    if diagonal <= 0:
        return None
    cell = SEAM_MATCH_CELL * diagonal
    grid = _grid(positions, cell)
    maps = []
    for axis in axes:
        index = "XYZ".index(axis)
        reflected = positions.copy()
        reflected[:, index] = 2.0 * float(center[index]) - reflected[:, index]
        matched = _partial_match(positions, grid, reflected, cell)
        maps.append({v: int(m) for v, m in enumerate(matched.tolist()) if m >= 0})
    if not any(maps):
        return None
    return maps


def mirror_permutations(mesh, center, axes):
    """Per-axis vertex maps, as dicts, sending each mirror-symmetric loose
    part onto itself or its twin across the axis plane through center.

    Coverage is per part: a part with any vertex or face that has no mirror
    image drops out, along with everything mapped into a dropped part, so a
    wrench keeps its symmetric handle while the asymmetric worm gear opts
    out. A part must qualify on every chosen axis. None when no part does."""
    if len(mesh.vertices) == 0:
        return None
    positions = _mesh_positions(mesh)
    diagonal = numpy.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    if diagonal <= 0:
        return None
    totals, corners = _topology(mesh)
    corner_list = corners.tolist()
    faces = []
    parent = list(range(len(positions)))
    start = 0
    for total in totals.tolist():
        face = corner_list[start : start + total]
        faces.append(face)
        for v in face[1:]:
            ra, rb = _find(parent, face[0]), _find(parent, v)
            if ra != rb:
                parent[ra] = rb
        start += total
    part_of = [_find(parent, v) for v in range(len(positions))]
    face_keys = {_canonical_cycle(face) for face in faces}

    cell = MATCH_CELL * diagonal
    grid = _grid(positions, cell)
    per_axis = []
    covered = None
    for axis in axes:
        index = "XYZ".index(axis)
        reflected = positions.copy()
        reflected[:, index] = 2.0 * float(center[index]) - reflected[:, index]
        permutation = _partial_match(positions, grid, reflected, cell)
        bad = {part_of[v] for v in numpy.nonzero(permutation < 0)[0].tolist()}
        for face in faces:
            if part_of[face[0]] in bad:
                continue
            mapped = [int(permutation[v]) for v in face]
            if _canonical_cycle(mapped) not in face_keys:
                bad.add(part_of[face[0]])
        per_axis.append(permutation)
        axis_covered = set(part_of) - bad
        covered = axis_covered if covered is None else covered & axis_covered

    # a covered part's image must be covered too, or its mirrored seams
    # land on a part that will not mirror back
    changed = True
    while changed and covered:
        changed = False
        for permutation in per_axis:
            for v in range(len(positions)):
                if part_of[v] not in covered:
                    continue
                if part_of[int(permutation[v])] not in covered:
                    covered.discard(part_of[v])
                    changed = True
    if not covered:
        return None
    return [
        {v: int(permutation[v]) for v in range(len(positions)) if part_of[v] in covered}
        for permutation in per_axis
    ]


# intentional stacks are exact copies (mirror modifier, twin outputs from the
# same vt text), so round tightly: loose rounding merges distinct islands that
# average_islands_scale happens to land on the same spot
STACK_DECIMALS = 9


def find_stacks(groups, uvs):
    """Islands whose loop uvs match exactly, as (kept island, duplicate
    islands) pairs. Those are intentional stacks (symmetry twins, detected
    duplicates, artist stacks): only the kept one should pack, the duplicates
    follow it."""
    by_key = {}
    for group in groups:
        key = tuple(
            sorted(
                (round(u, STACK_DECIMALS), round(v, STACK_DECIMALS))
                for fi in group
                for u, v in uvs[fi]
            )
        )
        by_key.setdefault(key, []).append(group)
    return [
        (members[0], members[1:]) for members in by_key.values() if len(members) > 1
    ]


def write_twin_output(source_path, target_path, matrix):
    """Write the twin's engine output: the representative's output with the
    vertex positions moved by matrix and the uvs kept, so the islands stack
    exactly. A reflection flips each face's winding so normals stay outward."""
    rotation = numpy.array(matrix)[:3, :3]
    translation = numpy.array(matrix)[:3, 3]
    mirrored = numpy.linalg.det(rotation) < 0
    lines = []
    for line in source_path.read_text().splitlines(keepends=True):
        if line.startswith("v "):
            position = rotation @ [float(x) for x in line.split()[1:4]] + translation
            lines.append("v {:.9f} {:.9f} {:.9f}\n".format(*position))
        elif mirrored and line.startswith("f "):
            tokens = line.split()
            lines.append(" ".join([tokens[0], *reversed(tokens[1:])]) + "\n")
        else:
            lines.append(line)
    target_path.write_text("".join(lines))
