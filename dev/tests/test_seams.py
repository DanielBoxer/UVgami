import collections
import importlib.util
import math
import sys
from pathlib import Path

import pytest

# loaded from file so it doesn't need the bpy-only addon package
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import (  # noqa: E402
    CREASE_ANGLE,
    LOW_ANGLE,
    TURN_COST,
    absorb,
    boundary_edges,
    build,
    close_rings,
    component_faces,
    crease_relief,
    cross,
    crosses,
    cut_path,
    detect_width,
    diagonal,
    disk_cuts,
    face_edges,
    flatten_distortion,
    flatten_teeth,
    is_hard_surface,
    island_eulers,
    island_groups,
    island_layout,
    island_ruined,
    islands_overlap,
    pair,
    partition,
    path_cost,
    rectify_targets,
    region_topology,
    reroute_boundaries,
    seam_edges,
    signed_area,
    snap_paths,
    split_islands,
    split_moves,
    split_sweeps,
    surface_genus,
    sweep_rims,
    turn_angle,
    unfold_hinges,
    uv_area_fit,
    uv_fit,
    uv_island_groups,
    uv_seams,
    uv_topology,
    uvs_collapsed,
    vertex_components,
)
from seams.islands import absorb_fragments, split_pieces  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
HEX_HEAD = Path(__file__).parents[1] / "bench/models/hard-surface/sharp/fastener_03.obj"


def read_obj(path):
    verts, faces = [], []
    for line in open(path):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return verts, faces


def region_count(verts, faces, width):
    weighted, areas, edges = build(verts, faces)
    find = partition(faces, weighted, edges, LOW_ANGLE)
    if width == "auto":
        width = detect_width(verts, faces, areas, edges, find, diagonal(verts))
    label, _ = absorb(verts, faces, weighted, areas, edges, find, width)
    return len(set(label.values()))


def test_beveled_cube_merges_to_faces():
    # 2-segment bevel: no per-edge angle can give 6 charts, strip merging must
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    assert region_count(verts, faces, "auto") == 6


def test_beveled_cube_zero_width_keeps_bands():
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    assert region_count(verts, faces, 0.0) == 54


def test_plain_cube_is_untouched():
    # no narrow regions, so auto width must not merge the 6 faces
    verts, faces = read_obj(FIXTURES / "cube.obj")
    assert region_count(verts, faces, "auto") == 6


def test_seam_edges_on_plain_cube():
    # the cross: five hinges stay uncut, the other seven cube edges are the
    # seams, and the triangulation diagonals stay interior
    verts, faces = read_obj(FIXTURES / "cube.obj")
    seams = seam_edges(verts, faces)
    assert len(seams) == 7
    assert all(v0 < v1 for v0, v1 in seams)
    assert len(island_groups(faces, seams, face_edges(faces))) == 1


def tube(sides=4, height=1.0):
    """Open-ended tube: merging its way around the ring would make an annulus."""
    verts, faces = [], []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        verts.append([math.cos(angle), math.sin(angle), 0.0])
        verts.append([math.cos(angle), math.sin(angle), height])
    for i in range(sides):
        low, high = 2 * i, 2 * i + 1
        next_low, next_high = 2 * ((i + 1) % sides), 2 * ((i + 1) % sides) + 1
        faces.append([low, next_low, high])
        faces.append([next_low, next_high, high])
    return verts, faces


def test_tube_stops_merging_before_it_closes_the_ring():
    verts, faces = tube()
    weighted, areas, edges = build(verts, faces)
    find = partition(faces, weighted, edges, LOW_ANGLE)
    # width far over the tube's own, so every side is a merge candidate
    label, _ = absorb(verts, faces, weighted, areas, edges, find, 100.0)
    ec, _, _ = region_topology(edges, label)
    assert len(set(label.values())) == 2
    assert all(value == 1 for value in ec.values())


def test_coarse_tube_unrolls_with_one_cut():
    # 22.5 degrees a segment shatters the partition into 16 columns, but the
    # wall is a verified sweep, so it partitions as one annulus and a single
    # cut opens it, however long the strip: slicing long islands is the
    # finish pass's job
    verts, faces = tube(16)
    seams = seam_edges(verts, faces)
    assert len(seams) == 1
    ((v0, v1),) = seams
    assert {verts[v0][2], verts[v1][2]} == {0.0, 1.0}


def test_tall_coarse_tube_closes_ring_for_one_cut():
    # same wall at height 2 unrolls to aspect pi, so close_rings merges the
    # halves back into an annulus and disk_cuts opens it with a single cut
    verts, faces = tube(16, height=2.0)
    seams = seam_edges(verts, faces)
    assert len(seams) == 1
    ((v0, v1),) = seams
    assert {verts[v0][2], verts[v1][2]} == {0.0, 2.0}


def test_ring_closing_refuses_a_crease():
    # two halves of an octagonal prism pass the aspect bound, but their
    # boundaries turn 45 degrees: creases, so the ring must stay open
    verts, faces = tube(8, height=2.0)
    weighted, areas, edges = build(verts, faces)
    label = {i: 0 if i < 8 else 1 for i in range(len(faces))}
    assert close_rings(verts, weighted, areas, edges, label) == label


def test_faceted_tube_unrolls_whole():
    # 45 degrees a segment reads as a crease, so the facets survive as
    # regions, but they are flat panels: the unfold opens the ring at one
    # boundary and the prism unrolls as a single strip
    verts, faces = tube(8)
    seams = seam_edges(verts, faces)
    assert len(seams) == 1
    assert len(island_groups(faces, seams, face_edges(faces))) == 1


@pytest.mark.skipif(not HEX_HEAD.exists(), reason="needs the bench models")
def test_smeared_closed_mesh_still_flattens_at_a_high_angle():
    # a hex head smears every feature to just under 60, so at 66 close_rings
    # seals the closed mesh into one region. the straight runs still find
    # the bolt's tube structure, so it flattens with fewer islands instead
    # of falling back to the CREASE_ANGLE retry
    verts, faces = read_obj(HEX_HEAD)
    faces = [tuple(f) for f in faces]
    at_floor = seam_edges(verts, faces, CREASE_ANGLE)
    high = seam_edges(verts, faces, 66)
    assert high
    edges = face_edges(faces)
    floor_islands = island_groups(faces, at_floor, edges)
    high_islands = island_groups(faces, high, edges)
    assert 1 < len(high_islands) < len(floor_islands)


def test_lower_feature_angle_keeps_shallow_seams():
    # 22.5 degree panel boundaries merge away at the default 30 but survive 15
    verts, faces = elbow(sides=16)
    assert len(seam_edges(verts, faces, angle=15)) > len(seam_edges(verts, faces))


def test_beveled_cube_unfolds_into_one_island():
    # the six faces survive the merges (test_beveled_cube_merges_to_faces),
    # then the unfold hinges them into a cross like the plain cube
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    seams = seam_edges(verts, faces)
    edges = face_edges(faces)
    assert len(island_groups(faces, seams, edges)) == 1


def panel_hinges(verts, faces):
    weighted, _, edges = build(verts, faces)
    root = partition(faces, weighted, edges, LOW_ANGLE)
    label = {i: root(i) for i in range(len(faces))}
    return unfold_hinges(verts, faces, weighted, edges, label)


def test_unfold_drops_the_hinge_that_would_overlap():
    # a unit base with two 3x3 wings folded up from adjacent edges: either
    # wing unfolds fine alone, opened together they land on the same corner
    # area, so one of the two hinges must ship as a seam
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 2.0, 0.0),
        (1.0, 2.0, 3.0),
        (1.0, -1.0, 3.0),
        (-1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (2.0, 1.0, 3.0),
        (-1.0, 1.0, 3.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (7, 6, 5),
        (7, 5, 2),
        (7, 2, 1),
        (7, 1, 4),
        (11, 8, 3),
        (11, 3, 2),
        (11, 2, 9),
        (11, 9, 10),
    ]
    hinges = panel_hinges(verts, faces)
    assert len(hinges) == 1
    assert hinges < {pair(1, 2), pair(2, 3)}


def test_unfold_keeps_hinges_that_open_clear():
    # same shape with wings that stay inside their own quadrant when opened,
    # so the overlap check must not reject either hinge
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.5),
        (0.0, 1.0, 0.5),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 2),
        (4, 2, 1),
        (7, 3, 2),
        (7, 2, 6),
    ]
    assert panel_hinges(verts, faces) == {pair(1, 2), pair(2, 3)}


def test_absorbed_bevel_still_reads_as_a_crease():
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    weighted, areas, edges = build(verts, faces)
    find = partition(faces, weighted, edges, LOW_ANGLE)
    width = detect_width(verts, faces, areas, edges, find, diagonal(verts))
    _, bounds = absorb(verts, faces, weighted, areas, edges, find, width)
    live = [key for key in bounds.length if bounds.length[key] > 0]
    # every boundary left is a cube corner with its bevel absorbed into one
    # side, so all of them must still turn the full 90 degrees. a corner patch
    # carries two bevels at once, which reads a little over
    angles = [bounds.turn[key] / bounds.length[key] for key in live]
    assert angles
    assert min(angles) > CREASE_ANGLE
    assert max(angles) < 120
    # the turn at the boundary's own edges is only the last bevel segment, so
    # the carry is what the crease reading rests on here
    assert min(bounds.step[key] / bounds.length[key] for key in live) < 45
    # and it is spread over a bevel's width, not a surface's, which is what
    # keeps the smooth merge from reading it as curvature
    assert max(bounds.spread[key] / bounds.length[key] for key in live) < width


def capped_tube(sides=32, height=2.0):
    """A tube with a flat triangle-fan cap on top: one region of it is the
    smooth-model sock, a wall merged over its end cap."""
    verts, faces = tube(sides, height)
    top = len(verts)
    verts.append([0.0, 0.0, height])
    for i in range(sides):
        faces.append([2 * i + 1, 2 * ((i + 1) % sides) + 1, top])
    return verts, faces


def elbow(rings=12, sides=12, bend_radius=3.0, tube_radius=1.0):
    """A quarter-torus tube: swept, but around a bending axis."""
    verts, faces = [], []
    for i in range(rings + 1):
        t = (math.pi / 2) * i / rings
        for j in range(sides):
            p = 2 * math.pi * j / sides
            spoke = bend_radius + tube_radius * math.cos(p)
            verts.append(
                [spoke * math.cos(t), tube_radius * math.sin(p), spoke * math.sin(t)]
            )
    for i in range(rings):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


def sweep_regions(verts, faces):
    weighted, areas, edges = build(verts, faces)
    label = {i: 0 for i in range(len(faces))}
    return split_sweeps(verts, faces, weighted, areas, edges, label), len(faces)


def test_sweep_split_lifts_the_cap_off_a_sock():
    verts, faces = capped_tube()
    label, count = sweep_regions(verts, faces)
    regions = collections.defaultdict(set)
    for i, r in label.items():
        regions[r].add(i)
    assert len(regions) == 2
    # the cap fan is exactly the faces added after the wall quads
    fan = set(range(count - 32, count))
    assert fan in regions.values()


def test_sweep_split_cuts_a_bent_tube_into_straight_runs():
    # mid-bend normals sit between wall and cap against any axis, so an
    # elbow is not a sock, it relabels into straight runs instead, and every
    # run must be one connected piece
    verts, faces = elbow()
    label, _ = sweep_regions(verts, faces)
    regions = collections.defaultdict(list)
    for i, r in label.items():
        regions[r].append(i)
    assert len(regions) > 1
    edges = face_edges(faces)
    for members in regions.values():
        assert len(component_faces(members, edges)) == 1


def filleted_tube(sides=32, height=2.0, fillet=8, radius=0.4):
    """A tube whose flat cap meets the wall through a rounded fillet, so no
    boundary in it turns like a crease: the smooth-model sock."""
    verts, faces = [], []
    rings = [(1.0, 0.0), (1.0, height)]
    for k in range(1, fillet + 1):
        t = (math.pi / 2) * k / fillet
        rings.append((1 - radius * (1 - math.cos(t)), height + radius * math.sin(t)))
    for r, z in rings:
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            verts.append([r * math.cos(angle), r * math.sin(angle), z])
    for ring in range(len(rings) - 1):
        for i in range(sides):
            a = ring * sides + i
            b = ring * sides + (i + 1) % sides
            c = (ring + 1) * sides + (i + 1) % sides
            d = (ring + 1) * sides + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    top = len(verts)
    verts.append([0.0, 0.0, rings[-1][1]])
    last = (len(rings) - 1) * sides
    for i in range(sides):
        faces.append([last + i, last + (i + 1) % sides, top])
    return verts, faces


def test_filleted_cap_is_cut_off_at_its_rim():
    # the wall merges straight over a filleted cap, no boundary turns like a
    # crease, so only the sweep split separates them
    verts, faces = filleted_tube()
    edges = face_edges(faces)
    with_rims = seam_edges(verts, faces)
    without = seam_edges(verts, faces, rims=False)
    assert len(island_groups(faces, without, edges)) == 1
    assert len(island_groups(faces, with_rims, edges)) == 2


def test_sweep_split_leaves_a_shallow_shell_alone():
    # a quarter of the capped tube: the wall barely turns, so this is a
    # curved plate with a flange, not a sock, and rims make no sense on it
    verts, faces = capped_tube()
    quarter = faces[0:16] + faces[64:72]
    label, _ = sweep_regions(verts, quarter)
    assert len(set(label.values())) == 1


def euler_after_cut(ec, cuts):
    """Slitting a region along a boundary-to-boundary path splits every vertex
    on it and every one of its edges, so EC goes up by one per cut."""
    return ec + len({v for edge in cuts for v in edge}) - len(cuts)


def test_annulus_region_is_cut_open():
    # the whole tube as one region, which is what a low-curvature tube wall
    # partitions into and no merge can repair
    verts, faces = tube(8)
    _, _, edges = build(verts, faces)
    label = {i: 0 for i in range(len(faces))}
    ec, _, _ = region_topology(edges, label)
    assert ec[0] == 0

    cuts = disk_cuts(verts, edges, label)
    assert len(cuts) == 1  # one rim to the other, along a single side edge
    v0, v1 = next(iter(cuts))
    assert {verts[v0][2], verts[v1][2]} == {0.0, 1.0}
    assert euler_after_cut(ec[0], cuts) == 1


def test_painted_restriction_moves_the_cut():
    # the tube's one cut is a side edge, and every other side edge is the
    # same length, so painting the chosen one has to send it elsewhere
    verts, faces = tube(8)
    _, _, edges = build(verts, faces)
    label = {i: 0 for i in range(len(faces))}
    plain = disk_cuts(verts, edges, label)
    painted = {v: 1.0 for edge in plain for v in edge}

    cuts = disk_cuts(verts, edges, label, painted)
    assert len(cuts) == 1
    assert not painted.keys() & {v for edge in cuts for v in edge}


def test_paint_cannot_block_a_cut_that_has_to_happen():
    # painting everything leaves the region non-disk if the cut is dropped,
    # so a restriction must repel, never veto
    verts, faces = tube(8)
    _, _, edges = build(verts, faces)
    label = {i: 0 for i in range(len(faces))}
    painted = {v: 1.0 for v in range(len(verts))}
    assert len(disk_cuts(verts, edges, label, painted)) == 1


def folded_pair(z):
    """Two triangles sharing the edge (0, 1), the second lifted to z."""
    verts = [[0, 0, 0], [0, 1, 0], [-1, 0.5, 0], [1, 0.5, z]]
    faces = [[0, 1, 2], [1, 0, 3]]
    return verts, faces


def test_crease_relief_orders_concave_convex_flat():
    def relief_at(z):
        verts, faces = folded_pair(z)
        weighted, _, edges = build(verts, faces)
        relief = crease_relief(verts, faces, weighted, edges)
        return relief.get(pair(0, 1), 1.0)

    concave, convex, flat = relief_at(1.0), relief_at(-1.0), relief_at(0.0)
    assert concave < convex < flat == 1.0


def folded_flap():
    """Two flat quad columns and a third folded straight up at x=1, so the
    fold line is the only crease."""
    rows = 7
    verts = (
        [[0, y, 0] for y in range(rows)]
        + [[1, y, 0] for y in range(rows)]
        + [[1, y, 1] for y in range(rows)]
    )
    faces = []
    for y in range(rows - 1):
        faces.append([y, rows + y, rows + y + 1, y + 1])
        faces.append([rows + y, 2 * rows + y, 2 * rows + y + 1, rows + y + 1])
    return verts, faces


def test_cut_path_rides_the_crease():
    # the fold route is 8 long against 6 direct, so only the crease
    # discount can make it win
    verts, faces = folded_flap()
    weighted, _, edges = build(verts, faces)
    relief = crease_relief(verts, faces, weighted, edges)
    adjacent = vertex_adjacency(faces)

    plain = cut_path(verts, adjacent, {0}, {6})
    assert all(verts[v][0] == 0 for v in plain)

    creased = cut_path(verts, adjacent, {0}, {6}, relief=relief)
    assert sum(1 for v in creased if verts[v][0] == 1 and verts[v][2] == 0) == 7


def rook_grid():
    """3x3 vertex grid with only horizontal and vertical edges, so every
    corner-to-corner path is 4 long and only turn count tells them apart."""
    verts = [(x, y, 0.0) for y in range(3) for x in range(3)]
    adjacent = collections.defaultdict(set)
    for y in range(3):
        for x in range(3):
            a = 3 * y + x
            if x < 2:
                adjacent[a].add(a + 1)
                adjacent[a + 1].add(a)
            if y < 2:
                adjacent[a].add(a + 3)
                adjacent[a + 3].add(a)
    return verts, adjacent


def turn_count(verts, path):
    turns = 0
    for u, v, w in zip(path, path[1:], path[2:]):
        a = [verts[v][i] - verts[u][i] for i in range(3)]
        b = [verts[w][i] - verts[v][i] for i in range(3)]
        if cross(a, b) != [0, 0, 0]:
            turns += 1
    return turns


def test_dull_cut_is_a_line():
    # every corner-to-corner path is 4 long, so without the turn penalty the
    # staircase can win on heap order. with it the single-corner L must win
    verts, adjacent = rook_grid()
    path = cut_path(verts, adjacent, {0}, {8}, relief={})
    assert len(path) == 5
    assert turn_count(verts, path) == 1


def test_creased_staircase_beats_the_line():
    # the same grid with a creased staircase: crease edges are exempt from
    # the turn penalty and discounted, so the seam follows the crease
    verts, adjacent = rook_grid()
    stairs = [(0, 1), (1, 4), (4, 5), (5, 8)]
    relief = {pair(a, b): 0.85 for a, b in stairs}
    path = cut_path(verts, adjacent, {0}, {8}, relief=relief)
    assert set(path) == {0, 1, 4, 5, 8}


def test_path_cost_prices_turns():
    verts, _ = rook_grid()
    straight = path_cost(verts, [0, 1, 2], relief={})
    bent = path_cost(verts, [0, 1, 4], relief={})
    assert straight == 2.0
    assert bent == 2.0 + TURN_COST * 0.5


def flat_grid():
    """2x2 quad grid of 8 triangles on z=0, split down the middle line."""
    verts = [[x, y, 0] for y in range(3) for x in range(3)]
    faces = []
    for y in range(2):
        for x in range(2):
            a = 3 * y + x
            faces.append([a, a + 1, a + 4])
            faces.append([a, a + 4, a + 3])
    return verts, faces


def test_tooth_on_a_flat_boundary_is_flattened():
    verts, faces = flat_grid()
    weighted, _, edges = build(verts, faces)
    # left column region 0, right column region 1, except one right triangle
    # sticking into the left as a tooth
    label = {0: 0, 1: 0, 4: 0, 5: 0, 2: 1, 3: 1, 6: 1, 7: 1}
    label[3] = 0

    flat = flatten_teeth(weighted, faces, edges, label)
    assert flat[3] == 1
    assert boundary_edges(edges, flat) == {(1, 4), (4, 7)}


def test_forced_seam_survives_the_tooth_flip():
    verts, faces = flat_grid()
    weighted, _, edges = build(verts, faces)
    label = {0: 0, 1: 0, 4: 0, 5: 0, 2: 1, 3: 1, 6: 1, 7: 1}
    label[3] = 0

    flat = flatten_teeth(weighted, faces, edges, label, forced={(1, 5)})
    assert flat[3] == 0


def folded_planes():
    """Two quad planes meeting at 90 degrees along the line y=0."""
    verts = (
        [[x, 0, 0] for x in range(3)]
        + [[x, -1, 0] for x in range(3)]
        + [[x, 0, 1] for x in range(3)]
    )
    faces = []
    for x in range(2):
        faces.append([x, x + 1, x + 4])
        faces.append([x, x + 4, x + 3])
    for x in range(2):
        faces.append([x, x + 1, x + 7])
        faces.append([x, x + 7, x + 6])
    return verts, faces


def test_tooth_flip_returns_the_seam_to_the_fold():
    verts, faces = folded_planes()
    weighted, _, edges = build(verts, faces)
    # one vertical triangle mislabeled onto the flat plane: its kept edge is
    # the fold itself, so the flip wins even though its lost edges are dull
    label = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    label[4] = 0

    flat = flatten_teeth(weighted, faces, edges, label)
    assert flat[4] == 1
    assert boundary_edges(edges, flat) == {(0, 1), (1, 2)}


def test_corner_on_a_crease_is_not_cut():
    verts, faces = read_obj(FIXTURES / "cube.obj")
    weighted, _, edges = build(verts, faces)
    # top face one region, the rest another: each top triangle has two sharp
    # boundary edges and only a dull diagonal to keep, so no flip may round
    # the corner off
    top = max(v[2] for v in verts)
    label = {
        f: 0 if all(verts[v][2] == top for v in face) else 1
        for f, face in enumerate(faces)
    }

    assert flatten_teeth(weighted, faces, edges, label) == label


def bump_grid():
    """3x3 quad grid on z=0 split down x=1, with the middle right quad
    labeled across the line so the boundary detours around it."""
    verts = [[x, y, 0] for y in range(4) for x in range(4)]
    faces = []
    for y in range(3):
        for x in range(3):
            a = 4 * y + x
            faces.append([a, a + 1, a + 5])
            faces.append([a, a + 5, a + 4])
    label = {f: 0 if (f // 2) % 3 == 0 else 1 for f in range(18)}
    label[8] = label[9] = 0
    return verts, faces, label


def test_reroute_straightens_a_boundary_bump():
    verts, faces, label = bump_grid()
    weighted, areas, edges = build(verts, faces)
    relief = crease_relief(verts, faces, weighted, edges)

    moved = reroute_boundaries(verts, faces, areas, edges, label, relief)
    assert moved[8] == 1 and moved[9] == 1
    assert boundary_edges(edges, moved) == {(1, 5), (5, 9), (9, 13)}


def test_reroute_leaves_a_forced_chain():
    verts, faces, label = bump_grid()
    weighted, areas, edges = build(verts, faces)
    relief = crease_relief(verts, faces, weighted, edges)

    moved = reroute_boundaries(
        verts, faces, areas, edges, label, relief, forced={(5, 6)}
    )
    assert moved == label


def walled_floor():
    """A floor with a wall folded up along y=0."""
    verts = [[x, y, 0] for y in range(3) for x in range(9)] + [
        [x, 0, 1] for x in range(9)
    ]
    faces = []
    for y in range(2):
        for x in range(8):
            a = 9 * y + x
            faces.append([a, a + 1, a + 10])
            faces.append([a, a + 10, a + 9])
    for x in range(8):
        faces.append([x, x + 1, 28 + x])
        faces.append([x, 28 + x, 27 + x])
    return verts, faces


def test_reroute_drops_the_seam_onto_the_fold():
    verts, faces = walled_floor()
    # the wall region reaches over the fold onto the middle of the floor, so
    # the seam runs mostly on flat ground: only the relief discount along the
    # fold can pay for the longer straight route
    label = {f: 1 if f >= 32 else 0 for f in range(48)}
    for f in range(4, 12):
        label[f] = 1
    weighted, areas, edges = build(verts, faces)
    relief = crease_relief(verts, faces, weighted, edges)

    moved = reroute_boundaries(verts, faces, areas, edges, label, relief)
    assert all(moved[f] == (1 if f >= 32 else 0) for f in range(48))
    assert boundary_edges(edges, moved) == {(x, x + 1) for x in range(8)}


def capped_prism(sides=12):
    """Open-bottomed prism with a fan-triangulated top cap and a sharp rim."""
    verts = [[0.0, 0.0, 1.0]]
    for ring_z in (1.0, 0.0):
        for i in range(sides):
            a = 2 * math.pi * i / sides
            verts.append([math.cos(a), math.sin(a), ring_z])
    faces = []
    for i in range(sides):
        faces.append([0, 1 + i, 1 + (i + 1) % sides])
    for i in range(sides):
        a, b = 1 + i, 1 + (i + 1) % sides
        faces.append([b, a, sides + a])
        faces.append([b, sides + a, sides + b])
    return verts, faces


def test_reroute_snaps_a_closed_loop_to_its_rim():
    verts, faces = capped_prism()
    # a rim loop with no junction to anchor it, so only loop_arcs pulls it back
    label = {f: 0 if f < 12 else 1 for f in range(36)}
    label[0] = 1
    weighted, areas, edges = build(verts, faces)
    relief = crease_relief(verts, faces, weighted, edges)

    moved = reroute_boundaries(verts, faces, areas, edges, label, relief)
    assert moved[0] == 0
    rim = {pair(1 + i, 1 + (i + 1) % 12) for i in range(12)}
    assert boundary_edges(edges, moved) == rim


def test_forced_seam_splits_a_flat_face():
    # both merges would take these coplanar triangles, nothing about the shape
    # says cut here, so only the mark can
    verts, faces = read_obj(FIXTURES / "cube.obj")
    weighted, _, edges = build(verts, faces)
    flat = next(
        key
        for key, owners in edges.items()
        if len(owners) == 2 and turn_angle(weighted, owners) < 1
    )
    seams = seam_edges(verts, faces, forced={flat})
    assert flat in seams
    # the unfold takes the rest to one island, the mark stays as its slit
    assert len(island_groups(faces, seams, edges)) == 1


def test_forced_seam_takes_a_detected_one_over():
    # the swept wall needs one cut and detection picks where. marking a panel
    # boundary has to become that cut, not add a second slit beside it
    verts, faces = tube(16)
    edges = face_edges(faces)
    base = seam_edges(verts, faces)
    side = next(
        pair(2 * i, 2 * i + 1) for i in range(16) if pair(2 * i, 2 * i + 1) not in base
    )
    seams = seam_edges(verts, faces, forced={side})
    assert seams == {side}
    assert len(base) == 1
    assert len(island_groups(faces, seams, edges)) == 1


def test_forced_seam_moves_the_band_it_blocks():
    # a mark on one side of a bevel band: absorb has to dissolve the band into
    # the far side, so the boundary lands on the mark instead of a two edge
    # ribbon surviving between the two. read at the region level, the unfold
    # hides ribbons from island counts
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    weighted, areas, edges = build(verts, faces)
    # 22.5 degrees is where a bevel band meets the face beside it
    band = next(
        key
        for key, owners in edges.items()
        if len(owners) == 2 and 20 < turn_angle(weighted, owners) < 25
    )
    find = partition(faces, weighted, edges, LOW_ANGLE, {band})
    width = detect_width(verts, faces, areas, edges, find, diagonal(verts))
    label, _ = absorb(verts, faces, weighted, areas, edges, find, width, {band})
    assert len(set(label.values())) == 6
    a, b = edges[band]
    assert label[a] != label[b]


def test_ring_closing_refuses_a_forced_seam():
    verts, faces = tube(16, height=2.0)
    weighted, areas, edges = build(verts, faces)
    label = {i: 0 if i < 16 else 1 for i in range(len(faces))}
    assert close_rings(verts, weighted, areas, edges, label) != label
    # a vertical edge on the two halves' own boundary
    forced = {pair(0, 1)}
    merged = close_rings(verts, weighted, areas, edges, label, forced=forced)
    assert merged == label


def test_disk_regions_are_left_alone():
    verts, faces = read_obj(FIXTURES / "cube.obj")
    weighted, areas, edges = build(verts, faces)
    find = partition(faces, weighted, edges, LOW_ANGLE)
    label = {i: find(i) for i in range(len(faces))}
    assert disk_cuts(verts, edges, label) == set()


def strip_island(quads, scale=1.0, angle=0.0):
    """A strip of unit quads laid out flat in uv, as the verts, faces and
    per-face corner uvs split_islands takes. Verts sit in the z=0 plane so 3d
    path lengths match uv lengths."""
    points, faces = [], []
    cos, sin = math.cos(angle), math.sin(angle)
    for i in range(quads + 1):
        for y in (0.0, 1.0):
            x = float(i)
            points.append((scale * (x * cos - y * sin), scale * (x * sin + y * cos)))
    for i in range(quads):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces.append([a, c, b])
        faces.append([c, d, b])
    uvs = [[points[v] for v in f] for f in faces]
    verts = [(x, y, 0.0) for x, y in points]
    return verts, faces, uvs


def fold_face(uvs, fi):
    """Swap two corner uvs so the face inverts in uv, like a SLIM fold."""
    uvs[fi] = [uvs[fi][1], uvs[fi][0], uvs[fi][2]]


def island_count(faces, seams):
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            owners.setdefault(pair(f[i], f[(i + 1) % len(f)]), []).append(fi)
    for key, fs in owners.items():
        if len(fs) == 2 and key not in seams:
            a, b = find(fs[0]), find(fs[1])
            if a != b:
                parent[a] = b
    return len({find(i) for i in range(len(faces))})


def merge_islands(a, b):
    """Two islands in one mesh, the second's verts reindexed after the first.
    They share no edge, so they stay separate islands whatever their uvs do."""
    verts_a, faces_a, uvs_a = a
    verts_b, faces_b, uvs_b = b
    shift = len(verts_a)
    faces = faces_a + [[v + shift for v in f] for f in faces_b]
    return verts_a + verts_b, faces, uvs_a + uvs_b


def test_clean_long_island_splits_for_packing():
    # flip-free, but alone it is far longer than the square its own uv area
    # needs, and one long island caps how far the pack can scale everything
    verts, faces, uvs = strip_island(30)
    extra = split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 4


def test_clean_short_strip_is_not_split():
    # a thin strip next to a much bigger island, the kind a bevel band
    # leaves, and cutting those would shatter every beveled model. the big
    # island sets the cap, and the strip is nowhere near it
    verts, faces, uvs = merge_islands(
        strip_island(30, scale=1 / 200), strip_island(2, scale=10.0)
    )
    assert split_islands(verts, faces, set(), uvs) == set()


def test_clean_compact_island_is_not_split():
    verts, faces, uvs = strip_island(4)
    assert split_islands(verts, faces, set(), uvs) == set()


def test_folded_long_island_is_split_into_even_pieces():
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 20)
    extra = split_islands(verts, faces, set(), uvs)
    assert extra
    # 29.3 long against a cap of 7.7, so 4 slices, each under the cap
    assert island_count(faces, extra) == 4


def test_split_is_rotation_invariant():
    verts, faces, uvs = strip_island(30, angle=0.7)
    fold_face(uvs, 20)
    extra = split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 4


def test_split_accepts_shared_edges_and_relief_cache():
    # a caller scanning a joined mesh piece by piece passes one edge map and
    # one relief cache, the cuts must match a default full scan
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 20)
    edges = face_edges(faces)
    groups = island_groups(faces, set(), edges)
    relief_cache = []
    extra = split_islands(verts, faces, set(), uvs, None, groups, edges, relief_cache)
    assert extra == split_islands(verts, faces, set(), uvs)
    assert relief_cache


def test_split_moves_part_the_sliced_strips():
    # the whole scan on plain data: the long strip slices into 4, and the
    # returned uv moves shrink each piece so blender reads them as islands
    verts, faces, uvs = strip_island(30)
    starts = []
    base = 0
    for face in faces:
        starts.append(base)
        base += len(face)
    edges = face_edges(faces)
    assert len(uv_island_groups(faces, uvs, edges)) == 1

    result = split_moves(verts, faces, uvs, starts)
    moves, seams, pieces = result
    assert moves
    face_of = {}
    for face_index, start in enumerate(starts):
        for corner in range(len(faces[face_index])):
            face_of[start + corner] = (face_index, corner)
    moved = [list(corners) for corners in uvs]
    for loop_index, u, v in moves:
        face_index, corner = face_of[loop_index]
        moved[face_index][corner] = (u, v)
    after = uv_island_groups(faces, moved, edges)
    assert len(after) == 4
    assert seams == uv_seams(faces, moved, edges)
    assert sorted(pieces) == sorted(after)

    # one range covering everything scans exactly like the whole-mesh path
    assert split_moves(verts, faces, uvs, starts, [(0, len(faces))]) == result


def test_island_eulers_match_uv_topology():
    verts, faces, uvs = strip_island(6)
    fold_face(uvs, 2)
    annulus_verts, annulus_faces, annulus_uvs = annulus_island()
    offset = len(verts)
    faces = faces + [[v + offset for v in face] for face in annulus_faces]
    uvs = uvs + annulus_uvs
    edges = face_edges(faces)
    seams = {pair(4, 5)}
    groups = island_groups(faces, seams, edges)
    assert len(groups) == 3
    expected = [uv_topology(group, faces, edges, seams)[0] for group in groups]
    assert island_eulers(groups, faces, seams) == expected
    assert sorted(expected) == [0, 1, 1]

    # enough corners that a corner pair key passes int32
    verts, faces, uvs = strip_island(15000)
    assert island_eulers([list(range(len(faces)))], faces, set()) == [1]


def test_folded_compact_island_is_halved():
    # not a strip, but island_ruined all the same, so it still gets one cut
    verts, faces, uvs = strip_island(4)
    fold_face(uvs, 2)
    extra = split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 2


def test_split_is_scale_invariant():
    # the same folded shape at 1/200 scale: packing is scale blind, the cap
    # shrinks with the area, so it slices into the same 4 pieces
    verts, faces, uvs = strip_island(30, scale=1 / 200)
    fold_face(uvs, 20)
    extra = split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 4


def test_halving_cut_takes_the_shortest_path():
    # 5 quads put the halving line mid-quad, where the raw bin cut takes the
    # sqrt(2) diagonal: straightening must slide a face across so the cut
    # lands on a unit column edge, either side of the line
    verts, faces, uvs = strip_island(5)
    fold_face(uvs, 2)
    extra = split_islands(verts, faces, set(), uvs)
    assert extra in ({pair(4, 5)}, {pair(6, 7)})


def test_painted_column_moves_the_halving_cut():
    # both columns either side of the halving line are equally short, so
    # paint on one decides which the straightened cut lands on
    verts, faces, uvs = strip_island(5)
    fold_face(uvs, 2)
    plain = split_islands(verts, faces, set(), uvs)
    painted = {v: 1.0 for edge in plain for v in edge}

    extra = split_islands(verts, faces, set(), uvs, painted)
    assert len(extra) == 1
    assert not painted.keys() & {v for edge in extra for v in edge}


def tube_island(rows, cols, shear):
    """A triangulated tube cut open along one lengthwise column, unrolled with
    a shear so the seam's two sides sit rows apart in uv. Returns the verts,
    faces, per-face corner uvs, the cut seams and each vert's mesh row."""
    radius = cols / (2 * math.pi)
    verts, row_of = [], []
    for r in range(rows + 1):
        for c in range(cols):
            a = 2 * math.pi * c / cols
            verts.append((radius * math.cos(a), radius * math.sin(a), float(r)))
            row_of.append(r)
    faces, uvs = [], []
    for r in range(rows):
        for c in range(cols):
            lo = r * cols
            hi = (r + 1) * cols
            a, b = lo + c, lo + (c + 1) % cols
            d, e = hi + c, hi + (c + 1) % cols
            # the wrapping column keeps counting in uv, which opens the cut
            far = c + 1
            uv = {
                a: (float(c), r + shear * c),
                b: (float(far), r + shear * far),
                d: (float(c), r + 1 + shear * c),
                e: (float(far), r + 1 + shear * far),
            }
            faces.append([a, b, d])
            uvs.append([uv[a], uv[b], uv[d]])
            faces.append([b, e, d])
            uvs.append([uv[b], uv[e], uv[d]])
    seams = {pair(r * cols, (r + 1) * cols) for r in range(rows)}
    return verts, faces, uvs, seams, row_of


def test_sheared_tube_splits_into_rings_not_slivers():
    # 40 rows of 6 unit quads: uv area 240, cap sqrt(480) = 21.9, length 42.1
    # and aspect 7.4, so it is a strip cut into 2 bins. the shear puts the
    # seam's two sides 3 rows apart, so the bin cut reaches the boundary at
    # two far apart points, and the shortest path between those runs
    # lengthwise along the seam. taking it would shave off a 3 face sliver
    # and leave the bins joined, so the straightener has to refuse it
    verts, faces, uvs, seams, row_of = tube_island(40, 6, 0.5)
    extra = split_islands(verts, faces, seams, uvs)
    assert island_count(faces, seams | extra) == 2

    groups = island_groups(faces, seams | extra, face_edges(faces))
    assert min(len(g) for g in groups) >= len(faces) // 4

    # a clean ring: the cut stays within two quad rows
    rows_cut = [row_of[v] for edge in extra for v in edge]
    assert max(rows_cut) - min(rows_cut) <= 2


def cone_island(rows, cols, sector, inner):
    """A cone frustum cut open along one column and unrolled exactly: uv is
    an annulus sector spanning sector radians, slant radius inner to
    inner + rows, unit quads. Returns the verts, faces, per-face corner
    uvs, the cut seams and each vert's mesh column."""
    k = sector / (2 * math.pi)
    rise = math.sqrt(1 - k * k)
    verts, col_of = [], []
    for r in range(rows + 1):
        s = inner + r
        for c in range(cols):
            a = 2 * math.pi * c / cols
            verts.append((k * s * math.cos(a), k * s * math.sin(a), rise * s))
            col_of.append(c)

    def unrolled(r, c):
        t = sector * c / cols
        return ((inner + r) * math.cos(t), (inner + r) * math.sin(t))

    faces, uvs = [], []
    for r in range(rows):
        for c in range(cols):
            lo = r * cols
            hi = (r + 1) * cols
            a, b = lo + c, lo + (c + 1) % cols
            d, e = hi + c, hi + (c + 1) % cols
            # the wrapping column keeps counting in uv, which opens the cut
            far = c + 1
            uv = {
                a: unrolled(r, c),
                b: unrolled(r, far),
                d: unrolled(r + 1, c),
                e: unrolled(r + 1, far),
            }
            faces.append([a, b, d])
            uvs.append([uv[a], uv[b], uv[d]])
            faces.append([b, e, d])
            uvs.append([uv[b], uv[e], uv[d]])
    seams = {pair(r * cols, (r + 1) * cols) for r in range(rows)}
    return verts, faces, uvs, seams, col_of


def test_cone_fan_is_cut_along_radii():
    # a cone frustum unrolls into a 300 degree fan of uv area 199 and arc
    # length 50 against a cap of 20, so it fills with 2 cuts. a cut binned
    # on the principal axis is a chord of the fan: on the mesh it climbs to
    # the thin rim, runs along it, and comes back down, so each cut must
    # instead hold to one column of the cone
    verts, faces, uvs, seams, col_of = cone_island(4, 48, 5 * math.pi / 3, 7.5)
    extra = split_islands(verts, faces, seams, uvs)
    assert island_count(faces, seams | extra) == 3

    remaining = set(extra)
    spans = []
    while remaining:
        chain = {remaining.pop()}
        grew = True
        while grew:
            grew = False
            for other in list(remaining):
                if any(set(other) & set(edge) for edge in chain):
                    remaining.discard(other)
                    chain.add(other)
                    grew = True
        cols_cut = [col_of[v] for edge in chain for v in edge]
        spans.append(max(cols_cut) - min(cols_cut))
    assert len(spans) == 2
    assert max(spans) <= 1


def test_boxed_in_fragment_rejoins_one_side():
    # a chain of 9 faces with face 4 boxed in alone: the fragment reopens
    # only its cuts toward the side it shares more cut edges with, and the
    # halving cut between the two full-sized pieces stays
    links = {f: [] for f in range(9)}
    edge_names = iter("abcdefgh")
    for f in range(8):
        name = next(edge_names)
        links[f].append((f + 1, name))
        links[f + 1].append((f, name))
    links[4].append((5, "x"))
    links[5].append((4, "x"))
    cuts = {"d", "e", "x"}
    pieces = split_pieces(list(range(9)), links, cuts)
    assert sorted(len(p) for p in pieces) == [1, 4, 4]
    pieces = absorb_fragments(pieces, links, cuts, lambda f: 1.0, 2.0)
    assert cuts == {"d"}
    assert sorted(sorted(p) for p in pieces) == [[0, 1, 2, 3], [4, 5, 6, 7, 8]]


def test_split_pieces_among_tiny_pieces_keep_their_cut():
    # halving a 4-face island leaves two pieces under the floor: that is
    # a cut made on purpose, so nothing rejoins
    links = {f: [] for f in range(4)}
    for f, name in zip(range(3), "abc"):
        links[f].append((f + 1, name))
        links[f + 1].append((f, name))
    cuts = {"b"}
    pieces = split_pieces(list(range(4)), links, cuts)
    pieces = absorb_fragments(pieces, links, cuts, lambda f: 1.0, 3.0)
    assert cuts == {"b"}
    assert len(pieces) == 2


def test_large_area_fragment_keeps_its_cut():
    # a one-face piece whose face is big is a packable island, not a crumb,
    # so the count of faces must not decide the rejoin
    links = {f: [] for f in range(9)}
    edge_names = iter("abcdefgh")
    for f in range(8):
        name = next(edge_names)
        links[f].append((f + 1, name))
        links[f + 1].append((f, name))
    cuts = {"d", "e"}
    pieces = split_pieces(list(range(9)), links, cuts)
    assert sorted(len(p) for p in pieces) == [1, 4, 4]

    def area(f):
        return 10.0 if f == 4 else 1.0

    pieces = absorb_fragments(pieces, links, cuts, area, 2.0)
    assert cuts == {"d", "e"}
    assert sorted(len(p) for p in pieces) == [1, 4, 4]


def blob_strip_island(blob, strip):
    """A blob x blob quad square with a strip x 1 quad tail off its lower
    right corner, uvs matching the grid, so the width profile steps hard at
    the join."""
    index = {}
    verts = []

    def vid(x, y):
        key = (x, y)
        if key not in index:
            index[key] = len(verts)
            verts.append((float(x), float(y), 0.0))
        return index[key]

    faces = []
    for x in range(blob):
        for y in range(blob):
            faces.append([vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    for x in range(blob, blob + strip):
        faces.append([vid(x, 0), vid(x + 1, 0), vid(x + 1, 1), vid(x, 1)])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    return verts, faces, uvs


def test_long_island_cut_at_the_feature_neck():
    # a 20x20 blob with a 20x1 tail: length ~39 passes half the cap
    # (sqrt(840) / 2), the width step at the join is ~20x, and the tail
    # alone stays under the cap, so the one cut lands at the neck
    verts, faces, uvs = blob_strip_island(20, 20)
    extra = split_islands(verts, faces, set(), uvs)
    groups = sorted(island_groups(faces, extra, face_edges(faces)), key=len)
    assert len(groups) == 2
    # the small piece is the tail, give or take a slab of blob columns
    assert 15 <= len(groups[0]) <= 60


def test_neck_scan_needs_a_long_island():
    # the same blob and tail next to a much larger island: now under half
    # the cap, and blanket neck cutting measured worse than packing the
    # concave island whole
    verts, faces, uvs = merge_islands(
        blob_strip_island(20, 20), strip_island(1, scale=60.0)
    )
    assert split_islands(verts, faces, set(), uvs) == set()


def test_neck_and_fill_compose():
    # a 45 long tail: the neck cut comes first, and the tail piece alone
    # is still past the cap, so it also fills with one even cut
    verts, faces, uvs = blob_strip_island(20, 45)
    extra = split_islands(verts, faces, set(), uvs)
    assert island_count(faces, extra) == 3


def test_second_neck_found_on_the_piece():
    # a second 12x12 blob on the far end of the tail: the first pass cuts
    # the strongest neck, and only the re-scan of the leftover piece on its
    # own axis finds the second one, freeing the strip from both
    verts, faces, uvs = blob_strip_island(20, 20)

    index = {(x, y): i for i, (x, y, _) in enumerate(verts)}

    def vid(x, y):
        key = (x, y)
        if key not in index:
            index[key] = len(verts)
            verts.append((float(x), float(y), 0.0))
        return index[key]

    for x in range(40, 52):
        for y in range(12):
            faces.append([vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    extra = split_islands(verts, faces, set(), uvs)
    groups = sorted(island_groups(faces, extra, face_edges(faces)), key=len)
    assert [len(g) for g in groups] == [20, 144, 400]


def test_split_scan_restricted_to_given_groups():
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 20)
    assert split_islands(verts, faces, set(), uvs, None, []) == set()
    everything = [list(range(len(faces)))]
    assert split_islands(verts, faces, set(), uvs, None, everything)


def test_split_respects_existing_seams():
    # a seam already cuts the strip in half, so each folded island is 14.3
    # against a cap of 7.7 and splits in two, and the seam edge itself must
    # not come back
    verts, faces, uvs = strip_island(30)
    fold_face(uvs, 10)
    fold_face(uvs, 50)
    mid = pair(30, 31)
    extra = split_islands(verts, faces, {mid}, uvs)
    assert mid not in extra
    assert island_count(faces, extra | {mid}) == 4


def test_uv_islands_follow_the_uv_map():
    # a flat strip is one island until its uvs split mid-column, no seam
    # marks involved
    verts, faces, uvs = strip_island(4)
    edges = face_edges(faces)
    assert len(uv_island_groups(faces, uvs, edges)) == 1
    for fi in range(4, 8):
        uvs[fi] = [(u + 5.0, v) for u, v in uvs[fi]]
    groups = uv_island_groups(faces, uvs, edges)
    assert sorted(len(g) for g in groups) == [4, 4]


def vertex_adjacency(faces):
    adjacent = collections.defaultdict(set)
    for a, b in face_edges(faces):
        adjacent[a].add(b)
        adjacent[b].add(a)
    return adjacent


def test_snap_paths_redraws_a_cut_on_the_dense_mesh():
    # a two segment cut whose three corners land on grid vertices comes back
    # as one connected run of real edges
    verts, faces, _ = grid_island(4, 4)
    mapped = [0, 2, 12]
    paths = snap_paths(verts, vertex_adjacency(faces), mapped, {(0, 1), (1, 2)})
    assert paths == {(0, 1), (1, 2), (2, 7), (7, 12)}


def test_snap_paths_drops_a_cut_with_nowhere_to_go():
    # both ends on one vertex, then an end nothing connects to
    verts, faces, _ = grid_island(4, 4)
    adjacent = vertex_adjacency(faces)
    mapped = [3, 3, len(verts)]
    assert snap_paths(verts, adjacent, mapped, {(0, 1)}) == set()
    assert snap_paths(verts, adjacent, mapped, {(0, 2)}) == set()


def test_snap_paths_drops_a_cut_between_loose_parts():
    # two copies of the grid side by side, a cut with an end on each is
    # dropped and the same-part cut still snaps
    verts, faces, _ = grid_island(4, 4)
    offset = len(verts)
    verts = verts + [(x + 10.0, y, z) for x, y, z in verts]
    faces = faces + [[v + offset for v in face] for face in faces]
    mapped = [0, offset + 2, 2]
    adjacent = vertex_adjacency(faces)
    assert snap_paths(verts, adjacent, mapped, {(0, 1)}) == set()
    assert snap_paths(verts, adjacent, mapped, {(0, 2)}) == {(0, 1), (1, 2)}


def test_uv_fit_scales_into_old_bounds():
    move = uv_fit([(0, 0), (2, 1)], (10, 10, 11, 10.5))
    assert move((0, 0)) == (10, 10)
    assert move((2, 1)) == (11, 10.5)
    # aspect mismatch keeps the scale uniform and the island inside
    move = uv_fit([(0, 0), (2, 1)], (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0.25)
    assert move((2, 1)) == (1, 0.75)


def test_uv_area_fit_keeps_the_old_uv_area():
    # two charts packed in a square, the island they came from was a 4:1 strip
    charts = [
        [(0, 0), (1, 0), (1, 0.4), (0, 0.4)],
        [(0, 0.6), (1, 0.6), (1, 1), (0, 1)],
    ]
    move = uv_area_fit(charts, 0.8, (0, 0, 0.8, 0.2))
    moved = [[move(uv) for uv in chart] for chart in charts]
    area = sum(abs(signed_area(chart)) for chart in moved)
    assert math.isclose(area, 0.8)
    # centered on the old spot, and a bbox fit would have shrunk it instead
    xs = [u for chart in moved for u, _ in chart]
    ys = [v for chart in moved for _, v in chart]
    assert math.isclose((min(xs) + max(xs)) / 2, 0.4)
    assert math.isclose((min(ys) + max(ys)) / 2, 0.1)
    assert max(ys) - min(ys) > 0.2


def test_uv_area_fit_falls_back_to_the_bbox_without_an_area():
    square = [[(0, 0), (2, 0), (2, 2), (0, 2)]]
    move = uv_area_fit(square, 0, (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0)
    assert move((2, 2)) == (1, 1)


def test_islands_overlap_detects_stacked_boxes():
    apart = [(0, 0, 1, 1), (1.1, 0, 2, 1), (0, 1.2, 1, 2)]
    assert not islands_overlap(apart)
    assert islands_overlap([*apart, (0.5, 0.5, 1.5, 1.5)])


def test_island_layout_separates_and_unmirrors():
    boxes = [(0, 0, 1, 1), (0.2, 0.1, 0.8, 0.9)]
    transforms = island_layout(boxes, [0.5, -0.3])
    moved = []
    for (x0, y0, x1, y1), (flip, du, dv) in zip(boxes, transforms):
        us = [x0, x1] if flip is None else [flip - x0, flip - x1]
        moved.append((min(us) + du, y0 + dv, max(us) + du, y1 + dv))
    assert not islands_overlap(moved)
    # only the negative-area island mirrors, within its own bounds
    assert transforms[0][0] is None
    assert transforms[1][0] == 1.0
    assert moved[1][2] - moved[1][0] == boxes[1][2] - boxes[1][0]


def grid_island(cols, rows):
    """A triangulated vertex grid laid out flat in uv, one island."""
    verts = [
        (float(x), float(y), 0.0) for y in range(rows + 1) for x in range(cols + 1)
    ]
    faces = []
    for cy in range(rows):
        for cx in range(cols):
            a = cy * (cols + 1) + cx
            b, c, d = a + 1, a + cols + 2, a + cols + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    return verts, faces, uvs


def annulus_island():
    """A flat 3x3 quad ring with the middle cell missing, identity uvs: no
    flips or crossings, ruined by topology alone."""
    verts, faces, uvs = grid_island(3, 3)
    hole = [faces.index([5, 6, 10]), faces.index([5, 10, 9])]
    for fi in sorted(hole, reverse=True):
        del faces[fi]
        del uvs[fi]
    return verts, faces, uvs


def test_uv_topology_reads_the_unwrap_not_the_mesh():
    verts, faces, uvs = strip_island(4)
    edges = face_edges(faces)
    ec, loops = uv_topology(list(range(len(faces))), faces, edges, set())
    assert ec == 1 and len(loops) == 1
    # a seam splits corners apart: still one disk per side of the cut
    seams = {pair(4, 5)}
    for group in island_groups(faces, seams, edges):
        ec, loops = uv_topology(group, faces, edges, seams)
        assert ec == 1 and len(loops) == 1


def test_annulus_island_is_ruined_and_opened_not_split():
    verts, faces, uvs = annulus_island()
    edges = face_edges(faces)
    group = list(range(len(faces)))
    ec, loops = uv_topology(group, faces, edges, set())
    assert ec == 0 and len(loops) == 2
    assert island_ruined(group, faces, uvs, edges, set())

    extra = split_islands(verts, faces, set(), uvs)
    assert extra
    # opened, not split: still one island, and a disk once the cut is a seam
    assert island_count(faces, extra) == 1
    ec, loops = uv_topology(group, faces, edges, extra)
    assert ec == 1 and len(loops) == 1


def test_slit_sides_crossing_counts_as_ruined():
    # a dangling seam between two interior verts makes a slit whose sides
    # share both mesh verts, and once the sides separate in uv they can
    # cross like any other boundary pair
    verts, faces, uvs = grid_island(3, 2)
    edges = face_edges(faces)
    seams = {pair(5, 6)}
    group = list(range(len(faces)))
    below = faces.index([1, 6, 5])
    above = faces.index([5, 6, 10])
    assert not island_ruined(group, faces, uvs, edges, seams)
    uvs[below] = [(1.0, 0.0), (1.8, 0.85), (1.2, 0.85)]
    uvs[above] = [(1.4, 0.7), (1.6, 1.0), (2.0, 2.0)]
    assert island_ruined(group, faces, uvs, edges, seams)


def test_crosses_includes_collinear_overlap():
    # the branch a naive segment test misses
    assert crosses((0, 0), (2, 0), (1, -1), (1, 1))
    assert not crosses((0, 0), (2, 0), (0, 1), (2, 1))
    assert crosses((0, 0), (2, 0), (3, 0), (1, 0))
    assert not crosses((0, 0), (2, 0), (3, 0), (5, 0))


def sphere(rings=12, sides=24):
    verts, faces = [], []
    for i in range(1, rings):
        t = math.pi * i / rings
        for j in range(sides):
            p = 2 * math.pi * j / sides
            verts.append(
                [math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]
            )
    bottom = len(verts)
    verts.append([0.0, 0.0, 1.0])
    verts.append([0.0, 0.0, -1.0])
    for j in range(sides):
        faces.append([bottom, (j + 1) % sides, j])
        base = (rings - 2) * sides
        faces.append([bottom + 1, base + j, base + (j + 1) % sides])
    for i in range(rings - 2):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


def test_vertex_components_join_on_a_shared_vertex():
    verts, faces = tube(sides=4)
    apart_verts, apart_faces = tube(sides=4)
    offset = len(verts)
    verts += [[v[0] + 5.0, v[1], v[2]] for v in apart_verts]
    faces += [[i + offset for i in f] for f in apart_faces]
    comps = vertex_components(faces)
    assert sorted(len(c) for c in comps) == [8, 8]
    # welding one vertex joins them, the connectivity mesh.separate uses
    welded = [[0 if i == offset else i for i in f] for f in faces]
    assert len(vertex_components(welded)) == 1


def test_beveled_cube_reads_hard():
    verts, faces = read_obj(FIXTURES / "cube-bevel2.obj")
    assert is_hard_surface(verts, faces)


def test_smooth_blob_reads_organic():
    # dense enough that the surface never turns past the partition angle:
    # one region covers everything, the no-structure case
    verts, faces = sphere(rings=24, sides=48)
    assert not is_hard_surface(verts, faces)


def test_coarse_blob_reads_organic():
    # coarse enough that every edge turns into the spread band instead
    verts, faces = sphere(rings=12, sides=24)
    assert not is_hard_surface(verts, faces)


def test_smooth_cylinder_reads_hard():
    # no crease anywhere, the hard call comes from the sweep rims, the
    # screwdriver case
    verts, faces = capped_tube(sides=48)
    assert is_hard_surface(verts, faces)


def ngon_capped_cylinder(sides=32, height=2.0):
    """Blender's default cylinder: a quad wall closed by two ngon caps."""
    verts, faces = [], []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        verts.append([math.cos(angle), math.sin(angle), 0.0])
        verts.append([math.cos(angle), math.sin(angle), height])
    for i in range(sides):
        low, high = 2 * i, 2 * i + 1
        next_low = 2 * ((i + 1) % sides)
        faces.append([low, next_low, next_low + 1, high])
    faces.append([2 * i for i in reversed(range(sides))])
    faces.append([2 * i + 1 for i in range(sides)])
    return verts, faces


def test_default_cylinder_reads_hard():
    verts, faces = ngon_capped_cylinder()
    assert is_hard_surface(verts, faces)


def test_default_cylinder_cuts_rims_and_one_seam():
    # both rims plus a single axial cut: the wall as one island, each cap its
    # own. absorb used to dissolve the wall columns into the caps instead
    verts, faces = ngon_capped_cylinder()
    seams = seam_edges(verts, faces)
    groups = island_groups(faces, seams, face_edges(faces))
    assert sorted(len(g) for g in groups) == [1, 1, 32]


def test_sweep_rims_leave_blobs_alone():
    verts, faces = sphere(rings=12, sides=24)
    rims, walls = sweep_rims(verts, faces)
    assert not rims and not walls


def test_sweep_rims_split_bent_tubes_into_straight_runs():
    # a quarter-torus tube is too bent for one axis, so it parts into
    # straight runs with a rim ring between them instead of staying whole
    verts, faces = elbow(rings=24, sides=16)
    rims, walls = sweep_rims(verts, faces)
    assert walls == set(range(len(faces)))
    # at least one full ring of forced rim edges between two runs
    assert len(rims) >= 16


def stadium_tube(height=4.0, width=2.0, radius=0.3, arc=8, flats=4):
    """Open tube over a stadium profile, a flat bar: two flat sides joined
    by semicircular ends whose segments turn under the crease angle."""
    profile = []
    half = width / 2 - radius
    for k in range(arc + 1):
        a = -math.pi / 2 + math.pi * k / arc
        profile.append((half + radius * math.cos(a), radius * math.sin(a)))
    for k in range(1, flats):
        profile.append((half - 2 * half * k / flats, radius))
    for k in range(arc + 1):
        a = math.pi / 2 + math.pi * k / arc
        profile.append((-half + radius * math.cos(a), radius * math.sin(a)))
    for k in range(1, flats):
        profile.append((-half + 2 * half * k / flats, -radius))
    verts, faces = [], []
    for x, y in profile:
        verts.append([x, y, 0.0])
        verts.append([x, y, height])
    sides = len(profile)
    for i in range(sides):
        low, high = 2 * i, 2 * i + 1
        next_low = 2 * ((i + 1) % sides)
        faces.append([low, next_low, high])
        faces.append([next_low, next_low + 1, high])
    return verts, faces


def test_flat_bar_wall_panels_at_its_soft_ridges():
    # the wall splits lengthwise at the rounded ends into two panels, the
    # cut an artist puts on the ridge
    verts, faces = stadium_tube()
    rims, walls = sweep_rims(verts, faces)
    assert walls == set(range(len(faces)))
    groups = island_groups(faces, rims, face_edges(faces))
    assert len(groups) == 2
    assert min(len(g) for g in groups) > len(faces) // 4


def test_round_tube_wall_keeps_one_wrap():
    # a circular profile has no ridge to cut, coarse or fine: facet spikes
    # alone must not shred the wall into lengthwise strips
    for sides in (16, 32):
        verts, faces = tube(sides=sides, height=4.0)
        rims, walls = sweep_rims(verts, faces)
        assert walls == set(range(len(faces)))
        assert not rims


def test_sweep_rims_still_reject_shattered_coarse_elbows():
    # at 12 sides the cross-tube edges turn past the partition angle, the
    # cluster shatters into lengthwise strips and no strip is a wall
    verts, faces = elbow()
    rims, walls = sweep_rims(verts, faces)
    assert not rims and not walls


WRENCH = Path(__file__).parents[1] / "bench/models/hard-surface/bevel/pipe_wrench.obj"


@pytest.mark.skipif(not WRENCH.exists(), reason="needs the bench models")
def test_sweep_walls_stay_flattenable_on_the_wrench():
    # the handle's hanging loop must be trimmed off the wall: claiming a
    # surface with a handle through it ruins the whole strip at the engine
    verts, faces = read_obj(WRENCH)
    faces = [tuple(f) for f in faces]
    claimed = 0
    for comp in vertex_components(faces):
        part = [faces[i] for i in comp]
        rims, walls = sweep_rims(verts, part)
        edges = face_edges(part)
        for piece in component_faces(walls, edges):
            assert surface_genus(piece, part, edges) <= 0
        claimed += len(walls)
    assert claimed


def uv_groups(faces, uvs):
    return uv_island_groups(faces, uvs, face_edges(faces))


def target_box(targets):
    xs = [p[0] for p in targets.values()]
    ys = [p[1] for p in targets.values()]
    return min(xs), min(ys), max(xs), max(ys)


def on_perimeter(point, box, tol=1e-6):
    x0, y0, x1, y1 = box
    x, y = point
    inside = x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol
    edge = min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1)) < tol
    return inside and edge


def test_rectify_straightens_a_wavy_strip():
    verts, faces, uvs = grid_island(8, 1)

    def wobble(uv):
        x, y = uv
        lean = 0.05 if int(x) % 2 else -0.05
        return (x, y + lean) if y in (0.0, 1.0) else uv

    uvs = [[wobble(uv) for uv in face] for face in uvs]
    plans = rectify_targets(uvs, uv_groups(faces, uvs))
    assert len(plans) == 1
    group, targets, inner = plans[0]
    assert sorted(group) == list(range(len(faces)))
    box = target_box(targets)
    assert all(on_perimeter(t, box) for t in targets.values())
    corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    hits = sum(
        1 for c in corners if any(math.dist(c, t) < 1e-6 for t in targets.values())
    )
    assert hits == 4
    # the wavy long sides actually moved onto the straight edges
    assert any(math.dist(p, t) > 0.01 for p, t in targets.items())


def arc_island(radius=2.0, width=0.5, span=240, segments=24, rows=2):
    """A curled strip: an annular arc in uv, the shape a bent panel
    flattens into. Nearest-to-box corner picking lands on the outer bulge
    of a curl this deep, only turning finds the real ends."""
    points = []
    for k in range(segments + 1):
        a = math.radians(span) * k / segments
        for r in range(rows + 1):
            rad = radius - width / 2 + width * r / rows
            points.append((rad * math.cos(a), rad * math.sin(a)))
    faces, uvs = [], []
    for k in range(segments):
        for r in range(rows):
            base = k * (rows + 1) + r
            quad = (base, base + 1, base + rows + 2, base + rows + 1)
            faces.append(quad)
            uvs.append([points[v] for v in quad])
    return faces, uvs


def test_rectify_straightens_a_curled_strip():
    radius, width, span, segments = 2.0, 0.5, 240, 24
    faces, uvs = arc_island(radius, width, span, segments)
    plans = rectify_targets(uvs, uv_groups(faces, uvs))
    assert len(plans) == 1
    _, targets, inner = plans[0]
    # unrolled to its true arc length: the chord-based box fit reads a 240
    # degree curl far shorter than its spine
    spine = math.radians(span) * radius
    diagonal = math.sqrt(spine**2 + width**2)
    reach = max(math.dist(a, b) for a in targets.values() for b in targets.values())
    assert abs(reach - diagonal) / diagonal < 0.05
    # the mid ring rides along, unrolled evenly down the rectangle's middle
    assert inner
    step = spine / segments
    mids = [
        inner[
            (
                radius * math.cos(math.radians(span) * k / segments),
                radius * math.sin(math.radians(span) * k / segments),
            )
        ]
        for k in range(1, segments)
    ]
    for a, b in zip(mids, mids[1:]):
        assert 0.8 * step < math.dist(a, b) < 1.2 * step


def test_rectify_skips_a_blob():
    points = [(0.0, 0.0)] + [
        (math.cos(a * math.pi / 3), math.sin(a * math.pi / 3)) for a in range(6)
    ]
    faces = [(0, i + 1, (i + 1) % 6 + 1) for i in range(6)]
    uvs = [[points[v] for v in face] for face in faces]
    assert rectify_targets(uvs, uv_groups(faces, uvs)) == []


def test_rectify_admits_a_toothed_strip_by_its_boundary():
    # deep teeth push the strip under the share gate, but the boundary
    # elongation lets it in, and the rectangle takes the unrolled tooth
    # lengths instead of the box width
    verts, faces, uvs = grid_island(12, 1)

    def tooth(uv):
        x, y = uv
        lean = 0.5 if int(x) % 2 else -0.5
        return (x, y + lean) if y in (0.0, 1.0) else uv

    uvs = [[tooth(uv) for uv in face] for face in uvs]
    plans = rectify_targets(uvs, uv_groups(faces, uvs))
    assert len(plans) == 1
    _, targets, inner = plans[0]
    box = target_box(targets)
    assert all(on_perimeter(t, box) for t in targets.values())
    assert box[2] - box[0] > 14


def test_rectify_reaches_islands_bordering_their_own_cut():
    # a cut open tube carries two uvs on every cut vertex, so the boundary
    # must be walked in uv points, not mesh vertices
    verts, faces, uvs, seams, row_of = tube_island(6, 8, 0.0)
    plans = rectify_targets(uvs, uv_groups(faces, uvs))
    assert len(plans) == 1
    group, targets, inner = plans[0]
    assert sorted(group) == list(range(len(faces)))
    box = target_box(targets)
    assert all(on_perimeter(t, box) for t in targets.values())


def test_rectify_skips_an_island_with_a_hole():
    verts, faces, uvs = annulus_island()
    assert rectify_targets(uvs, uv_groups(faces, uvs)) == []


FLAT_QUAD_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
FLAT_QUAD_UVS = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]


def test_flatten_distortion_is_four_at_isometry():
    value = flatten_distortion(FLAT_QUAD_VERTS, [(0, 1, 2, 3)], FLAT_QUAD_UVS, [0])
    assert abs(value - 4.0) < 1e-9


def test_flatten_distortion_grows_with_stretch():
    stretched = [[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]]
    value = flatten_distortion(FLAT_QUAD_VERTS, [(0, 1, 2, 3)], stretched, [0])
    assert value > 4.5


def test_flatten_distortion_flags_a_real_flip():
    faces = [(0, 1, 2), (0, 2, 3)]
    uvs = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    ]
    assert flatten_distortion(FLAT_QUAD_VERTS, faces, uvs, [0, 1]) == math.inf


def test_flatten_distortion_reads_a_mirrored_island_like_its_source():
    mirrored = [[(0.0, 0.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0)]]
    value = flatten_distortion(FLAT_QUAD_VERTS, [(0, 1, 2, 3)], mirrored, [0])
    assert abs(value - 4.0) < 1e-9


def test_collapsed_uvs_are_detected():
    point = (0.1, 0.9987)
    assert uvs_collapsed([[point, point, point]] * 100)


def test_real_uvs_are_not_collapsed():
    quad = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]
    assert not uvs_collapsed([quad])
