import importlib.util
from pathlib import Path

import numpy

# loaded from file, the addon package imports bpy
spec = importlib.util.spec_from_file_location(
    "addon_similar", Path(__file__).parents[2] / "src" / "similar.py"
)
similar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similar)

ROTATE_Z = numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MIRROR_X = numpy.diag([-1.0, 1.0, 1.0])
IDENTITY = numpy.identity(4)


class Elements:
    def __init__(self, count, values):
        self.count = count
        self.values = values

    def __len__(self):
        return self.count

    def foreach_get(self, attr, array):
        array[:] = self.values[attr]


class FakePolygon:
    def __init__(self, material_index):
        self.material_index = material_index
        self.use_smooth = True


class Polygons(Elements):
    def __init__(self, count, values, material_index):
        super().__init__(count, values)
        self.material_index = material_index

    def __iter__(self):
        for _ in range(self.count):
            yield FakePolygon(self.material_index)


class FakeMesh:
    def __init__(self, coords, faces, materials=(), material_index=0):
        corners = [v for face in faces for v in face]
        self.vertices = Elements(len(coords), {"co": [c for co in coords for c in co]})
        self.loops = Elements(len(corners), {"vertex_index": corners})
        self.polygons = Polygons(
            len(faces), {"loop_total": [len(face) for face in faces]}, material_index
        )
        self.materials = list(materials)


class FakeObject:
    def __init__(self, coords, faces, materials=(), material_index=0):
        self.data = FakeMesh(coords, faces, list(materials), material_index)
        self.matrix_world = IDENTITY


TETRA_COORDS = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)]
TETRA_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def moved_coords(rotation, translation):
    return [tuple(rotation @ co + translation) for co in TETRA_COORDS]


def test_rigid_fit_recovers_rotation():
    source = numpy.array(TETRA_COORDS, dtype=float)
    target = source @ ROTATE_Z.T + (5.0, 2.0, -1.0)

    matrix, error = similar.rigid_fit(source, target)

    assert error < 1e-9
    assert numpy.allclose(matrix[:3, :3], ROTATE_Z)
    assert numpy.allclose(matrix[:3, 3], (5.0, 2.0, -1.0))


def test_find_twins_matches_moved_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    twin = FakeObject(moved_coords(ROTATE_Z, numpy.array([4.0, 0, 0])), TETRA_FACES)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    matched_rep, matrix, exact = twins[twin]
    assert matched_rep is rep
    assert exact
    moved = numpy.array(TETRA_COORDS, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
    assert numpy.allclose(
        moved, numpy.array(twin.data.vertices.values["co"]).reshape(-1, 3)
    )


def test_find_twins_matches_mirrored_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    twin = FakeObject(moved_coords(MIRROR_X, numpy.array([9.0, 1, 2])), TETRA_FACES)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    _, matrix, _ = twins[twin]
    assert numpy.linalg.det(matrix[:3, :3]) < 0


def test_find_twins_rejects_deformed_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    stretched = [(x * 1.1, y, z) for x, y, z in TETRA_COORDS]
    other = FakeObject(stretched, TETRA_FACES)

    assert similar.find_twins([rep, other]) == {}


def reordered(coords, faces, order):
    new_coords = [None] * len(coords)
    for old, new in enumerate(order):
        new_coords[new] = coords[old]
    new_faces = [tuple(order[v] for v in face) for face in faces]
    return new_coords, new_faces


def test_find_twins_matches_reordered_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    coords, faces = reordered(
        moved_coords(ROTATE_Z, numpy.array([4.0, 0, 0])), TETRA_FACES, [2, 0, 3, 1]
    )
    twin = FakeObject(coords, faces[::-1])

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    matched_rep, matrix, exact = twins[twin]
    assert matched_rep is rep
    assert not exact
    moved = numpy.array(TETRA_COORDS, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
    twin_coords = numpy.array(coords, dtype=float)
    assert sorted(map(tuple, numpy.round(moved, 6))) == sorted(
        map(tuple, numpy.round(twin_coords, 6))
    )


def test_find_twins_refuses_reordered_copy_with_other_material():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES, materials=["red"])
    coords, faces = reordered(
        moved_coords(ROTATE_Z, numpy.array([4.0, 0, 0])), TETRA_FACES, [2, 0, 3, 1]
    )
    twin = FakeObject(coords, faces[::-1], materials=["blue"])

    assert similar.find_twins([rep, twin]) == {}


def test_find_twins_matches_mirrored_reordered_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    coords, faces = reordered(
        moved_coords(MIRROR_X, numpy.array([9.0, 1, 2])), TETRA_FACES, [1, 3, 0, 2]
    )
    twin = FakeObject(coords, faces)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    _, matrix, _ = twins[twin]
    assert numpy.linalg.det(matrix[:3, :3]) < 0


def octagon_tube():
    coords = []
    for level in (0.0, 1.0):
        for step in range(8):
            angle = step * numpy.pi / 4
            coords.append((numpy.cos(angle), numpy.sin(angle), level))
    faces = []
    for step in range(8):
        after = (step + 1) % 8
        faces.append((step, after, after + 8, step + 8))
    return coords, faces


# pca axes are arbitrary in the tube's round plane, needs the rotation search
def test_find_twins_matches_symmetric_rotated_copy():
    coords, faces = octagon_tube()
    rep = FakeObject(coords, faces)
    turn = numpy.pi / 6
    rotation = numpy.array(
        [
            [numpy.cos(turn), -numpy.sin(turn), 0.0],
            [numpy.sin(turn), numpy.cos(turn), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = [tuple(rotation @ co + (2.0, -1.0, 0.5)) for co in coords]
    order = [(index * 5 + 3) % 16 for index in range(16)]
    twin_coords, twin_faces = reordered(moved, faces, order)
    twin = FakeObject(twin_coords, twin_faces)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    _, matrix, _ = twins[twin]
    fitted = numpy.array(coords) @ matrix[:3, :3].T + matrix[:3, 3]
    assert sorted(map(tuple, numpy.round(fitted, 6))) == sorted(
        map(tuple, numpy.round(numpy.array(twin_coords), 6))
    )


CUBE_COORDS = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
CUBE_FACES = [
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
]


# all three pca axes are arbitrary on a cube, needs the permutation search
def test_find_twins_matches_symmetric_reordered_copy():
    rep = FakeObject(CUBE_COORDS, CUBE_FACES)
    moved = [tuple(ROTATE_Z @ co + (5.0, 0.0, 0.0)) for co in CUBE_COORDS]
    order = [(index * 3 + 1) % 8 for index in range(8)]
    twin_coords, twin_faces = reordered(moved, CUBE_FACES, order)
    twin = FakeObject(twin_coords, twin_faces)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    matched_rep, matrix, exact = twins[twin]
    assert matched_rep is rep
    assert not exact
    fitted = numpy.array(CUBE_COORDS) @ matrix[:3, :3].T + matrix[:3, 3]
    assert sorted(map(tuple, numpy.round(fitted, 6))) == sorted(
        map(tuple, numpy.round(numpy.array(twin_coords), 6))
    )


# same quad region and vertex positions, opposite diagonal: not a duplicate
def test_find_twins_rejects_different_topology():
    coords = [(0, 0, 0), (3, 0, 0), (2, 1, 0), (0, 1, 0)]
    rep = FakeObject(coords, [(0, 1, 2), (0, 2, 3)])
    other = FakeObject(coords, [(0, 1, 3), (1, 2, 3)])

    assert similar.find_twins([rep, other]) == {}


SQUARE = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]


def test_find_stacks_groups_exact_copies():
    uvs = [SQUARE, SQUARE, [(0.5, 0.5), (0.7, 0.5), (0.7, 0.7), (0.5, 0.7)]]
    groups = [[0], [1], [2]]

    stacks = similar.find_stacks(groups, uvs)

    assert stacks == [([0], [[1]])]


def test_find_stacks_ignores_partial_overlap():
    shifted = [(u + 0.05, v) for u, v in SQUARE]
    uvs = [SQUARE, shifted]

    assert similar.find_stacks([[0], [1]], uvs) == []


def test_write_twin_output_transforms_vertices(tmp_path):
    source = tmp_path / "rep.obj"
    source.write_text("o rep\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvt 0 0\nf 1/1 2/1 3/1\n")
    target = tmp_path / "twin.obj"
    matrix = numpy.identity(4)
    matrix[:3, :3] = ROTATE_Z
    matrix[:3, 3] = (10.0, 0.0, 0.0)

    similar.write_twin_output(source, target, matrix)
    lines = target.read_text().splitlines()

    assert lines[1] == "v 10.000000000 1.000000000 0.000000000"
    assert lines[4] == "vt 0 0"
    assert lines[5] == "f 1/1 2/1 3/1"


def test_write_twin_output_flips_mirrored_winding(tmp_path):
    source = tmp_path / "rep.obj"
    source.write_text("v 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
    target = tmp_path / "twin.obj"
    matrix = numpy.identity(4)
    matrix[:3, :3] = MIRROR_X

    similar.write_twin_output(source, target, matrix)
    lines = target.read_text().splitlines()

    assert lines[0] == "v -1.000000000 0.000000000 0.000000000"
    assert lines[3] == "f 3 2 1"


WING_COORDS = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (-1, 0, 0), (-2, 0, 0), (-1, 1, 0)]
WING_FACES = [(0, 1, 2), (3, 4, 5)]


def test_mirror_permutations_maps_a_symmetric_mesh():
    mesh = FakeMesh(WING_COORDS, WING_FACES)

    maps = similar.mirror_permutations(mesh, (0, 0, 0), ["X"])

    assert maps == [{0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}]


def test_mirror_permutations_rejects_asymmetric_positions():
    coords = [co for co in WING_COORDS]
    coords[4] = (-2.5, 0, 0)
    mesh = FakeMesh(coords, WING_FACES)

    assert similar.mirror_permutations(mesh, (0, 0, 0), ["X"]) is None


def test_mirror_permutations_rejects_asymmetric_triangulation():
    # both squares of the symmetric shape split, but along opposite diagonals
    coords = [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (1, 1, 0),
        (-1, 0, 0),
        (-2, 0, 0),
        (-2, 1, 0),
        (-1, 1, 0),
    ]
    mirrored = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]
    flipped = [(0, 1, 2), (0, 2, 3), (4, 5, 7), (5, 6, 7)]

    assert similar.mirror_permutations(FakeMesh(coords, mirrored), (0, 0, 0), ["X"])
    assert (
        similar.mirror_permutations(FakeMesh(coords, flipped), (0, 0, 0), ["X"]) is None
    )


def test_mirror_matches_snaps_drift_and_drops_the_asymmetric_vertex():
    # one wing tip drifted within tolerance, plus a vertex with no counterpart
    coords = [co for co in WING_COORDS] + [(0.5, 2.0, 0.0)]
    coords[4] = (-2.002, 0, 0)
    mesh = FakeMesh(coords, WING_FACES)

    maps = similar.mirror_matches(mesh, (0, 0, 0), ["X"])

    assert maps == [{0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}]


def test_mirror_matches_ignores_topology():
    # the asymmetric triangulation mirror_permutations rejects
    coords = [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (1, 1, 0),
        (-1, 0, 0),
        (-2, 0, 0),
        (-2, 1, 0),
        (-1, 1, 0),
    ]
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 7), (5, 6, 7)]

    maps = similar.mirror_matches(FakeMesh(coords, faces), (0, 0, 0), ["X"])

    assert maps == [{0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}]


def test_mirror_matches_returns_none_when_nothing_matches():
    coords = [(1, 0, 0), (2, 0, 0), (1, 1, 0)]
    mesh = FakeMesh(coords, [(0, 1, 2)])

    assert similar.mirror_matches(mesh, (0, 0, 0), ["X"]) is None


def test_mirror_permutations_covers_only_the_symmetric_parts():
    # the wing pair qualifies, the lone off-center triangle drops out
    coords = WING_COORDS + [(0.5, 2.0, 0.0), (1.5, 2.0, 0.0), (0.5, 3.0, 0.0)]
    faces = WING_FACES + [(6, 7, 8)]
    mesh = FakeMesh(coords, faces)

    maps = similar.mirror_permutations(mesh, (0, 0, 0), ["X"])

    assert maps == [{0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}]
