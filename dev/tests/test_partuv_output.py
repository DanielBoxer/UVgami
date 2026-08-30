"""Output-layer tests for partuv's connected OBJ writer, no native core needed."""

import numpy as np
import pytest
from partuv import output
from partuv.common import UnwrapError


class FakeComponent:
    def __init__(self, V, F, UV, source_vid):
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int32)
        self.UV = np.asarray(UV, dtype=np.float64)
        self.source_vid = np.asarray(source_vid, dtype=np.int32)


class FakeParts:
    def __init__(self, component):
        self._component = component

    def to_components(self):
        return self._component


def read_obj(path):
    vertices, uvs, faces = [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:]])
        elif parts[0] == "vt":
            uvs.append([float(x) for x in parts[1:]])
        elif parts[0] == "f":
            faces.append(
                [tuple(int(i) - 1 for i in token.split("/")) for token in parts[1:]]
            )
    return vertices, uvs, faces


# a unit square as two triangle charts split on the diagonal: the engine
# duplicates the diagonal vertices per chart, provenance maps them back
SQUARE_V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
SQUARE_F = [[0, 1, 2], [0, 2, 3]]


def two_chart_square():
    V = [SQUARE_V[i] for i in (0, 1, 2, 0, 2, 3)]
    F = [[0, 1, 2], [3, 4, 5]]
    UV = [[0, 0], [0.4, 0], [0.4, 0.4], [0.6, 0], [1, 0], [1, 0.4]]
    return FakeComponent(V, F, UV, [0, 1, 2, 0, 2, 3])


def test_charts_share_geometry_but_not_uv_indices(tmp_path):
    output.save_results(tmp_path, FakeParts(two_chart_square()), [], SQUARE_V, SQUARE_F)

    vertices, uvs, faces = read_obj(tmp_path / "final_components.obj")
    assert vertices == [[float(x) for x in v] for v in SQUARE_V]
    assert len(uvs) == 6
    # both charts reference source vertices 0 and 2 with distinct uv indices
    assert [[v for v, _ in face] for face in faces] == SQUARE_F
    assert faces[0][0] == (0, 0)
    assert faces[1][0] == (0, 3)
    assert faces[0][2][1] != faces[1][1][1]


def test_coincident_source_vertices_stay_separate(tmp_path):
    # two triangles touching at a doubled vertex: ids 1 and 4 share a position
    source_V = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [1, 0, 0], [2, 1, 0]]
    source_F = [[0, 1, 2], [3, 4, 5]]
    comp = FakeComponent(
        source_V,
        source_F,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 0], [2, 1]],
        [0, 1, 2, 3, 4, 5],
    )

    output.save_results(tmp_path, FakeParts(comp), [], source_V, source_F)

    vertices, _, faces = read_obj(tmp_path / "final_components.obj")
    assert len(vertices) == 6
    assert faces[0][1][0] == 1
    assert faces[1][1][0] == 4


def reject_cases():
    missing = two_chart_square()
    missing.source_vid = np.array([], dtype=np.int32)
    out_of_range = two_chart_square()
    out_of_range.source_vid = np.array([0, 1, 2, 0, 2, 9], dtype=np.int32)
    wrong_position = two_chart_square()
    wrong_position.source_vid = np.array([0, 1, 2, 1, 2, 3], dtype=np.int32)
    # both faces map onto source face 0: face 1 is doubled, face 2 missing
    duplicate = two_chart_square()
    duplicate.V = np.asarray([SQUARE_V[i] for i in (0, 1, 2, 0, 1, 2)], float)
    duplicate.source_vid = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    bad_uv = two_chart_square()
    bad_uv.UV = bad_uv.UV.copy()
    bad_uv.UV[3, 0] = np.nan
    flipped = two_chart_square()
    flipped.F = np.asarray([[0, 1, 2], [3, 5, 4]], dtype=np.int32)
    return {
        "missing": missing,
        "out_of_range": out_of_range,
        "wrong_position": wrong_position,
        "duplicate_face": duplicate,
        "non_finite_uv": bad_uv,
        "flipped_winding": flipped,
    }


@pytest.mark.parametrize("name,comp", reject_cases().items())
def test_invalid_provenance_rejected_before_writing(tmp_path, name, comp):
    with pytest.raises(UnwrapError) as error:
        output.save_results(tmp_path, FakeParts(comp), [], SQUARE_V, SQUARE_F)
    assert error.value.exit_code == 5
    assert not (tmp_path / "final_components.obj").exists()
