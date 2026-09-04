import importlib.util
import sys
import time
from pathlib import Path

import pytest

# loaded from file, the addon package imports bpy
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import (  # noqa: E402
    FlattenEngine,
    FlattenError,
    check_manifold,
    preseed_uvs,
)
from seams.preseed import _read_uvs, submesh  # noqa: E402

BUNDLED = Path(__file__).parents[2] / "engine-builds" / "windows" / "optcuts.exe"

CUBE_VERTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]
CUBE_FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


class GridEngine:
    def __init__(self):
        self.flatten_calls = []

    def flatten(self, verts, faces, seams, cancelled=None):
        self.flatten_calls.append((len(faces), frozenset(seams)))
        out = []
        for i, face in enumerate(faces):
            x = float(i)
            corners = [(x, 0.0), (x + 1.0, 0.0), (x + 1.0, 1.0), (x, 1.0)]
            corners = corners[: len(face)]
            out.append(corners[::-1] if i == 0 else corners)
        return out


def test_submesh_compacts_and_remaps_seams():
    sub_verts, sub_faces, sub_seams = submesh(
        CUBE_VERTS, CUBE_FACES, [0, 2], {(0, 1), (6, 7)}
    )
    assert len(sub_faces) == 2
    assert len(sub_verts) == 6
    # (0, 1) survives remapped, (6, 7) is outside the subset
    assert len(sub_seams) == 1
    (a, b) = next(iter(sub_seams))
    assert sub_verts[a] == CUBE_VERTS[0]
    assert sub_verts[b] == CUBE_VERTS[1]


def test_read_uvs_rejects_face_mismatch(tmp_path):
    out = tmp_path / "flatten.obj"
    out.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")
    with pytest.raises(FlattenError):
        _read_uvs(out, 2)


def test_engine_failure_raises(tmp_path):
    engine = FlattenEngine(tmp_path / "missing.exe", tmp_path)
    with pytest.raises(FlattenError):
        engine.flatten(CUBE_VERTS, CUBE_FACES, set())


def test_check_manifold_rejects_three_owner_edge():
    faces = list(CUBE_FACES) + [(0, 3, 6)]
    with pytest.raises(FlattenError, match="Non Manifold"):
        check_manifold(faces)


def test_preseed_marked_only_uses_given_seams():
    engine = GridEngine()
    marked = {(0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7)}
    seams, uvs = preseed_uvs(
        engine, CUBE_VERTS, CUBE_FACES, marked="ONLY", marked_seams=marked
    )
    assert marked <= seams
    assert all(uv is not None for uv in uvs)
    # the ruined layout ships as-is, the engine recuts the bad island itself
    assert len(engine.flatten_calls) == 1
    first = engine.flatten_calls[0]
    assert first[0] == len(CUBE_FACES)
    # marked seams reach the engine remapped but complete
    assert len(first[1]) == len(marked)
    # the flipped first face comes back exactly as the fake produced it
    assert uvs[0] == [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]


def test_preseed_only_leaves_other_faces_alone():
    # two loose quads, a preseed holds whole loose parts, never a slice
    verts = CUBE_VERTS[:4] + [(x + 5.0, y, z) for x, y, z in CUBE_VERTS[:4]]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    engine = GridEngine()
    seams, uvs = preseed_uvs(engine, verts, faces, marked="ONLY", only={0})
    assert uvs[0] is not None
    assert uvs[1] is None


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled engine missing")
def test_preseed_cube_with_real_engine(tmp_path):
    engine = FlattenEngine(BUNDLED, tmp_path)
    seams, uvs = preseed_uvs(engine, CUBE_VERTS, CUBE_FACES)
    assert all(uv is not None and len(uv) == 4 for uv in uvs)
    flat = [p for face in uvs for p in face]
    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in flat)
    # a seamed cube flattens without repair only if the layout held up
    assert seams


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled engine missing")
def test_flatten_start_poll_matches_flatten(tmp_path):
    engine = FlattenEngine(BUNDLED, tmp_path)
    seams = {(0, 1), (1, 2), (2, 3), (0, 3)}
    run = engine.start(CUBE_VERTS, CUBE_FACES, seams)
    while run.poll() is None:
        time.sleep(0.05)
    uvs = run.result()
    assert not run.workdir.exists()
    assert uvs == engine.flatten(CUBE_VERTS, CUBE_FACES, seams)


def test_preseed_mirrors_close_the_seam_set():
    engine = GridEngine()
    # the unit cube is symmetric across x = 0.5
    mirror = dict(enumerate([1, 0, 3, 2, 5, 4, 7, 6]))

    seams, _ = preseed_uvs(engine, CUBE_VERTS, CUBE_FACES, mirrors=[mirror])

    assert seams
    mirrored = {
        (min(mirror[a], mirror[b]), max(mirror[a], mirror[b])) for a, b in seams
    }
    assert mirrored == seams
