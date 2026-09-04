import importlib.util
import sys
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
    Cancelled,
    seam_edges,
    seam_edges_parallel,
    vertex_components,
)
from seams.parallel import balanced_chunks  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
# the venv interpreter stands in for blender's bundled one
PYTHON = (sys.executable, ())


def read_obj(path):
    verts, faces = [], []
    for line in open(path):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return verts, faces


def parts_mesh():
    verts, faces = [], []
    for name, offset in (("cube-bevel2", 0.0), ("cylinder", 5.0), ("cube", 10.0)):
        part_verts, part_faces = read_obj(FIXTURES / f"{name}.obj")
        base = len(verts)
        verts += [[x + offset, y, z] for x, y, z in part_verts]
        faces += [[v + base for v in face] for face in part_faces]
    return verts, faces


def test_parts_match_the_whole_run():
    verts, faces = parts_mesh()
    assert len(vertex_components(faces)) == 3
    whole = seam_edges(verts, faces)
    assert whole
    assert seam_edges_parallel(PYTHON, verts, faces) == whole


def test_forced_and_weights_reach_the_workers():
    verts, faces = parts_mesh()
    forced = {tuple(sorted(faces[0][:2]))}
    weights = {v: 1.0 for v in faces[-1]}
    whole = seam_edges(verts, faces, weights=weights, forced=forced)
    parallel = seam_edges_parallel(PYTHON, verts, faces, weights=weights, forced=forced)
    assert parallel == whole
    assert forced <= parallel


def test_cancel_before_the_workers_start():
    verts, faces = parts_mesh()
    with pytest.raises(Cancelled):
        seam_edges_parallel(PYTHON, verts, faces, cancelled=lambda: True)


def test_worker_failure_is_reported():
    verts, faces = parts_mesh()
    with pytest.raises(RuntimeError, match="seam worker failed"):
        seam_edges_parallel(
            (sys.executable, ("-c", "raise SystemExit(3)")), verts, faces
        )


def test_balanced_chunks_level_the_face_counts():
    parts = [[0] * n for n in (5, 4, 3, 2, 1)]
    chunks = balanced_chunks(parts, 2)
    assert sorted(sum(len(p) for p in chunk) for chunk in chunks) == [7, 8]
    assert balanced_chunks(parts, 8) == [
        [p] for p in sorted(parts, key=len, reverse=True)
    ]
