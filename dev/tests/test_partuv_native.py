"""Integration tests running the native partuv core. Skipped without the
compiled core and an NVIDIA GPU, so CI without a GPU passes."""

from pathlib import Path

import numpy as np
import partuv
import pytest
from partuv.cli import DEFAULT_CONFIG, _cuda_available

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    partuv._CORE_ERROR is not None or not _cuda_available(),
    reason="needs the native partuv core and an NVIDIA GPU",
)


def read_obj(path):
    """Vertices and faces as geometry (v) indices, ignoring uv indices."""
    vertices = []
    faces = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
    return np.array(vertices), faces


def count_components(num_vertices, faces):
    parent = list(range(num_vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for face in faces:
        for b in face[1:]:
            parent[find(face[0])] = find(b)
    return len({find(v) for v in range(num_vertices)})


def test_output_keeps_processed_mesh_connectivity(tmp_path):
    from partuv._core import pipeline_numpy
    from partuv.geometric import preprocess_geometric
    from partuv.output import save_results

    mesh, tree = preprocess_geometric(str(FIXTURES / "cylinder.obj"))
    assert count_components(len(mesh.vertices), mesh.faces.tolist()) == 1

    source_V = np.asarray(mesh.vertices, dtype=np.float64)
    source_F = np.asarray(mesh.faces, dtype=np.int32)
    final_part, parts = pipeline_numpy(
        source_V, source_F, tree, str(DEFAULT_CONFIG), 1.25
    )
    save_results(tmp_path, final_part, parts, source_V, source_F)
    vertices, faces = read_obj(tmp_path / "final_components.obj")

    # geometry must be the processed mesh: same vertices, same connectivity.
    # uv seams belong in vt indices, not duplicated v entries.
    assert len(vertices) == len(mesh.vertices)
    assert np.allclose(vertices, mesh.vertices)
    assert len(faces) == len(mesh.faces)
    assert count_components(len(vertices), faces) == 1

    output_triples = sorted(sorted(f) for f in faces)
    source_triples = sorted(sorted(f) for f in mesh.faces.tolist())
    assert output_triples == source_triples
