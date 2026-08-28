import importlib.util
import sys
from pathlib import Path

import numpy
import pytest

# loaded from file, the addon package imports bpy
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import Cancelled, face_edges  # noqa: E402
from seams.proxy_transfer import (  # noqa: E402
    AffineMaps,
    dense_subset,
    snap_cuts,
    transfer_projected,
    uv_tears,
)

IDENTITY = numpy.eye(4)
GRID = 6
SHIFT = 10.0
# the proxy cut sits between dense columns 2 and 3
CUT_X = 2.5


def quad_grid(size, spacing=1.0):
    """size by size quads on the z=0 plane, verts row major."""
    side = size + 1
    verts = [(x * spacing, y * spacing, 0.0) for y in range(side) for x in range(side)]
    faces = [
        [y * side + x, y * side + x + 1, (y + 1) * side + x + 1, (y + 1) * side + x]
        for y in range(size)
        for x in range(size)
    ]
    return verts, faces


def triangulated(faces):
    return [
        f
        for quad in faces
        for f in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]])
    ]


def grid_uvs(verts, faces):
    return [[(verts[v][0], verts[v][1]) for v in face] for face in faces]


def grid_edges(faces):
    return numpy.array(sorted(face_edges(faces)), dtype=numpy.int64)


def dense_arrays(verts, faces, matrix=IDENTITY):
    return {
        "positions": verts,
        "normals": [(0.0, 0.0, 1.0)] * len(verts),
        "matrix": matrix,
        "corners": numpy.array([v for face in faces for v in face], dtype=numpy.int64),
        "face_sizes": numpy.array([len(face) for face in faces], dtype=numpy.int64),
    }


def proxy_arrays(verts, faces, uvs, matrix=IDENTITY):
    return {"positions": verts, "faces": faces, "corner_uvs": uvs, "matrix": matrix}


def split_proxy(right_corner_uv=None):
    """Two proxy panels over the dense grid, torn at CUT_X: the left panel's
    uvs shifted by SHIFT, split into two triangles along its diagonal.
    right_corner_uv replaces the uv of the left panel's top left corner, to
    bend one triangle's map away from the other's."""
    verts = [
        (0.0, 0.0, 0.0),
        (CUT_X, 0.0, 0.0),
        (CUT_X, GRID, 0.0),
        (0.0, GRID, 0.0),
        (GRID, 0.0, 0.0),
        (GRID, GRID, 0.0),
    ]
    faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]
    uvs = grid_uvs(verts, faces)
    for f in (0, 1):
        uvs[f] = [(u + SHIFT, v) for u, v in uvs[f]]
    if right_corner_uv is not None:
        uvs[1][2] = right_corner_uv
    return proxy_arrays(verts, faces, uvs)


def plane_locator(proxy):
    """nearest_faces for flat proxies: the triangle a point falls in, or the
    one it is least outside of."""
    positions = numpy.asarray(proxy["positions"], dtype=numpy.float64)[:, :2]
    triangles = positions[numpy.array([face[:3] for face in proxy["faces"]])]

    def barycentric(point, tri):
        a, b, c = tri
        matrix = numpy.array([b - a, c - a]).T
        s, t = numpy.linalg.solve(matrix, point - a)
        return numpy.array([1 - s - t, s, t])

    def nearest_faces(points, normals):
        found = []
        for point in numpy.asarray(points)[:, :2]:
            lowest = [barycentric(point, tri).min() for tri in triangles]
            found.append(int(numpy.argmax(lowest)))
        return numpy.array(found), numpy.asarray(points, dtype=numpy.float64)

    return nearest_faces


def test_uv_tears_finds_only_the_torn_interior_edge():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + SHIFT, v) for u, v in uvs[0]]
    assert uv_tears(faces, uvs) == {(1, 4), (3, 4)}


def test_uv_tears_skips_boundary_edges():
    verts, faces = quad_grid(1)
    uvs = grid_uvs(verts, faces)
    assert uv_tears(faces, uvs) == set()


def test_uv_tears_ignores_a_gap_under_the_tolerance():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + 1e-7, v) for u, v in uvs[0]]
    assert uv_tears(faces, uvs) == {(1, 4), (3, 4)}
    assert uv_tears(faces, uvs, tolerance=1e-6) == set()


def test_dense_subset_renumbers_the_kept_faces():
    verts, faces = quad_grid(2)
    subset, used = dense_subset(dense_arrays(verts, faces), [3, 0])
    assert used.tolist() == [0, 1, 3, 4, 5, 7, 8]
    assert subset["face_sizes"].tolist() == [4, 4]
    assert subset["corners"].tolist() == [3, 4, 6, 5, 0, 1, 3, 2]
    numpy.testing.assert_array_equal(subset["positions"], numpy.asarray(verts)[used])


def test_snap_cuts_follows_real_edges():
    verts, faces = quad_grid(3)
    edges = grid_edges(faces)
    mapped = [0, 3]
    assert snap_cuts(verts, edges, mapped, {(0, 1)}) == {(0, 1), (1, 2), (2, 3)}


def test_affine_map_continues_past_the_face():
    maps = AffineMaps(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [[0, 1, 2]],
        [[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]],
    )
    uv = maps.uv(numpy.array([0, 0]), [(0.25, 0.25, 0.0), (3.0, -1.0, 5.0)])
    assert numpy.allclose(uv, [(0.5, 0.5), (6.0, -2.0)])


@pytest.mark.parametrize("faces_of", [lambda faces: faces, triangulated])
def test_transfer_projected_reads_a_continuous_map_exactly(faces_of):
    verts, faces = quad_grid(GRID)
    faces = faces_of(faces)
    proxy_verts, proxy_faces = quad_grid(1, GRID)
    proxy_faces = triangulated(proxy_faces)
    proxy = proxy_arrays(proxy_verts, proxy_faces, grid_uvs(proxy_verts, proxy_faces))
    dense = dense_arrays(verts, faces)
    reported = []

    seams, uvs = transfer_projected(dense, proxy, plane_locator(proxy), reported.append)

    expected = numpy.array(verts)[dense["corners"]][:, :2]
    assert numpy.allclose(uvs, expected)
    assert seams == set()
    assert reported == sorted(reported)
    assert reported[-1] == 1.0


def test_transfer_projected_matches_the_meshes_in_the_proxy_space():
    verts, faces = quad_grid(GRID)
    offset = numpy.eye(4)
    offset[0, 3] = 5.0
    proxy_verts, proxy_faces = quad_grid(1, GRID)
    proxy_faces = triangulated(proxy_faces)
    proxy_verts = [(x + 5.0, y, z) for x, y, z in proxy_verts]
    proxy = proxy_arrays(proxy_verts, proxy_faces, grid_uvs(proxy_verts, proxy_faces))
    dense = dense_arrays(verts, faces, offset)

    _, uvs = transfer_projected(dense, proxy, plane_locator(proxy))

    expected = numpy.array(verts)[dense["corners"]][:, :2] + (5.0, 0.0)
    assert numpy.allclose(uvs, expected)


def torn_columns(faces, verts):
    """Face indices by the dense column the proxy cut runs through."""
    xs = numpy.array([[verts[v][0] for v in face] for face in faces])
    left = numpy.flatnonzero(xs.max(axis=1) <= 2)
    straddling = numpy.flatnonzero((xs.min(axis=1) == 2) & (xs.max(axis=1) == 3))
    right = numpy.flatnonzero(xs.min(axis=1) >= 3)
    return left, straddling, right


def test_transfer_projected_tears_the_dense_mesh_along_the_proxy_cut():
    verts, faces = quad_grid(GRID)
    proxy = split_proxy()
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer_projected(dense, proxy, plane_locator(proxy))

    xy = numpy.array(verts)[dense["corners"]][:, :2]
    corner_face = numpy.repeat(numpy.arange(len(faces)), 4)
    left, straddling, right = torn_columns(faces, verts)
    # a straddling face is drawn on its first corner's side, the shifted one
    shifted = numpy.isin(corner_face, numpy.concatenate([left, straddling]))
    assert numpy.allclose(uvs[shifted], xy[shifted] + (SHIFT, 0.0))
    kept = numpy.isin(corner_face, right)
    assert numpy.allclose(uvs[kept], xy[kept])
    side = GRID + 1
    assert seams == {(y * side + 3, (y + 1) * side + 3) for y in range(GRID)}


def test_transfer_projected_never_tears_between_linked_proxy_faces():
    """The left panel's two triangles share vertices at the same uvs, so
    however far their maps disagree the dense edges between them are not
    tears, and the straddling faces above and below the diagonal, drawn
    through one map each, weld at their shared corners."""
    verts, faces = quad_grid(GRID)
    proxy = split_proxy(right_corner_uv=(SHIFT + 2.0, GRID + 2.0))
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer_projected(dense, proxy, plane_locator(proxy))

    side = GRID + 1
    assert seams == {(y * side + 3, (y + 1) * side + 3) for y in range(GRID)}
    _, straddling, _ = torn_columns(faces, verts)
    corners = dense["corners"]
    for v in range(side + 3, GRID * side, side):
        on_straddlers = [
            f * 4 + i for f in straddling for i in range(4) if corners[f * 4 + i] == v
        ]
        assert len(on_straddlers) == 2
        assert numpy.array_equal(uvs[on_straddlers[0]], uvs[on_straddlers[1]])


def test_transfer_projected_pulls_a_zigzag_seam_onto_one_edge_row():
    """A cut through the middle of a triangle strip gives the up and down
    triangles opposite sides, so the seam would zigzag along every
    diagonal. It is pulled onto one of the strip's two edge rows."""
    verts, faces = quad_grid(GRID)
    faces = triangulated(faces)
    proxy = split_proxy()
    # the cut runs along y instead of x
    proxy["positions"] = [(y, x, z) for x, y, z in proxy["positions"]]
    dense = dense_arrays(verts, faces)

    seams, _ = transfer_projected(dense, proxy, plane_locator(proxy))

    side = GRID + 1
    rows = [{(y * side + x, y * side + x + 1) for x in range(GRID)} for y in (2, 3)]
    assert seams in rows


def test_transfer_projected_straightens_a_staircase_onto_the_diagonals():
    """A cut at 45 degrees between two rows of a triangulated grid: the side
    labelling gives a staircase of horizontal and vertical edges, but the
    grid's diagonals run parallel to the cut and are shorter, so the seam
    is redrawn along them."""
    verts, faces = quad_grid(GRID)
    faces = triangulated(faces)
    # two proxy triangles either side of the line y = x + 0.5
    proxy_verts = [
        (-1.0, -0.5, 0.0),
        (7.0, 7.5, 0.0),
        (-1.0, 8.0, 0.0),
        (8.0, -1.0, 0.0),
    ]
    proxy_faces = [[0, 1, 2], [0, 3, 1]]
    uvs = grid_uvs(proxy_verts, proxy_faces)
    uvs[0] = [(u + SHIFT, v) for u, v in uvs[0]]
    proxy = proxy_arrays(proxy_verts, proxy_faces, uvs)
    dense = dense_arrays(verts, faces)

    seams, _ = transfer_projected(dense, proxy, plane_locator(proxy))

    side = GRID + 1

    def diagonal(a, b):
        return b - a == side + 1

    off_diagonal = [edge for edge in seams if not diagonal(*edge)]
    # the run's two ends sit on the grid boundary and may keep a step each
    assert len(off_diagonal) <= 2, off_diagonal
    assert len(seams) - len(off_diagonal) >= GRID - 1


def test_transfer_projected_stops_on_cancel():
    verts, faces = quad_grid(GRID)
    proxy = split_proxy()
    calls = []

    def counting(points, normals):
        calls.append(len(points))
        return plane_locator(proxy)(points, normals)

    with pytest.raises(Cancelled):
        transfer_projected(
            dense_arrays(verts, faces), proxy, counting, cancelled=lambda: True
        )
    assert calls == []
