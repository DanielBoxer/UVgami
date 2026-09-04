import importlib.util
import math
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
from seams.uv_transfer import transfer_exact  # noqa: E402

# unit square as two triangles sharing edge v0-v2
SQUARE_POS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
SQUARE_FACES = [[0, 1, 2], [0, 2, 3]]


def approx_uvs(expected):
    return {k: pytest.approx(v, abs=1e-9) for k, v in expected.items()}


def test_exact_reordered_faces_and_verts():
    # output remaps vertices and lists faces in a different order
    out_pos = [(1, 1, 0), (0, 0, 0), (0, 1, 0), (1, 0, 0)]
    out_faces = [[1, 0, 2], [1, 3, 0]]
    out_uvs = [
        [(0, 0), (1, 1), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
    ]

    plan = transfer_exact(SQUARE_POS, SQUARE_FACES, out_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    # every input loop gets the uv of its own vertex position (planar uv)
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 0.0),
        4: (1.0, 1.0),
        5: (0.0, 1.0),
    }


def test_seam_duplicates_map_many_to_one():
    # output cuts the shared edge: v0 and v2 each become two coincident verts
    out_pos = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ]
    out_faces = [[0, 1, 2], [3, 4, 5]]
    # different uvs on each side prove they land on different input loops
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(2, 0), (2, 1), (3, 1)],
    ]
    plan = transfer_exact(SQUARE_POS, SQUARE_FACES, out_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (2.0, 0.0),
        4: (2.0, 1.0),
        5: (3.0, 1.0),
    }
    assert plan.seam_edges == {(0, 2)}


def test_triangulated_quad_assigns_all_corners():
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 1), (0, 1)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }


def test_seam_through_quad_welds_the_cut_off():
    # two quads side by side, a uv cut runs across the first one's diagonal
    in_pos = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (2, 0, 0),
        (2, 1, 0),
    ]
    in_faces = [[0, 1, 2, 3], [1, 4, 5, 2]]
    out_faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1, 0), (2, 0), (2, 1)],
        [(1, 0), (2, 1), (1, 1)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
            4: (1.0, 0.0),
            5: (2.0, 0.0),
            6: (2.0, 1.0),
            7: (1.0, 1.0),
        }
    )
    # the welded diagonal is no input edge
    assert plan.seam_edges == set()


def test_far_chart_scale_is_ignored():
    # the far piece's own chart sits at half the anchor's scale
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (5.5, 5.5), (5, 5.5)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
        }
    )
    assert plan.seam_edges == set()


def test_mirrored_anchor_reflects_the_flap():
    # the anchor chart is mirrored, its uv winding is clockwise
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (0, 1), (1, 1)],
        [(5, 5), (5.5, 5.5), (5, 5.5)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (0.0, 1.0),
            2: (1.0, 1.0),
            3: (1.0, 0.0),
        }
    )


def test_stretched_anchor_does_not_square_its_stretch():
    # the anchor chart is stretched 2x along u
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (2, 0), (2, 1)],
        [(5, 5), (6, 6), (5, 6)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (2.0, 0.0),
            2: (2.0, 1.0),
            3: (0.0, 1.0),
        }
    )


def test_bent_quad_flap_keeps_its_3d_shape():
    # the quad is folded along the cut diagonal and the flap is equilateral in 3d
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: ((1 - math.sqrt(3)) / 2, (1 + math.sqrt(3)) / 2),
        }
    )


def test_ngon_chain_welds_part_by_part():
    # pentagon fan, the two far pieces sit in a translated frame
    in_pos = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (2, 5, 0), (0, 4, 0)]
    in_faces = [[0, 1, 2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    out_uvs = [
        [(0, 0), (4, 0), (4, 4)],
        [(10, 0), (14, 4), (12, 5)],
        [(10, 0), (12, 5), (10, 4)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (4.0, 0.0),
            2: (4.0, 4.0),
            3: (2.0, 5.0),
            4: (0.0, 4.0),
        }
    )


def test_weld_landing_on_its_island_splits():
    # the far piece would land where a face of the same island already sits
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
    in_faces = [[0, 1, 2, 3], [2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        # same island as the anchor, its long edge crosses the glued piece
        [(1, 1), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {
        4: (1.0, 1.0),
        5: (0.3, 0.7),
        6: (0.9, 1.5),
    }
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([0, 2, 3], [(5.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }
    # edge 2-3 is where the far piece's island meets the neighbouring triangle
    assert plan.seam_edges == {(0, 2), (2, 3)}


def test_weld_moves_the_cut_to_the_faces_outer_edge():
    # the same blocking face, in its own island, which the pack pulls clear
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
    in_faces = [[0, 1, 2, 3], [2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1.05, 1.0), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
            4: (1.05, 1.0),
            5: (0.3, 0.7),
            6: (0.9, 1.5),
        }
    )
    assert plan.seam_edges == {(2, 3)}


def test_without_a_pack_another_islands_overlap_splits():
    # the layout the weld stands on, but with no pack to pull the island clear
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
    in_faces = [[0, 1, 2, 3], [2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1.05, 1.0), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs, repack=False)

    assert plan.ok
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([0, 2, 3], [(5.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }
    assert plan.loop_uvs == {
        4: (1.05, 1.0),
        5: (0.3, 0.7),
        6: (0.9, 1.5),
    }
    assert plan.seam_edges == {(0, 2), (2, 3)}


def test_vertex_only_cut_splits_in_input_winding():
    # the pieces share only vertex 2, no weld edge exists
    in_pos = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 3, 0), (0, 2, 0)]
    in_faces = [[0, 1, 2, 3, 4]]
    # rotated corner orders prove the reorder
    out_faces = [[2, 0, 1], [4, 2, 3]]
    out_uvs = [
        [(1, 1), (0, 0), (1, 0)],
        [(5, 6), (6, 5), (6, 6)],
    ]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {}
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([2, 3, 4], [(6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }


def test_conflicting_triangle_cannot_be_split():
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    in_faces = [[0, 1, 2]]
    # the output holds the same triangle twice with different uvs
    out_faces = [[0, 1, 2], [0, 1, 2]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 5), (6, 6)],
    ]

    result = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert not result.ok
    assert result.reason == "ambiguous_geometry"


def test_coincident_input_faces_each_take_their_own_output():
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    # stacked input triangles, one output face per input face
    in_pos = [a, b, c, a, b, c]
    in_faces = [[0, 1, 2], [3, 4, 5]]
    out_faces = [[0, 1, 2], [3, 4, 5]]
    out_uvs = [[(0, 0), (1, 0), (0, 1)], [(2, 0), (3, 0), (2, 1)]]

    plan = transfer_exact(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (2.0, 0.0),
        4: (3.0, 0.0),
        5: (2.0, 1.0),
    }


def test_cut_copy_near_a_second_vertex_is_ambiguous():
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    # vertex 3 is a needle away from vertex 0
    in_pos = [a, b, c, (0, 0, 1e-6)]
    in_faces = [[0, 1, 2], [3, 1, 0]]
    out_pos = [a, b, c, (0, 0, 1e-6), a]
    out_faces = [[0, 1, 2], [3, 1, 4]]
    out_uvs = [[(0, 0), (1, 0), (0, 1)], [(2, 0), (3, 0), (2, 1)]]

    result = transfer_exact(in_pos, in_faces, out_pos, out_faces, out_uvs)

    assert not result.ok
    assert result.reason == "ambiguous_geometry"


def test_merged_output_still_matches_by_position():
    a, b, c, d = (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)
    # two input triangles with their shared edge doubled, merged by the engine
    in_pos = [a, b, c, b, c, d]
    in_faces = [[0, 1, 2], [3, 4, 5]]
    out_pos = [a, b, c, d]
    out_faces = [[0, 1, 2], [1, 2, 3]]
    out_uvs = [[(0, 0), (1, 0), (0, 1)], [(1, 0), (0, 1), (1, 1)]]

    plan = transfer_exact(in_pos, in_faces, out_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 0.0),
        4: (0.0, 1.0),
        5: (1.0, 1.0),
    }


def test_small_piece_far_from_the_origin_matches():
    # a 0.1 wide triangle 138 units out, at the engine's 7 digits
    in_pos = [(138.0, 0, 0), (138.1, 0, 0), (138.0, 0.1, 0)]
    out_pos = [tuple(float("%.6e" % (x + 3e-5)) for x in p) for p in in_pos]
    out_faces = [[0, 1, 2]]
    out_uvs = [[(0, 0), (1, 0), (0, 1)]]

    plan = transfer_exact(in_pos, [[0, 1, 2]], out_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}


def test_unmatched_output_face_fails():
    # verts 1, 3, 0 never share an input face
    out_faces = [[1, 3, 0]]
    out_uvs = [[(0, 0), (1, 0), (1, 1)]]

    result = transfer_exact(SQUARE_POS, SQUARE_FACES, SQUARE_POS, out_faces, out_uvs)

    assert not result.ok
    assert result.reason == "face_match"


# two loose triangles, the output only unwrapped the first
LOOSE_POS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 0, 0), (6, 0, 0), (5, 1, 0)]
LOOSE_FACES = [[0, 1, 2], [3, 4, 5]]
FIRST_TRIANGLE_UVS = [[(0, 0), (1, 0), (0, 1)]]


def test_missing_piece_fails_unless_partial():
    result = transfer_exact(
        LOOSE_POS, LOOSE_FACES, LOOSE_POS[:3], [[0, 1, 2]], FIRST_TRIANGLE_UVS
    )

    assert not result.ok
    assert result.reason == "incomplete_coverage"


def test_partial_leaves_the_missing_piece_alone():
    plan = transfer_exact(
        LOOSE_POS,
        LOOSE_FACES,
        LOOSE_POS[:3],
        [[0, 1, 2]],
        FIRST_TRIANGLE_UVS,
        partial=True,
    )

    assert plan.ok
    assert plan.untouched_faces == {1}
    assert plan.loop_uvs == approx_uvs({0: (0, 0), 1: (1, 0), 2: (0, 1)})
    assert not plan.split_faces
    assert not plan.seam_edges


def test_partial_with_nothing_covered_fails():
    result = transfer_exact(LOOSE_POS, LOOSE_FACES, [], [], [], partial=True)

    assert not result.ok
    assert result.reason == "incomplete_coverage"
