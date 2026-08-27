import bpy
import pytest
from bl_ext.user_default.UVgami.src.ops.uv import pack_objects, show_seams
from bl_ext.user_default.UVgami.src.utils.mesh import edit_restore

ADDON_MODULE = "bl_ext.user_default.UVgami"
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
MARGIN = 0.02


def two_quads(make_mesh, second_uvs):
    """One object, two faces that share no vertices, so they are two uv islands
    the packer can move independently."""
    return make_mesh(
        "quads",
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        [(0, 1, 2, 3), (4, 5, 6, 7)],
        [SQUARE, second_uvs],
    )


@pytest.fixture(autouse=True)
def pack_settings():
    settings = bpy.context.scene.uvgami
    settings.margin = MARGIN
    settings.fix_scale = False
    settings.combine_uvs = False
    return settings


def test_addon_is_enabled():
    assert ADDON_MODULE in bpy.context.preferences.addons
    assert hasattr(bpy.types.Scene, "uvgami")
    assert hasattr(bpy.ops.uvgami, "pack")


def test_addon_survives_a_disable_enable_round_trip():
    """A property or class the addon fails to unregister only shows up on the
    second enable, which is what a reload during development does."""
    bpy.ops.preferences.addon_disable(module=ADDON_MODULE)
    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    assert hasattr(bpy.types.Scene, "uvgami")
    assert bpy.context.scene.uvgami.margin is not None


def test_pack_moves_islands_into_the_unit_square(make_mesh, face_uvs):
    obj = two_quads(make_mesh, [(2, 2), (3, 2), (3, 3), (2, 3)])
    pack_objects([obj])
    corners = [uv for face in face_uvs(obj) for uv in face]
    assert min(u for u, _ in corners) >= 0
    assert max(u for u, _ in corners) <= 1
    assert max(v for _, v in corners) <= 1


def test_pack_keeps_stacked_duplicates_exactly_together(make_mesh, face_uvs):
    """Two islands with identical uvs are an intentional stack (a symmetry
    twin, an artist stack). Blender's packer would separate them, so the
    duplicate sits out the pack and snaps back onto the island that ran."""
    obj = two_quads(make_mesh, SQUARE)
    pack_objects([obj])
    kept, duplicate = face_uvs(obj)
    assert kept == duplicate
    assert kept != SQUARE


def test_pack_separates_islands_that_are_not_duplicates(make_mesh, face_uvs):
    """The control for the stack case: without identical uvs the two islands
    have to end up somewhere different, or the test above proves nothing."""
    obj = two_quads(make_mesh, [(0, 0), (2, 0), (2, 1), (0, 1)])
    pack_objects([obj])
    first, second = face_uvs(obj)
    assert first != second


def test_stalled_marker_flags_frozen_progress():
    """A running piece whose progress value stops changing reads as stalled
    after the threshold, and a repeated identical line doesn't reset it."""
    import pathlib

    from bl_ext.user_default.UVgami.src.unwrap import PROGRESS_STALL_SECONDS, Unwrap

    unwrap = Unwrap("p", "p", pathlib.Path("p.obj"), (None,) * 5, "OFF")
    unwrap.is_active = True
    assert not unwrap.is_stalled

    unwrap.progress_data.append("0.5 0.3 0.2")
    unwrap.update_progress()
    assert not unwrap.is_stalled

    unwrap.progress_changed_at -= PROGRESS_STALL_SECONDS + 1
    assert unwrap.is_stalled
    unwrap.progress_data.append("0.5 0.3 0.2")
    unwrap.update_progress()
    assert unwrap.is_stalled

    unwrap.progress_data.append("0.6 0.25 0.15")
    unwrap.update_progress()
    assert not unwrap.is_stalled


def test_fix_inconsistent_winding_rewinds_the_flipped_face(make_mesh):
    from bl_ext.user_default.UVgami.src.ops.start import (
        fix_inconsistent_winding,
        has_inconsistent_winding,
    )

    corner_uvs = [(0, 0), (1, 0), (0, 1)]
    # both faces walk the shared edge 1-2 in the same direction
    obj = make_mesh(
        "flipped",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        [(0, 1, 2), (1, 2, 3)],
        [corner_uvs, corner_uvs],
    )
    layer = obj.data.uv_layers.active.data
    uv_by_vertex = [
        {obj.data.loops[i].vertex_index: tuple(layer[i].uv) for i in face.loop_indices}
        for face in obj.data.polygons
    ]
    assert has_inconsistent_winding(obj.data)

    fix_inconsistent_winding(obj)
    assert not has_inconsistent_winding(obj.data)
    # the uvs stay on their corners through the rewind
    layer = obj.data.uv_layers.active.data
    for face, before in zip(obj.data.polygons, uv_by_vertex):
        for i in face.loop_indices:
            assert tuple(layer[i].uv) == before[obj.data.loops[i].vertex_index]


def test_fix_inconsistent_winding_leaves_a_consistent_mesh_alone(make_mesh):
    from bl_ext.user_default.UVgami.src.ops.start import (
        fix_inconsistent_winding,
        has_inconsistent_winding,
    )

    corner_uvs = [(0, 0), (1, 0), (0, 1)]
    obj = make_mesh(
        "consistent",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        [(0, 1, 2), (2, 1, 3)],
        [corner_uvs, corner_uvs],
    )
    assert not has_inconsistent_winding(obj.data)
    order_before = [loop.vertex_index for loop in obj.data.loops]
    fix_inconsistent_winding(obj)
    assert [loop.vertex_index for loop in obj.data.loops] == order_before


def test_show_seams_marks_the_uv_island_border(make_mesh, seam_edges):
    obj = make_mesh(
        "strip",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 1, 2, 3), (1, 4, 5, 2)],
        [SQUARE, [(2, 0), (3, 0), (3, 1), (2, 1)]],
    )
    edit_restore([obj], show_seams)
    # the two faces share the 3D edge 1-2 but no uv there, so it is the cut
    assert seam_edges(obj) == {frozenset((1, 2))}
