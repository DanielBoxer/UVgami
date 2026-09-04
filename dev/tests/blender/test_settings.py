import bpy
import pytest
from blender_fixtures import (
    island_count,
    logger,
    manager,
    needs_engine,
    seam_count,
)

pytestmark = [needs_engine, pytest.mark.smoke]

GRID_MATERIAL = "UVgami_grid"
# (islands, seams) on the balanced unwrap, recorded 2026-08-22 at optcuts 1.20
EXPECTED = {
    ("cylinder", False): (1, 30),
    ("cylinder", True): (3, 97),
    ("cube-bevel2", False): (1, 12),
    ("cube-bevel2", True): (2, 34),
}
QUALITY_LEVELS = ["LESS_STRETCH", "BALANCED", "FEWER_SEAMS"]


def add_cube(name, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def select(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def two_loose_cubes():
    a, b = add_cube("a"), add_cube("b", location=(5, 0, 0))
    select([a, b])
    bpy.ops.object.join()
    return bpy.context.active_object


def uv_area(obj):
    layer = obj.data.uv_layers.active.data
    total = 0
    for face in obj.data.polygons:
        corners = [layer[loop].uv for loop in face.loop_indices]
        total += (
            abs(
                sum(
                    a.x * b.y - b.x * a.y
                    for a, b in zip(corners, corners[1:] + corners[:1])
                )
            )
            / 2
        )
    return total


def uv_inside_unit_square(obj):
    return all(
        0 <= c <= 1 for datum in obj.data.uv_layers.active.data for c in datum.uv
    )


@pytest.mark.parametrize("model, hard_surface", EXPECTED)
def test_chart_and_seam_counts_are_pinned(
    load_obj, unwrap, outputs, model, hard_surface
):
    load_obj(model)
    bpy.context.scene.uvgami.optcuts.use_hard_surface = hard_surface

    unwrap()

    assert manager.summary == ["UV unwrap complete!"]
    (output,) = outputs().values()
    assert (island_count(output), seam_count(output)) == EXPECTED[model, hard_surface]


def test_quality_levels_trade_seams_for_stretch(load_obj, unwrap, outputs):
    seams = []
    for quality in QUALITY_LEVELS:
        bpy.ops.wm.read_homefile(use_empty=True)
        load_obj("cylinder")
        bpy.context.scene.uvgami.priority = quality
        unwrap()
        (output,) = outputs().values()
        seams.append(seam_count(output))
    assert seams == sorted(seams, reverse=True)
    assert len(set(seams)) == len(seams)


def test_one_core_gives_the_same_unwrap_as_many(unwrap, outputs):
    counts = []
    for cores in (1, 2):
        bpy.ops.wm.read_homefile(use_empty=True)
        joined = two_loose_cubes()
        bpy.context.scene.uvgami.max_cores = cores
        unwrap()
        output = outputs()[f"{joined.name}_unwrapped"]
        counts.append((island_count(output), seam_count(output)))
    assert counts[0] == counts[1]


def test_stack_similar_copies_the_twin_instead_of_unwrapping_it(unwrap, outputs):
    joined = two_loose_cubes()
    bpy.context.scene.uvgami.stack_similar = True

    unwrap()

    assert manager.summary == ["UV unwrap complete!"]
    engine_runs = [u for u, _ in manager.results if u.copy_of is None]
    copies = [u for u, _ in manager.results if u.copy_of is not None]
    assert len(engine_runs) == 1 and len(copies) == 1
    output = outputs()[f"{joined.name}_unwrapped"]
    layer = output.data.uv_layers.active.data
    # the two cubes sit at x<2 and x>2
    by_side = {False: set(), True: set()}
    for face in output.data.polygons:
        side = output.data.vertices[face.vertices[0]].co.x > 2
        for loop in face.loop_indices:
            by_side[side].add(tuple(round(c, 5) for c in layer[loop].uv))
    assert by_side[False] == by_side[True]


def test_combine_uvs_packs_both_outputs_into_one_square(unwrap, outputs):
    areas = {}
    for combine in (False, True):
        bpy.ops.wm.read_homefile(use_empty=True)
        a, b = add_cube("a"), add_cube("b", location=(5, 0, 0))
        select([a, b])
        props = bpy.context.scene.uvgami
        props.pack_after_unwrap = True
        props.combine_uvs = combine
        unwrap()
        for name, output in outputs().items():
            assert uv_inside_unit_square(output)
            areas[name, combine] = uv_area(output)
    for name in ("a_unwrapped", "b_unwrapped"):
        assert areas[name, True] < areas[name, False]


def test_grid_is_added_and_removed_without_losing_the_slots(unwrap, outputs):
    cube = add_cube("cube")
    cube.data.materials.append(bpy.data.materials.new("A"))
    cube.data.materials.append(bpy.data.materials.new("B"))
    for face in cube.data.polygons:
        face.material_index = face.index % 2
    unwrap()
    output = outputs()["cube_unwrapped"]
    select([output])
    indices_before = [face.material_index for face in output.data.polygons]

    bpy.ops.uvgami.add_grid()
    slots = [slot.material.name for slot in output.material_slots]
    assert slots == ["A", "B", GRID_MATERIAL]
    assert all(face.material_index == 2 for face in output.data.polygons)

    bpy.ops.uvgami.remove_grid()
    assert [slot.material.name for slot in output.material_slots] == ["A", "B"]
    assert [face.material_index for face in output.data.polygons] == indices_before


def test_auto_grid_lands_on_the_input_when_uvs_transfer(unwrap, outputs):
    cube = add_cube("cube")
    props = bpy.context.scene.uvgami
    props.auto_grid = True
    props.transfer_uvs = True

    unwrap()

    assert outputs() == {}
    assert any(
        slot.material and slot.material.name == GRID_MATERIAL
        for slot in cube.material_slots
    )


def test_reset_settings_restores_the_defaults():
    props = bpy.context.scene.uvgami
    props.margin = 0.5
    props.use_proxy = True
    props.max_cores = 1
    props.priority = "FEWER_SEAMS"

    bpy.ops.uvgami.reset_settings()

    assert props.margin == pytest.approx(0.001)
    assert props.use_proxy is False
    assert props.max_cores > 1
    assert props.priority == "BALANCED"


def test_summary_can_be_cleared(unwrap):
    add_cube("cube")
    unwrap()
    assert manager.summary
    assert manager.summary_failed is False

    bpy.ops.uvgami.clear_summary()
    assert manager.summary == []
    assert logger.get_all()
