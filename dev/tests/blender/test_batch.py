"""Several objects or pieces in one session: the per-part queue, a refusal
next to a good mesh, and stopping or cancelling pieces while it runs."""

import bpy
import pytest
from blender_fixtures import UNWRAP_SECONDS, manager, needs_engine

pytestmark = [needs_engine, pytest.mark.smoke]


def add_cube(name, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_sphere(name):
    """Enough faces that a cube queued behind it is still queued when the
    test looks."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def select(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def running_and_queued():
    """The piece the engine is on and the one waiting behind it, once both
    exist, else None."""
    running = [u for u in manager.active if u.process is not None]
    queued = [u for u in manager.active if u.process is None and u.is_exported]
    if running and queued:
        return running[0], queued[0]
    return None


def test_two_objects_each_get_an_output(unwrap, outputs):
    a, b = add_cube("a"), add_cube("b", location=(5, 0, 0))
    select([a, b])

    unwrap()

    assert manager.summary == ["UV unwrap complete!"]
    assert sorted(outputs()) == ["a_unwrapped", "b_unwrapped"]
    assert a.hide_get() and b.hide_get()
    assert len(manager.results) == 2


def test_loose_parts_unwrap_as_pieces_and_join_back(unwrap, outputs):
    a, b = add_cube("a"), add_cube("b", location=(5, 0, 0))
    select([a, b])
    bpy.ops.object.join()
    joined = bpy.context.active_object

    unwrap()

    assert manager.summary == ["UV unwrap complete!"]
    assert list(outputs()) == [f"{joined.name}_unwrapped"]
    output = outputs()[f"{joined.name}_unwrapped"]
    positions = {tuple(round(c, 3) for c in v.co) for v in output.data.vertices}
    assert len(positions) == 16
    # one engine run per loose part
    assert len(manager.results) == 2


def test_refused_mesh_fails_next_to_a_good_cube(
    make_mesh, unwrap, outputs, invalid_objects
):
    # 1e30 is the one refusal the engine still makes on a mesh blender
    # accepts, exit 113
    huge = make_mesh(
        "huge",
        [(1e30, 0, 0), (1, 0, 0), (0, 1, 0)],
        [(0, 1, 2)],
        [[(0, 0), (0, 0), (0, 0)]],
    )
    cube = add_cube("cube", location=(5, 0, 0))
    select([huge, cube])

    unwrap()

    assert manager.summary[0] == "1 of 2 parts failed"
    assert manager.summary_failed
    assert list(outputs()) == ["cube_unwrapped"]
    (invalid,) = invalid_objects().values()
    assert invalid.hide_get()
    assert invalid.name.startswith("Invalid Coordinates: ")


def test_cancel_all_kills_the_engine_and_keeps_the_inputs(unwrap, outputs):
    sphere, cube = add_sphere("sphere"), add_cube("cube", location=(5, 0, 0))
    select([sphere, cube])
    bpy.context.scene.uvgami.max_cores = 1

    unwrap(until=running_and_queued)
    running, _ = running_and_queued()
    bpy.ops.uvgami.cancel_all()

    assert not manager.is_active
    assert running.process.poll() is not None
    assert outputs() == {}
    assert not sphere.hide_get() and not cube.hide_get()
    assert manager.summary == []


def test_cancel_one_queued_piece_lets_the_other_finish(unwrap, outputs):
    sphere, cube = add_sphere("sphere"), add_cube("cube", location=(5, 0, 0))
    select([sphere, cube])
    bpy.context.scene.uvgami.max_cores = 1

    pump = unwrap(until=running_and_queued)
    _, queued = running_and_queued()
    bpy.ops.uvgami.cancel(stem=queued.path.stem)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert manager.summary == ["UV unwrap complete!", "1 object cancelled"]
    assert len(outputs()) == 1
    assert f"{queued.input_name}_unwrapped" not in outputs()
    assert not bpy.data.objects[queued.input_name].hide_get()


def test_stop_one_queued_piece_moves_it_to_not_unwrapped(
    unwrap, outputs, invalid_objects
):
    sphere, cube = add_sphere("sphere"), add_cube("cube", location=(5, 0, 0))
    select([sphere, cube])
    bpy.context.scene.uvgami.max_cores = 1

    pump = unwrap(until=running_and_queued)
    _, queued = running_and_queued()
    bpy.ops.uvgami.stop(stem=queued.path.stem)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert manager.summary[0] == "1 of 2 parts stopped"
    assert len(outputs()) == 1
    (stopped,) = invalid_objects().values()
    assert stopped.name.startswith("Stopped: ")
    assert stopped.hide_get()
