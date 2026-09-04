import bpy
import pytest
from bl_ext.user_default.UVgami.src import manager as manager_module
from bl_ext.user_default.UVgami.src import unwrap as unwrap_module
from bl_ext.user_default.UVgami.src.engines import optcuts
from bl_ext.user_default.UVgami.src.job import Result
from blender_fixtures import UNWRAP_SECONDS, manager, needs_engine

pytestmark = [needs_engine, pytest.mark.smoke]

# enough ticks of the viewer to tell a live map from a stale one
SNAPSHOTS_WATCHED = 5


def add_cube(name, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_sphere(name, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def select(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def solving():
    return next((u for u in manager.active if u.is_solving and u.has_reported), None)


def running_and_queued():
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
    # 1e30 is the one refusal the engine makes on a mesh blender accepts
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


def test_stop_a_solving_piece_keeps_its_partial_map(unwrap, outputs, invalid_objects):
    sphere = add_sphere("sphere")
    select([sphere])

    pump = unwrap(until=solving)
    piece = solving()
    bpy.ops.uvgami.stop(stem=piece.path.stem)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    # the engine answers a stop with its current map and exits 0
    assert piece.result is Result.FINISHED
    assert manager.summary == ["UV unwrap complete!"]
    assert invalid_objects() == {}
    output = outputs()["sphere_unwrapped"]
    assert output.data.uv_layers.active is not None
    assert sphere.hide_get()


def test_cancel_a_solving_piece_lets_the_queued_one_finish(unwrap, outputs):
    sphere, cube = add_sphere("sphere"), add_cube("cube", location=(5, 0, 0))
    select([sphere, cube])
    bpy.context.scene.uvgami.max_cores = 1

    pump = unwrap(until=solving)
    piece = solving()
    bpy.ops.uvgami.cancel(stem=piece.path.stem)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert manager.summary == ["UV unwrap complete!", "1 object cancelled"]
    assert len(outputs()) == 1
    assert f"{piece.input_name}_unwrapped" not in outputs()
    assert not bpy.data.objects[piece.input_name].hide_get()


def test_group_stop_merges_the_queued_pieces_into_one_object(unwrap, invalid_objects):
    spheres = [
        add_sphere("a"),
        add_sphere("b", location=(5, 0, 0)),
        add_sphere("c", location=(10, 0, 0)),
    ]
    select(spheres)
    bpy.ops.object.join()
    bpy.context.scene.uvgami.max_cores = 1

    pump = unwrap(until=solving)
    piece = solving()
    queued = [u for u in manager.active if u is not piece]
    queued_vertices = sum(u.vertex_count for u in queued)
    bpy.ops.uvgami.stop(job_id=piece.join_job.job_id)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert manager.summary[0] == "2 of 3 parts stopped"
    (stopped,) = invalid_objects().values()
    assert stopped.name.startswith("Stopped: ")
    assert len(stopped.data.vertices) == queued_vertices


def test_group_cancel_discards_the_piece_that_already_finished(unwrap, outputs):
    cubes = [
        add_cube("a"),
        add_cube("b", location=(5, 0, 0)),
        add_cube("c", location=(10, 0, 0)),
    ]
    select(cubes)
    bpy.ops.object.join()
    joined = bpy.context.active_object
    bpy.context.scene.uvgami.max_cores = 1

    pump = unwrap(until=lambda: manager.results)
    job = manager.results[0][0].join_job
    assert job.finished
    bpy.ops.uvgami.cancel(job_id=job.job_id)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert job.discard
    assert outputs() == {}
    assert not joined.hide_get()
    assert manager.summary == []


def test_a_stop_the_engine_ignores_is_force_killed(
    unwrap, outputs, invalid_objects, monkeypatch
):
    sphere = add_sphere("sphere")
    select([sphere])
    # the stop is reported as delivered and then nothing comes back
    monkeypatch.setattr(optcuts.ENGINE, "request_early_stop", lambda process: True)
    monkeypatch.setattr(manager_module, "STOP_SECONDS", 0)

    pump = unwrap(until=solving)
    piece = solving()
    bpy.ops.uvgami.stop(stem=piece.path.stem)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert outputs() == {}
    (killed,) = invalid_objects().values()
    assert killed.name.startswith("Stop timed out (force killed): ")


def test_a_timeout_keeps_the_partial_map(unwrap, outputs, invalid_objects):
    sphere = add_sphere("sphere")
    select([sphere])
    bpy.context.scene.uvgami.unwrap_timeout = 1

    pump = unwrap(until=solving)
    piece = solving()
    # two minutes into a one minute budget, without the wait
    piece.started_at -= 120
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert manager.summary[0] == "UV unwrap finished with errors"
    assert "sphere: timed out, stopped with a partial result" in manager.summary
    assert invalid_objects() == {}
    assert outputs()["sphere_unwrapped"].data.uv_layers.active is not None


def test_a_timeout_bins_the_mesh_when_the_engine_cannot_stop(
    unwrap, outputs, invalid_objects, monkeypatch
):
    sphere = add_sphere("sphere")
    select([sphere])
    bpy.context.scene.uvgami.unwrap_timeout = 1
    # xatlas and partuv take this branch, optcuts is the one slow enough to catch
    monkeypatch.setattr(optcuts.ENGINE, "supports_early_stop", False)

    pump = unwrap(until=solving)
    piece = solving()
    piece.started_at -= 120
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert manager.summary[0] == "UV unwrap failed"
    assert outputs() == {}
    (timed_out,) = invalid_objects().values()
    assert timed_out.name.startswith("Timed out after 2.0 minutes: ")


def test_a_viewed_piece_streams_a_map_that_changes(unwrap, monkeypatch):
    sphere = add_sphere("sphere")
    select([sphere])
    snapshots = []
    # the real one builds gpu batches, which a background blender has no context for
    monkeypatch.setattr(
        unwrap_module,
        "set_snapshot",
        lambda uvs, indices: snapshots.append(tuple(uvs)),
    )

    pump = unwrap(until=solving)
    piece = solving()
    piece.viewing = True
    pump.run_until(lambda: len(snapshots) >= SNAPSHOTS_WATCHED, UNWRAP_SECONDS)
    bpy.ops.uvgami.cancel(stem=piece.path.stem)
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert all(snapshots)
    # the same map every tick would be a stale read, not a live solve
    assert len(set(snapshots)) > 1
