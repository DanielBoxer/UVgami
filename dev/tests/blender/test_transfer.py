import bpy
import pytest
from blender_fixtures import manager, needs_engine

pytestmark = [needs_engine, pytest.mark.smoke]

# the prop minimum, well under the sphere's face count
PROXY_FACES = 100


def add_primitive(name, add, **kwargs):
    add(**kwargs)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_cube(name="cube"):
    return add_primitive(name, bpy.ops.mesh.primitive_cube_add)


def add_sphere(name="sphere"):
    return add_primitive(
        name, bpy.ops.mesh.primitive_uv_sphere_add, segments=32, ring_count=16
    )


def has_uvs(obj):
    layer = obj.data.uv_layers.active
    return layer is not None and any(any(datum.uv) for datum in layer.data)


def test_transfer_writes_uvs_onto_the_input_and_keeps_its_quads(unwrap, outputs):
    cube = add_cube()
    cube.data.uv_layers.remove(cube.data.uv_layers[0])
    bpy.context.scene.uvgami.transfer_uvs = True

    unwrap()

    assert manager.summary == ["UV unwrap complete!"]
    assert manager.transfer_uv_failed is False
    assert outputs() == {}
    assert not cube.hide_get()
    assert has_uvs(cube)
    assert [len(face.vertices) for face in cube.data.polygons] == [4] * 6
    assert any(edge.use_seam for edge in cube.data.edges)


def test_transfer_on_an_organic_mesh(unwrap, outputs):
    sphere = add_sphere()
    bpy.context.scene.uvgami.transfer_uvs = True

    unwrap()

    assert manager.summary[0] == "UV unwrap complete!"
    assert manager.transfer_uv_failed is False
    assert outputs() == {}
    assert has_uvs(sphere)
    assert any(edge.use_seam for edge in sphere.data.edges)


def test_transfer_that_cannot_apply_keeps_the_output(unwrap, outputs):
    suzanne = add_primitive("suzanne", bpy.ops.mesh.primitive_monkey_add)
    bpy.context.scene.uvgami.transfer_uvs = True

    unwrap()

    assert manager.transfer_uv_failed is True
    assert "ambiguous_geometry" in manager.transfer_uv_fail_detail
    assert manager.transfer_uv_reason_known is True
    assert list(outputs()) == ["suzanne_unwrapped"]
    assert suzanne.hide_get()


@pytest.mark.parametrize("with_map", [True, False])
def test_import_uvs_runs_with_and_without_a_map(unwrap, outputs, with_map):
    cube = add_cube()
    if not with_map:
        cube.data.uv_layers.remove(cube.data.uv_layers[0])
    bpy.context.scene.uvgami.import_uvs = True

    unwrap()

    assert manager.summary == ["UV unwrap complete!"]
    assert manager.error_messages == []
    assert has_uvs(outputs()["cube_unwrapped"])


def test_proxy_with_transfer_lands_on_the_input(unwrap, outputs):
    sphere = add_sphere()
    props = bpy.context.scene.uvgami
    props.use_proxy = True
    props.proxy_faces = PROXY_FACES
    props.transfer_uvs = True

    unwrap()

    assert manager.summary[0] == "UV unwrap complete!"
    assert manager.transfer_uv_failed is False
    assert outputs() == {}
    assert not sphere.hide_get()
    assert has_uvs(sphere)


def test_proxy_on_a_multi_part_mesh_decimates_only_the_part_over_the_budget(
    unwrap, outputs
):
    sphere = add_sphere()
    bpy.ops.mesh.primitive_cube_add(location=(5, 0, 0))
    sphere.select_set(True)
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.object.join()
    joined = bpy.context.active_object
    props = bpy.context.scene.uvgami
    props.use_proxy = True
    props.proxy_faces = PROXY_FACES
    props.transfer_uvs = True

    unwrap()

    assert manager.summary[0] == "UV unwrap complete!"
    assert manager.transfer_uv_failed is False
    assert outputs() == {}
    assert has_uvs(joined)
    cube_faces = [
        face
        for face in joined.data.polygons
        if all(joined.data.vertices[v].co.x > 2.5 for v in face.vertices)
    ]
    assert len(cube_faces) == 6
    layer = joined.data.uv_layers.active
    for face in cube_faces:
        assert any(any(layer.data[i].uv) for i in face.loop_indices)


def test_proxy_without_transfer_replaces_the_output_with_a_full_copy(unwrap, outputs):
    sphere = add_sphere()
    props = bpy.context.scene.uvgami
    props.use_proxy = True
    props.proxy_faces = PROXY_FACES
    props.transfer_uvs = False

    unwrap()

    assert manager.summary[0] == "UV unwrap complete!"
    assert manager.transfer_uv_failed is False
    assert list(outputs()) == ["sphere_unwrapped"]
    output = outputs()["sphere_unwrapped"]
    assert sphere.hide_get()
    assert has_uvs(output)
    # the copy is the original triangulated, not the decimated proxy
    assert len(output.data.polygons) > PROXY_FACES
