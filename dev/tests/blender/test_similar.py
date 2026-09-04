import bpy
from blender_fixtures import addon


def find_twins(objects):
    # a fresh object's matrix_world is stale until an update
    bpy.context.view_layer.update()
    return addon.src.similar.find_twins(objects)


def add_cube(location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    return bpy.context.active_object


def mirrored_cube(location):
    obj = add_cube(location)
    obj.scale.x = -1
    bpy.ops.object.transform_apply(scale=True)
    return obj


def test_exact_twin_keeps_its_own_materials():
    first = add_cube()
    red = bpy.data.materials.new("red")
    first.data.materials.append(red)
    bpy.ops.object.duplicate()
    second = bpy.context.active_object
    second.location.x = 6
    bpy.ops.object.transform_apply(location=True)
    second.data.materials.clear()
    second.data.materials.append(bpy.data.materials.new("blue"))

    twins = find_twins([first, second])
    assert second in twins
    _, _, exact = twins[second]
    assert exact


def test_reordered_twin_with_shared_material_matches():
    red = bpy.data.materials.new("red")
    first = add_cube()
    first.data.materials.append(red)
    second = mirrored_cube((6, 0, 0))
    second.data.materials.append(red)

    twins = find_twins([first, second])
    assert second in twins
    _, _, exact = twins[second]
    assert not exact


def test_reordered_twin_with_other_material_is_refused():
    first = add_cube()
    first.data.materials.append(bpy.data.materials.new("red"))
    second = mirrored_cube((6, 0, 0))
    second.data.materials.append(bpy.data.materials.new("blue"))

    assert second not in find_twins([first, second])
