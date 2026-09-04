import bpy
from blender_fixtures import addon

make_proxy = addon.src.proxy.make_proxy
needs_proxy = addon.src.proxy.needs_proxy
part_triangles = addon.src.proxy.part_triangles
triangle_count = addon.src.proxy.triangle_count

PROXY_FACES = 500


def add_dense_sphere(location=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=location)
    return bpy.context.active_object


def join(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def test_only_the_part_over_the_budget_is_decimated():
    sphere = add_dense_sphere()
    bpy.ops.mesh.primitive_cube_add(location=(5, 0, 0))
    obj = join([sphere, bpy.context.active_object])

    assert needs_proxy(obj, PROXY_FACES)
    assert make_proxy(obj, PROXY_FACES)

    _, _, counts = part_triangles(obj.data)
    assert sorted(counts.tolist()) == [12, PROXY_FACES]


def test_parts_all_under_the_budget_are_left_alone():
    cubes = []
    for i in range(3):
        bpy.ops.mesh.primitive_cube_add(location=(3 * i, 0, 0))
        cubes.append(bpy.context.active_object)
    obj = join(cubes)

    assert triangle_count(obj) > 20
    assert not needs_proxy(obj, 20)
    assert not make_proxy(obj, 20)
    assert triangle_count(obj) == 36


def test_a_single_part_lands_on_the_budget():
    obj = add_dense_sphere()

    assert needs_proxy(obj, PROXY_FACES)
    assert make_proxy(obj, PROXY_FACES)
    assert triangle_count(obj) == PROXY_FACES
