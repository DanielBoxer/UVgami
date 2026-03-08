import bmesh
import bpy


def new_bmesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    return bm


def set_bmesh(bm, obj):
    if obj.mode == "EDIT":
        bmesh.update_edit_mesh(obj.data)
    else:
        bm.to_mesh(obj.data)
    bm.free()


def move_to_collection(obj, target):
    for collection in obj.users_collection:
        collection.objects.unlink(obj)
    target.objects.link(obj)


def check_collection(name, parent):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if not bpy.context.scene.user_of_id(collection):
        parent.children.link(collection)
    return collection


def check_exists(reference):
    try:
        reference.name
        return True
    except ReferenceError:
        return False


def validate_obj(op, obj, report=False, check_uvs=False):
    if obj.type != "MESH":
        if report:
            op.report({"ERROR"}, "Selected object is not a mesh")
        return False
    if len(obj.data.polygons) == 0:
        if report:
            op.report({"ERROR"}, "Selected object has zero polygons")
        return False
    if check_uvs and not obj.data.uv_layers:
        if report:
            op.report({"ERROR"}, "Selected object doesn't have a UV map")
        return False
    return True


def deselect_all():
    for object in bpy.context.selected_objects:
        object.select_set(False)


def select_uvs():
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")


def set_active_any():
    # find any mesh and set it to active
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # don't set an unlinked object to active
            if len(obj.users_collection) != 0:
                bpy.context.view_layer.objects.active = obj
                return obj


def edit_restore(input, func, *args, **kwargs):
    old_selection = bpy.context.selected_objects
    old_active = bpy.context.view_layer.objects.active

    if old_active is None:
        old_active = set_active_any()

    old_mode = old_active.mode

    bpy.ops.object.mode_set(mode="OBJECT")

    deselect_all()
    for obj in input:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = input[0]
    bpy.ops.object.mode_set(mode="EDIT")

    func(*args, **kwargs)

    bpy.ops.object.mode_set(mode="OBJECT")

    deselect_all()
    bpy.context.view_layer.objects.active = old_active
    for obj in old_selection:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode=old_mode)
