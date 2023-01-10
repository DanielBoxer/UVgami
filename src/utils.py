# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import bmesh
import mathutils
import pathlib


def get_preferences():
    return bpy.context.preferences.addons[get_dir_path().stem].preferences


def get_dir_path():
    return pathlib.Path(__file__).parents[1]


def get_linux_path(path):
    return f'"/mnt/c{str(pathlib.PurePosixPath(path))[3:]}"'


def newline_label(label, layout):
    for line in label:
        layout.label(text=line)


def popup(msg, title, icon):
    def draw(self, context):
        newline_label(msg, self.layout)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def deselect_all():
    for object in bpy.context.selected_objects:
        object.select_set(False)


def select_uvs():
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")


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


def set_active_any():
    # find any mesh and set it to active
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # don't set an unlinked object to active
            if len(obj.users_collection) != 0:
                bpy.context.view_layer.objects.active = obj
                return obj


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


def import_obj(path, name=""):
    before = set(bpy.context.scene.objects)
    previous_select = bpy.context.selected_objects

    # blender 3.1 doesn't have the new importer
    if bpy.app.version >= (3, 2, 0):
        # new obj importer
        forward = "Y"
        up = "Z"
        if bpy.app.version < (3, 3, 0):
            # axis names renamed
            forward = "Y_FORWARD"
            up = "Z_UP"

        bpy.ops.wm.obj_import(
            "EXEC_DEFAULT", filepath=str(path), forward_axis=forward, up_axis=up
        )

    else:
        # old
        bpy.ops.import_scene.obj(
            "EXEC_DEFAULT",
            filepath=str(path),
            split_mode="OFF",
            axis_forward="Y",
            axis_up="Z",
        )
    # push to undo stack
    bpy.ops.ed.undo_push()

    after = set(bpy.context.scene.objects)

    # get imported object
    imported_obj = after.difference(before).pop()
    imported_obj.select_set(False)

    for obj in previous_select:
        obj.select_set(True)

    if name != "":
        imported_obj.name = name

    return imported_obj


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


def switch_shading(type):
    for area in bpy.context.screen.areas:
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = type
                break


def set_origin(obj, point):
    mw = obj.matrix_world
    obj.data.transform(mathutils.Matrix.Translation(-(mw.inverted() @ point)))
    mw.translation += point - mw.translation


def calc_center(obj):
    lc = 0.125 * sum((mathutils.Vector(co) for co in obj.bound_box), mathutils.Vector())
    center = obj.matrix_world @ lc
    return center


def print_stdin(process, msg):
    if process.poll() is not None:
        return False
    try:
        print(msg, file=process.stdin, flush=True)
    except OSError:
        return False
    return True
