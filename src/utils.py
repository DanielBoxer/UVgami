# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import math
import pathlib

import bmesh
import bpy
import mathutils
import numpy


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


def export_obj(obj, path, export_uv):
    obj.select_set(True)

    if bpy.app.version >= (3, 1, 0):
        # new obj exporter
        args = {
            "filepath": str(path),
            "export_selected_objects": True,
            "export_normals": False,
            "export_uv": export_uv,
            "export_materials": False,
            "apply_modifiers": False,
            "forward_axis": "Y",
            "up_axis": "Z",
        }

        if bpy.app.version < (3, 2, 0):
            # apply_modifiers doesn't exist in 3.1
            del args["apply_modifiers"]

        if bpy.app.version < (3, 3, 0):
            # axis enums were renamed
            args["forward_axis"] = "Y_FORWARD"
            args["up_axis"] = "Z_UP"

        bpy.ops.wm.obj_export("EXEC_DEFAULT", **args)

    else:
        # old
        bpy.ops.export_scene.obj(
            "EXEC_DEFAULT",
            filepath=str(path),
            use_selection=True,
            use_normals=False,
            use_uvs=export_uv,
            use_materials=False,
            use_blen_objects=False,
            use_mesh_modifiers=False,
            axis_forward="Y",
            axis_up="Z",
        )

    obj.select_set(False)


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


def apply_transforms(obj):
    location, _, scale = obj.matrix_basis.decompose()
    actual = (
        mathutils.Matrix.Translation(location)
        @ obj.matrix_basis.to_3x3().normalized().to_4x4()
        @ mathutils.Matrix.Diagonal(scale).to_4x4()
    )
    obj.data.transform(actual)
    for c in obj.children:
        c.matrix_local = actual @ c.matrix_local
    obj.matrix_basis = mathutils.Matrix()


def cut(num, center, length, dim, bm):
    start = center[dim] - length / 2
    end = center[dim] + length / 2
    rot = [0, 0, 0]
    rot[dim] = math.radians(90)
    # n + 2 for endpoints, 1:-1 to remove endpoints
    for s in numpy.linspace(start, end, num + 2)[1:-1]:
        loc = center.copy()
        loc[dim] = s
        cut = bmesh.ops.bisect_plane(
            bm,
            geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            plane_co=loc,
            plane_no=rot,
            dist=1e-7,
        )["geom_cut"]
        bmesh.ops.split_edges(
            bm, edges=[e for e in cut if isinstance(e, bmesh.types.BMEdge)]
        )
