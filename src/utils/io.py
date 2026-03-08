import bpy


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


def print_stdin(process, msg):
    if process.poll() is not None:
        return False
    try:
        print(msg, file=process.stdin, flush=True)
    except OSError:
        return False
    return True
