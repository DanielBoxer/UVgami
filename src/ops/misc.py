# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import math
import pathlib
import subprocess
import shutil
from ..ui.panels import expand
from ..utils import (
    calc_center,
    validate_obj,
    get_linux_path,
    get_preferences,
    check_exists,
    deselect_all,
)

sym_planes = {}


class UVGAMI_OT_expand(bpy.types.Operator):
    bl_idname = "uvgami.expand"
    bl_label = "Expand"
    bl_description = "Expand"

    index: bpy.props.IntProperty()

    def execute(self, context):
        expand[self.index] = not expand[self.index]
        return {"FINISHED"}


class UVGAMI_OT_reset_settings(bpy.types.Operator):
    bl_idname = "uvgami.reset_settings"
    bl_label = "Reset Settings"
    bl_description = "Reset all settings to their default values"
    bl_options = {"UNDO"}

    def execute(self, context):
        props = context.scene.uvgami
        for prop in props.__annotations__.keys():
            props.property_unset(prop)
        return {"FINISHED"}


class UVGAMI_OT_open_preferences(bpy.types.Operator):
    bl_idname = "uvgami.open_preferences"
    bl_label = "Preferences"
    bl_description = "Open UVgami preferences"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = "ADDONS"
        bpy.data.window_managers["WinMan"].addon_search = "UVgami"
        bpy.ops.preferences.addon_show(module="UVgami")
        return {"FINISHED"}


class UVGAMI_OT_preview_symmetry(bpy.types.Operator):
    bl_idname = "uvgami.preview_symmetry"
    bl_label = "Preview"
    bl_description = (
        "Add a plane meshes to verify symmetry of selected meshes."
        " Press again to delete the planes"
    )
    bl_options = {"UNDO"}

    def execute(self, context):
        sym = context.scene.uvgami.sym_axes
        old_select = context.selected_objects
        old_active = context.view_layer.objects.active
        for obj in context.selected_objects:
            if obj not in sym_planes:
                if validate_obj(self, obj):
                    center = calc_center(obj)
                    before = set(context.scene.objects)
                    if "X" in sym:
                        bpy.ops.mesh.primitive_plane_add(
                            size=obj.dimensions.y * 2,
                            location=center,
                            rotation=(0, math.radians(90), 0),
                        )
                    if "Y" in sym:
                        bpy.ops.mesh.primitive_plane_add(
                            size=obj.dimensions.x * 2,
                            location=center,
                            rotation=(math.radians(90), 0, 0),
                        )
                    if "Z" in sym:
                        bpy.ops.mesh.primitive_plane_add(
                            size=obj.dimensions.z * 2, location=center
                        )
                    sym_planes[obj] = set(context.scene.objects).difference(before)
            else:
                for plane in sym_planes[obj]:
                    if check_exists(plane):
                        bpy.data.objects.remove(plane, do_unlink=True)
                del sym_planes[obj]

        deselect_all()
        for obj in old_select:
            obj.select_set(True)
        context.view_layer.objects.active = old_active
        return {"FINISHED"}


class UVGAMI_OT_setup_wsl(bpy.types.Operator):
    bl_idname = "uvgami.setup_wsl"
    bl_label = "Setup WSL"
    bl_description = (
        "Setup WSL. This only needs to be done when using a new engine file."
        " Press if you are using WSL for the first time or if the engine was updated"
    )

    def execute(self, context):
        # wsl check
        if shutil.which("wsl") is None:
            self.report(
                {"ERROR"},
                (
                    "WSL is not installed."
                    " Either install WSL or use UVgami for Windows"
                ),
            )
            return {"CANCELLED"}

        # copy uvgami to wsl
        prefs = get_preferences()
        path = get_linux_path(pathlib.Path(prefs.engine_path))
        subprocess.run(["bash", "-c", f"cp {path} ~/"])
        prefs.is_wsl_setup = True

        self.report({"INFO"}, ("Successfully setup WSL"))
        return {"FINISHED"}
