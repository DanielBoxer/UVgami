# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
from ..utils import validate_obj

_old_mode = None
_old_active_group = None


def is_draw_active():
    active_object = bpy.context.active_object
    vertex_groups = active_object.vertex_groups
    if "UVgami_seam_restrictions" not in vertex_groups:
        return False
    return (
        active_object.mode == "WEIGHT_PAINT"
        and vertex_groups.active_index
        == vertex_groups["UVgami_seam_restrictions"].index
    )


class UVGAMI_OT_draw_guides(bpy.types.Operator):
    bl_idname = "uvgami.draw_guides"
    bl_label = "Draw"
    bl_description = (
        "Draw seam restrictions to change the unwrap."
        " Seams will not be placed on red areas"
    )
    bl_options = {"UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        if active_obj is None:
            return {"CANCELLED"}
        if not validate_obj(self, active_obj, report=True):
            return {"CANCELLED"}

        vertex_groups = active_obj.vertex_groups
        if not is_draw_active():
            global _old_mode
            _old_mode = active_obj.mode
            global _old_active_group
            _old_active_group = vertex_groups.active_index

        bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
        if "UVgami_seam_restrictions" not in vertex_groups:
            vertex_groups.new(name="UVgami_seam_restrictions")
        vertex_groups.active_index = vertex_groups["UVgami_seam_restrictions"].index
        return {"FINISHED"}


class UVGAMI_OT_exit_draw(bpy.types.Operator):
    bl_idname = "uvgami.exit_draw"
    bl_label = "Exit"
    bl_description = "Go back to previous mode"
    bl_options = {"UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        if active_obj is None:
            return {"CANCELLED"}

        if is_draw_active():
            active_obj.vertex_groups.active_index = _old_active_group
            bpy.ops.object.mode_set(mode=_old_mode)
        return {"FINISHED"}


class UVGAMI_OT_clear_draw(bpy.types.Operator):
    bl_idname = "uvgami.clear_draw"
    bl_label = "Clear"
    bl_description = "Clear current seam restrictions"
    bl_options = {"UNDO"}

    def execute(self, context):
        for obj in context.selected_objects:
            if not validate_obj(self, obj):
                continue

            vertex_groups = obj.vertex_groups
            if "UVgami_seam_restrictions" in vertex_groups:
                group_idx = vertex_groups["UVgami_seam_restrictions"].index
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == group_idx:
                            g.weight = 0
                            break
        return {"FINISHED"}
