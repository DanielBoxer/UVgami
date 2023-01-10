# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import math
from ..utils import edit_restore, select_uvs, validate_obj


def pack():
    select_uvs()
    if bpy.context.scene.uvgami.fix_scale:
        bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(margin=bpy.context.scene.uvgami.margin)
    bpy.ops.uv.select_all(action="DESELECT")


def show_seams():
    select_uvs()
    bpy.ops.uv.mark_seam(clear=True)
    bpy.ops.uv.seams_from_islands()
    bpy.ops.uv.select_all(action="DESELECT")
    bpy.ops.mesh.select_all(action="DESELECT")


class UVGAMI_OT_show_seams(bpy.types.Operator):
    bl_idname = "uvgami.show_seams"
    bl_label = "Show Seams"
    bl_description = "Make the UV seams visible"
    bl_options = {"UNDO"}

    def execute(self, context):
        for obj in context.selected_objects:
            if not validate_obj(self, obj, check_uvs=True):
                continue
            edit_restore([obj], show_seams)
        return {"FINISHED"}


class UVGAMI_OT_unwrap_sharp(bpy.types.Operator):
    bl_idname = "uvgami.unwrap_sharp"
    bl_label = "Unwrap Sharp"
    bl_description = "Unwrap by sharp edges"
    bl_options = {"UNDO", "REGISTER"}

    angle: bpy.props.FloatProperty(
        name="",
        description="Angle used to decide if an edge is sharp enough",
        default=math.radians(30),
        subtype="ANGLE",
        min=math.radians(1),
        max=math.radians(180),
    )
    margin: bpy.props.FloatProperty(name="", min=0, max=1)

    def execute(self, context):
        for obj in context.selected_objects:
            if validate_obj(self, obj):
                # active object is needed
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                # must be in edge select mode
                context.tool_settings.mesh_select_mode = (False, True, False)
                # clear seams
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.mark_seam(clear=True)
                # select and mark sharp
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.edges_select_sharp(sharpness=self.angle)
                bpy.ops.mesh.mark_seam()
                # unwrap and pack
                if not context.scene.uvgami.preview_unwrap_sharp:
                    if bpy.ops.uv.unwrap() == {"CANCELLED"}:
                        self.report({"WARNING"}, "Unwrap failed, use a lower angle")
                    select_uvs()
                    bpy.ops.uv.pack_islands(margin=self.margin)
            else:
                obj.select_set(False)
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        split = layout.split(factor=0.3)
        split.label(text="Sharpness")
        split.prop(self, "angle", slider=True)
        if not context.scene.uvgami.preview_unwrap_sharp:
            split = layout.split(factor=0.3)
            split.label(text="Margin")
            split.prop(self, "margin", slider=True)


class UVGAMI_OT_mark_seams_sharp(bpy.types.Operator):
    bl_idname = "uvgami.mark_seams_sharp"
    bl_label = "Mark Seams Sharp"
    bl_description = "Mark all seams as sharp"
    bl_options = {"UNDO"}

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")
        for obj in context.selected_objects:
            if validate_obj(self, obj):
                for edge in obj.data.edges:
                    edge.use_edge_sharp = True if edge.use_seam else False
        return {"FINISHED"}


class UVGAMI_OT_pack(bpy.types.Operator):
    bl_idname = "uvgami.pack"
    bl_label = "Pack"
    bl_description = "Pack UVs with Blender's packer"
    bl_options = {"UNDO"}

    def execute(self, context):
        combine_uvs = context.scene.uvgami.combine_uvs
        valid_objs = []

        for obj in context.selected_objects:
            if not validate_obj(self, obj, check_uvs=True):
                continue
            if combine_uvs:
                valid_objs.append(obj)
            else:
                edit_restore([obj], pack)

        if combine_uvs:
            edit_restore(valid_objs, pack)

        return {"FINISHED"}
