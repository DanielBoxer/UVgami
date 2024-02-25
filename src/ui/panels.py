# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
from ..manager import manager
from ..logger import logger
from ..utils import get_preferences, newline_label


expand = []


class UVGAMI_PT_main(bpy.types.Panel):
    bl_label = "UVgami"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw(self, context):
        box = self.layout.box()
        props = context.scene.uvgami

        row = box.row()
        row.scale_y = 2
        row.operator("uvgami.start", icon="UV")

        active_unwraps = manager.active
        groups = {}
        active_groups = []
        if active_unwraps:
            row = box.box().row()
            row.alignment = "CENTER"
            row.label(text="UV unwrap in progress")

            if manager.is_viewer_active:
                viewer_ui = box.box().row()
                viewer_ui.alignment = "CENTER"
                viewer_ui.label(text="Press ESC to exit viewer")

            for unwrap_idx, unwrap in enumerate(active_unwraps):
                # check for join jobs
                # meshes with matching join jobs will be grouped together
                found = False
                if unwrap.join_job is not None:
                    job = unwrap.join_job
                    if not job in groups:
                        groups[job] = []
                    groups[job].append(unwrap)
                    found = True
                    if unwrap.is_active and job not in active_groups:
                        active_groups.append(job)
                if not found:
                    # add to dictionary with unique int key
                    groups[unwrap_idx] = [unwrap]

            expand_idx = 0
            cancel_index = 0
            for id, group in groups.items():
                display_box = box.box()
                row = display_box.row()
                label_text = ""
                # if the key isn't an int, it's part of a group, and can be expanded
                expand_layout = not isinstance(id, int)
                # make sure there are enough items in the expand list
                if len(expand) < expand_idx + 1:
                    expand.append(False)

                # draw active icon and name
                is_active = False
                if expand_layout:
                    row.operator(
                        "uvgami.expand",
                        text="",
                        icon=f"DISCLOSURE_TRI_{'DOWN' if expand[expand_idx] else 'RIGHT'}",
                        emboss=False,
                    ).index = expand_idx
                    label_text = group[0].input_name
                    is_active = True if id in active_groups else False
                else:
                    label_text = group[0].name
                    is_active = True if group[0].is_active else False
                row.label(
                    text=label_text,
                    icon=f"RADIOBUT_{'ON' if is_active else 'OFF'}",
                )

                # group stop and cancel button
                if expand_layout:
                    if is_active:
                        stop_op = row.operator("uvgami.stop", text="", icon="SNAP_FACE")
                        stop_op.start_idx = cancel_index
                        stop_op.end_idx = cancel_index + len(group)
                    cancel_op = row.operator("uvgami.cancel", text="", icon="CANCEL")
                    cancel_op.start_idx = cancel_index
                    cancel_op.end_idx = cancel_index + len(group)
                    cancel_op.expand_idx = expand_idx

                # draw buttons
                if not expand_layout or expand[expand_idx]:
                    # if the group is expanded, show all items

                    for item in group:
                        if expand_layout:
                            row = display_box.row()
                            row.label(
                                text=item.name,
                                icon=f"LAYER_{'ACTIVE' if item.is_active else 'USED'}",
                            )

                        if item.progress != (0, 0, 1):
                            # viewer button
                            view_op = row.operator(
                                "uvgami.view_unwrap", text="", icon="HIDE_OFF"
                            )
                            view_op.index = manager.active.index(item)
                            # stop button
                            stop_op = row.operator(
                                "uvgami.stop", text="", icon="SNAP_FACE"
                            )
                            stop_op.start_idx = cancel_index
                            stop_op.end_idx = cancel_index + 1
                        # cancel button
                        cancel_op = row.operator(
                            "uvgami.cancel", text="", icon="CANCEL"
                        )
                        cancel_op.start_idx = cancel_index
                        cancel_op.end_idx = cancel_index + 1
                        if expand_layout:
                            cancel_op.expand_idx = expand_idx

                        cancel_index += 1
                elif expand_layout and not expand[expand_idx]:
                    # the length of the group needs to be added
                    cancel_index += len(group)

                expand_idx += 1

            if len(groups) > 1:
                row = box.row()
                row.operator("uvgami.cancel_all", icon="TRASH")

            box.separator()

        row = box.row()
        row.label(icon="SOLO_OFF", text="Quality")
        row.prop(props, "quality", text="")

        split = box.split(factor=0.7)
        split.label(icon="IMPORT", text="Import UVs")
        split.prop(props, "import_uvs")

        split = box.split(factor=0.7)
        split.label(icon="MOD_TRIANGULATE", text="Preserve Mesh")
        split.prop(props, "untriangulate")

        if props.untriangulate:
            row = box.row()
            row.prop(props, "maintain_mode", expand=True)


class UVGAMI_PT_speed(bpy.types.Panel):
    bl_label = "Speed"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        box = self.layout.box()
        props = context.scene.uvgami

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Speed", icon="SORTTIME")

        split = box.split(factor=0.7)
        split.label(icon="CON_ROTLIKE", text="Concurrent")
        split.prop(props, "concurrent")

        if props.concurrent:
            split = box.split()
            split.label(icon="SYSTEM", text="Cores")
            split.prop(props, "max_cores", slider=True)

        row = box.row()
        row.label(text="Finish", icon="TEMP")
        row.prop(props, "early_stop")

        split = box.split(factor=0.7)
        if props.use_symmetry:
            split.active = False
        split.label(text="Cuts", icon="MESH_GRID")
        split.prop(props, "use_cuts")

        if props.use_cuts:
            row = box.row()
            row.prop(props, "cut_type", expand=True)

        if props.use_cuts and props.cut_type == "EVEN":
            split = box.split()
            if props.use_symmetry:
                split.active = False
            split.prop(props, "cuts", slider=True)
            split.row().prop(props, "cut_axes")


class UVGAMI_PT_guides(bpy.types.Panel):
    bl_label = "Seam Restrictions"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_guided_mode")

    def draw(self, context):
        box = self.layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(
            text="Seam Restrictions",
            icon="SNAP_MIDPOINT" if bpy.app.version >= (2, 90, 0) else "IPO_LINEAR",
        )

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.draw_guides", icon="GREASEPENCIL")

        row = box.row()
        row.operator("uvgami.clear_draw", icon="FILE_REFRESH")
        row.operator("uvgami.exit_draw", icon="PANEL_CLOSE")

        row = box.row()
        row.label(text="Weight", icon="MOD_VERTEX_WEIGHT")
        row.prop(context.scene.uvgami, "weight_value", slider=True)


class UVGAMI_PT_symmetry(bpy.types.Panel):
    bl_label = "Symmetry"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_symmetry")

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Symmetry", icon="MOD_MIRROR")

        row = box.row()
        row.scale_y = 1.5
        row.prop(props, "sym_axes")

        row = box.row()
        row.operator("uvgami.preview_symmetry", icon="EMPTY_AXIS")
        row.prop(props, "sym_merge")


class UVGAMI_PT_grid(bpy.types.Panel):
    bl_label = "Grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Grid", icon="TEXTURE")

        split = box.split(factor=0.8)
        split.scale_y = 1.5
        split.operator("uvgami.add_grid", icon="UV_DATA")
        split.operator("uvgami.remove_grid", icon="TRASH", text="")

        row = box.row()
        row.prop(props, "grid_type", expand=True)
        box.prop(props, "grid_res")
        box.prop(props, "auto_grid")


class UVGAMI_PT_pack(bpy.types.Panel):
    bl_label = "Pack"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Pack", icon="PACKAGE")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.pack", icon="UGLYPACKAGE")

        row = box.split(factor=0.425)
        row.label(text="Margin", icon="IMGDISPLAY")
        row.prop(props, "margin", slider=True)

        box.prop(props, "combine_uvs")
        box.prop(props, "fix_scale")


class UVGAMI_PT_uv(bpy.types.Panel):
    bl_label = "UV Operations"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        box = self.layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="UV Operations", icon="GROUP_UVS")

        box.operator("uvgami.show_seams", icon="HIDE_OFF")

        split = box.split(factor=0.7)
        split.operator("uvgami.unwrap_sharp", icon="EDGESEL")

        row = split.row()
        row.scale_x = 0.85
        row.label(icon="VIEWZOOM")
        row.prop(context.scene.uvgami, "preview_unwrap_sharp")

        box.operator("uvgami.mark_seams_sharp", icon="SHARPCURVE")

        box.operator("uvgami.view_uvs", icon="VIEWZOOM")


class UVGAMI_PT_info(bpy.types.Panel):
    bl_label = "Info"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        box = self.layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Info", icon="INFO")

        if logger.unwrap_info:
            row = box.row()
            row.operator("uvgami.copy_logs", icon="COPYDOWN")
            row.operator("uvgami.clear_logs", icon="TRASH")
            col = box.column()
            newline_label(logger.get_all(), col)
        else:
            row = box.row()
            row.alignment = "CENTER"
            row.label(
                text=(
                    "No previous unwraps"
                    if get_preferences().show_info
                    else "Info is off"
                )
            )


class UVGAMI_PT_misc(bpy.types.Panel):
    bl_label = "Misc"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        box = self.layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Misc", icon="TOOL_SETTINGS")
        box.operator(
            "uvgami.reset_settings", text="Reset Settings", icon="FILE_REFRESH"
        )
        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.open_preferences", text="Preferences", icon="PREFERENCES")
