# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import pathlib
import platform
from ..utils import get_dir_path, get_preferences


class UVGAMI_PG_properties(bpy.types.PropertyGroup):
    quality: bpy.props.EnumProperty(
        name="Unwrap Quality",
        description=(
            (
                "A higher quality unwrap will have less stretching, "
                "but it will take longer to finish"
            )
        ),
        items=(
            ("HIGH", "High", ""),
            ("MEDIUM", "Medium", ""),
            ("LOW", "Low", ""),
        ),
        default="MEDIUM",
    )
    # preserve mesh
    untriangulate: bpy.props.BoolProperty(
        name="",
        description="Untriangulate mesh after unwrap. N-gons might not be preserved",
    )
    maintain_mode: bpy.props.EnumProperty(
        name="Preserve",
        description="",
        items=(
            (
                "FULL",
                "Full",
                (
                    "Fully untriangulate mesh and reroute seams."
                    " This might cause some areas to overlap slightly."
                    " There might also be a small amount of increased stretching."
                    " N-gons will remain triangulated."
                ),
            ),
            ("PARTIAL", "Partial", "Untriangulate all areas except for the seams"),
        ),
    )
    concurrent: bpy.props.BoolProperty(
        name="",
        description=(
            "Unwrap multiple meshes at the same time."
            " This only has an effect if you are unwrapping multiple meshes, "
            "or if the mesh is made up of multiple joined meshes"
        ),
    )
    use_guided_mode: bpy.props.BoolProperty(
        name="", description="Avoid placing seams on parts of the mesh"
    )
    weight_value: bpy.props.IntProperty(
        name="",
        description=(
            "A higher weight will follow the seam restrictions more "
            "but will take longer to finish the unwrap"
        ),
        min=1,
        max=5,
        default=3,
    )
    # symmetry
    use_symmetry: bpy.props.BoolProperty(
        name="",
        description=(
            "Use this setting for symmetrical meshes only."
            " This will result in a quicker unwrap with an symmetrical UV map"
        ),
    )
    sym_axes: bpy.props.EnumProperty(
        name="Axes",
        description=(
            "The axis or axes of symmetry of the input mesh."
            " Hold down Shift to select or deselect multiple axes"
        ),
        items=(
            ("X", "X", "X axis"),
            ("Y", "Y", "Y axis"),
            ("Z", "Z", "Z axis"),
        ),
        # allows for selection of multiple items
        options={"ENUM_FLAG"},
    )
    sym_merge: bpy.props.BoolProperty(
        name="Merge",
        description=(
            "Overlap and combine symmetrical UVs."
            " This will remove the seam on the axis"
        ),
        default=True,
    )
    # grid
    grid_type: bpy.props.EnumProperty(
        name="Grid Type",
        description="The type of grid material that will be added",
        items=(
            ("UV", "UV", "Normal UV grid"),
            ("COLOUR", "Colour", "Coloured UV grid"),
        ),
    )
    grid_res: bpy.props.IntProperty(
        name="Resolution",
        description="",
        default=1024,
        subtype="PIXEL",
        min=1,
        max=16384,
    )
    auto_grid: bpy.props.BoolProperty(
        name="Auto Grid", description="Automatically add a UV grid after unwrapping"
    )
    # pack
    margin: bpy.props.FloatProperty(
        name="", description="The space between UV islands", min=0, max=1
    )
    combine_uvs: bpy.props.BoolProperty(
        name="Combine UVs",
        description="Pack UVs of all selected objects into a single UV map",
    )
    fix_scale: bpy.props.BoolProperty(
        name="Average Islands Scale",
        description="Scale UV islands based on their actual size",
    )
    preview_unwrap_sharp: bpy.props.BoolProperty(
        name="",
        description=(
            "Preview: Only mark sharp edges as seams. Use this for high poly meshes"
        ),
    )


class UVGAMI_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = get_dir_path().stem

    autosave: bpy.props.BoolProperty(
        name="Autosave",
        description=(
            "Automatically save the Blender file before unwrapping "
            "to avoid losing work. This is recommended"
        ),
        default=True,
    )
    show_popup: bpy.props.BoolProperty(
        name="Show Popup",
        description=(
            "Show a popup when all meshes are finished unwrapping."
            " This might contain other information like if any objects were invalid or "
            "if there were any errors"
        ),
        default=True,
    )
    engine_path: bpy.props.StringProperty(
        name="",
        description="The path to the unwrapper application stored on your computer",
        subtype="FILE_PATH",
    )
    cleanup: bpy.props.EnumProperty(
        name="Input Cleanup",
        description="The action to perform on the original input mesh",
        items=(
            ("NONE", "None", "Leave the input mesh as it is"),
            ("HIDE", "Hide", "Hide the original input mesh"),
            ("DELETE", "Delete", "Delete the original input mesh"),
        ),
        default="HIDE",
    )
    invalid_collection: bpy.props.BoolProperty(
        name="Invalid Collection",
        description="Add all invalid meshes to a collection",
        default=True,
    )
    show_progress_bar: bpy.props.BoolProperty(
        name="Progress Bar",
        description="Display a progress bar in the 3D view during an unwrap",
        default=True,
    )
    show_info: bpy.props.BoolProperty(
        name="Info",
        description="Show information about previous unwraps in the info panel",
        default=True,
    )
    license_key: bpy.props.StringProperty(name="", description="Enter your license key")
    viewer_workspace: bpy.props.StringProperty(
        name="Viewer Workspace",
        description=(
            "The name of the workspace that will be opened when viewing an unwrap."
            " If this is empty, the UV editor will be opened instead"
        ),
    )
    # non ui
    is_wsl_setup: bpy.props.BoolProperty()

    def draw(self, context):
        layout = self.layout
        prefs = get_preferences()

        box = layout.box()
        row = box.row()
        row.scale_y = 1.5
        split = row.split(factor=0.2)
        split.scale_x = 1.5
        split.label(text="Engine Path")
        split.prop(self, "engine_path")

        engine_path = pathlib.Path(prefs.engine_path)
        if (
            str(engine_path) != "."
            and engine_path.is_file()
            and engine_path.stem == "uvgami"
            and engine_path.suffix == ""
            and platform.system() == "Windows"
        ):
            row.operator("uvgami.setup_wsl")

        split = box.split(factor=0.2)
        split.label(text="License Key")
        split.prop(self, "license_key")

        box = layout.box()

        cf = box.column_flow(columns=3)

        row = cf.row()
        row.label(icon="FILE_TICK")
        row.prop(self, "autosave")

        row = cf.row()
        row.label(icon="WINDOW")
        row.prop(self, "show_popup")

        row = cf.row()
        row.label(icon="SORTTIME")
        row.prop(self, "show_progress_bar")

        row = cf.row()
        row.label(icon="INFO")
        row.prop(self, "show_info")

        row = cf.row()
        row.label(
            icon="OUTLINER_COLLECTION" if bpy.app.version >= (2, 92, 0) else "GROUP"
        )
        row.prop(self, "invalid_collection")

        box.separator()

        row = box.row()
        row.label(icon="MOD_WIREFRAME")
        row.prop(self, "cleanup")

        if prefs.cleanup == "DELETE":
            row = box.row()
            row.label(
                text=(
                    "Warning: Use 'Input Cleanup: Delete' at your own risk, "
                    "losing work is possible"
                )
            )

        row = box.row()
        row.label(icon="WORKSPACE")
        row.prop(self, "viewer_workspace")
