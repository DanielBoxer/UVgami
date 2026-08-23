import multiprocessing

import bpy

from ..engines import ENGINES, installed_engines
from ..utils.paths import get_addon_id

# hang backstop, generous so a slow legitimate piece never hits it
UNWRAP_TIMEOUT_DEFAULT_MINUTES = 60

# built once so the item strings stay referenced, which blender requires for
# dynamic enum callbacks. explicit numbers keep saved files stable as the
# installed set changes
_ENGINE_ITEMS = {
    e.id: (e.id, e.label, e.description, e.enum_value) for e in ENGINES.values()
}


def _engine_items(self, context):
    return [_ENGINE_ITEMS[e.id] for e in installed_engines()]


# the getter clamps to an installed engine without touching the stored value,
# so the widget can't go blank when the selected engine is deleted, and
# reinstalling it restores the old selection
def _engine_get(self):
    stored = self.get("engine", -1)
    installed = installed_engines()
    if any(_ENGINE_ITEMS[e.id][3] == stored for e in installed):
        return stored
    return _ENGINE_ITEMS[installed[0].id][3] if installed else 0


def _engine_set(self, value):
    self["engine"] = value


class UVGAMI_PG_properties(bpy.types.PropertyGroup):
    engine: bpy.props.EnumProperty(
        name="Engine",
        description="The unwrapping engine to use",
        items=_engine_items,
        get=_engine_get,
        set=_engine_set,
    )
    import_uvs: bpy.props.BoolProperty(
        name="", description="Use the UV map on the mesh as input"
    )
    # TODO: preserve mesh is off, delete these props and the untriangulate code
    # untriangulate: bpy.props.BoolProperty(
    #     name="",
    #     description="Untriangulate mesh after unwrap. N-gons might not be preserved",
    # )
    # maintain_mode: bpy.props.EnumProperty(
    #     name="Preserve",
    #     description="How much of the mesh to untriangulate after unwrap",
    #     items=(
    #         (
    #             "FULL",
    #             "Full",
    #             (
    #                 "Fully untriangulate mesh and reroute seams."
    #                 " This might cause some areas to overlap slightly."
    #                 " There might also be a small amount of increased stretching."
    #                 " N-gons will remain triangulated."
    #             ),
    #         ),
    #         ("PARTIAL", "Partial", "Untriangulate all areas except for the seams"),
    #     ),
    #     default="FULL",
    # )
    # speed
    max_cores: bpy.props.IntProperty(
        name="",
        description="How many meshes to unwrap at the same time",
        default=max(1, multiprocessing.cpu_count() // 2),
        max=multiprocessing.cpu_count(),
        min=1,
    )
    unwrap_timeout: bpy.props.IntProperty(
        name="",
        description="Maximum time in minutes for each unwrap. Set to 0 to disable",
        min=0,
        max=120,
        default=UNWRAP_TIMEOUT_DEFAULT_MINUTES,
    )
    area_expand: bpy.props.IntProperty(
        name="",
        description=(
            "Grow the selected area by this many face rings to affect a larger area"
        ),
        min=0,
        max=10,
        default=1,
    )
    stack_similar: bpy.props.BoolProperty(
        name="",
        description="Stack repeated mesh pieces by only unwrapping one and copying it",
    )
    use_proxy: bpy.props.BoolProperty(
        name="",
        description=(
            "Unwrap a low poly copy, then transfer the seams to the original."
            " Much faster for high poly meshes"
        ),
    )
    proxy_faces: bpy.props.IntProperty(
        name="",
        description="How many triangles the decimated copy keeps",
        min=100,
        max=100000,
        default=2000,
    )
    # weights
    use_weights: bpy.props.BoolProperty(
        name="", description="Use the painted weights to change the unwrap"
    )
    weight_mode: bpy.props.EnumProperty(
        name="Mode",
        description="What the painted weights do",
        items=(
            ("SEAMS", "Avoid Seams", "Keep seams off the painted faces"),
            (
                "STRETCH",
                "Reduce Stretching",
                "Prioritize the painted faces to have less stretching"
                " relative to the other faces",
            ),
        ),
        default="SEAMS",
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
            " This will result in a symmetrical UV map"
        ),
    )
    sym_preview: bpy.props.BoolProperty(
        name="Preview",
        description="Show the symmetry planes of the selected meshes in the viewport",
        default=True,
    )
    sym_axes: bpy.props.EnumProperty(
        name="Axes",
        description=(
            "The axis of symmetry of the input mesh."
            " Hold down Shift to select or deselect multiple axes"
        ),
        items=(
            ("X", "X", "X axis"),
            ("Y", "Y", "Y axis"),
            ("Z", "Z", "Z axis"),
        ),
        default={"X"},
        # allows for selection of multiple items
        options={"ENUM_FLAG"},
    )
    sym_merge: bpy.props.BoolProperty(
        name="Merge",
        description=(
            "Overlap and combine symmetrical UVs. This will remove the seam on the axis"
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
        description="The resolution of the grid texture in pixels",
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
        name="",
        description="The space between UV islands",
        min=0,
        max=1,
        default=0.001,
        precision=3,
    )
    combine_uvs: bpy.props.BoolProperty(
        name="Combine UVs",
        description="Pack UVs of all selected objects into a single UV map",
    )
    fix_scale: bpy.props.BoolProperty(
        name="Average Islands Scale",
        description="Scale UV islands based on their actual size",
        default=True,
    )
    pack_after_unwrap: bpy.props.BoolProperty(
        name="Pack After Unwrap",
        description="Automatically pack UVs after each unwrap finishes",
        default=True,
    )
    transfer_uvs: bpy.props.BoolProperty(
        name="",
        description=(
            "Transfer the UV map from the output mesh to the original input mesh"
        ),
        default=False,
    )

    @property
    def preserve_mesh(self):
        # a transfer writes onto the original, which never lost its quads.
        # engines without preserve renumber verts, dissolving the wrong edges
        return False
        # engine = ENGINES.get(self.engine)
        # return (
        #     self.untriangulate
        #     and not self.transfer_uvs
        #     and engine is not None
        #     and engine.supports_preserve
        # )

    @property
    def avoid_seams(self):
        return self.use_weights and self.weight_mode == "SEAMS"

    @property
    def reduce_stretching(self):
        return self.use_weights and self.weight_mode == "STRETCH"


# each engine contributes a pointer to its own settings group, keyed by engine id
for engine in ENGINES.values():
    UVGAMI_PG_properties.__annotations__[engine.id.lower()] = bpy.props.PointerProperty(
        type=engine.property_group
    )


class UVGAMI_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = get_addon_id()

    autosave: bpy.props.BoolProperty(
        name="Autosave",
        description=("Automatically save the Blender file before unwrapping"),
        default=False,
    )
    show_popup: bpy.props.BoolProperty(
        name="Show Popup",
        description="Show a popup when all meshes are finished unwrapping",
        default=False,
    )
    show_progress_bar: bpy.props.BoolProperty(
        name="Progress Bar",
        description="Display a progress bar in the 3D view and UV editor during an unwrap",
        default=True,
    )
    show_warnings: bpy.props.BoolProperty(
        name="Warnings",
        description="Show warning if modifier is applied before unwrap",
        default=True,
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        row = box.row()
        row.label(text="General", icon="PREFERENCES")

        grid = box.grid_flow(row_major=True, columns=3, even_columns=True)

        row = grid.row()
        row.label(icon="FILE_TICK")
        row.prop(self, "autosave")

        row = grid.row()
        row.label(icon="WINDOW")
        row.prop(self, "show_popup")

        row = grid.row()
        row.label(icon="SORTTIME")
        row.prop(self, "show_progress_bar")

        row = grid.row()
        row.label(icon="INFO")
        row.prop(self, "show_warnings")

        box.separator()

        row = box.row()
        row.operator(
            "uvgami.reset_settings", text="Reset Settings", icon="FILE_REFRESH"
        )

        box = layout.box()
        row = box.row()
        row.label(text="Engines", icon="TOOL_SETTINGS")

        for engine in ENGINES.values():
            engine_box = box.box()
            row = engine_box.row()
            row.label(text=engine.label, icon=engine.icon)
            row = engine_box.row()
            row.active = False
            row.label(text=engine.description)
            engine.draw_prefs(engine_box, self)
