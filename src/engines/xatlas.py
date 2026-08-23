import bpy

from ..utils.paths import get_extension_dir_path
from ..utils.ui import is_non_default, only_active
from .binary_engine import BinaryEngine, EngineRelease, InstallEngineTask

# must match engine/xatlas/VERSION (check-engine-versions.yml fails on drift)
XATLAS_VERSION = "0.2.3"
XATLAS = EngineRelease("xatlas", "xatlas", XATLAS_VERSION, "300 KB")


class UVGAMI_OT_install_xatlas(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_xatlas"
    bl_label = "Download xatlas Engine"
    owner = "xatlas"
    release = XATLAS


class UVGAMI_PG_xatlas(bpy.types.PropertyGroup):
    max_cost: bpy.props.FloatProperty(
        name="",
        description="Lower values cut the mesh into more UV islands",
        default=2.0,
        # above 10 the chart count stops dropping, xatlas' merge pass sets the floor
        min=0.1,
        max=10.0,
    )


class XatlasEngine(BinaryEngine):
    id = "XATLAS"
    enum_value = 2
    label = "xatlas"
    description = "Fast CPU engine for baking lightmaps and texture painting"
    icon = "MESH_GRID"
    property_group = UVGAMI_PG_xatlas
    classes = (UVGAMI_PG_xatlas, UVGAMI_OT_install_xatlas)
    release = XATLAS
    # xatlas packs its own atlas, so it never needs forced packing

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="UV_ISLANDSEL", text="Chart Cost")
        row.prop(props.xatlas, "max_cost")

    def active_settings(self, props):
        max_cost = props.xatlas.max_cost
        return only_active(
            (
                (
                    "UV_ISLANDSEL",
                    f"Chart Cost {max_cost:.2f}",
                    "xatlas.max_cost",
                    is_non_default(props, "xatlas.max_cost"),
                ),
            )
        )

    def build_args(self, ctx, input_path, props):
        output_path = get_extension_dir_path() / "output" / f"{input_path.stem}.obj"
        return [
            str(ctx),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--max-cost",
            f"{props.xatlas.max_cost:.4f}",
        ]

    def describe_failure(self, code):
        return {
            2: ("Invalid input mesh", True),
            3: ("Invalid geometry", True),
            4: ("Unwrap failed", True),
        }.get(code) or super().describe_failure(code)


ENGINE = XatlasEngine()
