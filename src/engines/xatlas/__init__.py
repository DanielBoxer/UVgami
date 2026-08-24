from ...utils.paths import get_extension_dir_path
from ..binary_engine import BinaryEngine
from .install import XATLAS, UVGAMI_OT_install_xatlas


# above 4.0 the output is identical
PRIORITY_VALUES = {"LESS_STRETCH": "0.1", "BALANCED": "2.0", "FEWER_SEAMS": "4.0"}


class XatlasEngine(BinaryEngine):
    id = "XATLAS"
    enum_value = 2
    label = "xatlas"
    description = "Fast CPU engine for baking lightmaps and texture painting"
    icon = "MESH_GRID"
    classes = (UVGAMI_OT_install_xatlas,)
    release = XATLAS
    # xatlas packs its own atlas, so it never needs forced packing

    def build_args(self, ctx, input_path, props):
        output_path = get_extension_dir_path() / "output" / f"{input_path.stem}.obj"
        return [
            str(ctx),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--max-cost",
            PRIORITY_VALUES[props.priority],
        ]

    def describe_failure(self, code):
        return {
            2: ("Invalid input mesh", True),
            3: ("Invalid geometry", True),
            4: ("Unwrap failed", True),
        }.get(code) or super().describe_failure(code)


ENGINE = XatlasEngine()
