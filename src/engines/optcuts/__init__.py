import bpy

from ...hard_surface import (
    auto_hard_faces,
    build_seam_uvs,
    preseed_work,
    seam_flags,
    seam_restrictions,
)
from ...seams import uvs_collapsed
from ...utils.io import print_stdin
from ...utils.mesh import corner_uvs
from ...utils.ui import is_non_default, only_active
from ..binary_engine import BinaryEngine
from .install import OPTCUTS, UVGAMI_OT_install_optcuts


# what counts as a sharp feature
HARD_SURFACE_ANGLE = 66

PRIORITY_VALUES = {"LESS_STRETCH": "4.05", "BALANCED": "4.2", "FEWER_SEAMS": "5.0"}


class UVGAMI_PG_optcuts(bpy.types.PropertyGroup):
    use_hard_surface: bpy.props.BoolProperty(
        name="",
        description="Cut seams on sharp features. Good for mechanical shapes",
    )
    # hard_surface_auto: bpy.props.BoolProperty(
    #     name="Auto",
    #     description=(
    #         "Automatically unwrap sharp objects in hard surface mode and"
    #         " otherwise in normal mode"
    #     ),
    # )
    # hard_surface_angle: bpy.props.FloatProperty(
    #     name="Angle",
    #     description="What counts as a sharp feature. Lower keeps more seams",
    #     subtype="ANGLE",
    #     default=math.radians(66),
    #     min=math.radians(1),
    #     max=math.radians(180),
    # )

    @property
    def is_auto(self):
        # return self.use_hard_surface and self.hard_surface_auto
        return False


# class UVGAMI_OT_quick_unwrap(bpy.types.Operator):
#     bl_idname = "uvgami.quick_unwrap"
#     bl_label = "Quick Unwrap"
#     bl_description = "Mark the hard surface seams and flatten them"
#     bl_options = {"UNDO"}
#
#     def execute(self, context):
#         props = context.scene.uvgami
#         optcuts = props.optcuts
#         angle = HARD_SURFACE_ANGLE
#         guided = props.avoid_seams
#         selected = list(context.selected_objects)
#         active = context.view_layer.objects.active
#         mode = active.mode if active is not None else "OBJECT"
#         if mode != "OBJECT":
#             bpy.ops.object.mode_set(mode="OBJECT")
#
#         counts = []
#         flatten_error = None
#         for obj in selected:
#             if not validate_obj(self, obj):
#                 continue
#             only = None
#             if optcuts.is_auto:
#                 only = auto_hard_faces(obj)
#                 if not only:
#                     counts.append("organic")
#                     continue
#                 if len(only) == len(obj.data.polygons):
#                     only = None
#             weights = seam_restrictions(obj) if guided else None
#             try:
#                 applied = build_seam_uvs(obj, angle, weights=weights, only=only)
#             except FlattenError as error:
#                 flatten_error = str(error)
#                 break
#             if not applied:
#                 counts.append("no seams")
#                 continue
#             counts.append(str(int(seam_flags(obj.data).sum())))
#
#         deselect_all()
#         for obj in selected:
#             obj.select_set(True)
#         context.view_layer.objects.active = active
#         if active is not None and mode != "OBJECT":
#             bpy.ops.object.mode_set(mode=mode)
#
#         if flatten_error is not None:
#             self.report({"ERROR"}, flatten_error)
#             # FINISHED so the undo step covers meshes already written
#             return {"FINISHED"}
#         if not counts:
#             self.report({"ERROR"}, "Select a mesh with faces")
#             return {"CANCELLED"}
#         self.report({"INFO"}, f"Seams: {', '.join(counts)}")
#         return {"FINISHED"}


# class UVGAMI_PT_hard_surface(bpy.types.Panel):
#     bl_label = "Hard Surface"
#     bl_space_type = "VIEW_3D"
#     bl_region_type = "UI"
#     bl_category = "UVgami"
#     bl_parent_id = "UVGAMI_PT_main"
#     bl_options = {"DEFAULT_CLOSED"}
#     bl_order = 0
#
#     @classmethod
#     def poll(cls, context):
#         # imported here: engines imports this module back
#         from .. import active_engine
#
#         return active_engine(context.scene.uvgami.engine) is ENGINE
#
#     def draw_header(self, context):
#         self.layout.prop(context.scene.uvgami.optcuts, "use_hard_surface")
#
#     def draw(self, context):
#         optcuts = context.scene.uvgami.optcuts
#         layout = self.layout
#         layout.active = optcuts.use_hard_surface
#         box = layout.box()
#
#         row = box.row()
#         row.alignment = "CENTER"
#         row.label(text="Hard Surface", icon="MOD_BEVEL")
#
#         box.prop(optcuts, "hard_surface_auto")
#
#         row = box.row()
#         row.label(icon="DRIVER_ROTATIONAL_DIFFERENCE", text="Angle")
#         row.prop(optcuts, "hard_surface_angle", text="")


class OptcutsEngine(BinaryEngine):
    id = "OPTCUTS"
    enum_value = 0
    label = "Optcuts"
    description = (
        "Default CPU engine. Highest quality but can be slow."
        " Includes the UV island operators"
    )
    icon = "UV"
    property_group = UVGAMI_PG_optcuts
    classes = (
        UVGAMI_PG_optcuts,
        UVGAMI_OT_install_optcuts,
        # UVGAMI_OT_quick_unwrap,
        # UVGAMI_PT_hard_surface,
    )
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_preserve = True
    supports_import_uvs = True
    supports_proxy = True
    release = OPTCUTS

    def draw_settings(self, layout, props):
        split = layout.split(factor=0.7)
        split.label(icon="MOD_BEVEL", text="Hard Surface")
        split.prop(props.optcuts, "use_hard_surface")

    def active_settings(self, props):
        return only_active(
            (
                (
                    "MOD_BEVEL",
                    "Hard Surface",
                    "optcuts.use_hard_surface",
                    is_non_default(props, "optcuts.use_hard_surface"),
                ),
                # (
                #     "AUTO",
                #     "Hard Surface Auto",
                #     "optcuts.hard_surface_auto",
                #     optcuts.is_auto,
                # ),
            )
        )

    def import_uvs_ignored(self, props):
        return props.optcuts.use_hard_surface

    def prepare_uvs(self, obj, props):
        optcuts = props.optcuts
        if not optcuts.use_hard_surface:
            return props.import_uvs
        only = None
        if optcuts.is_auto:
            only = auto_hard_faces(obj)
            if not only:
                return False
            if len(only) == len(obj.data.polygons):
                only = None
        applied = build_seam_uvs(
            obj,
            HARD_SURFACE_ANGLE,
            weights=seam_restrictions(obj) if props.avoid_seams else None,
            only=only,
        )
        return applied

    def preseed_work(self, obj, props, mirrors=None):
        optcuts = props.optcuts
        if not optcuts.use_hard_surface:
            return None
        return preseed_work(
            obj,
            HARD_SURFACE_ANGLE,
            weights=seam_restrictions(obj) if props.avoid_seams else None,
            auto=optcuts.is_auto,
            mirrors=mirrors,
        )

    def piece_uses_uvs(self, obj, props, has_uvs):
        # auto mode routes per loose part: a piece the preseed skipped has no
        # seams and goes to the engine bare, to be cut from scratch. with
        # import uvs on, organic pieces keep the user's map instead
        if props.optcuts.is_auto and not props.import_uvs:
            has_uvs = has_uvs and bool(seam_flags(obj.data).any())
        if not has_uvs or obj.data.uv_layers.active is None:
            return has_uvs
        # a collapsed flatten goes bare too, the engine can fail re-cutting it
        return not uvs_collapsed(corner_uvs(obj.data))

    def build_args(self, ctx, input_path, props):
        u = PRIORITY_VALUES[props.priority]
        pinned = (input_path.parent / f"{input_path.stem}_fixed").is_file()
        if pinned and props.priority == "FEWER_SEAMS":
            # a pinned repair under the loose bound is a no-op: the broken
            # patch already counts as feasible, so no cut gets added
            u = PRIORITY_VALUES["BALANCED"]
        s = {5: "200", 4: "150", 3: "100", 2: "50", 1: "25"}.get(props.weight_value, "")
        shared_args = f"-u {u} -s {s}"

        return [str(ctx), "-i", str(input_path)] + shared_args.split()

    def describe_failure(self, code):
        return {
            -1: ("Mesh needs cleanup", True),
            101: ("Non Manifold Edges", True),
            102: ("Non Manifold Vertices", True),
            105: ("Invalid Geometry", True),
            107: ("Invalid UV Input", True),
            108: ("Unsupported Mesh Topology", True),
            109: ("Initial Cut Failed", True),
            110: ("Area UVs Too Broken To Pin", True),
            111: ("Island UVs Too Broken To Combine", True),
            113: ("Invalid Coordinates", True),
            114: ("Island UVs Too Broken To Relax", True),
            115: ("Inconsistent Face Orientation", True),
            116: ("Zero Area Faces", True),
            # 90 (the engine's terminate handler) stays unmapped on purpose:
            # the unknown-code path surfaces the fatal line from stderr
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


ENGINE = OptcutsEngine()
