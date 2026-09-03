import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..utils.geometry import calc_center
from ..utils.ui import tag_redraw
from .stop import group_targets

_sym_handler = None
_sym_shader = None

_AXIS_COLORS = {
    "X": (1.0, 0.23, 0.33),
    "Y": (0.54, 0.83, 0.0),
    "Z": (0.16, 0.56, 0.9),
}


def reset_prop(group, prop):
    group.property_unset(prop)
    if prop in group:
        del group[prop]


def reset_group(group):
    # property_unset is a no-op on pointer props
    for prop in group.__annotations__:
        value = getattr(group, prop)
        if isinstance(value, bpy.types.PropertyGroup):
            reset_group(value)
        else:
            reset_prop(group, prop)


class UVGAMI_OT_expand(bpy.types.Operator):
    bl_idname = "uvgami.expand"
    bl_label = "Expand"
    bl_description = "Expand or collapse joined mesh details"

    job_id: bpy.props.IntProperty()

    def execute(self, context):
        targets = group_targets(self.job_id)
        if targets:
            job = targets[0].join_job
            job.is_expanded = not job.is_expanded
            # the arrow only flips on the next dispatch tick otherwise
            tag_redraw()
        return {"FINISHED"}


class UVGAMI_OT_reset_setting(bpy.types.Operator):
    bl_idname = "uvgami.reset_setting"
    bl_label = "Active Setting"
    bl_options = {"UNDO", "INTERNAL"}

    path: bpy.props.StringProperty()
    label: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        return f"{properties.label}. Click to reset to default"

    def execute(self, context):
        group = context.scene.uvgami
        path = self.path
        if "." in path:
            name, path = path.split(".", 1)
            group = getattr(group, name)
        reset_prop(group, path)
        # unset skips the notifier a normal click sends
        tag_redraw()
        return {"FINISHED"}


class UVGAMI_OT_reset_settings(bpy.types.Operator):
    bl_idname = "uvgami.reset_settings"
    bl_label = "Reset Settings"
    bl_description = "Reset all settings to their default values"
    bl_options = {"UNDO"}

    def execute(self, context):
        reset_group(context.scene.uvgami)
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


def _plane_batches(axis, center, dims):
    # half-extents of the in-plane axes, floored so flat meshes still show
    floor = 0.1 * max(dims)
    he = Vector(max(d, floor) for d in dims)
    if axis == "X":
        u, v = Vector((0, he.y, 0)), Vector((0, 0, he.z))
    elif axis == "Y":
        u, v = Vector((he.x, 0, 0)), Vector((0, 0, he.z))
    else:
        u, v = Vector((he.x, 0, 0)), Vector((0, he.y, 0))
    corners = [center - u - v, center + u - v, center + u + v, center - u + v]
    fill = batch_for_shader(
        _sym_shader, "TRIS", {"pos": corners}, indices=((0, 1, 2), (0, 2, 3))
    )
    wire = batch_for_shader(
        _sym_shader, "LINES", {"pos": corners}, indices=((0, 1), (1, 2), (2, 3), (3, 0))
    )
    return fill, wire


def _draw_sym_planes():
    global _sym_shader
    props = bpy.context.scene.uvgami
    axes = props.sym_axes
    if not (props.use_symmetry and props.sym_preview and axes):
        return
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        return
    if _sym_shader is None:
        _sym_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    # depth tested so the mesh cuts through the plane at the symmetry line
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.line_width_set(2)
    _sym_shader.bind()
    for obj in objects:
        center = calc_center(obj)
        dims = obj.dimensions
        for axis in axes:
            fill, wire = _plane_batches(axis, center, dims)
            r, g, b = _AXIS_COLORS[axis]
            _sym_shader.uniform_float("color", (r, g, b, 0.25))
            fill.draw(_sym_shader)
            _sym_shader.uniform_float("color", (r, g, b, 0.9))
            wire.draw(_sym_shader)
    gpu.state.line_width_set(1)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def start_symmetry_draw():
    global _sym_handler
    if _sym_handler is None:
        _sym_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_sym_planes, (), "WINDOW", "POST_VIEW"
        )


def stop_symmetry_draw():
    global _sym_handler
    if _sym_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sym_handler, "WINDOW")
        _sym_handler = None
