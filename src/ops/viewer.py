import blf
import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader

from ..manager import manager
from ..utils.mesh import check_exists
from ..utils.ui import tag_redraw
from .stop import group_targets, piece_target

VIEWER_WORKSPACE = "UVgami Viewer"

CYCLE_KEYS = {"RIGHT_ARROW": 1, "LEFT_ARROW": -1}

ARROW_FONT_SIZE = 24
DONE_FONT_SIZE = 14
GRID_GAP = 8

# the stretch overlay adds onto this grey, baked in so the fill covers the grid
EDITOR_BACKGROUND = 0.22

WIRE_WIDTH = 1.0
WIRE_OUTLINE_WIDTH = 3.0
# what the core of a blender uv edge measures at
WIRE_COLOR = (0.54, 0.54, 0.54)
# a wire over triangles this small covers the colors
WIRE_HIDDEN_PIXELS = 3.0
WIRE_FULL_PIXELS = 10.0

# workspace name and (object, mode) to restore when the viewer closes
old_workspace = None
old_mode = None

# POST_VIEW in the image editor draws in uv space, so snapshot uvs go
# straight into gpu batches, no viewer mesh or edit mode needed
_handler = None
_text_handler = None
# (typical uv edge, batch) per edge length, so short bevel edges fade alone
_wire_batches = []
_fill_batch = None
_wire_shader = None
_fill_shader = None
# edges within this factor of each other share a wire batch
WIRE_EDGE_BUCKET = 2.0
# PACK_MARGIN = 0.01
# per-corner 3d angles of the engine input, for stretch colors
_corner_angle = None


def _corner_angles(pts):
    """Interior angle at each corner of every triangle, (faces, 3). The cross
    magnitude comes from the lengths and the dot, so 2d and 3d points both
    take the same line."""
    to_next = pts[:, [1, 2, 0]] - pts
    to_prev = pts[:, [2, 0, 1]] - pts
    dot = numpy.einsum("ijk,ijk->ij", to_next, to_prev)
    squared = (to_next * to_next).sum(2) * (to_prev * to_prev).sum(2) - dot * dot
    return numpy.arctan2(numpy.sqrt(numpy.maximum(squared, 0)), dot)


def load_input_mesh(path):
    """Per-corner 3d angles of the input obj. The engine keeps face order, so
    snapshot faces line up with these by index."""
    global _corner_angle
    _corner_angle = None
    if not path.is_file():
        return
    verts = []
    tris = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line[2:].split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                tris.append(
                    [int(token.partition("/")[0]) - 1 for token in line[2:].split()]
                )
    pts = numpy.asarray(verts, dtype=numpy.float32)[
        numpy.asarray(tris, dtype=numpy.int32)
    ]
    _corner_angle = _corner_angles(pts)


def _weight_to_rgb(weight):
    """Blender's weight ramp, dark blue through cyan, green and yellow to red.
    Ported from BKE_defvert_weight_to_rgb, GPL-2.0-or-later, whose four
    branches come to the same curve as these three clamped ramps."""
    rgb = numpy.empty(weight.shape + (3,), dtype=numpy.float32)
    rgb[..., 0] = numpy.clip((weight - 0.5) * 4, 0, 1)
    rgb[..., 1] = numpy.clip(numpy.minimum(weight, 1 - weight) * 4, 0, 1)
    rgb[..., 2] = numpy.clip((0.5 - weight) * 4, 0, 1)
    rgb *= (weight / 2 + 0.5)[..., None]
    return rgb


def _stretch_colors(uv_pts):
    """Per-corner colors from how far each uv corner angle is off its 3d angle,
    the same measurement as the uv editor's angle stretch overlay."""
    off = numpy.abs(_corner_angles(uv_pts) - _corner_angle) / numpy.pi
    weight = 1 - (1 - off) ** 2
    colors = numpy.ones((weight.size, 4), dtype=numpy.float32)
    colors[:, :3] = _weight_to_rgb(weight).reshape(-1, 3) + EDITOR_BACKGROUND
    return numpy.clip(colors, 0, 1, out=colors)


# def _island_of_vertex(edges, vertex_count):
#     """Island index per snapshot vertex, 0 to island count - 1. Min label
#     propagation with pointer jumping, a few numpy rounds even on a long strip."""
#     label = numpy.arange(vertex_count)
#     e0, e1 = edges[:, 0], edges[:, 1]
#     while True:
#         before = label
#         low = numpy.minimum(label[e0], label[e1])
#         label = label.copy()
#         numpy.minimum.at(label, e0, low)
#         numpy.minimum.at(label, e1, low)
#         while True:
#             jumped = label[label]
#             if numpy.array_equal(jumped, label):
#                 break
#             label = jumped
#         if numpy.array_equal(label, before):
#             return numpy.unique(label, return_inverse=True)[1]


# def _pack_islands(co, island, island_count):
#     """Shelf pack the islands by bounding box into the unit square. The engine
#     never packs its layout, so straight from the snapshot the islands are
#     scattered and small."""
#     low = numpy.full((island_count, 2), numpy.inf, dtype=numpy.float32)
#     high = numpy.full((island_count, 2), -numpy.inf, dtype=numpy.float32)
#     numpy.minimum.at(low, island, co)
#     numpy.maximum.at(high, island, co)
#     size = high - low
#     width = float(numpy.sqrt((size[:, 0] * size[:, 1]).sum()))
#     margin = PACK_MARGIN * width
#     offset = numpy.empty_like(low)
#     x = y = margin
#     shelf = extent = 0.0
#     for i in numpy.argsort(-size[:, 1]):
#         if x > margin and x + size[i, 0] > width:
#             x, y, shelf = margin, y + shelf + margin, 0.0
#         offset[i] = (x, y)
#         x += size[i, 0] + margin
#         shelf = max(shelf, size[i, 1])
#         extent = max(extent, x, y + shelf + margin)
#     return (co - low[island] + offset[island]) / extent


def _wire_batches_by_edge_length(co, edges):
    """One (typical edge, batch) per bucket of similar edge lengths."""
    length = numpy.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
    bucket = numpy.floor(
        numpy.log(numpy.maximum(length, 1e-9)) / numpy.log(WIRE_EDGE_BUCKET)
    )
    batches = []
    for key in numpy.unique(bucket):
        chosen = bucket == key
        batch = batch_for_shader(
            _wire_shader, "LINES", {"pos": co[edges[chosen].reshape(-1)]}
        )
        batches.append((float(numpy.median(length[chosen])), batch))
    return batches


def set_snapshot(uv_co, uv_indices):
    """Build the fill and wire batches for the latest engine snapshot."""
    global _wire_batches, _fill_batch, _wire_shader, _fill_shader
    if _wire_shader is None:
        # polyline sets width in the shader, past the driver's line limit of 1
        _wire_shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        _fill_shader = gpu.shader.from_builtin("SMOOTH_COLOR")

    tris = numpy.asarray(uv_indices, dtype=numpy.int32)
    co = numpy.asarray(uv_co, dtype=numpy.float32)

    edges = numpy.concatenate((tris[:, :2], tris[:, 1:], tris[:, ::2]))
    edges.sort(axis=1)
    edges = numpy.unique(edges, axis=0)
    # island = _island_of_vertex(edges, len(co))
    # co = _pack_islands(co, island, island.max() + 1)
    _wire_batches = _wire_batches_by_edge_length(co, edges)

    _fill_batch = None
    if _corner_angle is not None and len(_corner_angle) == len(tris):
        uv_pts = co[tris]
        _fill_batch = batch_for_shader(
            _fill_shader,
            "TRIS",
            {"pos": uv_pts.reshape(-1, 2), "color": _stretch_colors(uv_pts)},
        )


def clear_snapshot():
    global _wire_batches, _fill_batch
    _wire_batches = []
    _fill_batch = None


def _wire_fade(typical_edge):
    """0 to 1 on how many pixels wide the typical triangle draws right now."""
    view2d = bpy.context.region.view2d
    left = view2d.view_to_region(0.0, 0.0, clip=False)[0]
    right = view2d.view_to_region(1.0, 0.0, clip=False)[0]
    pixels = typical_edge * (right - left)
    span = WIRE_FULL_PIXELS - WIRE_HIDDEN_PIXELS
    return min(max((pixels - WIRE_HIDDEN_PIXELS) / span, 0.0), 1.0)


def _draw():
    gpu.state.blend_set("NONE")
    if _fill_batch is not None:
        _fill_batch.draw(_fill_shader)
    gpu.state.blend_set("ALPHA")
    if _wire_batches:
        _wire_shader.bind()
        _wire_shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
        _wire_shader.uniform_bool("lineSmooth", True)
    for typical_edge, batch in _wire_batches:
        # without a fill the wire is the whole picture, so it never fades out
        fade = _wire_fade(typical_edge) if _fill_batch is not None else 1.0
        if fade <= 0:
            continue
        for width, rgb in (
            (WIRE_OUTLINE_WIDTH, (0.0, 0.0, 0.0)),
            (WIRE_WIDTH, WIRE_COLOR),
        ):
            _wire_shader.uniform_float("lineWidth", width)
            _wire_shader.uniform_float("color", (*rgb, fade))
            batch.draw(_wire_shader)
    gpu.state.blend_set("NONE")


def _trim_to_uv(window):
    """Collapse the viewer workspace down to a single uv editor area."""
    screen = window.screen
    while len(screen.areas) > 1:
        smallest = min(screen.areas, key=lambda a: a.width * a.height)
        with bpy.context.temp_override(window=window, area=smallest):
            bpy.ops.screen.area_close()
    area = screen.areas[0]
    area.ui_type = "UV"
    # every selected object draws its uv map under the snapshot
    area.spaces.active.uv_editor.show_uv = False
    region = next(r for r in area.regions if r.type == "WINDOW")
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.image.view_all(fit_view=True)


def _schedule_workspace_trim(window, name):
    """Workspace switches are deferred to the next main loop tick, so the
    layout edit has to wait on a timer until the new workspace is active."""
    attempts = [0]

    def tick():
        workspace = bpy.data.workspaces.get(name)
        if workspace is None or attempts[0] > 20:
            return None
        if window.workspace != workspace:
            attempts[0] += 1
            return 0.01
        _trim_to_uv(window)
        return None

    bpy.app.timers.register(tick, first_interval=0.0)


def _draw_hint_text():
    """Pixel-space companion to _draw: blf works in pixels, so the grid
    corner anchor is converted from uv space each redraw."""
    font = 0
    blf.color(font, 1.0, 1.0, 1.0, 0.9)
    view2d = bpy.context.region.view2d
    if manager.viewer_done:
        right, bottom = view2d.view_to_region(1.0, 0.0, clip=False)
        blf.size(font, DONE_FONT_SIZE)
        blf.position(font, right + GRID_GAP, bottom, 0)
        blf.draw(font, "Done, click to exit")
    if any(u is not manager.current_viewer for u in _viewable_unwraps()):
        _draw_switch_arrows(font, view2d)


def _draw_switch_arrows(font, view2d):
    """Outside the grid's top corners, so they clear the islands."""
    left, top = view2d.view_to_region(0.0, 1.0, clip=False)
    right = view2d.view_to_region(1.0, 1.0, clip=False)[0]
    blf.size(font, ARROW_FONT_SIZE)
    for arrow, edge, side in (("←", left, -1), ("→", right, 1)):
        width, height = blf.dimensions(font, arrow)
        x = edge + side * GRID_GAP - (width if side < 0 else 0)
        blf.position(font, x, top - height, 0)
        blf.draw(font, arrow)


def _viewable_unwraps():
    return [unwrap for unwrap in manager.active if unwrap.is_viewable]


def start_viewer_draw():
    global _handler, _text_handler
    if _handler is None:
        _handler = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW"
        )
        _text_handler = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_hint_text, (), "WINDOW", "POST_PIXEL"
        )


def stop_viewer_draw():
    global _handler, _text_handler, _corner_angle
    if _handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handler, "WINDOW")
        _handler = None
    if _text_handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_text_handler, "WINDOW")
        _text_handler = None
    clear_snapshot()
    _corner_angle = None
    manager.viewer_done = False


class UVGAMI_OT_view_unwrap(bpy.types.Operator):
    bl_idname = "uvgami.view_unwrap"
    bl_label = "View Unwrap"
    bl_description = "Show the unwrap live in the UV editor"

    # a group button sets job_id, a piece button sets stem
    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        if self.job_id:
            candidates = [u for u in group_targets(self.job_id) if u.is_viewable]
        else:
            candidates = piece_target(self.stem)
        if not candidates:
            # the unwrap settled between the panel draw and the click
            return {"CANCELLED"}
        unwrap = candidates[0]
        manager.is_viewer_active = True
        manager.exit_viewer = False
        manager.viewer_done = False

        global old_workspace, old_mode
        window = context.window
        old_workspace = window.workspace.name
        workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
        if workspace is None:
            existing = set(bpy.data.workspaces.keys())
            bpy.ops.workspace.duplicate()
            new_name = next(n for n in bpy.data.workspaces.keys() if n not in existing)
            workspace = bpy.data.workspaces[new_name]
            workspace.name = VIEWER_WORKSPACE
            _schedule_workspace_trim(window, workspace.name)
        else:
            # leftover from a crashed session
            window.workspace = workspace

        # an edit mode mesh would draw its own uv map under the snapshot
        old_mode = None
        active = context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            old_mode = (active, active.mode)
            bpy.ops.object.mode_set(mode="OBJECT")

        start_viewer_draw()
        self._show(unwrap)
        self.report({"INFO"}, "Click to exit viewer")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _show(self, unwrap):
        """Point the viewer at one unwrap. The new map arrives on the next
        manager tick."""
        if manager.current_viewer is not None:
            manager.current_viewer.viewing = False
        manager.viewer_done = False
        clear_snapshot()
        load_input_mesh(unwrap.path)
        manager.current_viewer = unwrap
        unwrap.viewing = True
        tag_redraw(("WINDOW",))

    def _cycle(self, step):
        viewable = _viewable_unwraps()
        if not viewable:
            return
        current = manager.current_viewer
        # the one being viewed drops out of the list once it finishes
        index = viewable.index(current) + step if current in viewable else 0
        target = viewable[index % len(viewable)]
        if target is not current:
            self._show(target)

    def modal(self, context, event):
        if event.type in CYCLE_KEYS and event.value == "PRESS":
            self._cycle(CYCLE_KEYS[event.type])
            return {"RUNNING_MODAL"}

        if event.type in {"LEFTMOUSE", "RIGHTMOUSE", "ESC"} or manager.exit_viewer:
            manager.is_viewer_active = False
            if manager.current_viewer is not None:
                manager.current_viewer.viewing = False
                manager.current_viewer = None
            stop_viewer_draw()

            # both ops are deferred a tick but process in order: the delete
            # removes the active viewer workspace, then the old one activates
            workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
            if workspace is not None and context.window.workspace == workspace:
                bpy.ops.workspace.delete()
            restore = bpy.data.workspaces.get(old_workspace) if old_workspace else None
            if restore is not None:
                context.window.workspace = restore

            if old_mode is not None:
                obj, mode = old_mode
                # a finished unwrap can have hidden or removed the source
                if check_exists(obj) and obj.visible_get() and obj.mode == "OBJECT":
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode=mode)

            return {"FINISHED"}

        # let navigation through so the map can be panned and zoomed
        return {"PASS_THROUGH"}
