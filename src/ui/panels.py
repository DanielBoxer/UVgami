import bpy

from ..engines import active_engine, get_engine, installed_engines
from ..engines.optcuts import QUALITY_LABELS
from ..engines.install_task import draw_progress, task_state
from ..job import Result
from ..logger import logger
from ..manager import manager
from ..utils.ui import (
    draw_active,
    header_icon_limit,
    is_non_default,
    only_active,
    toggle,
)


def unwrap_settings(props):
    """The settings a full unwrap will apply, as (icon, label, path) entries
    for draw_active. Only ones that change the result, so a sub-setting of
    something already listed is left out."""
    engine = active_engine(props.engine)
    return engine.active_settings(props) + only_active(
        (
            (
                "IMPORT",
                "Import UVs",
                "import_uvs",
                engine.supports_import_uvs
                and not engine.import_uvs_ignored(props)
                and is_non_default(props, "import_uvs"),
            ),
            (
                "MOD_VERTEX_WEIGHT",
                "Weights",
                "use_weights",
                engine.supports_guided and is_non_default(props, "use_weights"),
            ),
            (
                "MOD_DECIM",
                "Proxy",
                "use_proxy",
                engine.supports_proxy and is_non_default(props, "use_proxy"),
            ),
            (
                "LINKED",
                "Stack Similar",
                "stack_similar",
                is_non_default(props, "stack_similar"),
            ),
            # (
            #     "MOD_MIRROR",
            #     "Symmetry",
            #     "use_symmetry",
            #     is_non_default(props, "use_symmetry"),
            # ),
            (
                "TIME",
                "Timeout",
                "unwrap_timeout",
                is_non_default(props, "unwrap_timeout"),
            ),
            # ("MOD_TRIANGULATE", "Preserve Mesh", "untriangulate", props.preserve_mesh),
            (
                "MOD_DATA_TRANSFER",
                "Transfer UVs",
                "transfer_uvs",
                is_non_default(props, "transfer_uvs"),
            ),
        )
    )


def describe_settings(settings):
    """The active settings as one string for the log."""
    return ", ".join(label for _, label, _ in settings)


def fix_settings(props):
    """Same idea for the uv editor operators, which always run optcuts, so
    this is a different set from the main panel's."""
    return only_active(
        (
            (
                "SOLO_OFF",
                QUALITY_LABELS[props.optcuts.quality],
                "optcuts.quality",
                is_non_default(props, "optcuts.quality"),
            ),
            ("MOD_DECIM", "Proxy", "use_proxy", is_non_default(props, "use_proxy")),
            (
                "TIME",
                "Timeout",
                "unwrap_timeout",
                is_non_default(props, "unwrap_timeout"),
            ),
        )
    )


class EnginePanel:
    """Hidden until an engine is installed, so the body can assume one. The
    enum getter clamps to an installed engine, so installed non-empty means
    active_engine can't return None."""

    @classmethod
    def poll(cls, context):
        return bool(installed_engines())


def optcuts_installed():
    return get_engine("OPTCUTS") in installed_engines()


# a wide button with a small icon button beside it
ICON_BUTTON_SPLIT = 0.85


def draw_missing_engine(layout):
    """Stands in for a panel body that has no engine to run."""
    box = layout.box()
    if task_state["running"]:
        draw_progress(box, "Downloading engine")
        return
    row = box.row()
    row.alignment = "CENTER"
    row.label(text="Engine not installed", icon="INFO")
    row = box.row()
    row.scale_y = 1.5
    # skip the confirmation, this is the only way to get an engine
    row.operator_context = "EXEC_DEFAULT"
    row.operator("uvgami.install_optcuts", text="Download Engine", icon="IMPORT")


SUCCESS_ICON = "COLORSET_03_VEC"
FAILED_ICON = "COLORSET_01_VEC"


def draw_summary(layout):
    """Banner with the last session's summary, until dismissed or the next run."""
    if not manager.summary:
        return
    box = layout.box()
    # a split, not a row: a label sizes to its text and leaves the x stranded
    # at the far end of an empty row, a full width one centers the message
    split = box.split(factor=0.9)
    split.operator(
        "uvgami.clear_summary",
        text=manager.summary[0],
        icon=FAILED_ICON if manager.summary_failed else SUCCESS_ICON,
        emboss=False,
    )
    split.operator("uvgami.clear_summary", text="", icon="X", emboss=False)
    notes = manager.summary[1:] or [f"Finished in {logger.get_latest().time:.1f}s"]
    column = box.column(align=True)
    column.active = False
    for note in notes:
        row = column.row()
        row.alignment = "CENTER"
        row.label(text=note)


def draw_queue(box):
    """The running and queued unwraps, with their stop and cancel buttons."""
    active_unwraps = manager.active
    if not active_unwraps and not manager.preparing and not manager.pending_transfers:
        return
    row = box.box().row()
    row.alignment = "CENTER"
    row.label(text="UV unwrap in progress")

    if manager.is_viewer_active:
        viewer_ui = box.box().row()
        viewer_ui.alignment = "CENTER"
        viewer_ui.label(text="Press ESC to exit viewer")

    groups, active_groups = _build_unwrap_groups(active_unwraps)
    rows = len(groups) + len(manager.preparing) + len(manager.pending_transfers)
    if rows > 1:
        box.row().operator("uvgami.cancel_all", icon="CANCEL")

    _draw_unwrap_groups(box, groups, active_groups)

    # objects still preseeding, their pieces don't exist yet
    for entry in manager.preparing:
        _draw_background_row(box, entry.name, entry.name)

    # proxy unwraps whose finish is still running
    for transfer in manager.pending_transfers:
        _draw_background_row(box, f"{transfer.name} (finishing)", transfer.name)

    box.separator()


def _draw_background_row(box, label, name):
    """A preseed or proxy finish, with the cancel that drops it."""
    row = box.box().row()
    row.label(text=label, icon="SORTTIME")
    row.operator("uvgami.cancel_background", text="", icon="CANCEL").name = name


def _build_unwrap_groups(active_unwraps):
    """Unwraps keyed by their join job, or by index when they have none. The
    key type is what tells the two apart when drawing."""
    groups = {}
    active_groups = []
    for unwrap_idx, unwrap in enumerate(active_unwraps):
        job = unwrap.join_job
        if job is None:
            groups[unwrap_idx] = [unwrap]
            continue
        groups.setdefault(job, []).append(unwrap)
        if unwrap.is_running and job not in active_groups:
            active_groups.append(job)
    return groups, active_groups


def _draw_unwrap_groups(box, groups, active_groups):
    for group_id, group in groups.items():
        display_box = box.box()
        row = display_box.row()
        expand_layout = not isinstance(group_id, int)

        if expand_layout:
            row.operator(
                "uvgami.expand",
                text="",
                icon=f"DISCLOSURE_TRI_{'DOWN' if group_id.is_expanded else 'RIGHT'}",
                emboss=False,
            ).job_id = group_id.job_id
            label_text = group[0].input_name
            is_active = group_id in active_groups
        else:
            label_text = group[0].name
            is_active = group[0].is_running
        # on the header too, a collapsed group hides its piece rows
        if any(u.is_stalled for u in group):
            label_text += " (stalled)"
        row.label(
            text=label_text,
            icon=f"RADIOBUT_{'ON' if is_active else 'OFF'}",
        )

        if expand_layout:
            if manager.engine.supports_viewer:
                # opens the viewer on the first piece of the group still running
                _icon_button(
                    row,
                    any(_is_viewable(u) for u in group),
                    "uvgami.view_unwrap",
                    "HIDE_OFF",
                ).job_id = group_id.job_id
            if manager.engine.supports_early_stop:
                _icon_button(
                    row, is_active, "uvgami.stop", "SNAP_FACE"
                ).job_id = group_id.job_id
            row.operator(
                "uvgami.cancel", text="", icon="CANCEL"
            ).job_id = group_id.job_id
            if group_id.is_expanded:
                _draw_group_pieces(display_box, group_id)
        else:
            _draw_piece_buttons(row, group[0])


# a group over this size only lists its running pieces
PIECE_ROW_LIMIT = 10

_RESULT_ICONS = {
    Result.FINISHED: "CHECKMARK",
    Result.INVALID: "ERROR",
    Result.CANCELLED: "X",
    Result.STOPPED: "SNAP_FACE",
}


def _draw_group_pieces(box, job):
    """One row per piece. Settled pieces keep their rows until the whole group
    finishes, so the panel height only changes per group, not per piece."""
    small = job.expected <= PIECE_ROW_LIMIT
    for item in job.members:
        if item.result is not None:
            if small:
                box.row().label(text=item.name, icon=_RESULT_ICONS[item.result])
            continue
        if not small and not item.is_running:
            continue
        row = box.row()
        row.label(
            text=item.name + (" (stalled)" if item.is_stalled else ""),
            icon=f"LAYER_{'ACTIVE' if item.is_running else 'USED'}",
        )
        _draw_piece_buttons(row, item)
    if not small:
        box.row().label(text=f"{job.reported} of {job.expected} done", icon="INFO")


def _is_viewable(item):
    """Only one viewer at a time, cycling inside it reaches the rest."""
    return item.is_viewable and not manager.is_viewer_active


def _icon_button(row, enabled, operator, icon):
    """Greyed out rather than left out while it can't be used, so the row keeps
    its shape instead of growing a button at a time as the queue moves."""
    sub = row.row()
    sub.enabled = enabled
    return sub.operator(operator, text="", icon=icon)


def _draw_piece_buttons(row, item):
    if manager.engine.supports_viewer:
        _icon_button(
            row, _is_viewable(item), "uvgami.view_unwrap", "HIDE_OFF"
        ).stem = item.path.stem
    if manager.engine.supports_early_stop:
        _icon_button(
            row, item.is_active, "uvgami.stop", "SNAP_FACE"
        ).stem = item.path.stem
    cancel_op = row.operator("uvgami.cancel", text="", icon="CANCEL")
    cancel_op.stem = item.path.stem


class UVGAMI_PT_main(bpy.types.Panel):
    # blank so draw_header can put the name before the icons: blender draws
    # bl_label after the header content
    bl_label = ""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw_header(self, context):
        props = context.scene.uvgami
        self.layout.label(text="UVgami")
        if active_engine(props.engine) is None:
            return
        draw_active(self.layout, unwrap_settings(props), header_icon_limit(context))

    def draw(self, context):
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        if engine is None:
            draw_missing_engine(self.layout)
            return

        box = self.layout.box()

        row = box.row()
        row.scale_y = 2
        row.enabled = not manager.is_active
        row.operator("uvgami.start", icon="UV")

        if not manager.in_uv_editor:
            draw_summary(box)
            draw_queue(box)

        row = box.row()
        row.label(icon="TOOL_SETTINGS", text="Engine")
        row.prop(props, "engine", text="")

        engine.draw_update_notice(box)
        engine.draw_settings(box, props)

        if engine.supports_import_uvs:
            split = box.split(factor=0.7)
            split.active = not engine.import_uvs_ignored(props)
            split.label(icon="IMPORT", text="Import UVs")
            split.prop(props, "import_uvs")

        split = box.split(factor=0.7)
        split.label(icon="MOD_DATA_TRANSFER", text="Transfer UVs")
        split.prop(props, "transfer_uvs")


def draw_concurrent(layout, props, engine):
    # hidden instead of grayed out: ai mode batches all meshes into one
    # process, so concurrency doesn't apply
    if engine.batches_queue(props):
        return
    split = layout.split(factor=0.7)
    split.label(icon="MOD_ARRAY", text="Concurrent")
    split.prop(props, "max_cores", slider=True)


def draw_proxy(layout, props):
    """Shared with the uv editor settings."""
    sub = toggle(layout, props, "use_proxy", "Proxy", "MOD_DECIM")
    if sub is not None:
        row = sub.row()
        row.label(text="Proxy Faces", icon="MESH_DATA")
        row.prop(props, "proxy_faces")


def draw_timeout(layout, props):
    """Shared with the uv editor settings."""
    row = layout.row()
    row.label(text="Timeout", icon="TIME")
    row.prop(props, "unwrap_timeout")


class UVGAMI_PT_speed(EnginePanel, bpy.types.Panel):
    bl_label = "Speed"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        box = self.layout.box()
        props = context.scene.uvgami

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Speed", icon="SORTTIME")

        engine = active_engine(props.engine)
        draw_concurrent(box, props, engine)

        if engine.supports_proxy:
            draw_proxy(box, props)

        split = box.split(factor=0.7)
        # the proxy finish packs the whole mesh fresh
        split.active = not engine.uses_proxy(props)
        split.label(icon="LINKED", text="Stack Similar")
        split.prop(props, "stack_similar")

        draw_timeout(box, props)


class UVGAMI_PT_weights(bpy.types.Panel):
    bl_label = "Weights"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    @classmethod
    def poll(cls, context):
        engine = active_engine(context.scene.uvgami.engine)
        return engine is not None and engine.supports_guided

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_weights")

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        # the proxy flatten takes no stretch file
        engine = active_engine(props.engine)
        stretch_ignored = engine.uses_proxy(props) and props.reduce_stretching
        # active, not enabled: painting turns the checkbox on itself, so the
        # buttons have to stay clickable while the panel reads as off
        layout.active = props.use_weights and not stretch_ignored
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Weights", icon="MOD_VERTEX_WEIGHT")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.draw_guides", icon="GREASEPENCIL")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.clear_draw", icon="TRASH")
        row.operator("uvgami.exit_draw", icon="PANEL_CLOSE")

        row = box.row()
        # the engine only reads the strength for seam avoidance (-s), stretch
        # mode runs at a fixed face weight
        row.active = props.avoid_seams
        row.label(text="Strength", icon="MOD_VERTEX_WEIGHT")
        row.prop(props, "weight_value", slider=True)

        box.row().prop(props, "weight_mode", expand=True)

        generate = box.box()
        row = generate.row()
        row.label(text="Generate", icon="SHADERFX")
        row = generate.row()
        row.operator("uvgami.seed_restrictions", text="From View").mode = "VIEW"
        row.operator("uvgami.seed_restrictions", text="Crevices").mode = "CREVICES"
        row.operator("uvgami.seed_restrictions", text="Both").mode = "BOTH"


class UVGAMI_PT_symmetry(EnginePanel, bpy.types.Panel):
    bl_label = "Symmetry"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_symmetry")

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        layout.active = props.use_symmetry
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Symmetry", icon="MOD_MIRROR")

        row = box.row()
        row.scale_y = 1.5
        row.prop(props, "sym_axes")

        row = box.row()
        row.prop(props, "sym_preview", icon="EMPTY_AXIS")
        row.prop(props, "sym_merge")

        # only the hard surface preseed can mirror seams on the whole mesh
        engine = active_engine(props.engine)
        if engine is not get_engine("OPTCUTS") or not props.optcuts.use_hard_surface:
            row = box.row()
            row.label(text="Mesh will be cut and mirrored", icon="ERROR")


class UVGAMI_PT_grid(EnginePanel, bpy.types.Panel):
    bl_label = "Grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Grid", icon="TEXTURE")

        split = box.split(factor=ICON_BUTTON_SPLIT)
        split.scale_y = 1.5
        split.operator("uvgami.add_grid", icon="UV_DATA")
        split.operator("uvgami.remove_grid", icon="TRASH", text="")

        row = box.row()
        row.prop(props, "grid_type", expand=True)
        box.prop(props, "grid_res")
        box.prop(props, "auto_grid")


class UVGAMI_PT_island_uv(bpy.types.Panel):
    bl_label = ""
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw_header(self, context):
        self.layout.label(text="UVgami")
        if not optcuts_installed():
            return
        draw_active(
            self.layout,
            fix_settings(context.scene.uvgami),
            header_icon_limit(context),
        )

    def draw(self, context):
        props = context.scene.uvgami
        if not optcuts_installed():
            draw_missing_engine(self.layout)
            return

        box = self.layout.box()
        get_engine("OPTCUTS").draw_update_notice(box)

        island = box.box()
        row = island.row()
        row.alignment = "CENTER"
        row.label(text="Island Operators", icon="GROUP_UVS")
        col = island.column()
        col.scale_y = 1.5
        col.operator("uvgami.unwrap_island", icon="UV")
        col.operator("uvgami.relax_island", icon="UV_VERTEXSEL")
        col.operator("uvgami.combine_islands", icon="UV_ISLANDSEL")

        # expand only feeds the area operators, so it's grouped with them
        area = box.box()
        row = area.row()
        row.alignment = "CENTER"
        row.label(text="Area Operators", icon="FACESEL")
        col = area.column()
        col.scale_y = 1.5
        col.operator("uvgami.unwrap_area", icon="UV_FACESEL")
        col.operator("uvgami.relax_area", icon="UV_VERTEXSEL")
        row = area.row()
        row.label(icon="PROP_ON", text="Expand Area")
        row.prop(props, "area_expand", text="")

        if manager.in_uv_editor:
            draw_summary(box)
            draw_queue(box)


class UVGAMI_PT_island_settings(bpy.types.Panel):
    bl_label = "Settings"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_island_uv"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return optcuts_installed()

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        row.label(icon="SOLO_OFF", text="Priority")
        row.prop(props.optcuts, "quality", text="")

        # these operators always run optcuts, whatever the main panel is set to
        engine = get_engine("OPTCUTS")
        draw_concurrent(box, props, engine)
        draw_proxy(box, props)
        draw_timeout(box, props)


class UVGAMI_PT_pack(EnginePanel, bpy.types.Panel):
    bl_label = "Pack"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

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
        row.prop(props, "margin")

        box.prop(props, "combine_uvs")
        box.prop(props, "fix_scale")

        box.prop(props, "pack_after_unwrap")


class UVGAMI_PT_misc(EnginePanel, bpy.types.Panel):
    bl_label = "Misc"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    def draw(self, context):
        box = self.layout.box()

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.open_preferences", text="Preferences", icon="PREFERENCES")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.open_logs", icon="TEXT")
