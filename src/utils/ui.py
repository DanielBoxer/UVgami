import bpy


def newline_label(label, layout):
    for line in label:
        layout.label(text=line)


def toggle(layout, props, name, text, icon, active=True):
    """Checkbox row, returning a box for the options it reveals, or None."""
    split = layout.split(factor=0.7)
    split.active = active
    split.label(icon=icon, text=text)
    split.prop(props, name)
    if not getattr(props, name):
        return None
    box = layout.box()
    box.active = active
    return box


def only_active(entries):
    """Keep the (icon, label, path, on) entries that are on, trimming them to
    what draw_active takes."""
    return [entry[:3] for entry in entries if entry[3]]


def is_non_default(props, path):
    """Whether a setting differs from the default it was registered with, so
    changing that default can't leave a comparison behind. path may name a
    nested group, like "optcuts.quality". An ENUM_FLAG property keeps its
    default in default_flag instead, so this would always call it changed."""
    group, name = props, path
    if "." in path:
        head, name = path.split(".", 1)
        group = getattr(props, head)
    return getattr(group, name) != group.bl_rna.properties[name].default


# at ui scale 1. the fixed part is the disclosure triangle, the panel name and
# the drag grip
ICON_WIDTH = 20
HEADER_FIXED_WIDTH = 140


def header_icon_limit(context):
    """How many icons fit beside the panel name. A header row can't wrap, so
    anything past this would be cut off at the sidebar edge."""
    scale = context.preferences.system.ui_scale
    free = context.region.width - HEADER_FIXED_WIDTH * scale
    return max(0, int(free / (ICON_WIDTH * scale)))


def draw_active(layout, settings, limit):
    """Icon strip of the settings that will change the run. Hover names the
    setting, click resets it to default. Past the limit the rest become a
    count, so a setting that is on never goes missing."""
    if not settings or limit < 1:
        return
    row = layout.row(align=True)
    row.alignment = "RIGHT"
    hidden = len(settings) - limit
    if hidden > 0:
        # the count takes a slot of its own
        settings = settings[: limit - 1]
    for icon, label, path in settings:
        op = row.operator("uvgami.reset_setting", text="", icon=icon, emboss=False)
        op.label = label
        op.path = path
    if hidden > 0:
        row.label(text=f"+{hidden + 1}")


def tag_redraw(region_types=("WINDOW", "UI"), area_types=("VIEW_3D", "IMAGE_EDITOR")):
    """Repaint the editors that show unwrap progress. Goes through bpy.data
    because the manager calls this from a timer, where the context has no
    window or area. High frequency callers pass WINDOW only: rebuilding the
    sidebar mid click makes Blender drop the click, so the UI region only
    redraws when what it shows changed."""
    for wm in bpy.data.window_managers:
        for window in wm.windows:
            for area in window.screen.areas:
                if area.type in area_types:
                    for region in area.regions:
                        if region.type in region_types:
                            region.tag_redraw()


_status = None


def _draw_status(header, context):
    from bl_ui.space_statusbar import STATUSBAR_HT_header

    # ours first, so it sits at the far left instead of past the right aligned
    # stats, where it's easy to miss
    if _status is not None:
        header.layout.row().label(text=_status[0], icon=_status[1])
    STATUSBAR_HT_header._draw_orig(header, context)


def set_status(text, icon="CHECKMARK"):
    """Add a message to the end of the status bar, None clears it."""
    global _status
    _status = (text, icon) if text else None
    for wm in bpy.data.window_managers:
        for window in wm.windows:
            window.workspace.status_text_set(_draw_status if text else None)


def popup(msg, title, icon):
    # popup_menu segfaults a background blender
    if bpy.app.background:
        return

    def draw(self, context):
        newline_label(msg, self.layout)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def switch_shading(type):
    for area in bpy.context.screen.areas:
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = type
                break
