import platform

import bpy

from ..logger import logger
from ..manager import manager
from ..utils.paths import get_addon_version


class UVGAMI_OT_clear_summary(bpy.types.Operator):
    bl_idname = "uvgami.clear_summary"
    bl_label = "Dismiss"
    bl_description = "Hide the result of the last unwrap"

    def execute(self, context):
        manager.clear_summary()
        return {"FINISHED"}


LOG_TEXT_NAME = "UVgami Log"


def _session_header():
    """What a pasted log needs to say what produced it."""
    return (
        f"UVgami {get_addon_version()} | Blender {bpy.app.version_string}"
        f" | {platform.system()}"
    )


def _text_editor_area():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "TEXT_EDITOR":
                return area
    return None


class UVGAMI_OT_open_logs(bpy.types.Operator):
    bl_idname = "uvgami.open_logs"
    bl_label = "Log"
    bl_description = "Show the info in a text editor, where it can be selected"

    def execute(self, context):
        text = bpy.data.texts.get(LOG_TEXT_NAME) or bpy.data.texts.new(LOG_TEXT_NAME)
        text.clear()
        entries = logger.get_all() or ["No previous unwraps"]
        text.write("\n".join([_session_header(), ""] + entries))
        # the view follows the cursor, which write leaves on the last line
        text.cursor_set(0)

        area = _text_editor_area()
        if area is None:
            if not bpy.ops.wm.window_new.poll():
                self.report({"INFO"}, f"Info written to the text '{LOG_TEXT_NAME}'")
                return {"FINISHED"}
            bpy.ops.wm.window_new()
            # the new window copies the current layout
            areas = context.window_manager.windows[-1].screen.areas
            area = max(areas, key=lambda a: a.width * a.height)
            area.type = "TEXT_EDITOR"
        space = area.spaces.active
        space.text = text
        space.top = 0
        return {"FINISHED"}
