# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
from ..logger import logger


class UVGAMI_OT_clear_logs(bpy.types.Operator):
    bl_idname = "uvgami.clear_logs"
    bl_label = "Clear"
    bl_description = "Delete all info"

    def execute(self, context):
        logger.unwrap_info.clear()
        self.report({"INFO"}, "Cleared info")
        return {"FINISHED"}


class UVGAMI_OT_copy_logs(bpy.types.Operator):
    bl_idname = "uvgami.copy_logs"
    bl_label = "Copy"
    bl_description = "Copy logs to clipboard"

    def execute(self, context):
        context.window_manager.clipboard = "\n".join(logger.get_all())
        self.report({"INFO"}, "Copied to clipboard")
        return {"FINISHED"}
