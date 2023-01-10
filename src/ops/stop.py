# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
from ..manager import manager
from ..ui.panels import expand
from ..utils import print_stdin


class UVGAMI_OT_stop(bpy.types.Operator):
    bl_idname = "uvgami.stop"
    bl_label = "Stop"
    bl_description = "Stop UV unwrap"

    start_idx: bpy.props.IntProperty()
    end_idx: bpy.props.IntProperty()

    def execute(self, context):
        for unwrap in manager.active[self.start_idx : self.end_idx]:
            if unwrap.process is not None:
                # send stop command
                if not print_stdin(unwrap.process, "stop"):
                    self.report({"ERROR"}, "Could not stop unwrap")
            else:
                unwrap.is_stopped = True

        self.report({"INFO"}, "UV unwrap stop in progress")
        return {"FINISHED"}


class UVGAMI_OT_cancel(bpy.types.Operator):
    bl_idname = "uvgami.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel UV unwrap"

    start_idx: bpy.props.IntProperty()
    end_idx: bpy.props.IntProperty()
    expand_idx: bpy.props.IntProperty()

    def execute(self, context):
        for unwrap in manager.active[self.start_idx : self.end_idx]:
            for job in unwrap.jobs:
                job.count = job.count - 1
                # if it was the last one
                if job.type == "JOIN" and job.count - len(job.unwrapped) == 0:
                    del expand[self.expand_idx]
                    # this makes it so the popup doesn't show if all cancelled
                    manager.finished_count -= len(job.unwrapped)
                    manager.cancelled_count += len(job.unwrapped)

            unwrap.cancel_unwrap(start_next=True)

        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_all(bpy.types.Operator):
    bl_idname = "uvgami.cancel_all"
    bl_label = "Cancel All"
    bl_description = "Cancel all active UV unwraps"

    def execute(self, context):
        for unwrap in manager.active.copy():
            unwrap.cancel_unwrap()

        manager.finish()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}
