# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy

from ..job import Join
from ..manager import manager
from ..utils.io import import_obj, print_stdin
from ..utils.mesh import check_collection, move_to_collection
from ..utils.paths import get_preferences


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

    def execute(self, context):
        unwraps = manager.active[self.start_idx : self.end_idx]
        cancel_count = len(unwraps)

        for unwrap in unwraps:
            # individual cancel from a group: move to invalid collection
            is_individual_from_group = (
                cancel_count == 1
                and unwrap.join_job is not None
                and unwrap.join_job.count > 1
            )
            if is_individual_from_group and get_preferences().invalid_collection:
                if unwrap.path.is_file():
                    invalid_obj = import_obj(unwrap.path)
                    collection = check_collection(
                        "UVgami Invalid Input", context.scene.collection
                    )
                    move_to_collection(invalid_obj, collection)
                    invalid_name = f"{invalid_obj.name}: Cancelled (group)"
                    invalid_obj.name = invalid_name
                    invalid_obj.hide_set(True)
                    manager.found_invalid_objects = True

            for job in unwrap.jobs:
                job.count = job.count - 1
                # if it was the last one
                if isinstance(job, Join) and job.count - len(job.unwrapped) == 0:
                    # this makes it so the popup doesn't show if all cancelled
                    manager.finished_count -= len(job.unwrapped)
                    manager.cancelled_count += len(job.unwrapped)

            manager.cancel_unwrap(unwrap)

        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_all(bpy.types.Operator):
    bl_idname = "uvgami.cancel_all"
    bl_label = "Cancel All"
    bl_description = "Cancel all active UV unwraps"

    def execute(self, context):
        manager.stop_all()
        manager.finish()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}
