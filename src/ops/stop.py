import bpy

from ..job import Result
from ..manager import manager
from ..objfile import merge_obj_files
from ..utils.io import import_obj
from ..utils.mesh import mark_not_unwrapped
from ..utils.ui import tag_redraw


def group_targets(job_id):
    """Every unwrap still running for one join job. Keyed by job id, not by a
    member's stem: pieces settle and leave manager.active between the draw and
    the click, so the stem the button was drawn with may already be gone."""
    return [
        u
        for u in manager.active
        if u.join_job is not None and u.join_job.job_id == job_id
    ]


def piece_target(stem):
    """Resolved at execute time, so a stale click on a settled piece does
    nothing."""
    return [u for u in manager.active if u.path.stem == stem]


def drop_unwrap(context, unwrap, invalid_label, result):
    if invalid_label is not None and unwrap.path.is_file():
        # the import must happen before record_result, which deletes the input file
        mark_not_unwrapped(import_obj(unwrap.path), invalid_label)
        manager.moved_to_invalid = True
    manager.record_result(unwrap, result)


class UVGAMI_OT_stop(bpy.types.Operator):
    bl_idname = "uvgami.stop"
    bl_label = "Stop"
    bl_description = "Stop UV unwrap"

    # a group button sets job_id, a piece button sets stem
    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        unwraps = group_targets(self.job_id) if self.job_id else piece_target(self.stem)

        # collect cancellations so group members can be merged into one import
        to_cancel = []
        for unwrap in unwraps:
            if unwrap.batch_process is not None:
                # pending batch members are cancelled by deleting their input
                # so the cli skips them, in-flight ones finish normally
                if unwrap.path.stem not in unwrap.batch_process.started:
                    to_cancel.append(unwrap)
            elif unwrap.process is not None:
                if manager.engine.supports_early_stop:
                    # the flag re-sends the request each tick and arms the
                    # manager's STOP_SECONDS force kill
                    unwrap.is_stopped = True
                    delivered = manager.engine.request_early_stop(unwrap.process)
                    # a process that exited since the last tick finishes on its own
                    if not delivered and unwrap.process.poll() is None:
                        self.report({"ERROR"}, "Could not stop unwrap")
                # a running solo mesh on an engine without early stop just
                # finishes normally, like an in-flight batch member
            else:
                # queued: starting a mesh just to stop it gives a map with no
                # work in it, so drop it and let it show as not unwrapped
                to_cancel.append(unwrap)

        self._cancel_collected(context, to_cancel)
        if to_cancel:
            manager.exit_viewer = True
            tag_redraw()

        # without early stop the wait would be a full unwrap
        if manager.engine.supports_early_stop:
            manager.run_until_settled(unwraps)
        return {"FINISHED"}

    def _cancel_collected(self, context, to_cancel):
        # pieces of one separated mesh share a join_job, so merge them into
        # one import instead of adding an object per piece
        groups = {}
        singles = []
        for unwrap in to_cancel:
            if unwrap.join_job is None:
                singles.append(unwrap)
            else:
                groups.setdefault(id(unwrap.join_job), []).append(unwrap)

        for group in groups.values():
            if len(group) < 2:
                singles.extend(group)
                continue
            self._import_merged_group(context, group)
            # import already done above
            for unwrap in group:
                drop_unwrap(context, unwrap, None, Result.STOPPED)

        for unwrap in singles:
            drop_unwrap(context, unwrap, "Stopped", Result.STOPPED)

    def _import_merged_group(self, context, group):
        # merge before any settle: record_result deletes these input files
        paths = [unwrap.path for unwrap in group if unwrap.path.is_file()]
        if not paths:
            return
        merged_obj = import_obj(merge_obj_files(paths))
        mark_not_unwrapped(merged_obj, "Stopped", group[0].input_name)
        manager.moved_to_invalid = True


class UVGAMI_OT_cancel(bpy.types.Operator):
    bl_idname = "uvgami.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel UV unwrap"

    # a group button sets job_id, a piece button sets stem
    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def invoke(self, context, event):
        # only a group cancel can discard finished pieces, so only it confirms
        targets = group_targets(self.job_id) if self.job_id else []
        if targets and targets[0].join_job.finished:
            count = len(targets[0].join_job.finished)
            pieces = "piece" if count == 1 else "pieces"
            return context.window_manager.invoke_confirm(
                self,
                event,
                title="Cancel Unwrap",
                message=f"{count} finished {pieces} will be discarded",
                icon="WARNING",
            )
        return self.execute(context)

    def execute(self, context):
        unwraps = group_targets(self.job_id) if self.job_id else piece_target(self.stem)
        if self.job_id:
            # the user dropped the whole mesh, so the already finished pieces
            # get discarded instead of joined when the group settles
            for unwrap in unwraps:
                if unwrap.join_job is not None:
                    unwrap.join_job.discard = True

        for unwrap in unwraps:
            # an individual cancel from a group goes to the collection, so the
            # joined result visibly misses a piece
            is_individual_from_group = (
                not self.job_id
                and unwrap.join_job is not None
                and unwrap.join_job.expected > 1
            )
            label = "Cancelled (group)" if is_individual_from_group else None
            drop_unwrap(context, unwrap, label, Result.CANCELLED)

        if unwraps:
            manager.exit_viewer = True
            tag_redraw()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_background(bpy.types.Operator):
    bl_idname = "uvgami.cancel_background"
    bl_label = "Cancel"
    bl_description = "Cancel this preseed or proxy finish"

    name: bpy.props.StringProperty()

    def execute(self, context):
        # resolved at execute time, so a stale click is a no-op
        entry = next((p for p in manager.preparing if p.name == self.name), None)
        if entry is not None:
            entry.cancel()
        else:
            transfer = next(
                (t for t in manager.pending_transfers if t.name == self.name), None
            )
            if transfer is None:
                return {"CANCELLED"}
            manager.cancel_transfer(transfer)
        tag_redraw()
        self.report({"INFO"}, "Cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_all(bpy.types.Operator):
    bl_idname = "uvgami.cancel_all"
    bl_label = "Cancel All"
    bl_description = "Cancel all active UV unwraps"

    def invoke(self, context, event):
        jobs = {u.join_job for u in manager.active if u.join_job is not None}
        count = sum(len(job.finished) for job in jobs)
        if count:
            pieces = "piece" if count == 1 else "pieces"
            return context.window_manager.invoke_confirm(
                self,
                event,
                title="Cancel All",
                message=f"{count} finished {pieces} will be discarded",
                icon="WARNING",
            )
        return self.execute(context)

    def execute(self, context):
        manager.cancel_session()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}
