import functools
import time
import traceback
from collections import deque, namedtuple

import bpy
import numpy

from .batch import BatchProcess, last_meaningful_line
from .job import Join, ProxyUVs, Result, TransferReport
from .logger import logger
from .ops.grid import add_grid, make_grid_img, make_grid_mat
from .ops.uv import pack, show_seams
from .progress_bar import progress_bar
from .reroute_seams import reroute_seams
from .uv_transfer import AMBIGUOUS_GEOMETRY
from .similar import write_twin_output
from .utils.geometry import set_origin
from .utils.io import import_obj
from .utils.mesh import (
    add_shading_modifiers,
    check_collection,
    check_exists,
    edit_restore,
    move_to_collection,
)
from .utils.paths import clear_io_dir, get_io_dir_paths, get_preferences
from .utils.ui import popup, set_status, switch_shading, tag_redraw

# how long a clean run's status bar message stays up
STATUS_SECONDS = 5
SETTLE_TICK_SECONDS = 0.05
# a sidebar rebuild mid click drops the click
PANEL_REDRAW_SECONDS = 1.0


class Settings:
    """A frozen copy of scene.uvgami taken at session start. Everything the
    session does reads this, so a slider moved mid-run changes nothing."""

    def __init__(self, props):
        for prop in props.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            value = getattr(props, prop.identifier)
            if prop.type == "POINTER":
                value = Settings(value)
            elif isinstance(value, set):
                value = set(value)
            setattr(self, prop.identifier, value)
        # bl_rna skips the plain python properties like preserve_mesh
        for cls in type(props).__mro__:
            for name, attribute in vars(cls).items():
                if isinstance(attribute, property) and not hasattr(self, name):
                    setattr(self, name, getattr(props, name))


PendingTransfer = namedtuple(
    "PendingTransfer", ["job", "output", "pack_index", "name", "started_at"]
)
# an object being preseeded, before its pieces exist
Preparing = namedtuple("Preparing", ["name", "cancel"])
# how long to wait for the engine to stop before killing it
STOP_SECONDS = 180


class UnwrapManager:
    def __init__(self):
        self._queue = deque()
        self._running = []
        self._pack_output_objects = []
        self.input = {}
        self.engine = None
        # run context returned by engine.validate, opaque to the manager
        self.engine_ctx = None
        self.is_active = False
        # the session was started from the uv editor, so its queue ui goes there
        self.in_uv_editor = False
        self._dispatch_handle = None
        # one per Unwrap still feeding pieces in, blocks the session finishing
        self.pieces_still_arriving = 0
        # _reset_session must not clear this, the builders add and remove entries
        self.preparing = []
        # session progress as (done, running, remaining) fractions
        self.progress = numpy.zeros(3)
        # last session's summary, shown as a banner until dismissed
        self.summary = []
        self.summary_failed = False
        self._reset_session()

    def _reset_session(self):
        """Per-run state. Called from __init__ too, so the error path can reach
        finish() before a session ever starts."""
        # frozen scene.uvgami, set in start()
        self.props = None
        self.moved_to_invalid = False
        # (unwrap, result) per settled piece, the source for all counts
        self.results = []
        self._to_import = []
        # one per proxy finish still running in its worker thread
        self.pending_transfers = []
        self.transfer_uv_failed = False
        self.transfer_uv_fail_detail = ""
        self.transfer_uv_reason_known = False
        self.transfer_uv_split_count = 0
        self.error_code = 0
        self.error_stderr = ""
        self.error_messages = []
        self.current_viewer = None
        self.is_viewer_active = False
        self.exit_viewer = False
        self.viewer_done = False
        self._pack_output_objects = []
        self._drawn_panel_state = None
        self._panel_drawn_at = 0.0

    @property
    def active(self):
        """All unwraps (running and queued)"""
        return self._running + list(self._queue)

    def add(self, unwrap):
        self._queue.append(unwrap)

    def remove_unwrap(self, unwrap):
        if unwrap in self._running:
            self._running.remove(unwrap)
        elif unwrap in self._queue:
            self._queue.remove(unwrap)

    def start(self, uv_editor=False):
        self._reset_session()
        self.props = Settings(bpy.context.scene.uvgami)
        self._fill_slots()
        if get_preferences().show_progress_bar:
            progress_bar.start(uv_editor)
        self.in_uv_editor = uv_editor
        self.is_active = True
        self.clear_summary()
        self._dispatch_handle = functools.partial(self._dispatch)
        bpy.app.timers.register(self._dispatch_handle)

    def _fill_slots(self):
        """Start queued unwraps up to the concurrency limit."""
        props = self.props
        engine = self.engine
        if engine.batches_queue(props):
            if self.pieces_still_arriving > 0:
                # a batch needs the whole queue at once, so wait for every exporter
                return
            if any(u.batch_process is not None for u in self._running):
                return
            if len(self._queue) > 1:
                self._start_batch_process(engine, props)
                return
            # a single queued mesh runs the normal solo path
        max_concurrent = 1 if engine.batches_queue(props) else props.max_cores
        # pieces export in queue order, so an unexported piece means everything
        # behind it is unexported too. copy twins never run, their representative
        # settles them
        for unwrap in list(self._queue):
            if len(self._running) >= max_concurrent:
                break
            if not unwrap.is_exported:
                break
            if unwrap.copy_of is not None:
                continue
            self._queue.remove(unwrap)
            unwrap.start_unwrap()
            self._running.append(unwrap)

    def _start_batch_process(self, engine, props):
        """Unwrap every queued mesh in one engine process."""
        unwraps = [u for u in self._queue if u.copy_of is None]
        self._queue = deque(u for u in self._queue if u.copy_of is not None)
        args = engine.build_batch_args(
            self.engine_ctx, [u.path for u in unwraps], props
        )
        batch_process = BatchProcess(
            args,
            engine.build_env(self.engine_ctx),
            {unwrap.path.stem: unwrap for unwrap in unwraps},
        )
        for unwrap in unwraps:
            unwrap.join_batch(batch_process)
            self._running.append(unwrap)

    def _dispatch(self):
        if not self.is_active:
            return None

        try:
            completed = []
            failed = []
            requeued = []

            for unwrap in list(self._running):
                unwrap.update_progress()

                if unwrap.viewing:
                    unwrap.update_viewer()

                if unwrap.is_stopped:
                    self.engine.request_early_stop(unwrap.process)
                    if unwrap.stop_requested_at is None:
                        unwrap.stop_requested_at = time.monotonic()
                    elif time.monotonic() - unwrap.stop_requested_at > STOP_SECONDS:
                        unwrap.stop_process()
                        failed.append((unwrap, -3))
                        # already failed this tick, the poll below must not re-add it
                        continue

                timeout_minutes = bpy.context.scene.uvgami.unwrap_timeout
                if (
                    timeout_minutes > 0
                    and not unwrap.is_stopped
                    and unwrap.started_at is not None
                    and time.monotonic() - unwrap.started_at > timeout_minutes * 60
                ):
                    can_stop = (
                        self.engine.supports_early_stop and unwrap.batch_process is None
                    )
                    if can_stop:
                        # the stop flow above requests the map and force kills if ignored
                        unwrap.is_stopped = True
                        self.error_messages.append(
                            f"{unwrap.input_name}: timed out,"
                            " stopped with a partial result"
                        )
                    else:
                        unwrap.stop_process()
                        failed.append((unwrap, -2))
                        # already failed this tick, the poll below must not re-add it
                        continue

                ret_code = unwrap.poll_engine()
                if ret_code is not None:
                    if ret_code == 0 and unwrap.output_path.is_file():
                        completed.append(unwrap)
                    elif ret_code == 0:
                        # exit 0 with no output file, the unwrap would
                        # otherwise stay running forever
                        failed.append((unwrap, -4))
                    else:
                        # a batched mesh that never started goes back to the queue
                        # rather than take the dead process's exit code
                        stem = unwrap.path.stem
                        if (
                            unwrap.batch_process is not None
                            and unwrap.batch_process.should_retry(stem)
                            and unwrap.path.is_file()
                        ):
                            requeued.append(unwrap)
                        else:
                            failed.append((unwrap, ret_code))

            logger.update_time()
            self._update_progress_bar()

            for unwrap in completed:
                self._settle_step(unwrap, "finishing", self._process_completion)

            for unwrap, ret_code in failed:
                self._settle_step(
                    unwrap, "handling the failure of", self._handle_failure, ret_code
                )

            self._drain_imports()
            self._finish_transfers()

            for unwrap in requeued:
                if unwrap in self._running:
                    self._running.remove(unwrap)
                unwrap.leave_batch()
                self._queue.append(unwrap)
            if requeued:
                print(f"UVgami: requeued {len(requeued)} mesh(es) after batch ended")

            self._fill_slots()

            # an empty queue is not the end, an exporter or a proxy finish
            # may still be running
            if (
                not self._running
                and not self._queue
                and self.pieces_still_arriving == 0
                and not self.pending_transfers
            ):
                self._finish_batch()
                return None

        except Exception as e:
            from .handler import handle_error

            handle_error(e, "MIDDLE")
            return None

        return 0.1

    def run_until_settled(self, unwraps):
        while self.is_active and any(u.result is None for u in unwraps):
            if self._dispatch() is None:
                return
            time.sleep(SETTLE_TICK_SECONDS)

    def _update_progress_bar(self):
        # unexported pieces sit at (0, 0, 1) until they start reporting
        progress = [numpy.array(unwrap.progress) for unwrap in self.active]
        progress += [numpy.array((1, 0, 0))] * len(self.results)
        # a running proxy finish holds the bar below done until it applies
        progress += [
            numpy.array((t.job.progress, 0, 1 - t.job.progress))
            for t in self.pending_transfers
        ]
        if not progress:
            return
        self.progress = sum(progress) / len(progress)
        if get_preferences().show_progress_bar:
            progress_bar.update(self.progress)
            tag_redraw(("WINDOW",))
        state = self._panel_state()
        now = time.monotonic()
        if (
            state != self._drawn_panel_state
            and now - self._panel_drawn_at >= PANEL_REDRAW_SECONDS
        ):
            self._drawn_panel_state = state
            self._panel_drawn_at = now
            tag_redraw(("UI",))

    def _panel_state(self):
        """Everything the queue ui draws, as a comparable value."""
        return (
            len(self.results),
            self.is_viewer_active,
            tuple(p.name for p in self.preparing),
            tuple(t.name for t in self.pending_transfers),
            tuple(
                (
                    unwrap.path.stem,
                    unwrap.is_running,
                    unwrap.is_exported,
                    unwrap.is_viewable,
                    unwrap.is_stalled,
                )
                for unwrap in self.active
            ),
        )

    def record_result(self, unwrap, result):
        """Every piece ends here as finished, invalid, or cancelled."""
        if unwrap.result is not None:
            return
        unwrap.result = result
        self.results.append((unwrap, result))

        self.remove_unwrap(unwrap)
        unwrap.release_engine()
        unwrap.cleanup()

        group = unwrap.join_job
        if group is None:
            if result is Result.FINISHED:
                self._to_import.append(unwrap)
        else:
            group.record(unwrap, result)
            if group.is_settled() and group.finished and not group.discard:
                self._to_import.append(group)

        for twin in unwrap.twins:
            self._settle_twin(twin, result)

    def _settle_twin(self, twin, result):
        """A copy twin ends however its representative ended. One that hasn't
        exported yet can't finish (no metadata), the exporter settles it."""
        if twin.result is not None:
            return
        if result is Result.FINISHED:
            if not twin.is_exported:
                return
            write_twin_output(
                twin.copy_of.output_path, twin.output_path, twin.copy_matrix
            )
        self.record_result(twin, result)

    def _drain_imports(self):
        """Every result imports here on the timer."""
        for item in self._to_import:
            is_group = isinstance(item, Join)
            unwrap = item.finished[-1] if is_group else item
            try:
                if is_group and len(item.finished) > 1:
                    path, edge_path, added_edges = item.finish()
                else:
                    path = unwrap.output_path
                    edge_path, added_edges = unwrap.edge_path, []
                self._import_and_finalize(unwrap, path, edge_path, added_edges)
                # checkpoint each result, so a mid-session ctrl z can't
                # discard it. the last push is in finish()
                if self._running or self._queue or self.pieces_still_arriving:
                    bpy.ops.ed.undo_push(message="UVgami Unwrap")
            except Exception:
                error_list = traceback.format_exc().split("\n")[:-1]
                logger.add_data("errors", "Error finishing unwrap:")
                for line in error_list:
                    logger.add_data("errors", line)
                    print(line)
                self.error_messages.append(
                    f"Error finishing {unwrap.input_name}, see the console"
                )
        self._to_import.clear()

    def _settle_step(self, unwrap, doing, step, *args):
        """Run one piece's settle step. It settles either way, an exception
        here would leave its group waiting forever."""
        try:
            step(unwrap, *args)
        except Exception:
            message = f"Error {doing} {unwrap.input_name}"
            logger.add_data("errors", f"{message}:")
            for line in traceback.format_exc().split("\n")[:-1]:
                logger.add_data("errors", line)
                print(line)
            self.error_messages.append(f"{message}, see the console")
            self.record_result(unwrap, Result.INVALID)

    def _process_completion(self, unwrap):
        if unwrap.viewing:
            self.viewer_done = True
            tag_redraw()
        self.record_result(unwrap, Result.FINISHED)

    def _import_and_finalize(self, unwrap, path, edge_path, added_edges):
        props = self.props

        if unwrap.preserve_job is not None and unwrap.maintain_mode == "FULL":
            reroute_seams(path, edge_path)

        output = import_obj(path, f"{unwrap.input_name}_unwrapped")

        set_origin(output, unwrap.origin)

        for m_name in unwrap.materials:
            output.data.materials.append(
                None if m_name is None else bpy.data.materials.get(m_name)
            )

        # before the preserve job, so untriangulated faces carry these along
        self._restore_face_data(unwrap, output, "material_index", "material_indices")
        self._restore_face_data(unwrap, output, "use_smooth", "face_smooth")
        add_shading_modifiers(output, unwrap.shading_modifiers)

        if unwrap.preserve_job is not None:
            unwrap.preserve_job.finish(unwrap, output, added_edges)

        if unwrap.hide_job is not None:
            unwrap.hide_job.finish(self.input[unwrap.hide_job])

        symmetrize_job = unwrap.symmetrize_job
        half_rebuilt = False
        if symmetrize_job is not None:
            if not symmetrize_job.kept_whole:
                symmetrize_job.finish(output)
            elif symmetrize_job.whole is not None:
                output = symmetrize_job.rebuild(output, unwrap.origin)
                half_rebuilt = True

        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]
        if any(u.preseeded for u in pieces):
            # imported here: ops.island imports this module back
            from .ops.island import finish_preseed, rectify_islands

            ranges = None
            if len(pieces) > 1:
                ranges = []
                start = 0
                for u in pieces:
                    # material_indices is the piece's own face count
                    stop = start + len(u.material_indices)
                    if u.preseeded:
                        ranges.append((start, stop))
                    start = stop
                if start != len(output.data.polygons):
                    ranges = None
            # rebuild sliced the strips already, mirrored, and a second
            # pass in uv space would land differently on each side
            if not half_rebuilt:
                finish_preseed(output, ranges)
            rectify_islands(output)

        # after rectify, whose per-island solves would drift a stack apart
        kept_whole = symmetrize_job is not None and symmetrize_job.kept_whole
        if kept_whole and symmetrize_job.overlap:
            symmetrize_job.snap_overlap(output)

        if props.pack_after_unwrap:
            self._pack_output_objects.append(output)

        edit_restore([output], show_seams)

        # the whole copy kept its own groups, and the piece's indices point
        # at the half's vertices
        if not half_rebuilt:
            self._restore_vertex_groups(unwrap, output)

        logger.add_data("objects", unwrap.input_name)

        if unwrap.transfer_uvs_job is not None:
            job = unwrap.transfer_uvs_job
            # locate output in the pack list before the transfer deletes output
            pack_index = None
            if props.pack_after_unwrap:
                for i, obj in enumerate(self._pack_output_objects):
                    if obj == output:
                        pack_index = i
                        break
            # the proxy finish flattens the whole original, so it can't skip parts
            group = unwrap.join_job
            missing_pieces = group is not None and len(group.finished) < group.expected
            if missing_pieces and isinstance(job, ProxyUVs):
                failed = group.expected - len(group.finished)
                self.transfer_uv_reason_known = True
                report = TransferReport(
                    False, 0, f"{failed} of {group.expected} parts were not unwrapped"
                )
                self._settle_transfer(job, output, pack_index, report)
                return
            if isinstance(job, ProxyUVs):
                report = job.start(self.input[job], output)
                if report is None:
                    # _finish_transfers settles it when the process exits
                    self.pending_transfers.append(
                        PendingTransfer(
                            job,
                            output,
                            pack_index,
                            unwrap.input_name,
                            time.monotonic(),
                        )
                    )
                    # unlinked while it waits, or it sits in the scene
                    # beside the original
                    for collection in output.users_collection:
                        collection.objects.unlink(output)
                    return
            else:
                report = job.finish(self.input[job], output)
            self._settle_transfer(job, output, pack_index, report)
            return

        self._add_auto_grid(props, output)

        collection = check_collection("UVgami Unwrapped", bpy.context.scene.collection)
        move_to_collection(output, collection)

    def _settle_transfer(self, job, output, pack_index, report):
        """Everything after a transfer's report: pack list, hide state, grid,
        collection. Shared with the proxy finishes, which report later."""
        props = self.props
        input_mesh = self.input[job]
        if report.applied:
            self.transfer_uv_split_count += report.split_count
            if pack_index is not None and job.repack_input:
                self._pack_output_objects[pack_index] = input_mesh
            replacement = getattr(job, "replacement", None)
            if replacement is None:
                self._add_auto_grid(props, input_mesh)
                return
            # proxy with transfer off: a duplicate of the original replaces
            # the deleted output
            if pack_index is not None:
                self._pack_output_objects[pack_index] = replacement
            output = replacement
        else:
            # no hide job exists when a transfer job holds the slot, so
            # without this the original sits on top of the kept output
            if check_exists(input_mesh):
                input_mesh.hide_set(True)
            self.transfer_uv_failed = True
            self.transfer_uv_fail_detail = report.detail
            if report.reason == AMBIGUOUS_GEOMETRY:
                self.transfer_uv_reason_known = True
            logger.add_data(
                "errors",
                f"UV transfer failed ({report.detail}), keeping output",
            )
            if not check_exists(output):
                return

        self._add_auto_grid(props, output)

        collection = check_collection("UVgami Unwrapped", bpy.context.scene.collection)
        move_to_collection(output, collection)

    def _finish_transfers(self):
        """Apply proxy finishes whose worker thread is done."""
        timeout_minutes = bpy.context.scene.uvgami.unwrap_timeout
        for entry in list(self.pending_transfers):
            job, output, pack_index = entry.job, entry.output, entry.pack_index
            if (
                timeout_minutes > 0
                and time.monotonic() - entry.started_at > timeout_minutes * 60
            ):
                # cancel deletes the two objects the settle path works on
                job.cancel()
                self.pending_transfers.remove(entry)
                self.transfer_uv_failed = True
                self.transfer_uv_reason_known = True
                self.transfer_uv_fail_detail = f"{entry.name} timed out"
                continue
            try:
                report = job.poll()
            except Exception:
                error_list = traceback.format_exc().split("\n")[:-1]
                logger.add_data("errors", "Error finishing UV transfer:")
                for line in error_list:
                    logger.add_data("errors", line)
                    print(line)
                report = TransferReport(False, 0, "see the console")
            if report is None:
                continue
            self.pending_transfers.remove(entry)
            try:
                self._settle_transfer(job, output, pack_index, report)
            except Exception:
                error_list = traceback.format_exc().split("\n")[:-1]
                logger.add_data("errors", "Error finishing UV transfer:")
                for line in error_list:
                    logger.add_data("errors", line)
                    print(line)

    def _add_auto_grid(self, props, obj):
        """Grid goes on whichever object survives the run, which is the input
        mesh once a transfer has deleted the output."""
        if props.auto_grid:
            add_grid(obj, make_grid_mat(make_grid_img()))

    def _restore_face_data(self, unwrap, output, attribute, field):
        """Put a per-face list captured at export back on the output, in piece
        order when the mesh was separated."""
        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]
        values = []
        for piece in pieces:
            values.extend(getattr(piece, field))
        if len(values) == len(output.data.polygons):
            output.data.polygons.foreach_set(attribute, values)

    def _restore_vertex_groups(self, unwrap, output):
        if unwrap.join_job is not None and len(unwrap.join_job.finished) > 1:
            combined_groups = {}
            v_offset = 0
            for u in unwrap.join_job.finished:
                for group_name, weights in u.vertex_groups.items():
                    if group_name not in combined_groups:
                        combined_groups[group_name] = {}
                    for v_idx, weight in weights.items():
                        combined_groups[group_name][v_idx + v_offset] = weight
                v_offset += u.vertex_count
            groups_data = combined_groups
        else:
            groups_data = unwrap.vertex_groups

        for group_name, weights in groups_data.items():
            new_group = output.vertex_groups.new(name=group_name)
            for v_idx, weight in weights.items():
                if v_idx < len(output.data.vertices):
                    new_group.add([v_idx], weight, "REPLACE")

    def _handle_failure(self, unwrap, ret_code):
        msg = ""

        # windows reports exit codes unsigned
        if ret_code >= 2**31:
            ret_code -= 2**32

        move_to_invalid = False
        # manager-synthetic codes for timeout, force kill and missing output
        if ret_code == -2:
            elapsed = (time.monotonic() - unwrap.started_at) / 60
            msg = f"Timed out after {elapsed:.1f} minutes"
            move_to_invalid = True
        elif ret_code == -3:
            msg = "Stop timed out (force killed)"
            move_to_invalid = True
        elif ret_code == -4:
            msg = "Engine produced no output"
            move_to_invalid = True
        else:
            described = self.engine.describe_failure(ret_code)
            if described is not None:
                msg, move_to_invalid = described
                if not move_to_invalid:
                    # one code can cover several causes, stderr says which
                    last = last_meaningful_line(unwrap.get_stderr_tail())
                    self.error_messages.append(f"{msg} ({last})" if last else msg)
            else:
                msg = f"Unknown Engine Error ({ret_code})"
                move_to_invalid = True
                self.error_code = ret_code
                tail = unwrap.get_stderr_tail()
                last = last_meaningful_line(tail)
                if last:
                    self.error_stderr = last
                if tail:
                    # the whole traceback goes to the console
                    print(f"UVgami engine stderr (exit {ret_code}):")
                    for line in tail:
                        print(line)

        if move_to_invalid:
            invalid_obj = import_obj(unwrap.path)
            collection = check_collection(
                "UVgami Not Unwrapped", bpy.context.scene.collection
            )
            self.moved_to_invalid = True
            move_to_collection(invalid_obj, collection)
            label = f"{invalid_obj.name}: {msg}"
            invalid_obj.name = label
            invalid_obj.hide_set(True)
            logger.add_data("errors", label)

        self.record_result(unwrap, Result.INVALID)

    def _result_counts(self):
        """Per-result piece counts. A finished piece whose group was discarded
        or never settled was not imported, so it counts as cancelled."""
        counts = dict.fromkeys(Result, 0)
        for unwrap, result in self.results:
            group = unwrap.join_job
            if (
                result is Result.FINISHED
                and group is not None
                and (group.discard or not group.is_settled())
            ):
                result = Result.CANCELLED
            counts[result] += 1
        return counts

    def _finish_batch(self):
        """Called when all unwraps are done (completed, failed, or cancelled)."""
        props = self.props
        if props.pack_after_unwrap and self._pack_output_objects:
            valid_objects = [o for o in self._pack_output_objects if check_exists(o)]
            if valid_objects:
                if props.combine_uvs:
                    edit_restore(valid_objects, pack)
                else:
                    for obj in valid_objects:
                        edit_restore([obj], pack)

        counts = self._result_counts()
        self.finish()
        self.log_final_status()

        if counts[Result.CANCELLED] != len(self.results):
            msg = []

            # problems that don't fail a mesh but shouldn't read as a clean run
            had_error = bool(
                self.transfer_uv_failed or self.error_code or self.error_messages
            )

            # headline first, the banner shows it alone on the top row
            finished, invalid = counts[Result.FINISHED], counts[Result.INVALID]
            cancelled, stopped = counts[Result.CANCELLED], counts[Result.STOPPED]
            # the counts are per loose part, not per mesh
            if finished and invalid:
                msg.append(f"{invalid} of {finished + invalid} parts failed")
            elif finished and stopped:
                msg.append(f"{stopped} of {finished + stopped} parts stopped")
            elif finished and had_error:
                msg.append("UV unwrap finished with errors")
            elif finished:
                msg.append("UV unwrap complete!")
            elif stopped and not invalid:
                msg.append("UV unwrap stopped")
            else:
                msg.append("UV unwrap failed")

            if invalid:
                logger.add_data("errors", "Some meshes were not able to be unwrapped")
            if finished and cancelled:
                objects = "object" if cancelled == 1 else "objects"
                msg.append(f"{cancelled} {objects} cancelled")
            if self.moved_to_invalid:
                msg.append("Check 'UVgami Not Unwrapped'.")

            if self.transfer_uv_failed:
                detail = self.transfer_uv_fail_detail or "unknown reason"
                line = f"UV transfer failed: {detail}."
                if not self.transfer_uv_reason_known:
                    line += " This can happen with cuts or symmetry enabled."
                msg.append(line)

            if self.transfer_uv_split_count > 0:
                count = self.transfer_uv_split_count
                msg.append(f"UV transfer split {count} face(s) crossed by a seam.")

            if self.error_code != 0:
                err_msg = f"An unknown error occurred: {self.error_code}"
                if self.error_stderr:
                    err_msg += f" ({self.error_stderr})"
                msg.append(err_msg)
                logger.add_data("errors", err_msg)

            # a whole queue can fail the same way, show the line once
            for err in dict.fromkeys(self.error_messages):
                msg.append(err)
                logger.add_data("errors", err)

            self.summary = msg
            self.summary_failed = bool(invalid or had_error)
            self._show_status()

            if get_preferences().show_popup:
                popup(msg, "UVgami", "INFO")
        else:
            self.clear_summary()

        # the dispatch timer is gone, so repaint the queue ui and banner here
        tag_redraw()

    def _show_status(self):
        """Put the summary in the status bar. A clean run clears itself, a run
        with problems stays until the next one so it can't be missed."""
        text = f"UVgami: {self.summary[0]}"
        if self.summary_failed:
            set_status(f"{text}", "ERROR")
        else:
            set_status(text)
            bpy.app.timers.register(
                functools.partial(set_status, None), first_interval=STATUS_SECONDS
            )

    def log_final_status(self):
        counts = self._result_counts()
        cancelled = counts[Result.CANCELLED] == len(self.results)
        logger.change_status("Cancelled" if cancelled else "Complete")

    def clear_summary(self):
        """Drop the banner and the status bar message."""
        self.summary = []
        self.summary_failed = False
        set_status(None)

    def _unregister_dispatch(self):
        if self._dispatch_handle is not None:
            if bpy.app.timers.is_registered(self._dispatch_handle):
                bpy.app.timers.unregister(self._dispatch_handle)
            self._dispatch_handle = None

    def finish(self):
        # one undo step for the whole session, so a single ctrl z reverts it
        bpy.ops.ed.undo_push(message="UVgami Unwrap")
        self._unregister_dispatch()
        progress_bar.remove()
        self.is_active = False
        self._running.clear()
        self._queue.clear()
        self._pack_output_objects.clear()
        self.input.clear()

        # count first, the error path reaches here before start() set props
        if self._result_counts()[Result.FINISHED] > 0 and self.props.auto_grid:
            switch_shading("MATERIAL")

        for path in get_io_dir_paths():
            clear_io_dir(path)

    def drop_preparing(self, entry):
        """Tolerant: stop_all clears the list before the builder gets here."""
        if entry in self.preparing:
            self.preparing.remove(entry)

    def finished_adding(self):
        """Clamped: stop_all zeroes the count, so a timer that outlives it
        would otherwise go negative and the session could never finish."""
        self.pieces_still_arriving = max(0, self.pieces_still_arriving - 1)

    def stop_all(self):
        # late import: ops.viewer imports the manager
        from .ops.viewer import stop_viewer_draw

        for unwrap in list(self._running):
            unwrap.stop_process()
            unwrap.cleanup()
        for unwrap in list(self._queue):
            unwrap.cleanup()
        for transfer in self.pending_transfers:
            transfer.job.cancel()
        self.pending_transfers.clear()
        self._running.clear()
        self._queue.clear()
        # a file load kills the builder timers that would have cleared these
        self.preparing.clear()
        self.pieces_still_arriving = 0
        self._unregister_dispatch()
        progress_bar.remove()
        # the viewer modal dies with a file load, so remove its handler here
        stop_viewer_draw()
        self.exit_viewer = True
        self.is_viewer_active = False
        self.is_active = False

    def shutdown(self):
        """Drop everything, for a file load or the addon unloading. The engine
        processes and the draw handlers survive both, and a file load kills the
        timer that would have cleaned them up."""
        self.stop_all()
        self.clear_summary()
        # a log carried into the next file would list runs on the old one
        logger.unwrap_info.clear()


manager = UnwrapManager()
