# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import functools
import traceback
from collections import deque

import bmesh
import bpy
import numpy

from . import progress_bar
from .job import Join
from .logger import logger
from .ops.grid import add_grid, make_grid_img, make_grid_mat
from .reroute_seams import reroute_seams
from .utils.geometry import set_origin
from .utils.io import import_obj, print_stdin
from .utils.mesh import (
    check_collection,
    move_to_collection,
    new_bmesh,
    set_active_any,
    set_bmesh,
)
from .utils.paths import get_extension_dir_path, get_preferences
from .utils.ui import popup, switch_shading


class UnwrapManager:
    def __init__(self):
        self._queue = deque()
        self._running = []
        self.input = {}
        self.engine_path = None
        self.is_active = False
        self.is_viewer_active = False
        self._dispatch_handle = None

    @property
    def active(self):
        """All unwraps (running and queued)"""
        return self._running + list(self._queue)

    def add(self, unwrap):
        """Add an unwrap to the queue."""
        self._queue.append(unwrap)

    def remove_unwrap(self, unwrap):
        """Remove an unwrap from running or queue."""
        if unwrap in self._running:
            self._running.remove(unwrap)
        elif unwrap in self._queue:
            self._queue.remove(unwrap)

    def start(self):
        self.starting_count = len(self._queue) + len(self._running)
        # fill initial slots from queue
        self._fill_slots()
        if get_preferences().show_progress_bar:
            progress_bar.start()
        self.is_active = True
        self.found_invalid_objects = False
        self.finished_count = 0
        self.cancelled_count = 0
        self.error_code = 0
        self.license_error = None
        self.current_viewer = None
        self.is_viewer_active = False
        self.exit_viewer = False
        # register central dispatch timer
        self._dispatch_handle = functools.partial(self._dispatch)
        bpy.app.timers.register(self._dispatch_handle)

    def _fill_slots(self):
        """Start queued unwraps up to the concurrency limit."""
        props = bpy.context.scene.uvgami
        max_concurrent = props.max_cores if props.concurrent else 1
        while len(self._running) < max_concurrent and self._queue:
            unwrap = self._queue.popleft()
            unwrap.start_unwrap()
            self._running.append(unwrap)

    def _dispatch(self):
        """Central dispatch timer that monitors all running unwraps."""
        # guard against running after finish
        if not self.is_active:
            return None

        try:
            completed = []
            failed = []

            for unwrap in list(self._running):
                # update progress
                unwrap.update_progress()

                # check early stop
                early_stop = bpy.context.scene.uvgami.early_stop
                if early_stop != 100 and unwrap.progress[0] >= early_stop / 100:
                    unwrap.is_stopped = True

                # update viewer
                if unwrap.viewing:
                    unwrap.update_viewer()

                # if part of batch unwrap, hasn't started and stop button pressed
                if unwrap.is_stopped:
                    print_stdin(unwrap.process, "stop")

                # check process status
                ret_code = unwrap.process.poll()
                if ret_code is not None:
                    if ret_code == 0 and unwrap.output_path.is_file():
                        completed.append(unwrap)
                    elif ret_code != 0:
                        failed.append((unwrap, ret_code))

            logger.update_time()

            # update progress bar
            self._update_progress_bar()

            # process completions (each isolated so one failure doesn't block others)
            for unwrap in completed:
                try:
                    self._process_completion(unwrap)
                except Exception:
                    error_list = traceback.format_exc().split("\n")[:-1]
                    logger.add_data("errors", "Error finishing unwrap:")
                    for line in error_list:
                        logger.add_data("errors", line)
                        print(line)
                    # ensure unwrap is removed even on error
                    if unwrap in self._running:
                        self._running.remove(unwrap)
                    unwrap.cleanup()

            # process failures (each isolated)
            for unwrap, ret_code in failed:
                try:
                    self._handle_failure(unwrap, ret_code)
                except Exception:
                    error_list = traceback.format_exc().split("\n")[:-1]
                    logger.add_data("errors", "Error handling unwrap failure:")
                    for line in error_list:
                        logger.add_data("errors", line)
                        print(line)
                    if unwrap in self._running:
                        self._running.remove(unwrap)
                    unwrap.cleanup()

            # fill empty slots from queue
            self._fill_slots()

            # check if everything is done
            if not self._running and not self._queue:
                self._finish_batch()
                return None

        except Exception as e:
            # catastrophic dispatch error
            from .handler import handle_error

            handle_error(e, "MIDDLE")
            return None

        return 0.1

    def _update_progress_bar(self):
        """Update the overall progress bar."""
        if not get_preferences().show_progress_bar:
            return

        all_unwraps = self.active
        progress = [numpy.array(unwrap.progress) for unwrap in all_unwraps]
        # fill up progress bar with finished unwraps
        for _ in range(self.starting_count - len(progress)):
            progress.append(numpy.array((1, 0, 0)))
        if self.starting_count > 0:
            new_progress = sum(progress) / self.starting_count
            progress_bar.update(new_progress)
            # force redraw of view3D
            bpy.context.view_layer.objects.active = (
                bpy.context.view_layer.objects.active
            )

    def _process_completion(self, unwrap, invalid_pass=False):
        """Process a successfully completed unwrap."""
        props = bpy.context.scene.uvgami
        path = unwrap.output_path
        is_import_ready = True
        added_edges = []
        edge_path = unwrap.edge_path
        if not invalid_pass:
            self.finished_count += 1

        if unwrap.join_job is not None:
            if not invalid_pass:
                unwrap.join_job.unwrapped.append(unwrap)
            # get all paths of finished unwraps before joining
            if unwrap.join_job.is_completed() and unwrap.join_job.count > 1:
                data = unwrap.join_job.finish(unwrap)
                path = data[0]
                edge_path = data[1]
                added_edges = data[2]
            # if the count is 1, that means all but one unwrap of group was cancelled
            elif not (unwrap.join_job.is_completed() and unwrap.join_job.count == 1):
                # in all other cases, wait until last unwrap finishes before importing
                is_import_ready = False

        if is_import_ready:
            # reroute seams before importing
            if unwrap.preserve_job is not None and props.maintain_mode == "FULL":
                reroute_seams(path, edge_path)

            old_active = bpy.context.view_layer.objects.active
            if old_active is None:
                old_active = set_active_any()
            output = import_obj(path, f"{unwrap.input_name}_unwrapped")
            # the new obj importer changes the active object
            if bpy.app.version >= (3, 2, 0):
                bpy.context.view_layer.objects.active = old_active

            set_origin(output, unwrap.origin)

            # set materials
            materials = [
                bpy.data.materials.get(m_name)
                for m_name in unwrap.materials
                if m_name is not None
            ]
            for m in materials:
                output.data.materials.append(m)

            if unwrap.preserve_job is not None:
                unwrap.preserve_job.finish(unwrap, output, added_edges)

            if unwrap.cleanup_job is not None:
                unwrap.cleanup_job.finish(self.input[unwrap.cleanup_job])

            if unwrap.symmetrize_job is not None:
                unwrap.symmetrize_job.finish(output)

            if unwrap.merge_cuts:
                bm = new_bmesh(output)
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
                set_bmesh(bm, output)

            # automatically add grid material to final object
            if props.auto_grid:
                grid_img = make_grid_img()
                add_grid(output, make_grid_mat(grid_img))

            # shade smooth
            if unwrap.shade_smooth:
                output.data.polygons.foreach_set(
                    "use_smooth", [True] * len(output.data.polygons)
                )
                if unwrap.auto_smooth != -1:
                    if bpy.app.version >= (4, 1, 0):
                        pass
                    else:
                        output.data.use_auto_smooth = True
                        output.data.auto_smooth_angle = unwrap.auto_smooth

            # copy vertex groups from input
            if unwrap.input_name in bpy.data.objects:
                input_obj = bpy.data.objects[unwrap.input_name]

                for group in input_obj.vertex_groups:
                    new_group = output.vertex_groups.new(name=group.name)
                    for v_idx in range(unwrap.vertex_count):
                        try:
                            weight = group.weight(v_idx)
                            new_group.add([v_idx], weight, "REPLACE")
                        except RuntimeError:
                            # vertex not in group
                            continue
            else:
                logger.add_data(
                    "errors", "Input object not found, couldn't copy vertex groups"
                )

            logger.add_data("objects", unwrap.input_name)

        self.exit_viewer = True

        if not invalid_pass:
            # remove from running and clean up files
            if unwrap in self._running:
                self._running.remove(unwrap)
            unwrap.cleanup()

    def _handle_failure(self, unwrap, ret_code):
        """Handle an unwrap process that exited with a non-zero code."""
        prefs = get_preferences()
        msg = ""

        # convert unsigned int
        THRESHOLD = 2147483648
        ADJUSTMENT = 4294967296
        if ret_code >= THRESHOLD:
            ret_code -= ADJUSTMENT

        move_to_invalid = False
        if ret_code == -1:
            msg = "Mesh needs cleanup"
            move_to_invalid = True
        elif ret_code == 101:
            msg = "Non Manifold Edges"
            move_to_invalid = True
        elif ret_code == 102:
            msg = "Non Manifold Vertices"
            move_to_invalid = True
        elif ret_code == 105:
            msg = "Invalid Geometry"
            move_to_invalid = True
        elif ret_code == 107:
            msg = "Invalid UV Input"
            move_to_invalid = True
        else:
            self.error_code = ret_code

        if move_to_invalid:
            if prefs.invalid_collection:
                # move to collection for invalid meshes
                invalid_obj = import_obj(unwrap.path)
                collection = check_collection(
                    "UVgami Invalid Input", bpy.context.scene.collection
                )
                move_to_collection(invalid_obj, collection)
                invalid_name = f"{invalid_obj.name}: {msg}"
                invalid_obj.name = invalid_name
                invalid_obj.hide_set(True)
                logger.add_data("errors", invalid_name)

            self.found_invalid_objects = True

        found_job = None
        # count has to be reduced because this object won't be unwrapped
        for job in unwrap.jobs:
            if job.count > 1:
                job.count = job.count - 1
                # found_job can't be a Cleanup job because the unwrapped list
                # will be empty
                if isinstance(job, Join):
                    found_job = job

        # remove from running
        if unwrap in self._running:
            self._running.remove(unwrap)
        unwrap.stop_process()
        unwrap.cleanup()

        # if the invalid obj has jobs that are complete with the now reduced count
        # that means that this unwrap was the last of the group
        if found_job is not None and found_job.is_completed():
            # use the last completed unwrap
            self._process_completion(found_job.unwrapped[-1], invalid_pass=True)

    def _finish_batch(self):
        """Called when all unwraps are done (completed, failed, or cancelled)."""
        self.finish()

        # don't show popup if all unwraps were cancelled
        if self.cancelled_count != self.starting_count:
            logger.change_status("Complete")
            if get_preferences().show_popup:
                msg = []

                if self.finished_count > 0:
                    msg.append("UV unwrap complete!")

                if self.found_invalid_objects:
                    msg.append("Some meshes were not able to be unwrapped.")
                    msg.append("Check 'UVgami Invalid Input'.")
                    logger.add_data(
                        "errors", "Some meshes were not able to be unwrapped"
                    )

                if self.error_code != 0:
                    err_msg = f"An unknown error occurred: {self.error_code}"
                    msg.append(err_msg)
                    logger.add_data("errors", err_msg)

                if self.license_error is not None:
                    msg.append(self.license_error)
                    logger.add_data("errors", self.license_error)

                popup(msg, "UVgami", "INFO")
        else:
            logger.change_status("Cancelled")

    def _unregister_dispatch(self):
        """Unregister the dispatch timer if active."""
        if self._dispatch_handle is not None:
            if bpy.app.timers.is_registered(self._dispatch_handle):
                bpy.app.timers.unregister(self._dispatch_handle)
            self._dispatch_handle = None

    def finish(self):
        """Clean up everything."""
        self._unregister_dispatch()
        progress_bar.remove()
        self.is_active = False
        self._running.clear()
        self._queue.clear()

        if (
            bpy.context.scene.uvgami.auto_grid
            and getattr(self, "finished_count", 0) > 0
        ):
            switch_shading("MATERIAL")

        # clean up io folders
        for file in (get_extension_dir_path() / "input").iterdir():
            file.unlink()
        for file in (get_extension_dir_path() / "output").iterdir():
            file.unlink()

    def cancel_unwrap(self, unwrap):
        """Cancel a specific unwrap."""
        self.cancelled_count += 1
        unwrap.stop_process()
        self.remove_unwrap(unwrap)
        unwrap.cleanup()
        self.exit_viewer = True
        # update 3d view to remove progress bar
        bpy.context.view_layer.objects.active = bpy.context.view_layer.objects.active

    def stop_all(self):
        """Stop all running processes and clean up."""
        for unwrap in list(self._running):
            unwrap.stop_process()
            unwrap.cleanup()
        for unwrap in list(self._queue):
            unwrap.cleanup()
        self._running.clear()
        self._queue.clear()
        self._unregister_dispatch()
        progress_bar.remove()


manager = UnwrapManager()
