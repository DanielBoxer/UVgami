# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import multiprocessing
from .utils import (
    get_preferences,
    import_obj,
    popup,
    get_dir_path,
    set_active_any,
    switch_shading,
    set_origin,
)
from . import progress_bar
from .logger import logger
from .ops.grid import (
    make_grid_img,
    make_grid_mat,
    add_grid,
)
from .reroute_seams import reroute_seams


class UnwrapManager:
    def __init__(self):
        self.active = []
        self.input = {}
        self.engine_path = None
        self.is_active = False
        self.is_viewer_active = False

    def start(self):
        self.starting_count = len(self.active)
        if bpy.context.scene.uvgami.concurrent:
            active_copy = self.active.copy()
            for unwrap_idx in range(int(multiprocessing.cpu_count() / 2 - 1)):
                # if there are more cores than meshes
                if unwrap_idx == len(active_copy):
                    break
                active_copy[unwrap_idx].start_unwrap()
        else:
            self.active[0].start_unwrap()
        if get_preferences().show_progress_bar:
            progress_bar.start()
        self.is_active = True
        self.found_invalid_objects = False
        self.finished_count = 0
        self.cancelled_count = 0
        self.unknown_error = False
        self.license_error = None
        self.current_viewer = None
        self.is_viewer_active = False
        self.exit_viewer = False

    def finish(self):
        progress_bar.remove()
        self.is_active = False
        self.active.clear()

        if bpy.context.scene.uvgami.auto_grid and self.finished_count > 0:
            switch_shading("MATERIAL")

        # clean up io folders
        for file in (get_dir_path() / "input").iterdir():
            file.unlink()
        for file in (get_dir_path() / "output").iterdir():
            file.unlink()

    def finish_unwrap(self, unwrap, invalid_pass=False):
        try:
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
                elif not (
                    unwrap.join_job.is_completed() and unwrap.join_job.count == 1
                ):
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
                        output.data.use_auto_smooth = True
                        output.data.auto_smooth_angle = unwrap.auto_smooth

                logger.add_data("objects", unwrap.input_name)

            manager.exit_viewer = True

            if not invalid_pass:
                unwrap.remove()
                self.start_next()
        except Exception as e:
            # avoid circular import
            from .handler import handle_error

            handle_error(e, "FINISH")

    def start_next(self):
        if self.active:
            for unwrap in self.active:
                if not unwrap.is_active:
                    unwrap.start_unwrap()
                    break
        else:
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

                    if self.unknown_error:
                        msg.append("An unknown error occurred.")
                        logger.add_data("errors", "An unknown error occurred")

                    if self.license_error is not None:
                        msg.append(self.license_error)
                        logger.add_data("errors", self.license_error)

                    popup(msg, "UVgami", "INFO")
            else:
                logger.change_status("Cancelled")

    def stop_all(self):
        for unwrap in self.active:
            unwrap.cancel_unwrap()
        progress_bar.remove()


manager = UnwrapManager()
