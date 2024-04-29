# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bmesh
import bpy

from . import progress_bar
from .logger import logger
from .ops.grid import add_grid, make_grid_img, make_grid_mat
from .reroute_seams import reroute_seams
from .utils import (
    get_dir_path,
    get_preferences,
    import_obj,
    new_bmesh,
    popup,
    set_active_any,
    set_bmesh,
    set_origin,
    switch_shading,
)


class UnwrapManager:
    def __init__(self):
        self.active = []
        self.input = {}
        self.engine_path = None
        self.is_active = False
        self.is_viewer_active = False

    def start(self):
        self.starting_count = len(self.active)
        props = bpy.context.scene.uvgami
        if props.concurrent:
            active_copy = self.active.copy()
            for unwrap_idx in range(props.max_cores):
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
        self.error_code = 0
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

                if unwrap.merge_cuts:
                    bm = new_bmesh(output)
                    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
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

                    if self.error_code != 0:
                        # convert unsigned int
                        THRESHOLD = 2147483648
                        ADJUSTMENT = 4294967296
                        if self.error_code >= THRESHOLD:
                            self.error_code -= ADJUSTMENT
                        err_msg = f"An unknown error occurred: {self.error_code}"
                        msg.append(err_msg)
                        logger.add_data("errors", err_msg)

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
