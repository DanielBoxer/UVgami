# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import collections
import pathlib
import platform
import subprocess
import threading
import time

import bmesh
import bpy
import mathutils

from .logger import logger
from .manager import manager
from .utils.io import print_stdin
from .utils.mesh import check_exists
from .utils.paths import get_extension_dir_path, get_linux_path, get_preferences


class Unwrap:
    def __init__(
        self,
        name: str,
        input_name: str,
        path: pathlib.Path,
        guide_path: pathlib.Path,
        edge_path: pathlib.Path,
        jobs: tuple,
        origin: mathutils.Vector,
        materials: list,
        added_edges: list,
        vertex_count: int,
        shade_smooth: bool,
        auto_smooth: int,
        merge_cuts: bool,
    ):
        # unwrap name
        self.name = name
        self.input_name = input_name

        # paths
        self.path = path
        self.output_path = get_extension_dir_path() / "output" / f"{self.path.stem}.obj"
        # seam restrictions
        self.guide_path = guide_path
        # for untriangulation (added edges)
        self.edge_path = edge_path

        # jobs
        self.jobs = [j for j in jobs if j is not None]
        self.preserve_job = jobs[0]
        self.join_job = jobs[1]
        self.cleanup_job = jobs[2]
        self.symmetrize_job = jobs[3]

        # object info
        self.origin = mathutils.Vector(origin)
        self.materials = materials
        self.added_edges = added_edges
        self.vertex_count = vertex_count
        self.shade_smooth = shade_smooth
        self.auto_smooth = auto_smooth

        # other
        self.merge_cuts = merge_cuts

        # unwrap state
        self.is_active = False
        self.progress = (0, 0, 1)
        # unwrap process
        self.process = None
        # copy of input obj used for viewing
        self.viewer_obj = None
        self.viewing = False
        self.view_update_count = 0
        self.progress_data = collections.deque()
        self.uv_co = collections.deque()
        self.uv_indices = collections.deque()
        self.is_uv_data_ready = False
        self.is_stopped = False
        self.stop_requested_at = None

    def start_unwrap(self):
        prefs = get_preferences()
        # check for valid engine
        engine_path = pathlib.Path(prefs.engine_path)
        if (
            str(engine_path) == "."
            or not engine_path.is_file()
            or engine_path.stem != "uvgami"
        ):
            engine_path = str(manager.engine_path)

        quality = bpy.context.scene.uvgami.quality
        u = ""
        if quality == "HIGH":
            u = "4.05"
        elif quality == "MEDIUM":
            u = "4.1"
        else:
            u = "4.2"

        s_weight = bpy.context.scene.uvgami.weight_value
        s = ""
        if s_weight == 5:
            s = "200"
        elif s_weight == 4:
            s = "150"
        elif s_weight == 3:
            s = "100"
        elif s_weight == 2:
            s = "50"
        elif s_weight == 1:
            s = "25"

        args = []
        shared_args = f"-u {u} -s {s}"

        if platform.system() == "Windows" and engine_path.suffix == "":
            input_path = get_linux_path(self.path)
            output_path = get_linux_path(get_extension_dir_path() / "output")
            args = [
                "bash",
                "-c",
                f"~/uvgami -i {input_path} -o {output_path}/ {shared_args}",
            ]
        else:
            args = [str(engine_path), "-i", str(self.path)] + shared_args.split()

        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )

        # start reading thread
        thread = threading.Thread(target=self.get_output)
        thread.start()

        self.is_active = True
        self.started_at = time.monotonic()

    def stop_process(self):
        if self.process is not None and self.process.poll() is None:
            if platform.system() == "Windows" and manager.engine_path.suffix == "":
                # wsl
                print_stdin(self.process, "cancel")
            else:
                # windows
                self.process.kill()

    def get_output(self):
        # get lines until there are no more left
        for line in iter(self.process.stdout.readline, ""):
            if line.startswith("progress: "):
                self.progress_data.append(line[10:])
            elif line == "visual_begin:\n":
                self.uv_co.clear()
                self.uv_indices.clear()
                self.is_uv_data_ready = False
            elif line == "visual_end:\n":
                self.is_uv_data_ready = True
            elif line.startswith("vt"):
                uv_co = line[3:].split()
                self.uv_co.append((float(uv_co[0]), float(uv_co[1])))
            elif line.startswith("f"):
                uv_indices = line[2:].split()
                self.uv_indices.append(
                    (int(uv_indices[0]), int(uv_indices[1]), int(uv_indices[2]))
                )
        # process has ended, thread will exit here

    def update_progress(self):
        """Read progress from the stdout reader thread."""
        if len(self.progress_data) > 0:
            progress = self.progress_data.popleft()
            try:
                self.progress = tuple(float(num) for num in progress.split())
            except ValueError:
                # invalid progress string
                return
            self.progress_data.clear()

    def update_viewer(self):
        print_stdin(self.process, "snapshot")
        if self.is_uv_data_ready:
            uvs = list(self.uv_co)
            uv_idcs = list(self.uv_indices)
            self.is_uv_data_ready = False

            # need to use from_edit_mesh here so mesh is updated in edit mode
            bm = bmesh.from_edit_mesh(self.viewer_obj.data)
            uv_map = bm.loops.layers.uv.verify()

            for face in bm.faces:
                # set uvs
                uv_idx_triple = uv_idcs[face.index]
                face.loops[0][uv_map].uv = uvs[uv_idx_triple[0]]
                face.loops[1][uv_map].uv = uvs[uv_idx_triple[1]]
                face.loops[2][uv_map].uv = uvs[uv_idx_triple[2]]

            # need to use update_edit_mesh, don't call bm.free(), it will crash
            bmesh.update_edit_mesh(self.viewer_obj.data)

    def cleanup(self):
        """Clean up files and viewer objects."""
        try:
            if self.path.is_file():
                self.path.unlink()
            if self.guide_path is not None and self.guide_path.is_file():
                self.guide_path.unlink()
        except PermissionError:
            logger.add_data("errors", "Error deleting file")
        if self.viewer_obj is not None and check_exists(self.viewer_obj):
            bpy.data.objects.remove(self.viewer_obj, do_unlink=True)
