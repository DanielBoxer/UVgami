import collections
import pathlib
import subprocess
import threading
import time

import mathutils

from .batch import EngineOutput, read_stderr_tail
from .logger import logger
from .manager import manager
from .ops.viewer import set_snapshot
from .utils.paths import get_extension_dir_path
from .utils.ui import tag_redraw

# the readers finish with the process, so this only bounds a stuck one
READER_JOIN_SECONDS = 5

# healthy runs move progress every few seconds, minutes frozen is a hang
PROGRESS_STALL_SECONDS = 120


class Unwrap:
    def __init__(
        self,
        name: str,
        input_name: str,
        path: pathlib.Path,
        jobs: tuple,
        maintain_mode: str = "FULL",
        preseeded: bool = False,
    ):
        self.name = name
        self.input_name = input_name
        # this piece carries preseed uvs, so its output gets the finish pass
        self.preseeded = preseeded

        self.path = path
        self.output_path = get_extension_dir_path() / "output" / f"{self.path.stem}.obj"

        self.preserve_job = jobs[0]
        self.join_job = jobs[1]
        if self.join_job is not None:
            self.join_job.members.append(self)
        self.hide_job = jobs[2]
        self.symmetrize_job = jobs[3]
        self.transfer_uvs_job = jobs[4]

        # a duplicate piece skips the engine and takes copy_of's output moved
        # by copy_matrix. a reordered copy's indices don't line up with that
        # output, so it exports copy_of's metadata instead
        self.copy_of = None
        self.copy_matrix = None
        self.copy_reordered = False
        self.twins = []

        # snapshot: the edge file was written for this mode, so a later change
        # must not reach the untriangulate pass
        self.maintain_mode = maintain_mode

        # result state, set once through manager.record_result
        self.result = None

        # what the export needs about this piece, filled by the session builder
        self.has_uvs = False
        self.keeps_output = True
        self.matrix = None

        # export data, filled by set_export_data
        self.is_exported = False
        self.guide_path = None
        self.edge_path = None
        self.origin = None
        self.materials = []
        self.added_edges = []
        self.vertex_count = 0
        self.material_indices = []
        self.vertex_groups = {}
        self.face_smooth = []
        self.shading_modifiers = []

        self.is_active = False
        self.progress = (0, 0, 1)
        self.has_reported = False
        # shared with the other unwraps when part of a batch process
        self.process = None
        self.batch_process = None
        self.viewing = False
        self.progress_data = collections.deque()
        self.uv_co = collections.deque()
        self.uv_indices = collections.deque()
        self.is_uv_data_ready = False
        self.is_stopped = False
        self.progress_changed_at = None
        self.started_at = None
        self.stop_requested_at = None
        # bounded tail of the solo process's stderr, drained by a reader thread
        self.stderr_tail = collections.deque(maxlen=10)
        self._stderr_thread = None
        self._output_thread = None

    def set_export_data(
        self,
        *,
        origin,
        vertex_count,
        guide_path=None,
        edge_path=None,
        materials=(),
        added_edges=(),
        material_indices=(),
        vertex_groups=None,
        face_smooth=(),
        shading_modifiers=(),
    ):
        """The defaults cover a fix export, which carries no mesh metadata."""
        self.guide_path = guide_path
        self.edge_path = edge_path
        self.origin = mathutils.Vector(origin)
        self.materials = materials
        self.added_edges = added_edges
        self.vertex_count = vertex_count
        self.material_indices = material_indices
        self.vertex_groups = vertex_groups or {}
        self.face_smooth = face_smooth
        self.shading_modifiers = shading_modifiers
        self.is_exported = True

    def start_unwrap(self):
        args = manager.engine.build_args(manager.engine_ctx, self.path, manager.props)

        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=manager.engine.build_env(manager.engine_ctx),
        )

        self._output_thread = threading.Thread(target=self.get_output)
        self._output_thread.start()

        # drain stderr separately so it stays out of the stdout protocol
        self._stderr_thread = threading.Thread(
            target=read_stderr_tail,
            args=(self.process.stderr, self.stderr_tail),
            daemon=True,
        )
        self._stderr_thread.start()

        self.is_active = True
        self.started_at = time.monotonic()

    def join_batch(self, batch_process):
        """Run inside a shared batch process instead of spawning our own."""
        self.batch_process = batch_process
        self.process = batch_process.process
        self.is_active = True

    def leave_batch(self):
        """Detach from a dead batch process so this mesh can be re-queued into
        a fresh batch. Mirrors join_batch."""
        self.batch_process = None
        self.process = None
        self.is_active = False

    def poll_engine(self):
        """None while running, 0 on success, or a failure code."""
        if self.batch_process is None:
            return self.process.poll()
        # the engine reports when it reaches each mesh, which starts the
        # timeout clock
        if self.started_at is None and self.path.stem in self.batch_process.started:
            self.started_at = time.monotonic()
        return self.batch_process.poll_result(self.path.stem)

    def stop_process(self):
        """Hard stop: for a batch member this kills the whole batch process."""
        if self.process is not None and self.process.poll() is None:
            manager.engine.stop(self.process, manager.engine_ctx)

    def release_engine(self):
        """This unwrap no longer needs the engine. A batch process is left
        running for the other meshes, and deleting the input file in cleanup()
        is what makes the cli skip this mesh."""
        if self.batch_process is not None:
            return
        self.stop_process()
        if self.process is None:
            return
        # without this a model with many pieces runs out of open files
        for thread in (self._output_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=READER_JOIN_SECONDS)
        for pipe in (self.process.stdout, self.process.stderr, self.process.stdin):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                # close flushes stdin, which a dead engine refuses. it still closes
                pass
        self.process.wait()

    def get_stderr_tail(self):
        """Last stderr lines from this unwrap's process, batch or solo."""
        if self.batch_process is not None:
            return self.batch_process.stderr_lines()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        return self.stderr_tail

    def get_output(self):
        parser = EngineOutput(self)
        for line in iter(self.process.stdout.readline, ""):
            parser.feed(line)

    def update_progress(self):
        """Read progress from the stdout reader thread."""
        if len(self.progress_data) > 0:
            # only the newest queued line matters
            progress = self.progress_data.pop()
            self.progress_data.clear()
            try:
                parsed = tuple(float(num) for num in progress.split())
            except ValueError:
                return
            self.has_reported = True
            if parsed != self.progress:
                self.progress = parsed
                self.progress_changed_at = time.monotonic()

    @property
    def is_running(self):
        """Whether the engine is on this mesh now. A batch process takes the
        whole queue when it spawns, so is_active alone would show them all."""
        if self.batch_process is None:
            return self.is_active
        return self.is_active and self.path.stem in self.batch_process.started

    @property
    def is_stalled(self):
        """Progress frozen for minutes. False on engines that never report it."""
        return (
            self.is_active
            and self.progress_changed_at is not None
            and time.monotonic() - self.progress_changed_at > PROGRESS_STALL_SECONDS
        )

    @property
    def is_stoppable(self):
        """Reporting, so stopping keeps a uv map instead of dropping the piece."""
        return self.is_running and self.has_reported

    @property
    def is_viewable(self):
        """Running and reporting, so the engine has uvs to snapshot. The
        progress numbers can't say it, an all-high-distortion report is the
        0 0 1 this starts at."""
        return self.is_active and self.has_reported

    def update_viewer(self):
        manager.engine.request_snapshot(self.process)
        if self.is_uv_data_ready:
            uvs = list(self.uv_co)
            uv_idcs = list(self.uv_indices)
            self.is_uv_data_ready = False
            if uvs and uv_idcs:
                set_snapshot(uvs, uv_idcs)
                tag_redraw(("WINDOW",))

    def cleanup(self):
        """Clean up input files."""
        try:
            if self.path.is_file():
                self.path.unlink()
            if self.guide_path is not None and self.guide_path.is_file():
                self.guide_path.unlink()
            importance_path = self.path.parent / f"{self.path.stem}_importance"
            if importance_path.is_file():
                importance_path.unlink()
        except PermissionError:
            logger.add_data("errors", "Error deleting file")
