import collections
import functools
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .cancel import Cancelled, check_cancelled
from .mesh import face_edges
from .parallel import seam_edges_parallel
from .pipeline import seam_edges
from .regions import CREASE_ANGLE
from .symmetry import mirror_seams


class FlattenError(RuntimeError):
    pass


# how often a cancellable flatten checks on the engine
POLL_INTERVAL = 0.05


# the engine doesn't validate in flatten mode, a non-manifold mesh returns exit 0
def check_manifold(faces):
    if any(len(owners) > 2 for owners in face_edges(faces).values()):
        raise FlattenError("Non Manifold Edges")


# result() also removes the workdir
class FlattenRun:
    def __init__(self, process, workdir, out_path, face_count):
        self.process = process
        self.workdir = workdir
        self.out_path = out_path
        self.face_count = face_count
        self.progress = 0.0
        self.stderr_tail = collections.deque(maxlen=10)
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _read_stdout(self):
        for line in iter(self.process.stdout.readline, ""):
            if line.startswith("progress: "):
                try:
                    self.progress = float(line.split()[1])
                except (IndexError, ValueError):
                    pass

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_tail.append(line.rstrip("\r\n"))

    def poll(self):
        return self.process.poll()

    def wait(self):
        self.process.wait()
        return self.result()

    def result(self):
        code = self.process.wait()
        self._close()
        try:
            if code != 0:
                detail = " ".join(self.stderr_tail).strip()
                raise FlattenError(f"flatten engine exited {code}: {detail}")
            return _read_uvs(self.out_path, self.face_count)
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def stop(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self._close()
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _close(self):
        for thread in (self._stdout_thread, self._stderr_thread):
            thread.join(timeout=5)
        for pipe in (self.process.stdout, self.process.stderr):
            pipe.close()


# the preview operator and a builder thread can flatten at once
class FlattenEngine:
    def __init__(self, engine_path, workdir):
        self.engine_path = str(engine_path)
        self.workdir = Path(workdir)

    # with cancelled or progress the engine is polled instead of waited on
    def flatten(self, verts, faces, seams, cancelled=None, progress=None):
        run = self.start(verts, faces, seams)
        if cancelled is None and progress is None:
            return run.wait()
        while run.poll() is None:
            if cancelled is not None and cancelled():
                run.stop()
                raise Cancelled
            if progress is not None:
                progress(run.progress)
            time.sleep(POLL_INTERVAL)
        return run.result()

    def start(self, verts, faces, seams):
        self.workdir.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(dir=self.workdir))
        obj_path = workdir / "flatten.obj"
        seam_path = workdir / "flatten_seams"
        out_dir = workdir / "flatten_out"

        with obj_path.open("w") as f:
            for x, y, z in verts:
                f.write(f"v {x} {y} {z}\n")
            for face in faces:
                f.write("f " + " ".join(str(v + 1) for v in face) + "\n")

        if seams:
            seam_path.write_text("".join(f"{a} {b}\n" for a, b in sorted(seams)))

        args = [self.engine_path, "-i", str(obj_path), "-o", str(out_dir), "-flatten"]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
            )
        except OSError as error:
            shutil.rmtree(workdir, ignore_errors=True)
            raise FlattenError(f"flatten engine failed to start: {error}") from error
        return FlattenRun(process, workdir, out_dir / "flatten.obj", len(faces))


def _read_uvs(path, face_count):
    uvs = []
    face_uvs = []
    with Path(path).open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "vt":
                uvs.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "f":
                face_uvs.append(
                    [uvs[int(token.split("/")[1]) - 1] for token in parts[1:]]
                )
    if len(face_uvs) != face_count:
        raise FlattenError(
            f"flatten output has {len(face_uvs)} faces, expected {face_count}"
        )
    return face_uvs


# compact copy of just these faces, with the seams reindexed
def submesh(verts, faces, subset, seams):
    vmap = {}
    sub_verts = []
    sub_faces = []
    for f in subset:
        face = []
        for v in faces[f]:
            idx = vmap.get(v)
            if idx is None:
                idx = vmap[v] = len(sub_verts)
                sub_verts.append(verts[v])
            face.append(idx)
        sub_faces.append(tuple(face))
    sub_seams = set()
    for a, b in seams:
        ma, mb = vmap.get(a), vmap.get(b)
        if ma is not None and mb is not None:
            sub_seams.add((ma, mb) if ma < mb else (mb, ma))
    return sub_verts, sub_faces, sub_seams


# a closed part no seam touches folds onto itself in the flatten
def _flattenable(subset, edges, seams):
    parent = {f: f for f in subset}

    def find(f):
        root = f
        while parent[root] != root:
            root = parent[root]
        while parent[f] != root:
            parent[f], f = root, parent[f]
        return root

    keeps = set()
    for key, owners in edges.items():
        inside = [f for f in owners if f in parent]
        if not inside:
            continue
        root = find(inside[0])
        for f in inside[1:]:
            parent[find(f)] = root
        if len(inside) == 1 or key in seams:
            keeps.add(inside[0])
    kept_roots = {find(f) for f in keeps}
    return [f for f in subset if find(f) in kept_roots]


# a ruined island ships as-is, the engine's own cut search benches better
def preseed_uvs(
    engine,
    verts,
    faces,
    angle=CREASE_ANGLE,
    marked="NONE",
    weights=None,
    only=None,
    marked_seams=frozenset(),
    mirrors=None,
    cancelled=None,
    python=None,
):
    subset = list(range(len(faces))) if only is None else sorted(only)
    edges = face_edges(faces)
    if marked == "ONLY":
        seams = set(marked_seams)
    else:
        forced = set(marked_seams) if marked == "ADD" else None
        detect = faces if only is None else [faces[i] for i in subset]
        run = (
            seam_edges
            if python is None
            else functools.partial(seam_edges_parallel, python)
        )
        seams = run(
            verts, detect, angle, weights=weights, forced=forced, cancelled=cancelled
        )
    if mirrors:
        allowed = edges if only is None else face_edges([faces[i] for i in subset])
        seams = mirror_seams(seams, mirrors, allowed)
    subset = _flattenable(subset, edges, seams)
    if not subset:
        return None

    check_cancelled(cancelled)
    all_uvs = [None] * len(faces)
    sub_verts, sub_faces, sub_seams = submesh(verts, faces, subset, seams)
    flattened = engine.flatten(sub_verts, sub_faces, sub_seams, cancelled)
    for f, face_uv in zip(subset, flattened):
        all_uvs[f] = face_uv
    return seams, all_uvs, subset
