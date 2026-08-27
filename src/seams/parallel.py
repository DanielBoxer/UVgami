"""seam_edges one loose part at a time across worker processes.

Every pass reads only its own part once the auto width and the model area
come from the whole mesh (whole_mesh_inputs), so the parts' seams joined
are the whole run's seams, and 484 parts spread over the cores instead of
one 30s pass. The parts keep the mesh's vertex ids, reindexing would move
the tie breaks in the heap searches, and carry their face ids for the
sweep run seeds (WholeMesh)."""

import os
import pickle
import subprocess
from pathlib import Path

from .cancel import Cancelled, check_cancelled
from .mesh import vertex_components
from .pipeline import seam_edges, whole_mesh_inputs
from .regions import CREASE_ANGLE

WORKER_SCRIPT = Path(__file__).with_name("worker.py")
POLL_INTERVAL = 0.05


def balanced_chunks(parts, count):
    """Parts dealt to count workers, largest first onto the lightest, so
    the workers carry about the same face count."""
    loads = [0] * count
    chunks = [[] for _ in range(count)]
    for part in sorted(parts, key=len, reverse=True):
        lightest = loads.index(min(loads))
        loads[lightest] += len(part)
        chunks[lightest].append(part)
    return [chunk for chunk in chunks if chunk]


def seam_edges_parallel(
    python,
    verts,
    faces,
    angle=CREASE_ANGLE,
    rims=True,
    weights=None,
    forced=None,
    cancelled=None,
):
    """seam_edges with the loose parts run on worker processes, one per
    core. python is (executable, leading args) for the workers, Blender's
    bundled interpreter in the addon. A mesh with one part runs here."""
    parts = vertex_components(faces)
    count = min(os.cpu_count() or 1, len(parts))
    if count < 2:
        return seam_edges(verts, faces, angle, rims, weights, forced, cancelled)
    min_width, model_area = whole_mesh_inputs(verts, faces, rims, forced)
    check_cancelled(cancelled)
    job = {
        "angle": angle,
        "rims": rims,
        "weights": weights,
        "forced": forced,
        "min_width": min_width,
        "model_area": model_area,
    }
    chunks = [
        [(part, [faces[i] for i in part]) for part in chunk]
        for chunk in balanced_chunks(parts, count)
    ]
    executable, args = python
    command = [executable, *args, str(WORKER_SCRIPT)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    workers = [
        subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        for _ in chunks
    ]
    seams = set()
    closed = True
    try:
        for worker, chunk in zip(workers, chunks):
            # a worker that died at startup closes the pipe
            try:
                worker.stdin.write(pickle.dumps((verts, chunk, job)))
                worker.stdin.close()
            except OSError:
                pass
        for worker in workers:
            while True:
                try:
                    out, err = worker.communicate(timeout=POLL_INTERVAL)
                    break
                except subprocess.TimeoutExpired:
                    if cancelled is not None and cancelled():
                        raise Cancelled
            if worker.returncode != 0:
                tail = err.decode(errors="replace").strip()[-500:]
                raise RuntimeError(f"seam worker failed: {tail}")
            for part_seams, part_closed in pickle.loads(out):
                seams |= part_seams
                closed = closed and part_closed
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
    if not seams and angle > CREASE_ANGLE and closed:
        return seam_edges_parallel(
            python, verts, faces, CREASE_ANGLE, rims, weights, forced, cancelled
        )
    return seams
