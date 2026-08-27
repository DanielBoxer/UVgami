"""Entry point of a seam worker process, run by seam_edges_parallel.

Reads one pickled (verts, parts, job) from stdin, parts as (face ids,
faces) pairs, runs seams_at_angle on each and pickles the (seams, closed) list to stdout. Runs on the bare
interpreter, so the seams package is imported as a top-level package off
the addon's src directory, never through the addon and bpy."""

import pickle
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from seams.pipeline import WholeMesh, seams_at_angle

    verts, parts, job = pickle.load(sys.stdin.buffer)
    whole = WholeMesh(job.pop("min_width"), job.pop("model_area"), None)
    results = [
        seams_at_angle(
            verts, faces, cancelled=None, whole=whole._replace(face_ids=ids), **job
        )
        for ids, faces in parts
    ]
    sys.stdout.buffer.write(pickle.dumps(results))


if __name__ == "__main__":
    main()
