import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .common import (
    EXIT_ENGINE_FAILURE,
    UnwrapError,
    deliver,
    find_engine,
    log,
)

QUALITY_UPPER_BOUND = {"less-stretch": "4.05", "balanced": "4.2", "fewer-seams": "5.0"}
SEAM_WEIGHT_LEVELS = {1: "25", 2: "50", 3: "100", 4: "150", 5: "200"}


def build_args(engine_path, input_path, output_dir, quality, seam_weight, import_uvs):
    args = [
        str(engine_path),
        "-i",
        str(input_path),
        # optcuts appends the mesh name directly
        "-o",
        str(output_dir) + os.sep,
        "-u",
        QUALITY_UPPER_BOUND[quality],
        "-s",
        SEAM_WEIGHT_LEVELS[seam_weight],
    ]
    if not import_uvs:
        args.append("-g")
    return args


def run(
    input_path,
    output_path,
    quality,
    import_uvs,
    seam_weights,
    seam_weight,
    engine_path,
    timeout=None,
):
    engine = find_engine("optcuts", "OptCuts", engine_path)
    with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        work_input = in_dir / input_path.name
        shutil.copyfile(input_path, work_input)
        # optcuts reads "<stem>_weights" and "<stem>_fixed" next to the input
        for sidecar in input_path.parent.glob(f"{input_path.stem}_*"):
            if sidecar.is_file():
                shutil.copyfile(sidecar, in_dir / sidecar.name)
        if seam_weights is not None:
            shutil.copyfile(seam_weights, in_dir / f"{input_path.stem}_weights")

        args = build_args(engine, work_input, out_dir, quality, seam_weight, import_uvs)
        log(f"running: {' '.join(args)}", style="step")
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        # the stdout loop below would block past a wait(timeout=) deadline
        killed = threading.Event()
        timer = None
        if timeout is not None:

            def kill():
                killed.set()
                process.kill()

            timer = threading.Timer(timeout, kill)
            timer.start()
        try:
            for line in process.stdout:
                sys.stderr.write(line)
            returncode = process.wait()
        finally:
            if timer is not None:
                timer.cancel()
        if killed.is_set():
            raise UnwrapError(
                EXIT_ENGINE_FAILURE, f"OptCuts timed out after {timeout}s"
            )
        if returncode != 0:
            raise UnwrapError(
                EXIT_ENGINE_FAILURE, f"OptCuts exited with code {returncode}"
            )

        deliver(out_dir / f"{input_path.stem}.obj", output_path)
