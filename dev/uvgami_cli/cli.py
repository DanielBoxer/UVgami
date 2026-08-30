import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .common import (
    EXIT_INVALID_INPUT,
    EXIT_MISSING_RUNTIME,
    UnwrapError,
    log,
    unwrap_all,
)


@dataclass
class EngineSpec:
    name: str
    add_args: Callable  # populate an argparse argument group with the engine's flags
    flags: dict  # flag -> args dest, so other engines can reject it
    validate: Callable  # apply this engine's defaults and validation to args
    run: Callable  # (args, pairs) -> exit code


def _add_optcuts_args(group):
    group.add_argument(
        "--quality",
        choices=["less-stretch", "balanced", "fewer-seams"],
        help="default: balanced",
    )
    group.add_argument(
        "--import-uvs",
        action="store_true",
        default=None,
        help="keep existing UVs as a starting point",
    )
    group.add_argument("--seam-weights", type=Path, help="vertex weights file")
    group.add_argument(
        "--seam-weight",
        type=int,
        choices=range(1, 6),
        help="seam weight level, default: 3",
    )
    group.add_argument(
        "--optcuts-path", type=Path, help="default: local engines/ binary"
    )
    group.add_argument(
        "--timeout", type=float, help="kill the engine after this many seconds per mesh"
    )


# flags that only apply to optcuts, so the other engine can reject them
OPTCUTS_FLAGS = {
    "--quality": "quality",
    "--import-uvs": "import_uvs",
    "--seam-weights": "seam_weights",
    "--seam-weight": "seam_weight",
    "--optcuts-path": "optcuts_path",
    "--timeout": "timeout",
}


def _validate_optcuts(args):
    args.quality = args.quality or "balanced"
    args.import_uvs = bool(args.import_uvs)
    args.seam_weight = args.seam_weight or 3
    if args.seam_weights is not None and not args.seam_weights.is_file():
        raise UnwrapError(
            EXIT_INVALID_INPUT, f"seam weights file not found: {args.seam_weights}"
        )


def run_optcuts(args, pairs):
    from . import optcuts

    def unwrap_one(input_path, output_path):
        optcuts.run(
            input_path,
            output_path,
            args.quality,
            args.import_uvs,
            args.seam_weights,
            args.seam_weight,
            args.optcuts_path,
            args.timeout,
        )

    return unwrap_all(pairs, unwrap_one)


def _add_xatlas_args(group):
    group.add_argument(
        "--max-cost",
        type=float,
        help="chart growth cost ceiling, lower means more charts, default: 2.0",
    )
    group.add_argument(
        "--xatlas-path", type=Path, help="default: local engines/ binary"
    )


XATLAS_FLAGS = {"--max-cost": "max_cost", "--xatlas-path": "xatlas_path"}


def _validate_xatlas(args):
    if args.max_cost is not None and args.max_cost <= 0:
        raise UnwrapError(EXIT_INVALID_INPUT, "--max-cost must be positive")


def run_xatlas(args, pairs):
    from . import xatlas

    def unwrap_one(input_path, output_path):
        xatlas.run(input_path, output_path, args.xatlas_path, args.max_cost)

    return unwrap_all(pairs, unwrap_one)


def _add_partuv_args(group):
    group.add_argument(
        "--threshold", type=float, help="distortion threshold, default: 1.25"
    )
    group.add_argument(
        "--segmentation",
        choices=["ai", "geometric"],
        help="part segmentation, default: ai",
    )
    group.add_argument(
        "--checkpoint",
        type=Path,
        help="PartField model checkpoint, default: $UVGAMI_PARTUV_CHECKPOINT,"
        " then the repo checkpoint at engine/partuv/model_objaverse.ckpt",
    )
    group.add_argument("--config", type=Path, help="default: packaged config.yaml")


# flags that only apply to partuv, so the other engine can reject them
PARTUV_FLAGS = {
    "--threshold": "threshold",
    "--segmentation": "segmentation",
    "--checkpoint": "checkpoint",
    "--config": "config",
}


def _validate_partuv(args):
    args.segmentation = args.segmentation or "ai"
    if args.segmentation == "geometric" and args.checkpoint is not None:
        raise UnwrapError(
            EXIT_INVALID_INPUT, "--checkpoint only applies to --segmentation ai"
        )
    args.threshold = args.threshold if args.threshold is not None else 1.25


def run_partuv(args, pairs):
    try:
        import partuv.cli
        import partuv.common
    except ImportError as error:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            f"the PartUV engine is not installed ({error});"
            " install it with: uv sync --extra partuv",
        ) from error
    # partuv raises its own UnwrapError class, re-raise as ours so main's
    # handler and the --json error path keep working (exit codes match)
    try:
        checkpoint = None
        if args.segmentation == "ai":
            checkpoint = args.checkpoint
            if checkpoint is None and "UVGAMI_PARTUV_CHECKPOINT" not in os.environ:
                # the editable install serves partuv from site-packages, so its
                # package-relative default checkpoint misses the source tree
                repo_checkpoint = (
                    Path(__file__).parents[2]
                    / "engine"
                    / "partuv"
                    / "model_objaverse.ckpt"
                )
                if repo_checkpoint.is_file():
                    checkpoint = repo_checkpoint
            checkpoint = partuv.cli.resolve_checkpoint(checkpoint)
        return partuv.cli.run(
            pairs, checkpoint, args.config, args.threshold, args.segmentation
        )
    except partuv.common.UnwrapError as error:
        raise UnwrapError(error.exit_code, str(error)) from error


ENGINE_SPECS = {
    "optcuts": EngineSpec(
        name="optcuts",
        add_args=_add_optcuts_args,
        flags=OPTCUTS_FLAGS,
        validate=_validate_optcuts,
        run=run_optcuts,
    ),
    "xatlas": EngineSpec(
        name="xatlas",
        add_args=_add_xatlas_args,
        flags=XATLAS_FLAGS,
        validate=_validate_xatlas,
        run=run_xatlas,
    ),
    "partuv": EngineSpec(
        name="partuv",
        add_args=_add_partuv_args,
        flags=PARTUV_FLAGS,
        validate=_validate_partuv,
        run=run_partuv,
    ),
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="uvgami",
        description="UV unwrap OBJ files with the OptCuts, xatlas or PartUV engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    unwrap = subparsers.add_parser("unwrap", help="unwrap OBJ files")
    unwrap.add_argument("input", type=Path, nargs="+", help="input OBJ files")
    unwrap.add_argument(
        "--engine",
        choices=list(ENGINE_SPECS),
        default="optcuts",
        help="unwrapping engine, default: optcuts",
    )
    unwrap.add_argument(
        "-o",
        "--output",
        type=Path,
        action="append",
        help="output file, repeat once per input, default: <input stem>_uv.obj",
    )
    unwrap.add_argument(
        "--output-dir",
        type=Path,
        help="write each output as <input stem>.obj in this directory",
    )
    unwrap.add_argument(
        "--overwrite", action="store_true", help="replace existing output"
    )
    unwrap.add_argument(
        "--json",
        action="store_true",
        help="print a JSON result on stdout (single input only)",
    )

    for spec in ENGINE_SPECS.values():
        group = unwrap.add_argument_group(f"{spec.name} options")
        spec.add_args(group)
    return parser


def validate(args):
    for input_path in args.input:
        # in a batch a missing input fails per mesh instead, so a cancelled
        # mesh (its input file is deleted) doesn't abort the rest
        if len(args.input) == 1 and not input_path.is_file():
            raise UnwrapError(EXIT_INVALID_INPUT, f"input not found: {input_path}")
        if input_path.suffix.lower() != ".obj":
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"input must be an OBJ file: {input_path}"
            )
    if args.json and len(args.input) > 1:
        raise UnwrapError(EXIT_INVALID_INPUT, "--json only supports a single input")
    if args.output and args.output_dir:
        raise UnwrapError(
            EXIT_INVALID_INPUT, "-o and --output-dir are mutually exclusive"
        )
    if args.output and len(args.output) != len(args.input):
        raise UnwrapError(EXIT_INVALID_INPUT, "-o must be given once per input")
    if args.output_dir is not None:
        args.outputs = [args.output_dir / f"{p.stem}.obj" for p in args.input]
    elif args.output:
        args.outputs = list(args.output)
    else:
        args.outputs = [p.with_name(f"{p.stem}_uv.obj") for p in args.input]
    if len(set(args.outputs)) != len(args.outputs):
        raise UnwrapError(EXIT_INVALID_INPUT, "output paths collide, rename the inputs")
    for output_path in args.outputs:
        if output_path.exists() and not args.overwrite:
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"output exists (use --overwrite): {output_path}"
            )

    # reject the other engines' flags when explicitly passed, before defaults
    for name, spec in ENGINE_SPECS.items():
        if name == args.engine:
            continue
        for flag, attr in spec.flags.items():
            if getattr(args, attr) is not None:
                raise UnwrapError(
                    EXIT_INVALID_INPUT,
                    f"{flag} is not valid with --engine {args.engine}",
                )

    ENGINE_SPECS[args.engine].validate(args)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        pairs = list(zip(args.input, args.outputs))
        start = time.perf_counter()
        code = ENGINE_SPECS[args.engine].run(args, pairs)
        elapsed = time.perf_counter() - start
    except UnwrapError as error:
        log(f"error: {error}", style="error")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "exit_code": error.exit_code,
                        "message": str(error),
                    }
                )
            )
        return error.exit_code

    if len(pairs) == 1:
        log(f"wrote {args.outputs[0]} in {elapsed:.1f}s", style="success")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "engine": args.engine,
                        "input": str(args.input[0]),
                        "output": str(args.outputs[0]),
                        "seconds": round(elapsed, 2),
                    }
                )
            )
    else:
        log(
            f"batch finished in {elapsed:.1f}s, {code.ok} ok, {code.failed} failed",
            style="error" if code.failed else "success",
        )
    return code
