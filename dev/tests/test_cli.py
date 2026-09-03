import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from uvgami_cli import cli, optcuts
from uvgami_cli.common import UnwrapError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_optcuts(monkeypatch):
    calls = []

    def fake_run(*args):
        calls.append(args)
        output_path = args[1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(optcuts, "run", fake_run)
    return calls


@pytest.fixture
def fake_xatlas(monkeypatch):
    from uvgami_cli import xatlas

    calls = []

    def fake_run(*args):
        calls.append(args)
        output_path = args[1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(xatlas, "run", fake_run)
    return calls


@pytest.fixture
def fake_partuv(monkeypatch):
    import partuv.cli

    calls = {}

    def fake_run(pairs, checkpoint, config, threshold, segmentation="ai"):
        calls.update(
            pairs=pairs,
            checkpoint=checkpoint,
            config=config,
            threshold=threshold,
            segmentation=segmentation,
        )
        for _, output_path in pairs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")
        return 0

    monkeypatch.setattr(partuv.cli, "run", fake_run)
    monkeypatch.setattr(
        partuv.cli,
        "resolve_checkpoint",
        lambda flag: str(flag) if flag else "fake.ckpt",
    )
    return calls


def test_default_output_name(triangle, fake_optcuts):
    assert cli.main(["unwrap", str(triangle)]) == 0
    assert (triangle.parent / "triangle_uv.obj").is_file()


def test_explicit_output(triangle, tmp_path, fake_optcuts):
    out = tmp_path / "result" / "custom.obj"
    assert cli.main(["unwrap", str(triangle), "-o", str(out)]) == 0
    assert fake_optcuts[0][1] == out


def test_optcuts_defaults(triangle, fake_optcuts):
    cli.main(["unwrap", str(triangle)])
    (
        _,
        _,
        quality,
        import_uvs,
        seam_weights,
        seam_weight,
        engine_path,
        timeout,
    ) = fake_optcuts[0]
    assert quality == "balanced"
    assert import_uvs is False
    assert seam_weights is None
    assert seam_weight == 3
    assert engine_path is None
    assert timeout is None


def test_missing_input(tmp_path, fake_optcuts):
    code = cli.main(["unwrap", str(tmp_path / "nope.obj")])
    assert code == 2


def test_non_obj_input(tmp_path, fake_optcuts):
    bad = tmp_path / "mesh.stl"
    bad.write_text("solid")
    assert cli.main(["unwrap", str(bad)]) == 2


def test_overwrite_protection(triangle, fake_optcuts):
    existing = triangle.parent / "triangle_uv.obj"
    existing.write_text("v 0 0 0\n")
    assert cli.main(["unwrap", str(triangle)]) == 2
    assert cli.main(["unwrap", str(triangle), "--overwrite"]) == 0


def test_missing_seam_weights_file(triangle, tmp_path, fake_optcuts):
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--seam-weights",
            str(tmp_path / "nope_weights"),
        ]
    )
    assert code == 2


def test_json_success(triangle, fake_optcuts, capsys):
    assert cli.main(["unwrap", str(triangle), "--json"]) == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] == "ok"
    assert result["engine"] == "optcuts"
    assert result["output"].endswith("triangle_uv.obj")


def test_json_error(triangle, monkeypatch, capsys):
    def fail(*args):
        raise UnwrapError(4, "engine blew up")

    monkeypatch.setattr(optcuts, "run", fail)
    assert cli.main(["unwrap", str(triangle), "--json"]) == 4
    result = json.loads(capsys.readouterr().out)
    assert result == {"status": "error", "exit_code": 4, "message": "engine blew up"}


def test_stdout_empty_without_json(triangle, fake_optcuts, capsys):
    cli.main(["unwrap", str(triangle)])
    assert capsys.readouterr().out == ""


# the addon's optcuts path runs the cli as python -m uvgami_cli
def test_module_entry_point(triangle):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uvgami_cli",
            "unwrap",
            str(triangle),
            "--seam-weight",
            "not-a-number",
        ],
        capture_output=True,
        text=True,
    )
    # argparse rejects the seam weight before any engine import runs
    assert result.returncode == 2
    assert "invalid int value" in result.stderr


# the addon's partuv path runs the wheel as python -m partuv
def test_partuv_module_entry_point(triangle):
    env = os.environ.copy()
    engine = str(REPO_ROOT / "engine" / "partuv")
    env["PYTHONPATH"] = os.pathsep.join(
        [engine, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [engine]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "partuv",
            str(triangle),
            "--threshold",
            "not-a-number",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    # argparse rejects the threshold before the compiled core is touched
    assert result.returncode == 2
    assert "invalid float value" in result.stderr


def test_engine_error_code_passthrough(triangle, monkeypatch):
    def fail(*args):
        raise UnwrapError(5, "no output")

    monkeypatch.setattr(optcuts, "run", fail)
    assert cli.main(["unwrap", str(triangle)]) == 5


def test_batch_markers_and_output_dir(triangle, cube, tmp_path, fake_optcuts, capsys):
    out_dir = tmp_path / "out"
    code = cli.main(["unwrap", str(triangle), str(cube), "--output-dir", str(out_dir)])
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["start: triangle", "done: triangle", "start: cube", "done: cube"]
    assert fake_optcuts[0][1] == out_dir / "triangle.obj"
    assert fake_optcuts[1][1] == out_dir / "cube.obj"


def test_batch_isolates_failures(triangle, cube, monkeypatch, capsys):
    def run(input_path, output_path, *rest):
        if input_path.stem == "triangle":
            raise UnwrapError(4, "boom")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(optcuts, "run", run)
    code = cli.main(["unwrap", str(triangle), str(cube)])
    assert code == 4
    lines = capsys.readouterr().out.splitlines()
    assert "failed: triangle 4" in lines
    assert "done: cube" in lines
    assert (cube.parent / "cube_uv.obj").is_file()


def test_batch_isolates_unexpected_exceptions(triangle, cube, monkeypatch, capsys):
    def run(input_path, output_path, *rest):
        if input_path.stem == "cube":
            raise RuntimeError("segfault-ish")
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(optcuts, "run", run)
    code = cli.main(["unwrap", str(triangle), str(cube)])
    assert code == 4
    lines = capsys.readouterr().out.splitlines()
    assert "done: triangle" in lines
    assert "failed: cube 4" in lines


def test_single_input_emits_no_markers(triangle, tmp_path, fake_optcuts, capsys):
    out_dir = tmp_path / "out"
    code = cli.main(["unwrap", str(triangle), "--output-dir", str(out_dir)])
    assert code == 0
    assert capsys.readouterr().out == ""
    assert fake_optcuts[0][1] == out_dir / "triangle.obj"


def test_output_count_mismatch(triangle, cube, tmp_path, fake_optcuts):
    code = cli.main(["unwrap", str(triangle), str(cube), "-o", str(tmp_path / "x.obj")])
    assert code == 2
    assert not fake_optcuts


def test_output_and_output_dir_conflict(triangle, tmp_path, fake_optcuts):
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "-o",
            str(tmp_path / "x.obj"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 2


def test_json_rejected_for_multiple_inputs(triangle, cube, fake_optcuts):
    code = cli.main(["unwrap", str(triangle), str(cube), "--json"])
    assert code == 2


def test_batch_missing_input_fails_per_mesh(triangle, tmp_path, fake_optcuts, capsys):
    missing = tmp_path / "gone.obj"
    code = cli.main(["unwrap", str(triangle), str(missing)])
    assert code == 2
    lines = capsys.readouterr().out.splitlines()
    assert "done: triangle" in lines
    assert "failed: gone 2" in lines


def test_input_deleted_mid_batch_is_skipped(triangle, cube, tmp_path, capsys):
    """Cancelling a mesh in the add-on deletes its input file while the batch
    runs; the mesh must fail fast without aborting the rest."""
    from uvgami_cli.common import unwrap_all

    unwrapped = []

    def unwrap_one(input_path, output_path):
        # cancel the cube while the triangle is being unwrapped
        cube.unlink(missing_ok=True)
        unwrapped.append(input_path.stem)

    code = unwrap_all(
        [(triangle, tmp_path / "a.obj"), (cube, tmp_path / "b.obj")], unwrap_one
    )
    assert code == 2
    assert unwrapped == ["triangle"]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "start: triangle",
        "done: triangle",
        "start: cube",
        "failed: cube 2",
    ]


def test_partuv_dispatch(triangle, tmp_path, fake_partuv):
    out = tmp_path / "out.obj"
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--engine",
            "partuv",
            "-o",
            str(out),
            "--threshold",
            "1.5",
            "--segmentation",
            "geometric",
        ]
    )
    assert code == 0
    assert fake_partuv["pairs"] == [(triangle, out)]
    assert fake_partuv["threshold"] == 1.5
    assert fake_partuv["segmentation"] == "geometric"
    # geometric never resolves a checkpoint
    assert fake_partuv["checkpoint"] is None
    assert fake_partuv["config"] is None


def test_partuv_ai_resolves_checkpoint(triangle, tmp_path, monkeypatch, fake_partuv):
    # env var set so the CLI's repo-checkpoint fallback doesn't kick in here,
    # keeping this test independent of whether the real checkpoint is present
    monkeypatch.setenv("UVGAMI_PARTUV_CHECKPOINT", str(tmp_path / "env.ckpt"))
    config = tmp_path / "config.yaml"
    config.write_text("pamo: true\n")
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "partuv", "--config", str(config)]
    )
    assert code == 0
    assert fake_partuv["segmentation"] == "ai"
    assert fake_partuv["threshold"] == 1.25
    assert fake_partuv["checkpoint"] == "fake.ckpt"
    assert fake_partuv["config"] == config


def test_partuv_resolves_repo_checkpoint(triangle, monkeypatch, fake_partuv):
    monkeypatch.delenv("UVGAMI_PARTUV_CHECKPOINT", raising=False)
    repo_checkpoint = REPO_ROOT / "engine" / "partuv" / "model_objaverse.ckpt"
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda self: self == repo_checkpoint or original_is_file(self)
    )
    code = cli.main(["unwrap", str(triangle), "--engine", "partuv"])
    assert code == 0
    assert fake_partuv["checkpoint"] == str(repo_checkpoint)


def test_xatlas_dispatch(triangle, tmp_path, fake_xatlas, capsys):
    out = tmp_path / "out.obj"
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "xatlas", "-o", str(out), "--json"]
    )
    assert code == 0
    assert fake_xatlas[0][:2] == (triangle, out)
    # no engine path given, so the bundled binary is resolved inside run
    assert fake_xatlas[0][2] is None
    assert json.loads(capsys.readouterr().out)["engine"] == "xatlas"


def test_xatlas_path_flag_passed_through(triangle, tmp_path, fake_xatlas):
    engine = tmp_path / "xatlas.exe"
    engine.write_text("fake")
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "xatlas", "--xatlas-path", str(engine)]
    )
    assert code == 0
    assert fake_xatlas[0][2] == engine


def test_xatlas_max_cost_passed_through(triangle, fake_xatlas):
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "xatlas", "--max-cost", "0.8"]
    )
    assert code == 0
    assert fake_xatlas[0][3] == 0.8


def test_xatlas_max_cost_must_be_positive(triangle, fake_xatlas, capsys):
    code = cli.main(["unwrap", str(triangle), "--engine", "xatlas", "--max-cost", "0"])
    assert code == 2
    assert "--max-cost" in capsys.readouterr().err


@pytest.mark.parametrize(
    "engine, flag, value",
    [
        ("optcuts", "--max-cost", "0.8"),
        ("optcuts", "--xatlas-path", "xatlas.exe"),
        ("optcuts", "--threshold", "1.5"),
        ("xatlas", "--quality", "less-stretch"),
        ("partuv", "--quality", "less-stretch"),
    ],
)
def test_another_engines_flag_is_rejected(triangle, capsys, engine, flag, value):
    code = cli.main(["unwrap", str(triangle), "--engine", engine, flag, value])
    assert code == 2
    assert flag in capsys.readouterr().err


def test_checkpoint_rejected_for_geometric(triangle, tmp_path, capsys):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("x")
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--engine",
            "partuv",
            "--segmentation",
            "geometric",
            "--checkpoint",
            str(ckpt),
        ]
    )
    assert code == 2
    assert "--checkpoint" in capsys.readouterr().err


def test_partuv_error_maps_exit_code(triangle, monkeypatch, capsys):
    import partuv.cli
    from partuv.common import UnwrapError as PartuvError

    def boom(flag):
        raise PartuvError(3, "no checkpoint")

    monkeypatch.setattr(partuv.cli, "resolve_checkpoint", boom)
    code = cli.main(["unwrap", str(triangle), "--engine", "partuv", "--json"])
    assert code == 3
    result = json.loads(capsys.readouterr().out)
    assert result == {"status": "error", "exit_code": 3, "message": "no checkpoint"}


def test_json_success_reports_partuv_engine(triangle, fake_partuv, capsys):
    assert cli.main(["unwrap", str(triangle), "--engine", "partuv", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["engine"] == "partuv"


def test_colliding_outputs_rejected(triangle, tmp_path, fake_optcuts):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / "triangle.obj"
    other.write_bytes(triangle.read_bytes())
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            str(other),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
