import sys
import types
from pathlib import Path

import pytest
from partuv import cli
from partuv.common import UnwrapError


# windows runs native, so only non-windows non-linux errors here
def test_requires_linux(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        cli.run([(triangle, tmp_path / "out.obj")], checkpoint, None, 1.25)
    assert error.value.exit_code == 3


def test_resolve_checkpoint_precedence(tmp_path, monkeypatch):
    flagged = tmp_path / "flag.ckpt"
    monkeypatch.setenv("UVGAMI_PARTUV_CHECKPOINT", "/root/env.ckpt")
    assert cli.resolve_checkpoint(flagged) == str(flagged)
    assert cli.resolve_checkpoint(None) == "/root/env.ckpt"

    monkeypatch.delenv("UVGAMI_PARTUV_CHECKPOINT")
    default = tmp_path / "model_objaverse.ckpt"
    monkeypatch.setattr(cli, "DEFAULT_CHECKPOINT", default)
    with pytest.raises(UnwrapError) as error:
        cli.resolve_checkpoint(None)
    assert error.value.exit_code == 2
    default.write_text("ckpt")
    assert cli.resolve_checkpoint(None) == str(default)


def test_missing_checkpoint(triangle, tmp_path, monkeypatch, fake_partuv_runtime):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    with pytest.raises(UnwrapError) as error:
        cli.run(
            [(triangle, tmp_path / "out.obj")], tmp_path / "nope.ckpt", config, 1.25
        )
    assert error.value.exit_code == 3


def test_missing_config(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        cli.run(
            [(triangle, tmp_path / "out.obj")], checkpoint, tmp_path / "nope.yaml", 1.25
        )
    assert error.value.exit_code == 3


def test_input_list_resolves_same_pairs_as_positional(triangle, cube, tmp_path):
    out = tmp_path / "out"
    list_file = tmp_path / "inputs.txt"
    list_file.write_text(f"{triangle}\n{cube}\n", encoding="utf-8")

    seg = ["--segmentation", "geometric"]
    from_list = cli.build_parser().parse_args(
        ["--input-list", str(list_file), "--output-dir", str(out), *seg]
    )
    cli.validate(from_list)
    positional = cli.build_parser().parse_args(
        [str(triangle), str(cube), "--output-dir", str(out), *seg]
    )
    cli.validate(positional)

    assert from_list.input == positional.input
    assert from_list.outputs == positional.outputs


def test_input_list_combines_with_positional(triangle, cube, tmp_path):
    list_file = tmp_path / "inputs.txt"
    list_file.write_text(f"{cube}\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            str(triangle),
            "--input-list",
            str(list_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--segmentation",
            "geometric",
        ]
    )
    cli.validate(args)
    assert args.input == [triangle, cube]


def test_input_list_skips_blank_lines(triangle, cube, tmp_path):
    list_file = tmp_path / "inputs.txt"
    list_file.write_text(f"\n{triangle}\n   \n{cube}\n\t\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "--input-list",
            str(list_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--segmentation",
            "geometric",
        ]
    )
    cli.validate(args)
    assert args.input == [triangle, cube]


def test_input_list_missing_file_exits_2(tmp_path):
    assert cli.main(["--input-list", str(tmp_path / "nope.txt")]) == 2


def test_input_list_scales_past_command_line_limit(tmp_path):
    # ~491 mesh paths as argv exceed windows' 32k command-line cap
    out = tmp_path / "out"
    paths = [tmp_path / f"{'x' * 200}_{i}.obj" for i in range(500)]
    list_file = tmp_path / "inputs.txt"
    list_file.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "--input-list",
            str(list_file),
            "--output-dir",
            str(out),
            "--segmentation",
            "geometric",
        ]
    )
    cli.validate(args)
    assert len(args.input) == 500
    assert len(args.outputs) == 500


class FakeMesh:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    faces = [[0, 1, 2]]


@pytest.fixture
def fake_partuv_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    # a machine with a gpu, the cpu-fallback tests override this
    monkeypatch.setattr(cli, "_cuda_available", lambda: True)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    calls = {}

    core_module = types.ModuleType("partuv._core")
    preprocess_module = types.ModuleType("partuv.preprocess")
    output_module = types.ModuleType("partuv.output")
    geometric_module = types.ModuleType("partuv.geometric")

    class FakeModel:
        def __init__(self, checkpoint_path, device):
            calls["model"] = (checkpoint_path, device)
            calls["model_loads"] = calls.get("model_loads", 0) + 1

    def fake_preprocess(mesh_path, pf_model=None, output_path=None):
        calls["preprocess"] = mesh_path
        return FakeMesh(), None, {"tree": 1}, {}

    def fake_preprocess_geometric(mesh_path):
        calls["geometric"] = mesh_path
        return FakeMesh(), {"tree": 2}

    def fake_pipeline_numpy(V, F, tree_dict, config_path, threshold, visual=False):
        calls["pipeline"] = (tree_dict, config_path, threshold)
        # read now: a cpu-fallback temp config is deleted when run() returns
        calls["config_text"] = Path(config_path).read_text()
        calls["visual"] = visual
        return "final", ["part0"]

    def fake_save_results(output_dir, final_part, individual_parts, source_V, source_F):
        calls["save"] = (final_part, individual_parts)
        result = output_dir / "final_components.obj"
        result.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    core_module.pipeline_numpy = fake_pipeline_numpy
    preprocess_module.preprocess = fake_preprocess
    preprocess_module.PFInferenceModel = FakeModel
    output_module.save_results = fake_save_results
    geometric_module.preprocess_geometric = fake_preprocess_geometric
    monkeypatch.setitem(sys.modules, "partuv._core", core_module)
    monkeypatch.setitem(sys.modules, "partuv.preprocess", preprocess_module)
    monkeypatch.setitem(sys.modules, "partuv.output", output_module)
    monkeypatch.setitem(sys.modules, "partuv.geometric", geometric_module)
    return calls


def test_run_orchestration(triangle, tmp_path, fake_partuv_runtime):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    cli.run([(triangle, output)], checkpoint, config, 1.5)

    calls = fake_partuv_runtime
    assert calls["model"] == (str(checkpoint), "cuda")
    assert calls["preprocess"] == str(triangle)
    assert calls["pipeline"] == ({"tree": 1}, str(config), 1.5)
    assert calls["visual"] is False
    assert calls["save"] == ("final", ["part0"])
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_geometric_skips_torch_and_checkpoint(
    triangle, tmp_path, fake_partuv_runtime
):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    cli.run([(triangle, output)], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    assert calls["geometric"] == str(triangle)
    assert "model" not in calls
    assert calls["pipeline"] == ({"tree": 2}, str(config), 1.25)
    assert output.is_file()


def test_geometric_cpu_fallback_disables_pamo(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(cli, "_cuda_available", lambda: False)
    config = tmp_path / "config.yaml"
    config.write_text("unwrap:\n  pamo: true\n  usePamoFaceThreshold: 1000\n")

    cli.run([(triangle, tmp_path / "out.obj")], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    # pipeline sees a rewritten temp config, not the original
    assert calls["pipeline"][1] != str(config)
    assert "pamo: false" in calls["config_text"]
    assert "pamo: true" not in calls["config_text"]


def test_geometric_gpu_keeps_config(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(cli, "_cuda_available", lambda: True)
    config = tmp_path / "config.yaml"
    config.write_text("unwrap:\n  pamo: true\n")

    cli.run([(triangle, tmp_path / "out.obj")], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    assert calls["pipeline"][1] == str(config)
    assert "pamo: true" in calls["config_text"]


def test_ai_never_probes_cuda(triangle, tmp_path, fake_partuv_runtime, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("unwrap:\n  pamo: true\n")
    probed = []
    monkeypatch.setattr(cli, "_cuda_available", lambda: probed.append(True) or True)

    cli.run([(triangle, tmp_path / "out.obj")], checkpoint, config, 1.5)

    calls = fake_partuv_runtime
    # torch.cuda already decides the ai path, so the driver probe is not used
    assert probed == []
    assert calls["pipeline"][1] == str(config)
    assert "pamo: true" in calls["config_text"]


def test_visual_reaches_pipeline(triangle, tmp_path, fake_partuv_runtime):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")

    cli.run([(triangle, tmp_path / "out.obj")], None, config, 1.25, "geometric", True)

    assert fake_partuv_runtime["visual"] is True


def test_batch_loads_model_once(triangle, cube, tmp_path, fake_partuv_runtime, capsys):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    pairs = [(triangle, tmp_path / "a.obj"), (cube, tmp_path / "b.obj")]

    code = cli.run(pairs, checkpoint, config, 1.5)

    assert code == 0
    assert fake_partuv_runtime["model_loads"] == 1
    assert (tmp_path / "a.obj").is_file()
    assert (tmp_path / "b.obj").is_file()
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["start: triangle", "done: triangle", "start: cube", "done: cube"]


def test_windows_runs_native_when_partuv_installed(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    # no import error means the compiled core loaded
    monkeypatch.setattr("partuv._CORE_ERROR", None)
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    cli.run([(triangle, output)], None, config, 1.25, "geometric")

    assert fake_partuv_runtime["geometric"] == str(triangle)
    assert output.is_file()


def test_windows_reinstall_hint_when_native_fails(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.setattr("partuv._CORE_ERROR", ImportError("msvcp140 missing"))

    with pytest.raises(UnwrapError) as error:
        cli.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric")

    assert error.value.exit_code == 3
    message = str(error.value)
    assert "reinstall PartUV" in message
    assert "msvcp140 missing" in message


def test_preprocess_module_imports():
    # torch is the ai extra, not synced by default
    pytest.importorskip("torch")
    from importlib import import_module

    assert callable(import_module("partuv.preprocess").preprocess)
