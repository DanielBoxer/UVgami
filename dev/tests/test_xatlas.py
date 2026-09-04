import subprocess
from pathlib import Path

import pytest
from uvgami_cli import xatlas
from uvgami_cli.common import REPO_ROOT, UnwrapError, find_engine

BUNDLED = REPO_ROOT / "engine-builds" / "windows" / "xatlas.exe"


class FakeProcess:
    def __init__(
        self, argv, returncode=0, output_text="v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n"
    ):
        self.argv = argv
        self.returncode = returncode
        self.output_text = output_text
        self.stdout = iter(["start: triangle\n", "progress: 0.50 0 0.50\n"])

    def wait(self):
        if self.returncode == 0:
            Path(self.argv[self.argv.index("-o") + 1]).write_text(self.output_text)
        return self.returncode


@pytest.fixture
def fake_engine(tmp_path):
    path = tmp_path / "xatlas.exe"
    path.write_text("fake")
    return path


def popen_recorder(monkeypatch, **process_kwargs):
    calls = []

    def fake_popen(argv, **kwargs):
        process = FakeProcess(argv, **process_kwargs)
        calls.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_build_args(fake_engine):
    args = xatlas.build_args(fake_engine, Path("in.obj"), Path("out.obj"))
    assert args[args.index("-i") + 1] == "in.obj"
    assert args[args.index("-o") + 1] == "out.obj"


def test_build_args_omits_max_cost_by_default(fake_engine):
    args = xatlas.build_args(fake_engine, Path("in.obj"), Path("out.obj"))
    assert "--max-cost" not in args


def test_build_args_max_cost(fake_engine):
    args = xatlas.build_args(fake_engine, Path("in.obj"), Path("out.obj"), 0.75)
    assert float(args[args.index("--max-cost") + 1]) == 0.75


def test_run_passes_max_cost(triangle, tmp_path, fake_engine, monkeypatch):
    calls = popen_recorder(monkeypatch)
    xatlas.run(triangle, tmp_path / "out.obj", fake_engine, 3.5)
    argv = calls[0].argv
    assert float(argv[argv.index("--max-cost") + 1]) == 3.5


def test_run_success(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch)
    output = tmp_path / "result.obj"
    xatlas.run(triangle, output, fake_engine)
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_writes_to_temp_not_output(triangle, tmp_path, fake_engine, monkeypatch):
    calls = popen_recorder(monkeypatch)
    output = tmp_path / "result.obj"
    xatlas.run(triangle, output, fake_engine)
    argv = calls[0].argv
    assert Path(argv[argv.index("-o") + 1]) != output


def test_run_engine_failure(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch, returncode=4)
    with pytest.raises(UnwrapError) as error:
        xatlas.run(triangle, tmp_path / "out.obj", fake_engine)
    assert error.value.exit_code == 4


def test_run_output_missing_uvs(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch, output_text="v 0 0 0\nf 1 1 1\n")
    output = tmp_path / "out.obj"
    with pytest.raises(UnwrapError) as error:
        xatlas.run(triangle, output, fake_engine)
    assert error.value.exit_code == 5
    # a rejected result must not land on the requested output
    assert not output.exists()


def test_find_engine_explicit_path_missing(tmp_path):
    with pytest.raises(UnwrapError) as error:
        find_engine("xatlas", "xatlas", tmp_path / "nope.exe")
    assert error.value.exit_code == 3


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled xatlas binary not present")
def test_xatlas_smoke(cube, tmp_path):
    output = tmp_path / "cube_uv.obj"
    xatlas.run(cube, output, None)
    lines = output.read_text().splitlines()
    positions = [line for line in lines if line.startswith("v ")]
    faces = [line for line in lines if line.startswith("f ")]
    uvs = [line for line in lines if line.startswith("vt ")]
    # blender's seams-from-islands finds nothing unless the mesh stays welded
    assert len(positions) == 8
    assert len(faces) == 12
    assert uvs
    for line in uvs:
        u, v = (float(value) for value in line.split()[1:3])
        assert 0.0 <= u <= 1.0
        assert 0.0 <= v <= 1.0

    # a welded vertex on a seam carries more than one uv
    uv_indices = {}
    for line in faces:
        for token in line.split()[1:]:
            position, uv = token.split("/")
            uv_indices.setdefault(position, set()).add(uv)
    assert any(len(indices) > 1 for indices in uv_indices.values())
