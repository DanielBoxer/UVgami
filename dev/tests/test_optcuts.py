import math
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from uvgami_cli import optcuts
from uvgami_cli.common import REPO_ROOT, UnwrapError, find_engine

BUNDLED = REPO_ROOT / "engine-builds" / "windows" / "optcuts.exe"
FIXTURES = Path(__file__).parent / "fixtures"


class FakeProcess:
    def __init__(
        self, argv, returncode=0, output_text="v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n"
    ):
        self.argv = argv
        self.returncode = returncode
        self.output_text = output_text
        self.stdout = iter(["progress: 0 0 1\n"])

    def wait(self):
        input_path = Path(self.argv[self.argv.index("-i") + 1])
        # captured here because the workdir is deleted after run() returns
        self.sidecar_existed = (
            input_path.parent / f"{input_path.stem}_weights"
        ).is_file()
        if self.returncode == 0:
            out_dir = Path(self.argv[self.argv.index("-o") + 1])
            (out_dir / f"{input_path.stem}.obj").write_text(self.output_text)
        return self.returncode


@pytest.fixture
def fake_engine(tmp_path):
    path = tmp_path / "optcuts.exe"
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


def test_build_args_quality_and_weight_mapping(fake_engine):
    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "less-stretch", 5, False
    )
    assert args[args.index("-u") + 1] == "4.05"
    assert args[args.index("-s") + 1] == "200"
    assert "-g" in args

    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "fewer-seams", 1, True
    )
    assert args[args.index("-u") + 1] == "5.0"
    assert args[args.index("-s") + 1] == "25"
    assert "-g" not in args


def test_output_dir_arg_has_trailing_separator(fake_engine):
    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "balanced", 3, False
    )
    out_arg = args[args.index("-o") + 1]
    assert out_arg.endswith(("/", "\\"))


def test_run_success(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch)
    output = tmp_path / "result.obj"
    optcuts.run(triangle, output, "balanced", False, None, 3, fake_engine)
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_copies_weights_sidecar(triangle, tmp_path, fake_engine, monkeypatch):
    calls = popen_recorder(monkeypatch)
    weights = tmp_path / "weights.txt"
    weights.write_text("0,1.0")
    optcuts.run(
        triangle, tmp_path / "out.obj", "balanced", False, weights, 3, fake_engine
    )
    argv = calls[0].argv
    assert Path(argv[argv.index("-i") + 1]).name == "triangle.obj"
    assert calls[0].sidecar_existed


def test_run_copies_sidecars_next_to_the_input(
    triangle, tmp_path, fake_engine, monkeypatch
):
    seen = []

    def fake_popen(argv, **kwargs):
        input_path = Path(argv[argv.index("-i") + 1])
        seen.append(sorted(p.name for p in input_path.parent.iterdir()))
        return FakeProcess(argv)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    (tmp_path / "triangle_fixed").write_text("0,1")

    optcuts.run(triangle, tmp_path / "out.obj", "balanced", False, None, 3, fake_engine)
    assert "triangle_fixed" in seen[0]


def test_run_engine_failure(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch, returncode=7)
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle, tmp_path / "out.obj", "balanced", False, None, 3, fake_engine
        )
    assert error.value.exit_code == 4


def test_run_timeout_kills_the_engine(triangle, tmp_path, fake_engine, monkeypatch):
    class HangingProcess:
        def __init__(self, argv, **kwargs):
            self.returncode = 1
            self.killed = threading.Event()
            self.stdout = self._lines()

        def _lines(self):
            assert self.killed.wait(10), "the timer never killed the engine"
            yield from ()

        def kill(self):
            self.killed.set()

        def wait(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", HangingProcess)
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle,
            tmp_path / "out.obj",
            "balanced",
            False,
            None,
            3,
            fake_engine,
            0.05,
        )
    assert error.value.exit_code == 4
    assert "timed out" in str(error.value)


def test_run_output_missing_uvs(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch, output_text="v 0 0 0\nf 1 1 1\n")
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle, tmp_path / "out.obj", "balanced", False, None, 3, fake_engine
        )
    assert error.value.exit_code == 5


def test_find_engine_explicit_path_missing(tmp_path):
    with pytest.raises(UnwrapError) as error:
        find_engine("optcuts", "OptCuts", tmp_path / "nope.exe")
    assert error.value.exit_code == 3


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_smoke(cube, tmp_path):
    output = tmp_path / "cube_uv.obj"
    optcuts.run(cube, output, "fewer-seams", False, None, 3, None)
    text = output.read_text()
    assert "vt " in text
    assert "f " in text


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_shared_process_matches_solo(tmp_path):
    input_dir = tmp_path / "input"
    solo_dir = tmp_path / "solo"
    shared_dir = tmp_path / "shared"
    for directory in (input_dir, solo_dir, shared_dir):
        directory.mkdir()
    cube = input_dir / "cube.obj"
    shutil.copyfile(FIXTURES / "cube.obj", cube)
    patch = input_dir / "patch.obj"
    _write_bump_patch(patch)
    # the pinned patch first, its pins must not leak into the cube
    for obj in (patch, cube):
        subprocess.run(
            [str(BUNDLED), "-i", str(obj), "-o", str(solo_dir) + os.sep],
            check=True,
            timeout=300,
            capture_output=True,
        )
    result = subprocess.run(
        [str(BUNDLED), "-o", str(shared_dir) + os.sep],
        input=f"unwrap {patch}\nunwrap {cube}\nunwrap {input_dir / 'none.obj'}\n",
        text=True,
        timeout=300,
        capture_output=True,
    )
    assert result.returncode == 0
    markers = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("start:", "done:", "failed:"))
    ]
    assert markers == [
        "start: patch",
        "done: patch",
        "start: cube",
        "done: cube",
        "start: none",
        "failed: none 103",
    ]
    for obj in (patch, cube):
        assert (shared_dir / obj.name).read_bytes() == (
            solo_dir / obj.name
        ).read_bytes()


def _write_bump_patch(path, n=12, height=0.3, width=0.05):
    verts = []
    border = []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = i / n, j / n
            z = height * math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / width)
            verts.append((x, y, z))
            if i in (0, n) or j in (0, n):
                border.append(j * (n + 1) + i)
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2))
            faces.append((a, a + n + 2, a + n + 1))
    with path.open("w") as f:
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        for x, y, _ in verts:
            f.write(f"vt {x} {y}\n")
        for face in faces:
            f.write("f " + " ".join(f"{c + 1}/{c + 1}" for c in face) + "\n")
    (path.parent / f"{path.stem}_fixed").write_text(",".join(map(str, border)))
    return verts, faces, border


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_pinned_border(tmp_path):
    input_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    input_dir.mkdir()
    out_dir.mkdir()
    obj = input_dir / "patch.obj"
    verts, faces, border = _write_bump_patch(obj)

    subprocess.run(
        [str(BUNDLED), "-i", str(obj), "-o", str(out_dir) + os.sep],
        check=True,
        timeout=300,
        capture_output=True,
    )

    uvs_out = []
    faces_out = []
    for line in (out_dir / "patch.obj").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            uvs_out.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "f":
            faces_out.append(tuple(int(c.split("/")[1]) - 1 for c in parts[1:]))
    assert len(faces_out) == len(faces)

    # face order survives the engine, map input vts to output vts through it
    vt_map = {}
    for face_in, face_out in zip(faces, faces_out):
        for ti, to in zip(face_in, face_out):
            vt_map[ti] = to

    # the output is uniformly scaled into the unit box, undo it via two pins
    pin_pairs = [((verts[t][0], verts[t][1]), uvs_out[vt_map[t]]) for t in border]
    lo = min(pin_pairs, key=lambda p: p[0][0])
    hi = max(pin_pairs, key=lambda p: p[0][0])
    scale = (hi[0][0] - lo[0][0]) / (hi[1][0] - lo[1][0])
    du = lo[0][0] - scale * lo[1][0]
    dv = lo[0][1] - scale * lo[1][1]
    residual = max(
        max(abs(scale * out[0] + du - old[0]), abs(scale * out[1] + dv - old[1]))
        for old, out in pin_pairs
    )
    assert residual < 1e-6


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_nocut_relaxes_without_cutting(tmp_path):
    input_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    input_dir.mkdir()
    out_dir.mkdir()
    obj = input_dir / "patch.obj"
    verts, faces, border = _write_bump_patch(obj, n=20, height=1.5, width=0.01)
    sidecar = input_dir / "patch_fixed"
    sidecar.write_text(sidecar.read_text() + "\nnocut")

    subprocess.run(
        [str(BUNDLED), "-i", str(obj), "-o", str(out_dir) + os.sep],
        check=True,
        timeout=300,
        capture_output=True,
    )

    uvs_out = []
    faces_out = []
    for line in (out_dir / "patch.obj").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            uvs_out.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "f":
            faces_out.append(tuple(int(c.split("/")[1]) - 1 for c in parts[1:]))
    assert len(faces_out) == len(faces)
    # a cut duplicates the uv verts along it
    assert len(uvs_out) == len(verts)

    vt_map = {}
    for face_in, face_out in zip(faces, faces_out):
        for ti, to in zip(face_in, face_out):
            vt_map[ti] = to

    pin_pairs = [((verts[t][0], verts[t][1]), uvs_out[vt_map[t]]) for t in border]
    lo = min(pin_pairs, key=lambda p: p[0][0])
    hi = max(pin_pairs, key=lambda p: p[0][0])
    scale = (hi[0][0] - lo[0][0]) / (hi[1][0] - lo[1][0])
    du = lo[0][0] - scale * lo[1][0]
    dv = lo[0][1] - scale * lo[1][1]
    residual = max(
        max(abs(scale * out[0] + du - old[0]), abs(scale * out[1] + dv - old[1]))
        for old, out in pin_pairs
    )
    assert residual < 1e-6

    # the flat grid map is distorted over the bump, relaxing must move it
    interior = set(range(len(verts))) - set(border)
    moved = max(
        max(
            abs(scale * uvs_out[vt_map[t]][0] + du - verts[t][0]),
            abs(scale * uvs_out[vt_map[t]][1] + dv - verts[t][1]),
        )
        for t in interior
    )
    assert moved > 1e-3


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_nocut_without_pins(tmp_path):
    input_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    input_dir.mkdir()
    out_dir.mkdir()
    obj = input_dir / "patch.obj"
    verts, faces, _ = _write_bump_patch(obj, n=20, height=1.5, width=0.01)
    (input_dir / "patch_fixed").write_text("\nnocut")

    subprocess.run(
        [str(BUNDLED), "-i", str(obj), "-o", str(out_dir) + os.sep],
        check=True,
        timeout=300,
        capture_output=True,
    )

    uvs_out = []
    faces_out = []
    for line in (out_dir / "patch.obj").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            uvs_out.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "f":
            faces_out.append(tuple(int(c.split("/")[1]) - 1 for c in parts[1:]))
    assert len(faces_out) == len(faces)
    # a cut duplicates the uv verts along it
    assert len(uvs_out) == len(verts)

    # input is the unit grid and the output is normalized into the unit box
    vt_map = {}
    for face_in, face_out in zip(faces, faces_out):
        for ti, to in zip(face_in, face_out):
            vt_map[ti] = to
    moved = max(
        max(
            abs(uvs_out[vt_map[t]][0] - verts[t][0]),
            abs(uvs_out[vt_map[t]][1] - verts[t][1]),
        )
        for t in range(len(verts))
    )
    assert moved > 1e-3


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_nocut_keeps_hole_chart(tmp_path):
    n = 8
    input_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    input_dir.mkdir()
    out_dir.mkdir()
    obj = input_dir / "ring.obj"

    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = i / n, j / n
            z = 0.3 * math.exp(-((x - 0.2) ** 2 + (y - 0.2) ** 2) / 0.05)
            verts.append((x, y, z))
    hole = {n // 2 - 1, n // 2}
    faces = []
    for j in range(n):
        for i in range(n):
            if i in hole and j in hole:
                continue
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2))
            faces.append((a, a + n + 2, a + n + 1))
    # the vert inside the hole is unreferenced, drop it so vt counts compare
    used = sorted({v for face in faces for v in face})
    local = {v: i for i, v in enumerate(used)}
    verts = [verts[v] for v in used]
    faces = [tuple(local[v] for v in face) for face in faces]
    with obj.open("w") as f:
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        for x, y, _ in verts:
            f.write(f"vt {x} {y}\n")
        for face in faces:
            f.write("f " + " ".join(f"{c + 1}/{c + 1}" for c in face) + "\n")
    (input_dir / "ring_fixed").write_text("\nnocut")

    subprocess.run(
        [str(BUNDLED), "-i", str(obj), "-o", str(out_dir) + os.sep],
        check=True,
        timeout=300,
        capture_output=True,
    )

    out_text = (out_dir / "ring.obj").read_text()
    faces_out = [ln for ln in out_text.splitlines() if ln.startswith("f ")]
    vts_out = [ln for ln in out_text.splitlines() if ln.startswith("vt ")]
    assert len(faces_out) == len(faces)
    # cutting the hole open to a disk would duplicate uv verts
    assert len(vts_out) == len(verts)


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_nocut_rejects_broken_map(tmp_path):
    input_dir = tmp_path / "input"
    (tmp_path / "output").mkdir()
    input_dir.mkdir()
    obj = input_dir / "patch.obj"
    verts, _, _ = _write_bump_patch(obj)
    (input_dir / "patch_fixed").write_text("\nnocut")

    lines = obj.read_text().splitlines()
    center = len(verts) // 2
    lines[len(verts) + center] = "vt -5 -5"
    obj.write_text("\n".join(lines) + "\n")

    ran = subprocess.run(
        [str(BUNDLED), "-i", str(obj), "-o", str(tmp_path / "output") + os.sep],
        timeout=300,
        capture_output=True,
    )
    assert ran.returncode == 114


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_pinned_rejects_broken_map(tmp_path):
    input_dir = tmp_path / "input"
    (tmp_path / "output").mkdir()
    input_dir.mkdir()
    obj = input_dir / "patch.obj"
    verts, _, _ = _write_bump_patch(obj)

    # drag one interior uv far away, inverting its faces
    lines = obj.read_text().splitlines()
    center = len(verts) // 2
    lines[len(verts) + center] = "vt -5 -5"
    obj.write_text("\n".join(lines) + "\n")

    ran = subprocess.run(
        [str(BUNDLED), "-i", str(obj), "-o", str(tmp_path / "output") + os.sep],
        timeout=300,
        capture_output=True,
    )
    assert ran.returncode == 110
