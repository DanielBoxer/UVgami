import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

# loaded from file, the addon package imports bpy
spec = importlib.util.spec_from_file_location(
    "addon_batch", Path(__file__).parents[2] / "src" / "batch.py"
)
addon_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon_batch)


def start(script, sinks=None):
    return addon_batch.BatchProcess([sys.executable, "-c", script], sinks=sinks)


class Sink:
    def __init__(self):
        self.progress_data = deque()
        self.uv_co = deque()
        self.uv_indices = deque()
        self.is_uv_data_ready = False


def feed(lines, sink=None):
    sink = sink or Sink()
    parser = addon_batch.EngineOutput(sink)
    for line in lines:
        parser.feed(line + "\n")
    return sink


def wait_result(batch_process, stem, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = batch_process.poll_result(stem)
        if code is not None:
            return code
        time.sleep(0.05)
    raise AssertionError(f"no result for {stem}")


SHARED_ENGINE = (
    "import pathlib,sys\n"
    "for line in sys.stdin:\n"
    "    stem = pathlib.Path(line[7:].strip()).stem\n"
    "    print(f'start: {stem}', flush=True)\n"
    "    print('progress: 0.5 0 0.5', flush=True)\n"
    "    print(f'done: {stem}', flush=True)\n"
)


def wait_idle(batch_process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if batch_process.is_idle:
            return
        time.sleep(0.05)
    raise AssertionError("process never went idle")


def test_sent_meshes_run_in_turn():
    sinks = {"a": Sink(), "b": Sink()}
    batch_process = start(SHARED_ENGINE)
    assert batch_process.is_idle
    batch_process.send(Path("in/a.obj"), sinks["a"])
    assert wait_result(batch_process, "a") == 0
    wait_idle(batch_process)
    batch_process.send(Path("in/b.obj"), sinks["b"])
    assert wait_result(batch_process, "b") == 0
    assert sinks["a"].progress_data.popleft() == "0.5 0 0.5\n"
    assert sinks["b"].progress_data.popleft() == "0.5 0 0.5\n"
    assert batch_process.process.poll() is None
    batch_process.close()
    assert batch_process.process.poll() == 0


def test_shared_process_busy_until_done():
    batch_process = start(
        "import sys,time;sys.stdin.readline();"
        "print('start: a',flush=True);time.sleep(30)"
    )
    try:
        batch_process.send(Path("a.obj"), Sink())
        deadline = time.monotonic() + 10
        while "a" not in batch_process.started:
            assert time.monotonic() < deadline, "start marker never arrived"
            time.sleep(0.05)
        assert not batch_process.is_idle
    finally:
        batch_process.process.kill()


def test_send_to_dead_process_reports_its_exit_code():
    batch_process = start("import sys;print('start: a');print('done: a');sys.exit(3)")
    wait_dead(batch_process)
    batch_process.send(Path("b.obj"), Sink())
    assert not batch_process.is_idle
    assert wait_result(batch_process, "b") == 3


def test_stderr_lines_do_not_wait_on_a_live_process():
    batch_process = start(
        "import sys,time;print('start: a',flush=True);"
        "print('bad mesh',file=sys.stderr,flush=True);"
        "print('failed: a 101',flush=True);time.sleep(30)"
    )
    try:
        assert wait_result(batch_process, "a") == 101
        deadline = time.monotonic() + 10
        while not batch_process.stderr_tail:
            assert time.monotonic() < deadline, "stderr line never arrived"
            time.sleep(0.05)
        began = time.monotonic()
        assert addon_batch.last_meaningful_line(batch_process.stderr_lines()) == (
            "bad mesh"
        )
        assert time.monotonic() - began < 0.5
    finally:
        batch_process.process.kill()


def test_markers_reported():
    batch_process = start("print('start: a');print('done: a');print('failed: b -2')")
    assert wait_result(batch_process, "a") == 0
    assert wait_result(batch_process, "b") == -2
    assert "a" in batch_process.started


def test_unknown_lines_ignored():
    batch_process = start("print('loading model');print('start: a');print('done: a')")
    assert wait_result(batch_process, "a") == 0


def test_missing_marker_fails_after_exit():
    batch_process = start("print('done: a')")
    assert wait_result(batch_process, "a") == 0
    assert wait_result(batch_process, "b") != 0


def test_pending_while_running():
    batch_process = start("import time;print('start: a',flush=True);time.sleep(30)")
    try:
        deadline = time.monotonic() + 10
        while "a" not in batch_process.started:
            assert time.monotonic() < deadline, "start marker never arrived"
            time.sleep(0.05)
        assert batch_process.poll_result("a") is None
    finally:
        batch_process.process.kill()


def wait_dead(batch_process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            batch_process.process.poll() is not None
            and not batch_process._reader.is_alive()
        ):
            return
        time.sleep(0.05)
    raise AssertionError("process did not exit")


def test_should_retry_when_never_started_and_others_did():
    batch_process = start("print('start: a')")
    wait_dead(batch_process)
    assert batch_process.should_retry("b")


def test_should_not_retry_when_nothing_started():
    batch_process = start("print('nothing')")
    wait_dead(batch_process)
    assert not batch_process.should_retry("a")


def test_should_not_retry_when_stem_started():
    batch_process = start("print('start: a')")
    wait_dead(batch_process)
    assert not batch_process.should_retry("a")


def test_should_not_retry_while_running():
    batch_process = start("import time;print('start: a',flush=True);time.sleep(30)")
    try:
        deadline = time.monotonic() + 10
        while "a" not in batch_process.started:
            assert time.monotonic() < deadline, "start marker never arrived"
            time.sleep(0.05)
        assert not batch_process.should_retry("b")
    finally:
        batch_process.process.kill()


def test_should_not_retry_when_result_exists():
    batch_process = start("print('start: a');print('done: a')")
    wait_dead(batch_process)
    assert not batch_process.should_retry("a")


def test_parser_uvgami_snapshot():
    sink = feed(
        [
            "progress: 0.5 0.3 0.2",
            "visual_begin:",
            "vt 0.1 0.2",
            "vt 0.3 0.4",
            "f 0 1 1",
            "visual_end:",
        ]
    )
    assert sink.progress_data.popleft() == "0.5 0.3 0.2\n"
    assert list(sink.uv_co) == [(0.1, 0.2), (0.3, 0.4)]
    assert list(sink.uv_indices) == [(0, 1, 1)]
    assert sink.is_uv_data_ready


def test_parser_new_snapshot_replaces_old():
    sink = feed(["visual_begin:", "vt 0.1 0.2", "f 0 0 0", "visual_end:"])
    feed(["visual_begin:", "vt 0.5 0.6", "f 0 0 0", "visual_end:"], sink)
    assert list(sink.uv_co) == [(0.5, 0.6)]


def test_parser_ignores_loose_geometry_lines():
    # engine logs outside a visual block must not be parsed as uv data
    sink = feed(["f 0 1 2", "vt 0.5 0.5", "found 3 parts"])
    assert not sink.uv_co
    assert not sink.uv_indices


def test_stderr_tail_captured_on_failure():
    batch_process = start(
        "import sys;"
        "print('start: a');"
        "print('No module named partuv.__main__', file=sys.stderr);"
        "sys.exit(1)"
    )
    assert wait_result(batch_process, "a") != 0
    tail = list(batch_process.stderr_lines())
    assert addon_batch.last_meaningful_line(tail) == "No module named partuv.__main__"


def test_stderr_does_not_pollute_stdout_protocol():
    sinks = {"a": Sink()}
    batch_process = start(
        "import sys;"
        "print('start: a');"
        "print('progress: 0.5 0 0.5');"
        "print('noise on stderr', file=sys.stderr);"
        "print('done: a')",
        sinks,
    )
    assert wait_result(batch_process, "a") == 0
    assert sinks["a"].progress_data.popleft() == "0.5 0 0.5\n"


def test_last_meaningful_line_skips_blank():
    assert addon_batch.last_meaningful_line(deque(["real error", "  ", "\t", ""])) == (
        "real error"
    )
    assert addon_batch.last_meaningful_line(deque()) == ""


def test_batch_routes_output_to_sinks():
    sinks = {"a": Sink(), "b": Sink()}
    batch_process = start(
        "print('start: a');"
        "print('progress: 0.5 0 0.5');"
        "print('done: a');"
        "print('start: b');"
        "print('progress: 0.2 0 0.8');"
        "print('done: b')",
        sinks,
    )
    assert wait_result(batch_process, "a") == 0
    assert wait_result(batch_process, "b") == 0
    assert sinks["a"].progress_data.popleft() == "0.5 0 0.5\n"
    assert sinks["b"].progress_data.popleft() == "0.2 0 0.8\n"
