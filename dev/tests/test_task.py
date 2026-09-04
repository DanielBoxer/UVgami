import importlib.util
import threading
import time
from pathlib import Path

import pytest

# loaded from file, the addon package imports bpy
spec = importlib.util.spec_from_file_location(
    "addon_task", Path(__file__).parents[2] / "src" / "utils" / "task.py"
)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def wait(background):
    deadline = time.monotonic() + 5
    while not background.done():
        assert time.monotonic() < deadline, "task never finished"
        time.sleep(0.005)


def test_result_comes_back():
    background = task.BackgroundTask(lambda cancelled: 21 * 2)
    wait(background)
    assert background.result() == 42


def test_not_done_while_the_work_runs():
    started = threading.Event()
    release = threading.Event()

    def work(cancelled):
        started.set()
        release.wait(5)
        return "ok"

    background = task.BackgroundTask(work)
    assert started.wait(5)
    assert not background.done()
    release.set()
    wait(background)
    assert background.result() == "ok"


def test_error_is_raised_on_the_collecting_thread():
    collecting = threading.current_thread()
    raised_on = {}

    def work(cancelled):
        raise ValueError("from the worker")

    background = task.BackgroundTask(work)
    wait(background)
    try:
        background.result()
    except ValueError:
        raised_on["thread"] = threading.current_thread()
    assert raised_on["thread"] is collecting


def test_error_is_not_swallowed_by_done():
    background = task.BackgroundTask(lambda cancelled: 1 / 0)
    wait(background)
    with pytest.raises(ZeroDivisionError):
        background.result()


def test_the_work_sees_the_cancel():
    started = threading.Event()

    def work(cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.005)
        return "gave up"

    background = task.BackgroundTask(work)
    assert started.wait(5)
    background.cancel()
    wait(background)
    assert background.result() == "gave up"


def test_work_that_never_checks_still_finishes():
    background = task.BackgroundTask(lambda cancelled: "ignored it")
    background.cancel()
    wait(background)
    assert background.result() == "ignored it"
