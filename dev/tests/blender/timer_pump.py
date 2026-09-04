import contextlib
import time

import bpy

POLL_SECONDS = 0.01


class TimerPump:
    def __init__(self):
        self._pending = []

    def register(self, function, first_interval=0, persistent=False):
        self._pending.append((time.monotonic() + first_interval, function))

    def is_registered(self, function):
        return any(function is pending for _, pending in self._pending)

    def unregister(self, function):
        self._pending = [entry for entry in self._pending if entry[1] is not function]

    def run_until(self, condition, timeout):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"condition still false after {timeout}s,"
                    f" {len(self._pending)} timers pending"
                )
            self._tick()
            time.sleep(POLL_SECONDS)

    def _tick(self):
        now = time.monotonic()
        due = sorted(
            (entry for entry in self._pending if entry[0] <= now),
            key=lambda entry: entry[0],
        )
        self._pending = [entry for entry in self._pending if entry[0] > now]
        for _, function in due:
            interval = function()
            if interval is not None:
                self._pending.append((time.monotonic() + interval, function))


@contextlib.contextmanager
def pump_timers():
    originals = {
        name: getattr(bpy.app.timers, name)
        for name in ("register", "is_registered", "unregister")
    }
    pump = TimerPump()
    # blender never ticks bpy.app.timers with -b
    for name in originals:
        setattr(bpy.app.timers, name, getattr(pump, name))
    try:
        yield pump
    finally:
        for name, original in originals.items():
            setattr(bpy.app.timers, name, original)
