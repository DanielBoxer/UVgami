import threading


# the work must touch no bpy data, Blender's data is not thread safe
class BackgroundTask:
    def __init__(self, work):
        self._box = {}
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, args=(work,), daemon=True)
        self._thread.start()

    def _run(self, work):
        try:
            self._box["result"] = work(self.is_cancelled)
        except BaseException as error:
            self._box["error"] = error

    def done(self):
        return not self._thread.is_alive()

    def result(self):
        if "error" in self._box:
            raise self._box["error"]
        return self._box["result"]

    # a thread cannot be killed, the work has to check is_cancelled
    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled
