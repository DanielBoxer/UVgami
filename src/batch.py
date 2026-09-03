import collections
import subprocess
import threading


def read_stderr_tail(stderr, tail):
    for line in iter(stderr.readline, ""):
        tail.append(line.rstrip("\r\n"))


# tqdm and torch write carriage-return progress bars to stderr
def last_meaningful_line(tail):
    for line in reversed(tail):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class EngineOutput:
    def __init__(self, sink=None):
        self.sink = sink
        self._in_visual = False

    def feed(self, line):
        sink = self.sink
        if sink is None:
            return
        if line.startswith("progress: "):
            sink.progress_data.append(line[10:])
        elif line == "visual_begin:\n":
            sink.uv_co.clear()
            sink.uv_indices.clear()
            sink.is_uv_data_ready = False
            self._in_visual = True
        elif line == "visual_end:\n":
            sink.is_uv_data_ready = True
            self._in_visual = False
        elif self._in_visual:
            if line.startswith("vt"):
                uv_co = line[3:].split()
                sink.uv_co.append((float(uv_co[0]), float(uv_co[1])))
            elif line.startswith("f"):
                uv_indices = line[2:].split()
                sink.uv_indices.append(
                    (int(uv_indices[0]), int(uv_indices[1]), int(uv_indices[2]))
                )


class BatchProcess:
    def __init__(self, args, env=None, sinks=None):
        # passed in rather than set later, the reader may see a start marker right away
        self.sinks = sinks or {}
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
        )
        self.started = set()
        self._results = {}
        # None for an argv batch
        self._sent = None
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        # one tail shared by every mesh in the batch
        self.stderr_tail = collections.deque(maxlen=10)
        self._stderr_reader = threading.Thread(
            target=read_stderr_tail,
            args=(self.process.stderr, self.stderr_tail),
            daemon=True,
        )
        self._stderr_reader.start()

    def _read_output(self):
        parser = EngineOutput()
        for line in iter(self.process.stdout.readline, ""):
            if line.startswith("start: "):
                stem = line[7:].strip()
                self.started.add(stem)
                parser.sink = self.sinks.get(stem)
            elif line.startswith("done: "):
                self._results[line[6:].strip()] = 0
                parser.sink = None
            elif line.startswith("failed: "):
                stem, _, code = line[8:].strip().rpartition(" ")
                try:
                    self._results[stem] = int(code)
                except ValueError:
                    pass
                parser.sink = None
            else:
                parser.feed(line)

    # the exit code reports a write to a dead process
    def send(self, path, sink):
        self.sinks[path.stem] = sink
        self._sent = path.stem
        try:
            print(f"unwrap {path}", file=self.process.stdin, flush=True)
        except OSError:
            pass

    @property
    def is_idle(self):
        if self.process.poll() is not None:
            return False
        return self._sent is None or self._sent in self._results

    # a closed stdin is the process's signal to exit
    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            # a dead engine refuses the flush in close
            pass
        self.process.wait()
        # closing a pipe a reader is blocked on waits for that read
        for thread in (self._reader, self._stderr_reader):
            thread.join(timeout=1)
        for pipe in (self.process.stdout, self.process.stderr):
            pipe.close()

    # only a dead process's reader will drain
    def stderr_lines(self):
        if self.process.poll() is not None:
            self._stderr_reader.join(timeout=1)
        return self.stderr_tail

    # len(started) keeps a startup crash from requeuing forever
    def should_retry(self, stem):
        return (
            self.process.poll() is not None
            and stem not in self.started
            and stem not in self._results
            and len(self.started) > 0
        )

    # None while pending, 0 when unwrapped, nonzero exit code on failure
    def poll_result(self, stem):
        code = self._results.get(stem)
        if code is not None:
            return code
        # a marker can still be in flight until stdout is drained
        if self.process.poll() is None or self._reader.is_alive():
            return None
        ret = self.process.poll()
        # the process ended without reporting this mesh
        return ret if ret != 0 else 1
