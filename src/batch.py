import collections
import subprocess
import threading


def read_stderr_tail(stderr, tail):
    """Drain a process's stderr into a bounded tail of decoded lines."""
    for line in iter(stderr.readline, ""):
        tail.append(line.rstrip("\r\n"))


def last_meaningful_line(tail):
    """Newest non-empty stderr line, or empty string. tqdm and torch write
    carriage-return progress bars to stderr, so skip whitespace-only lines."""
    for line in reversed(tail):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class EngineOutput:
    """Parses engine stdout lines into an unwrap-like sink.

    The optcuts engine answers stdin snapshot requests with a full uv map
    (visual_begin:/vt/f/visual_end:); both engines emit progress: lines."""

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
    """One engine process unwrapping many meshes sequentially, either every
    mesh given on its argv or the ones sent to it one at a time.

    Tracks per-mesh state from the cli's start/done/failed stdout markers,
    keyed by input file stem."""

    def __init__(self, args, env=None, sinks=None):
        # unwrap-like sinks keyed by stem, engine output routes to the sink of
        # the mesh being unwrapped. passed in here because the reader thread
        # may see the first start marker right away
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
        # stderr stays out of the stdout protocol, read by its own thread into
        # a tail shared by every mesh in the batch
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

    def send(self, path, sink):
        """Write the next mesh to a process started without input paths. A
        dead process refuses the write and answers through its exit code."""
        self.sinks[path.stem] = sink
        self._sent = path.stem
        try:
            print(f"unwrap {path}", file=self.process.stdin, flush=True)
        except OSError:
            pass

    @property
    def is_idle(self):
        """Alive and done with the last mesh sent to it."""
        if self.process.poll() is not None:
            return False
        return self._sent is None or self._sent in self._results

    def close(self):
        """End an idle process: a closed stdin is its signal to exit."""
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

    def stderr_lines(self):
        """Last stderr lines, waiting briefly for the reader to drain. A live
        process keeps its reader busy, so only a dead one is waited for."""
        if self.process.poll() is not None:
            self._stderr_reader.join(timeout=1)
        return self.stderr_tail

    def should_retry(self, stem):
        """True when the process has died with this mesh never started and no
        result, but at least one other mesh did start. A mid-batch death has
        intact remaining work, while a startup crash starts nothing and must
        keep failing so requeuing can't loop forever."""
        return (
            self.process.poll() is not None
            and stem not in self.started
            and stem not in self._results
            and len(self.started) > 0
        )

    def poll_result(self, stem):
        """None while pending, 0 when unwrapped, nonzero exit code on failure."""
        code = self._results.get(stem)
        if code is not None:
            return code
        # only fall back to the process exit code once stdout is drained,
        # otherwise a marker could still be in flight
        if self.process.poll() is None or self._reader.is_alive():
            return None
        ret = self.process.poll()
        # the process ended without reporting this mesh
        return ret if ret != 0 else 1
