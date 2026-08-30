import platform
import time

import bpy

from .utils.paths import get_addon_version, get_extension_dir_path

LOG_FILE_NAME = "log.txt"
HEADER_PREFIX = "UVgami "
# a bug report wants the recent runs, not every run since install
LOG_LINE_LIMIT = 500
UNSAVED_FILE_NAME = "unsaved"


def get_log_path():
    return get_extension_dir_path() / LOG_FILE_NAME


def block_header():
    """What a pasted log needs to say what produced it. The .blend is in it so
    the runs under one header all came from that file."""
    name = bpy.path.basename(bpy.data.filepath) or UNSAVED_FILE_NAME
    return (
        f"{HEADER_PREFIX}{get_addon_version()} | Blender {bpy.app.version_string}"
        f" | {platform.system()} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {name}"
    )


def split_blocks(content):
    """The log's blocks, oldest first, each starting with its header line.
    Anything before the first header is dropped."""
    blocks = []
    for line in content.splitlines():
        if line.startswith(HEADER_PREFIX):
            blocks.append([])
        if blocks and line:
            blocks[-1].append(line)
    return blocks


def trim_blocks(blocks):
    """Drop the oldest blocks past the line limit. The newest is kept whole
    however long its file was open."""
    kept = []
    lines = 0
    for block in reversed(blocks):
        if kept and lines + len(block) > LOG_LINE_LIMIT:
            break
        kept.append(block)
        lines += len(block)
    kept.reverse()
    return kept


def _read_blocks():
    path = get_log_path()
    if not path.is_file():
        return []
    return split_blocks(path.read_text(encoding="utf-8"))


def _join_blocks(blocks):
    return "\n\n".join("\n".join(block) for block in blocks)


def _append_lines(blocks, header, lines):
    """Add the lines under header, starting a block when the last one belongs
    to another file or another Blender."""
    if not blocks or blocks[-1][0] != header:
        blocks.append([header])
    blocks[-1].extend(lines)


def format_triangles(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


class Info:
    def __init__(self):
        self.started = time.strftime("%H:%M:%S")
        self.time = 0
        # these three add up to self.time
        self.pre_processing_time = 0
        self.unwrap_time = 0
        self.post_processing_time = 0
        self.errors = []
        self.status = "In Progress"
        self.written_to_file = False
        # (name, triangle count) of every input mesh
        self.objects = []
        self.piece_counts = {}
        self.engine = ""
        self.settings = ""

    def get_time(self):
        text = f"{self.time:.2f}s"
        if self.unwrap_time:
            text += (
                f" (pre-processing {self.pre_processing_time:.2f}s,"
                f" unwrap {self.unwrap_time:.2f}s,"
                f" post-processing {self.post_processing_time:.2f}s)"
            )
        return text

    def get_objects(self):
        return ", ".join(
            f"{name} ({format_triangles(count)} tris)" for name, count in self.objects
        )

    def get_pieces(self):
        counts = ", ".join(
            f"{count} {result.value}"
            for result, count in self.piece_counts.items()
            if count
        )
        return f"Pieces: {counts}" if counts else ""

    def get_info(self):
        """One line for the run."""
        fields = [self.started, self.status, self.get_time()]
        if self.engine:
            fields.append(self.engine)
        fields.append(self.get_objects())
        pieces = self.get_pieces()
        if pieces:
            fields.append(pieces)
        if self.settings:
            fields.append(f"Settings: {self.settings}")
        if self.errors:
            fields.append(f"Errors: {'; '.join(self.errors)}")
        return " | ".join(fields)


class Logger:
    def __init__(self):
        self.unwrap_info = []
        self.start_time = 0
        self.unwrap_start = None
        self.unwrap_end = None
        self.header = None

    def reset(self):
        """Drop the runs on the file being closed. The next one opens a block
        of its own, under a header naming the file it ran on."""
        self.unwrap_info.clear()
        self.header = None

    def new_info(self):
        info = Info()
        self.unwrap_info.append(info)
        self.start_timer()
        return info

    def discard_info(self):
        """Drop the entry for a run that was refused before it started."""
        self.unwrap_info.pop()

    def add_data(self, target, data):
        getattr(self.get_latest(), target).append(data)

    def change_status(self, status):
        self.get_latest().status = status

    def get_latest(self):
        return self.unwrap_info[-1]

    def get_all(self):
        """One line per run, oldest first."""
        return [info.get_info() for info in self.unwrap_info]

    def write_latest(self):
        """Append the finished run to the log file. A crash or a file load takes
        the in-memory list, and a crash is what the log is asked for."""
        if self.header is None:
            self.header = block_header()
        info = self.get_latest()
        blocks = _read_blocks()
        _append_lines(blocks, self.header, [info.get_info()])
        info.written_to_file = True
        body = _join_blocks(trim_blocks(blocks))
        get_log_path().write_text(body + "\n", encoding="utf-8")

    def read_log(self):
        """The log file, plus the runs on this file that haven't finished."""
        blocks = _read_blocks()
        pending = [
            info.get_info() for info in self.unwrap_info if not info.written_to_file
        ]
        if pending:
            _append_lines(blocks, self.header or block_header(), pending)
        return _join_blocks(blocks)

    def start_timer(self):
        self.start_time = time.perf_counter()
        self.unwrap_start = None
        self.unwrap_end = None

    def mark_unwrapping(self):
        """Called while any engine still has work. Everything before the first
        call is pre-processing, everything after the last is post-processing."""
        self.unwrap_end = time.perf_counter()
        if self.unwrap_start is None:
            self.unwrap_start = self.unwrap_end

    def update_time(self):
        info = self.get_latest()
        info.time = time.perf_counter() - self.start_time
        if self.unwrap_start is None:
            info.pre_processing_time = info.time
            return
        info.pre_processing_time = self.unwrap_start - self.start_time
        info.unwrap_time = self.unwrap_end - self.unwrap_start
        info.post_processing_time = (
            info.time - info.pre_processing_time - info.unwrap_time
        )


logger = Logger()
