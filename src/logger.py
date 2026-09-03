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


# the .blend is in it so the runs under one header all came from that file
def block_header():
    name = bpy.path.basename(bpy.data.filepath) or UNSAVED_FILE_NAME
    return (
        f"{HEADER_PREFIX}{get_addon_version()} | Blender {bpy.app.version_string}"
        f" | {platform.system()} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {name}"
    )


# anything before the first header line is dropped
def split_blocks(content):
    blocks = []
    for line in content.splitlines():
        if line.startswith(HEADER_PREFIX):
            blocks.append([])
        if blocks and line:
            blocks[-1].append(line)
    return blocks


# the newest block is kept whole however long its file was open
def trim_blocks(blocks):
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


# starts a block when the last one belongs to another file or another Blender
def _append_lines(blocks, header, lines):
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

    # the next run opens a block of its own, under a header naming the file it ran on
    def reset(self):
        self.unwrap_info.clear()
        self.header = None

    def new_info(self):
        info = Info()
        self.unwrap_info.append(info)
        self.start_timer()
        return info

    # for a run that was refused before it started
    def discard_info(self):
        self.unwrap_info.pop()

    def add_data(self, target, data):
        getattr(self.get_latest(), target).append(data)

    def change_status(self, status):
        self.get_latest().status = status

    def get_latest(self):
        return self.unwrap_info[-1]

    # oldest first
    def get_all(self):
        return [info.get_info() for info in self.unwrap_info]

    # a crash or a file load loses the in-memory list
    def write_latest(self):
        if self.header is None:
            self.header = block_header()
        info = self.get_latest()
        blocks = _read_blocks()
        _append_lines(blocks, self.header, [info.get_info()])
        info.written_to_file = True
        body = _join_blocks(trim_blocks(blocks))
        get_log_path().write_text(body + "\n", encoding="utf-8")

    # the log file, plus the runs on this file that haven't finished
    def read_log(self):
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

    # the first and last call bound the unwrap phase
    def mark_unwrapping(self):
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
