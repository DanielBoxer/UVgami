import os
import platform
import shutil
import sys
import traceback
from pathlib import Path

EXIT_INVALID_INPUT = 2
EXIT_MISSING_RUNTIME = 3
EXIT_ENGINE_FAILURE = 4
EXIT_BAD_OUTPUT = 5

REPO_ROOT = Path(__file__).resolve().parents[2]


class UnwrapError(Exception):
    def __init__(self, exit_code, message):
        super().__init__(message)
        self.exit_code = exit_code


_STYLES = {"error": "31", "success": "32", "step": "2", "header": "36"}


def _color_enabled():
    if os.environ.get("NO_COLOR") is not None or not sys.stderr.isatty():
        return False
    if platform.system() == "Windows":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        standard_error_handle = -12
        handle = ctypes.c_void_p(kernel32.GetStdHandle(standard_error_handle))
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_virtual_terminal_processing = 0x0004
        return bool(
            kernel32.SetConsoleMode(
                handle, mode.value | enable_virtual_terminal_processing
            )
        )
    return True


_COLOR = _color_enabled()


def log(message, style=None):
    # stdout is reserved for --json output and batch markers
    if _COLOR and style in _STYLES:
        message = f"\033[{_STYLES[style]}m{message}\033[0m"
    print(message, file=sys.stderr, flush=True)


def emit(message):
    print(message, flush=True)


# int so a caller can use the return value as an exit code
class BatchResult(int):
    def __new__(cls, exit_code, ok, failed):
        result = super().__new__(cls, exit_code)
        result.ok = ok
        result.failed = failed
        return result


# a thread reading a stdin pipe stalls native module imports for minutes on windows
def unwrap_all(pairs, unwrap_one):
    batch = len(pairs) > 1
    first_code = 0
    ok = 0
    failed = 0
    for index, (input_path, output_path) in enumerate(pairs, 1):
        if batch:
            log(f"[{index}/{len(pairs)}] {input_path.stem}", style="header")
            emit(f"start: {input_path.stem}")
        try:
            if not input_path.is_file():
                raise UnwrapError(EXIT_INVALID_INPUT, f"input not found: {input_path}")
            unwrap_one(input_path, output_path)
        except Exception as error:
            if not batch:
                raise
            if isinstance(error, UnwrapError):
                code = error.exit_code
            else:
                code = EXIT_ENGINE_FAILURE
                log(traceback.format_exc())
            log(f"error: {input_path.name}: {error}", style="error")
            emit(f"failed: {input_path.stem} {code}")
            failed += 1
            if first_code == 0:
                first_code = code
        else:
            if batch:
                emit(f"done: {input_path.stem}")
            ok += 1
    return BatchResult(first_code, ok, failed)


def find_engine(name, label, explicit_path):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise UnwrapError(EXIT_MISSING_RUNTIME, f"{label} binary not found: {path}")
        return path

    system = platform.system()
    if system == "Windows":
        subdir, binary = "windows", f"{name}.exe"
    elif system == "Linux":
        subdir, binary = "linux", name
    elif system == "Darwin":
        machine = platform.machine().lower()
        subdir = "macos-arm64" if machine == "arm64" else "macos-x64"
        binary = name
    else:
        raise UnwrapError(EXIT_MISSING_RUNTIME, f"unsupported platform: {system}")

    path = REPO_ROOT / "engine-builds" / subdir / binary
    if not path.is_file():
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            f"local {label} binary not found: {path} (pass --{name}-path)",
        )
    return path


def validate_uv_obj(path):
    if not path.is_file():
        raise UnwrapError(EXIT_BAD_OUTPUT, f"engine produced no output: {path}")
    found = {"v": False, "vt": False, "f": False}
    with open(path) as file:
        for line in file:
            key = line.split(maxsplit=1)[0] if line.strip() else ""
            if key in found:
                found[key] = True
                if all(found.values()):
                    return
    missing = ", ".join(key for key, seen in found.items() if not seen)
    raise UnwrapError(EXIT_BAD_OUTPUT, f"output OBJ is missing data: {missing}")


def deliver(result_path, output_path):
    validate_uv_obj(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(result_path), str(output_path))
