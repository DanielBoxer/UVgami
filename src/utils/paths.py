import functools
import os
import pathlib
import platform
import tomllib

import bpy


def get_dir_path():
    return pathlib.Path(__file__).parents[2]


@functools.cache
def get_addon_version():
    with (get_dir_path() / "blender_manifest.toml").open("rb") as file:
        return tomllib.load(file)["version"]


def get_root_package():
    # the bl_ext install prefix isn't always the same depth
    return __package__.rsplit(".", 2)[0]


def get_addon_id():
    return get_root_package()


def get_preferences():
    return bpy.context.preferences.addons[get_addon_id()].preferences


def get_extension_dir_path():
    return pathlib.Path(bpy.utils.extension_path_user(get_root_package(), create=True))


def engine_file_stem(name):
    """The engine binaries read argv in the ansi codepage, so a non-ascii
    path arrives as question marks and never opens."""
    return "".join(c if c.isascii() else "_" for c in bpy.path.clean_name(name))


ENGINE_FILE_SUFFIXES = (
    ".obj",
    "_weights",
    "_importance",
    "_edges",
    "_fixed",
    "_inputs.txt",
)


def get_io_dir_paths():
    """The engine input and output folders, created if missing."""
    input_path = get_extension_dir_path() / "input"
    output_path = get_extension_dir_path() / "output"
    input_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)
    return input_path, output_path


def clear_io_dir(path):
    """Delete the engine files in path, leaving anything else alone. Never
    recurses, and refuses a path outside the extension dir."""
    root = get_extension_dir_path()
    if not path.is_relative_to(root):
        raise ValueError(f"refusing to clear {path}, outside {root}")
    for file in path.iterdir():
        if file.is_file() and file.name.endswith(ENGINE_FILE_SUFFIXES):
            file.unlink()


def get_platform_tag():
    """Platform name used for both the local engine folders and the engine
    release asset names, or None on an unsupported platform."""
    system = platform.system()
    if system == "Windows":
        return "windows"
    if system == "Linux":
        return "linux"
    if system == "Darwin":
        if platform.machine().lower() == "arm64":
            return "macos-arm64"
        return "macos-x64"
    return None


def get_engine_binary_name(name):
    return f"{name}.exe" if platform.system() == "Windows" else name


ENGINE_DIR_ENV_VAR = "UVGAMI_ENGINE_DIR"


def _local_engine_dirs():
    override = os.environ.get(ENGINE_DIR_ENV_VAR)
    if override:
        yield pathlib.Path(override)
    tag = get_platform_tag()
    if tag is not None:
        yield get_dir_path() / "engine-builds" / tag


def get_local_engine_path(name):
    """Path to an engine binary in UVGAMI_ENGINE_DIR or engine-builds/<platform>/, or
    None. No engines ship with the addon, so this only finds a build made in a dev
    checkout."""
    binary_name = get_engine_binary_name(name)
    for directory in _local_engine_dirs():
        engine_path = directory / binary_name
        if engine_path.is_file():
            return engine_path
    return None
