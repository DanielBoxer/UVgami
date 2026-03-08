import pathlib
import platform

import bpy


def get_dir_path():
    return pathlib.Path(__file__).parents[2]


def get_addon_name():
    return get_dir_path().stem


def get_root_package():
    parts = __package__.split(".")
    return ".".join(parts[:3])


def get_addon_id():
    if bpy.app.version >= (4, 2, 0):
        return get_root_package()
    return get_addon_name()


def get_preferences():
    return bpy.context.preferences.addons[get_addon_id()].preferences


def get_extension_dir_path():
    if bpy.app.version < (4, 2, 0):
        return get_dir_path()

    extension_folder = pathlib.Path(
        bpy.utils.extension_path_user(get_root_package(), create=True)
    )
    return extension_folder


def get_linux_path(path):
    return f'"/mnt/c{str(pathlib.PurePosixPath(path))[3:]}"'


def get_bundled_engine_path():
    """Return the path to the bundled engine binary, or None if not found."""
    engines_dir = get_dir_path() / "engines"
    if not engines_dir.is_dir():
        return None

    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        platform_dir = "windows"
        binary_name = "uvgami.exe"
    elif system == "Linux":
        platform_dir = "linux"
        binary_name = "uvgami"
    elif system == "Darwin":
        if machine == "arm64":
            platform_dir = "macos-arm64"
        else:
            platform_dir = "macos-x64"
        binary_name = "uvgami"
    else:
        return None

    engine_path = engines_dir / platform_dir / binary_name
    if engine_path.is_file():
        return engine_path
    return None
