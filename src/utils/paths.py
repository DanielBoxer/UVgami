import pathlib

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
