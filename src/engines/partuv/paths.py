import platform

from ...utils.paths import get_extension_dir_path


def get_partuv_venv_path():
    return get_extension_dir_path() / "partuv-venv"


def get_partuv_venv_python():
    venv = get_partuv_venv_path()
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def get_uv_path():
    name = "uv.exe" if platform.system() == "Windows" else "uv"
    return get_extension_dir_path() / "uv" / name


# PartField checkpoint, downloaded by the AI-tier install
def get_partuv_checkpoint_path():
    return get_extension_dir_path() / "model_objaverse.ckpt"
