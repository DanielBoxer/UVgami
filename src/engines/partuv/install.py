import json
import functools
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

import bpy

from ...utils.download import download_file
from ..install_task import (
    DELETE_DESCRIPTION,
    DELETED_MESSAGE,
    DOWNLOADED_MESSAGE,
    InstallTask,
    offline_error,
    report_progress,
    task_state,
)
from .venv_commands import VENV_PYTHON, build_install_commands
from .paths import (
    get_partuv_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
    get_uv_path,
)

# must match engine/partuv/pyproject.toml
PARTUV_VERSION = "0.1.4"
PARTUV_RELEASE_API = f"https://api.github.com/repos/DanielBoxer/UVgami/releases/tags/partuv-v{PARTUV_VERSION}"
# the hugging face original moves with its main branch
CHECKPOINT_URL = "https://github.com/DanielBoxer/UVgami/releases/download/checkpoint/model_objaverse.ckpt"
# partuv is an nvidia cuda wheel
PARTUV_PLATFORMS = ("Windows", "Linux")
GEOMETRIC_DOWNLOAD_SIZE = "200 MB"
AI_DOWNLOAD_SIZE = "5 GB"
PARTUV_PY_TAG = "cp311"
UV_VERSION = "0.11.25"
UV_ARCHIVES = {
    "Windows": "uv-x86_64-pc-windows-msvc.zip",
    "Linux": "uv-x86_64-unknown-linux-gnu.tar.gz",
}


# the panel calls this every redraw, cleared by invalidate_engine_caches
@functools.cache
def get_installed_partuv_version():
    """Version of the wheel in the venv, read from its dist-info, or None."""
    venv = get_partuv_venv_path()
    if platform.system() == "Windows":
        site = venv / "Lib" / "site-packages"
    else:
        site = venv / "lib" / f"python{VENV_PYTHON}" / "site-packages"
    info = next(site.glob("partuv-*.dist-info"), None)
    if info is None:
        return None
    return info.name[len("partuv-") : -len(".dist-info")]


def partuv_update_pending():
    version = get_installed_partuv_version()
    return version is not None and version != PARTUV_VERSION


def find_wheel_url():
    request = urllib.request.Request(
        PARTUV_RELEASE_API, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    # linux wheels ship as manylinux after auditwheel repair
    plat = "win_amd64" if platform.system() == "Windows" else "x86_64"
    for asset in release.get("assets", []):
        name = asset["name"]
        if (
            name.startswith(f"partuv-{PARTUV_VERSION}-")
            and PARTUV_PY_TAG in name
            and name.endswith(f"{plat}.whl")
        ):
            return asset["browser_download_url"]
    raise RuntimeError(
        f"no partuv {PARTUV_VERSION} wheel for {PARTUV_PY_TAG} {plat} in the"
        f" partuv-v{PARTUV_VERSION} release"
    )


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        raise RuntimeError(f"{args[0]} failed: {tail}")


def ensure_uv():
    """Download the standalone uv binary if it isn't already present."""
    uv = get_uv_path()
    if uv.is_file():
        return uv
    archive_name = UV_ARCHIVES[platform.system()]
    url = (
        f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{archive_name}"
    )
    uv.parent.mkdir(parents=True, exist_ok=True)
    tmp = uv.parent / archive_name
    task_state["phase"] = "Downloading uv"
    download_file(url, tmp, progress=report_progress)
    # the archives nest the binary in a per-target folder
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(tmp) as archive:
            member = next(n for n in archive.namelist() if n.endswith("uv.exe"))
            with archive.open(member) as src, open(uv, "wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(tmp) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith("/uv"))
            with archive.extractfile(member) as src, open(uv, "wb") as dst:
                shutil.copyfileobj(src, dst)
        uv.chmod(0o755)
    tmp.unlink()
    return uv


def run_venv_install(wheel_url, ai):
    uv = ensure_uv()
    # uv prints no byte counts
    task_state["phase"] = "Installing packages"
    task_state["bytes_total"] = None
    venv_python = get_partuv_venv_python()
    commands = build_install_commands(
        str(uv),
        str(venv_python),
        str(get_partuv_venv_path()),
        wheel_url,
        ai,
        create_venv=not venv_python.is_file(),
    )
    for command in commands:
        _run(command)


def download_checkpoint():
    target = get_partuv_checkpoint_path()
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    task_state["phase"] = "Downloading AI checkpoint"
    download_file(CHECKPOINT_URL, target, progress=report_progress)


def install_partuv(ai):
    run_venv_install(find_wheel_url(), ai)
    if ai:
        download_checkpoint()


def uninstall_partuv():
    task_state["phase"] = "Deleting engine"
    venv = get_partuv_venv_path()
    if venv.is_dir():
        shutil.rmtree(venv)
    get_partuv_checkpoint_path().unlink(missing_ok=True)
    uv_dir = get_uv_path().parent
    if uv_dir.is_dir():
        shutil.rmtree(uv_dir)


class PartuvTask(InstallTask):
    owner = "partuv"

    def precheck(self):
        if platform.system() not in PARTUV_PLATFORMS:
            return "PartUV is only available on Windows and Linux"
        return None


class UVGAMI_OT_install_partuv(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.install_partuv"
    bl_label = "Download PartUV Engine"
    done_message = DOWNLOADED_MESSAGE

    def precheck(self):
        return super().precheck() or offline_error()

    @classmethod
    def description(cls, context, properties):
        if properties.tier == "AI":
            return (
                f"Download the engine with AI segmentation, ~{AI_DOWNLOAD_SIZE}."
                " Includes geometric. Needs an NVIDIA GPU"
            )
        return (
            "Download the engine with geometric segmentation only, a much smaller"
            " download. Needs an NVIDIA GPU"
        )

    def invoke(self, context, event):
        ai = self.tier == "AI"
        return context.window_manager.invoke_confirm(
            self,
            event,
            title="Download PartUV AI" if ai else "Download PartUV",
            message=f"{AI_DOWNLOAD_SIZE}. The NVIDIA AI model is non-commercial only"
            if ai
            else GEOMETRIC_DOWNLOAD_SIZE,
            confirm_text="Download",
        )

    tier: bpy.props.EnumProperty(
        items=(
            ("GEOMETRIC", "Geometric", ""),
            ("AI", "AI", ""),
        ),
        default="GEOMETRIC",
        options={"HIDDEN"},
    )

    def build_task(self):
        ai = self.tier == "AI"
        return lambda: install_partuv(ai)


class UVGAMI_OT_uninstall_partuv(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.uninstall_partuv"
    bl_label = "Delete PartUV Engine"
    bl_description = DELETE_DESCRIPTION
    done_message = DELETED_MESSAGE

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def build_task(self):
        return uninstall_partuv
