import functools
import os
import pathlib
import platform
import shutil
import subprocess
from dataclasses import dataclass

import bpy

from .. import Engine
from ...utils.paths import get_dir_path, get_extension_dir_path
from ...utils.ui import is_non_default, only_active
from ..install_task import draw_progress, draw_update_row, task_state
from .install import (
    PARTUV_PLATFORMS,
    UVGAMI_OT_install_partuv,
    UVGAMI_OT_uninstall_partuv,
    get_installed_partuv_version,
    partuv_update_pending,
)
from .paths import (
    get_partuv_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
)


GEOMETRIC_SEGMENTATION = 0
AI_SEGMENTATION = 1

# built once so the item strings stay referenced, which blender requires for
# dynamic enum callbacks
_SEGMENTATION_ITEMS = (
    (
        "GEOMETRIC",
        "Geometric",
        "Geometric clustering. Faster but more seams",
        "",
        GEOMETRIC_SEGMENTATION,
    ),
    (
        "AI",
        "AI",
        "AI segmentation. Better results than geometric",
        "",
        AI_SEGMENTATION,
    ),
)


def _segmentation_items(self, context):
    if is_ai_segmentation_available():
        return list(_SEGMENTATION_ITEMS)
    return [_SEGMENTATION_ITEMS[GEOMETRIC_SEGMENTATION]]


# clamped without touching the stored value, so ai comes back as the selection
# once its checkpoint is downloaded
def _segmentation_get(self):
    stored = self.get("segmentation", AI_SEGMENTATION)
    if stored == AI_SEGMENTATION and not is_ai_segmentation_available():
        return GEOMETRIC_SEGMENTATION
    return stored


def _segmentation_set(self, value):
    self["segmentation"] = value


class UVGAMI_PG_partuv(bpy.types.PropertyGroup):
    segmentation: bpy.props.EnumProperty(
        name="Segmentation",
        description="How PartUV splits the mesh into parts",
        items=_segmentation_items,
        get=_segmentation_get,
        set=_segmentation_set,
    )
    threshold: bpy.props.FloatProperty(
        name="",
        description="Distortion threshold. Lower values cut the mesh into more UV islands",
        default=1.25,
        min=1.0,
        max=10.0,
    )


# cached: validate runs on every panel redraw and shutil.which scans the
# whole PATH. a dev checkout doesn't change mid-session
@functools.cache
def find_partuv_dev_repo():
    """Return the repo path if the developer CLI is usable, else None."""
    repo = get_dir_path()
    if (
        (repo / "dev" / "uvgami_cli").is_dir()
        and (repo / ".venv").is_dir()
        and shutil.which("uv") is not None
    ):
        return repo
    return None


def is_partuv_installed():
    return get_partuv_venv_python().is_file()


def is_partuv_ai_installed():
    return is_partuv_installed() and get_partuv_checkpoint_path().is_file()


def is_ai_segmentation_available():
    # a dev checkout runs against the repo checkpoint, not the downloaded one
    return find_partuv_dev_repo() is not None or is_partuv_ai_installed()


@dataclass
class PartuvRun:
    # dev runs the workspace partuv through uv, installed runs the wheel's
    # python -m partuv from the install operator's venv
    mode: str
    path: pathlib.Path


class PartuvEngine(Engine):
    id = "PARTUV"
    enum_value = 1
    label = "PartUV"
    description = "GPU engine. Fewer islands and can be faster on dense meshes"
    icon = "MOD_EXPLODE"
    property_group = UVGAMI_PG_partuv
    classes = (UVGAMI_PG_partuv, UVGAMI_OT_install_partuv, UVGAMI_OT_uninstall_partuv)

    def is_available(self):
        return platform.system() in PARTUV_PLATFORMS

    def batches_queue(self, props):
        # one process loads the model once for every queued mesh, running more
        # than one runs out of vram
        return props.partuv.segmentation == "AI"

    def validate(self, prefs):
        repo = find_partuv_dev_repo()
        if repo is not None:
            return PartuvRun("dev", repo), None
        if is_partuv_installed():
            return PartuvRun("installed", get_partuv_venv_path()), None
        return None, "PartUV is not installed. Download it in the add-on preferences"

    def invalidate_caches(self):
        get_installed_partuv_version.cache_clear()

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="MOD_EXPLODE", text="Segmentation")
        row.prop(props.partuv, "segmentation", text="")

        row = layout.row()
        row.label(icon="MOD_LENGTH", text="Threshold")
        row.prop(props.partuv, "threshold")

    def active_settings(self, props):
        partuv = props.partuv
        return only_active(
            (
                (
                    "MOD_EXPLODE",
                    "Geometric Segmentation",
                    "partuv.segmentation",
                    partuv.segmentation == "GEOMETRIC"
                    and is_ai_segmentation_available(),
                ),
                (
                    "MOD_LENGTH",
                    f"Threshold {partuv.threshold:.2f}",
                    "partuv.threshold",
                    is_non_default(props, "partuv.threshold"),
                ),
            )
        )

    def describe(self):
        if find_partuv_dev_repo() is not None:
            return f"{self.label} (local build)"
        version = get_installed_partuv_version()
        return f"{self.label} {version}" if version else self.label

    def update_pending(self):
        return find_partuv_dev_repo() is None and partuv_update_pending()

    def draw_update_notice(self, layout):
        row = draw_update_row(
            layout, "partuv", "Installing PartUV", self.update_pending()
        )
        if row is not None:
            row.operator("uvgami.install_partuv", text="Update", icon="IMPORT").tier = (
                "AI" if is_partuv_ai_installed() else "GEOMETRIC"
            )

    def draw_prefs(self, layout, prefs):
        if not self.is_available():
            layout.row().label(
                text="PartUV needs Windows or Linux with an NVIDIA GPU", icon="ERROR"
            )
            return
        if find_partuv_dev_repo() is not None:
            layout.row().label(text="Using the local build", icon="CHECKMARK")
            return

        if task_state["running"] and task_state["owner"] == "partuv":
            draw_progress(layout, "Installing PartUV")
            return

        installed = is_partuv_installed()
        ai_installed = is_partuv_ai_installed()
        update_pending = partuv_update_pending()
        row = layout.row()
        if update_pending:
            row.label(text="Engine update available", icon="FILE_REFRESH")
        elif ai_installed:
            row.label(text="Installed with AI segmentation", icon="CHECKMARK")
        elif installed:
            row.label(
                text="Installed with geometric segmentation only", icon="CHECKMARK"
            )
        else:
            row.label(text="Not installed", icon="X")

        if update_pending or not ai_installed:
            row = layout.row()
            row.scale_y = 1.5
            if update_pending:
                row.operator(
                    "uvgami.install_partuv", text="Update", icon="IMPORT"
                ).tier = "AI" if ai_installed else "GEOMETRIC"
            if not installed:
                row.operator(
                    "uvgami.install_partuv",
                    text="Download Geometric",
                    icon="IMPORT",
                ).tier = "GEOMETRIC"
            if not ai_installed:
                row.operator(
                    "uvgami.install_partuv", text="Download AI", icon="IMPORT"
                ).tier = "AI"
        if installed:
            row = layout.row()
            row.scale_y = 1.5
            row.operator("uvgami.uninstall_partuv", text="Delete PartUV", icon="TRASH")

        if task_state["error"] is not None and task_state["owner"] == "partuv":
            layout.row().label(text=task_state["error"], icon="ERROR")

    def build_args(self, ctx, input_path, props):
        return self.build_batch_args(ctx, [input_path], props)

    def build_batch_args(self, ctx, input_paths, props):
        if ctx.mode == "dev":
            base = [
                "uv",
                "run",
                "--project",
                str(ctx.path),
                "--no-sync",
                "python",
                "-m",
                "partuv",
            ]
        else:
            base = [str(get_partuv_venv_python()), "-m", "partuv"]
        # windows caps a command line near 32k chars, so a large batch of mesh
        # paths as argv overflows CreateProcess. named per invocation since solo
        # mode spawns several over one session, and it goes in the input dir so
        # manager.finish cleans it up
        input_list = (
            get_extension_dir_path() / "input" / f"{input_paths[0].stem}_inputs.txt"
        )
        input_list.write_text(
            "\n".join(str(path) for path in input_paths) + "\n", encoding="utf-8"
        )
        return base + [
            "--input-list",
            str(input_list),
            "--output-dir",
            str(get_extension_dir_path() / "output"),
            "--overwrite",
            "--segmentation",
            props.partuv.segmentation.lower(),
            "--threshold",
            f"{props.partuv.threshold:.3f}",
            # drives the progress bar and the live chart viewer
            "--visual",
        ]

    def build_env(self, ctx):
        env = os.environ.copy()
        # the checkpoint isn't shipped in the wheel, and the cli's source-tree
        # default resolves relative to the installed package
        if ctx.mode == "dev":
            checkpoint = ctx.path / "engine" / "partuv" / "model_objaverse.ckpt"
        else:
            checkpoint = get_partuv_checkpoint_path()
        env["UVGAMI_PARTUV_CHECKPOINT"] = str(checkpoint)
        return env

    def describe_failure(self, code):
        return {
            2: ("Invalid input mesh", True),
            # missing module, config, checkpoint or cuda, stderr says which
            3: ("PartUV could not start", False),
            4: ("PartUV failed on this mesh", True),
            5: ("PartUV produced invalid output", True),
        }.get(code) or super().describe_failure(code)

    def stop(self, process, ctx):
        if platform.system() == "Windows":
            # kill the whole tree, uv spawns the engine as a child
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
            )
        else:
            process.kill()


ENGINE = PartuvEngine()
