import shutil
import zipfile

import bpy

from ..utils.download import download_file
from ..utils.paths import (
    get_engine_binary_name,
    get_extension_dir_path,
    get_local_engine_path,
    get_platform_tag,
)
from . import Engine
from .install_task import (
    DELETE_DESCRIPTION,
    NOT_DOWNLOADED_ERROR,
    DELETED_MESSAGE,
    DOWNLOADED_MESSAGE,
    InstallTask,
    draw_error,
    draw_online_access,
    draw_progress,
    draw_update_row,
    parse_version,
    UPDATE_ICON,
    offline_error,
    report_progress,
    task_state,
)

RELEASES_URL = "https://github.com/DanielBoxer/UVgami/releases"

DOWNLOAD_PHASE = "Downloading engine"


def engine_install_root(name):
    """Where every downloaded version of one engine lives."""
    return get_extension_dir_path() / name


class EngineRelease:
    """An engine binary published as a <name>-v<version> GitHub release. The
    addon ships no binaries, so this download is how the engine arrives."""

    def __init__(self, name, label, version, minimum_version, download_size):
        self.name = name
        self.label = label
        self.version = version
        self.minimum_version = minimum_version
        self.download_size = download_size
        self.install_op = f"uvgami.install_{name}"

    def install_dir(self):
        # named by version, which is how installed_version reads it back
        return engine_install_root(self.name) / self.version

    def is_downloaded(self):
        """Whether any version of this engine was downloaded, which is what the
        delete button clears. A local build or the engine path can be what
        actually runs while a download still sits there."""
        root = engine_install_root(self.name)
        return root.is_dir() and any(d.is_dir() for d in root.iterdir())

    def installed_version(self):
        """The newest version whose binary is there, or None. A dead download
        leaves the folder without one."""
        root = engine_install_root(self.name)
        if not root.is_dir():
            return None
        binary_name = get_engine_binary_name(self.name)
        names = [
            d.name
            for d in root.iterdir()
            if parse_version(d.name) is not None and (d / binary_name).is_file()
        ]
        return max(names, key=parse_version, default=None)

    def installed_path(self):
        version = self.installed_version()
        if version is None or self.install_too_old():
            return None
        return (
            engine_install_root(self.name) / version / get_engine_binary_name(self.name)
        )

    def installed_below(self, bound):
        version = self.installed_version()
        return version is not None and parse_version(version) < parse_version(bound)

    def install_too_old(self):
        """Whether the download is older than the addon can run."""
        return self.installed_below(self.minimum_version)

    def update_available(self):
        """Whether the download still runs but a newer engine is pinned."""
        return self.installed_below(self.version)

    def install(self):
        install_dir = self.install_dir()
        install_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.fetch_binary(install_dir)
        except Exception:
            # an empty version dir counts as downloaded
            shutil.rmtree(install_dir, ignore_errors=True)
            raise

        # drop engines left behind by older addon versions
        for old in install_dir.parent.iterdir():
            if old.is_dir() and old != install_dir:
                shutil.rmtree(old)

    def fetch_binary(self, install_dir):
        asset = f"{self.name}-engine-{self.version}-{get_platform_tag()}.zip"
        url = f"{RELEASES_URL}/download/{self.name}-v{self.version}/{asset}"
        archive_path = install_dir / asset
        task_state["phase"] = DOWNLOAD_PHASE
        download_file(url, archive_path, progress=report_progress)

        binary = install_dir / get_engine_binary_name(self.name)
        with zipfile.ZipFile(archive_path) as archive:
            # the cpack zip nests the binary in a package folder
            member = next(
                (
                    name
                    for name in archive.namelist()
                    if name.rsplit("/", 1)[-1] == binary.name
                ),
                None,
            )
            if member is None:
                raise RuntimeError(f"{asset} has no {binary.name}")
            with archive.open(member) as src, open(binary, "wb") as dst:
                shutil.copyfileobj(src, dst)
        archive_path.unlink()
        binary.chmod(0o755)


class UVGAMI_OT_delete_engine(InstallTask, bpy.types.Operator):
    """Shared by every downloaded engine, so it takes the engine name."""

    bl_idname = "uvgami.delete_engine"
    bl_label = "Delete Engine"
    done_message = DELETED_MESSAGE
    bl_description = DELETE_DESCRIPTION

    engine_name: bpy.props.StringProperty(options={"HIDDEN"})

    @property
    def owner(self):
        return self.engine_name

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def build_task(self):
        root = engine_install_root(self.engine_name)

        def delete():
            task_state["phase"] = "Deleting engine"
            shutil.rmtree(root, ignore_errors=True)

        return delete


class InstallEngineTask(InstallTask):
    """Operator body for downloading one engine binary."""

    done_message = DOWNLOADED_MESSAGE
    release = None

    @classmethod
    def description(cls, context, properties):
        action = "Update" if cls.release.update_available() else "Download"
        return f"{action} the engine"

    def precheck(self):
        if get_platform_tag() is None:
            return f"{self.release.label} has no build for this platform"
        return offline_error()

    def invoke(self, context, event):
        action = "Update" if self.release.update_available() else "Download"
        return context.window_manager.invoke_confirm(
            self,
            event,
            title=f"{action} {self.release.label}",
            message=self.release.download_size,
            confirm_text=action,
        )

    def build_task(self):
        return self.release.install


class BinaryEngine(Engine):
    """Engine that runs a binary downloaded from its own GitHub release."""

    release = None

    def validate(self, prefs):
        # a local build wins over the download, for dev checkouts
        path = get_local_engine_path(self.release.name) or self.release.installed_path()
        if path is not None:
            return path, None
        if self.release.install_too_old():
            return (
                None,
                "This update needs a newer engine. Download it in the add-on"
                " preferences",
            )
        return None, NOT_DOWNLOADED_ERROR

    def describe(self):
        if get_local_engine_path(self.release.name) is not None:
            return f"{self.label} {self.release.version} (local build)"
        version = self.release.installed_version() or self.release.version
        return f"{self.label} {version}"

    def update_pending(self):
        if get_local_engine_path(self.release.name) is not None:
            return False
        return self.release.update_available()

    def draw_update_notice(self, layout):
        required = self.release.install_too_old()
        state = "required" if required else "available"
        text = f"{self.label} update {state}" if self.update_pending() else None
        draw_update_row(layout, self.release.name, DOWNLOAD_PHASE, text, required)

    def draw_prefs(self, layout, prefs):
        release = self.release
        if task_state["running"] and task_state["owner"] == release.name:
            draw_progress(layout, DOWNLOAD_PHASE)
            return

        _, error = self.validate(prefs)
        needs_download = error is not None
        update_pending = self.update_pending()
        if release.install_too_old():
            status = ("Engine update required", "ERROR")
        elif update_pending:
            status = ("Engine update available", "FILE_REFRESH")
        elif needs_download:
            status = ("Not downloaded", "X")
        elif error is not None:
            status = (error, "ERROR")
        elif get_local_engine_path(release.name) is not None:
            status = ("Local build", "CHECKMARK")
        else:
            status = ("Downloaded", "CHECKMARK")

        layout.row().label(text=status[0], icon=status[1])

        wants_download = needs_download or update_pending
        if wants_download or release.is_downloaded():
            row = layout.row()
            row.scale_y = 1.5
            if wants_download and not draw_online_access(row):
                row.operator(
                    release.install_op,
                    text="Update Engine" if update_pending else "Download Engine",
                    icon=UPDATE_ICON if update_pending else "IMPORT",
                )
            if release.is_downloaded():
                delete = row.operator(
                    "uvgami.delete_engine", text="Delete Engine", icon="TRASH"
                )
                delete.engine_name = release.name

        draw_error(layout, release.name)
