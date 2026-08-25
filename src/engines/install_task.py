import threading

import bpy

from ..utils.ui import tag_redraw

# written by the install or uninstall thread, read by the preferences ui.
# shared across engines, so only one install task runs at a time. owner names
# the engine whose task ran last, so each prefs section shows only its own.
task_state = {
    "running": False,
    "owner": "",
    "error": None,
    "phase": "",
    "bytes_done": 0,
    "bytes_total": None,
}


# the unwrap panels show what is installed
INSTALL_AREAS = ("PREFERENCES", "VIEW_3D", "IMAGE_EDITOR")

DOWNLOADED_MESSAGE = "Engine downloaded"
DELETED_MESSAGE = "Engine deleted"
DELETE_DESCRIPTION = "Delete the engine"
NOT_DOWNLOADED_ERROR = "Engine not downloaded. Download it in the add-on preferences"


def parse_version(version):
    """(1, 20, 2) for "1.20.2", or None when the name is not a version."""
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def offline_error():
    """Blender requires add-ons to check this before any download."""
    if bpy.app.online_access:
        return None
    if bpy.app.online_access_override:
        return "Blender was started in offline mode"
    return "Turn on Allow Online Access in the preferences"


def draw_online_access(layout):
    """Blender's own button for the Allow Online Access preference. True when it
    drew, meaning no download can run yet."""
    if bpy.app.online_access:
        return False
    row = layout.row()
    row.scale_y = 1.5
    # its poll explains itself when blender was started in offline mode
    row.operator(
        "extensions.userpref_allow_online", text="Allow Online Access", icon="URL"
    )
    return True


def report_progress(done, total):
    task_state["bytes_done"] = done
    task_state["bytes_total"] = total


def _run_task(task):
    try:
        task()
    except Exception as error:
        task_state["error"] = str(error)
    finally:
        task_state["running"] = False


class InstallTask:
    """Runs an engine install or uninstall on a thread, with a modal that
    redraws the preferences while it works. Subclasses return the work as a
    callable from build_task and may reject the run from precheck."""

    done_message = ""
    owner = ""

    def build_task(self):
        raise NotImplementedError

    def precheck(self):
        """Return an error message to block the run, or None to proceed."""
        return None

    def execute(self, context):
        if task_state["running"]:
            self.report({"WARNING"}, "An engine install or delete is already running")
            return {"CANCELLED"}
        error = self.precheck()
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        task_state["running"] = True
        task_state["owner"] = self.owner
        task_state["error"] = None
        task_state["phase"] = ""
        task_state["bytes_done"] = 0
        task_state["bytes_total"] = None
        # built here so the thread never reads operator properties
        threading.Thread(
            target=_run_task, args=(self.build_task(),), daemon=True
        ).start()

        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if task_state["running"]:
            tag_redraw(area_types=INSTALL_AREAS)
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        # imported late: the engines package imports this module while loading
        from . import invalidate_engine_caches

        invalidate_engine_caches()
        tag_redraw(area_types=INSTALL_AREAS)
        if task_state["error"] is not None:
            self.report({"ERROR"}, f"{self.bl_label} failed: {task_state['error']}")
            return {"CANCELLED"}
        self.report({"INFO"}, self.done_message)
        return {"FINISHED"}


UPDATE_ICON = "FILE_REFRESH"
UPDATE_LABEL_SPLIT = 0.9


def draw_update_row(layout, owner, default_phase, text, required=False):
    """Shared body for Engine.draw_update_notice: progress while this engine's
    task runs, else the message. text is None when no update is pending. The
    download button for it is in the preferences, not here."""
    if task_state["running"] and task_state["owner"] == owner:
        draw_progress(layout, default_phase)
        return
    if text is None:
        return
    split = layout.split(factor=UPDATE_LABEL_SPLIT)
    split.label(text=text, icon="ERROR" if required else "INFO")
    split.row().operator("uvgami.open_preferences", text="", icon="PREFERENCES")


def draw_error(layout, owner):
    """The last failure for this engine, kept after the operator report is gone.
    Only the first line: a label cannot wrap."""
    if task_state["error"] is None or task_state["owner"] != owner:
        return
    layout.row().label(text=task_state["error"].partition("\n")[0], icon="ERROR")


def draw_progress(layout, default_phase):
    """Draw the running task's progress row in the preferences."""
    row = layout.row()
    phase = task_state["phase"] or default_phase
    total = task_state["bytes_total"]
    if total:
        factor = task_state["bytes_done"] / total
        row.progress(factor=factor, type="BAR", text=f"{phase}  {factor * 100:.0f}%")
    else:
        row.label(text=phase, icon="SORTTIME")
