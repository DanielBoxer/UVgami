import threading

import bpy

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


def offline_error():
    """Blender requires add-ons to check this before any download."""
    if bpy.app.online_access:
        return None
    if bpy.app.online_access_override:
        return "Blender was started in offline mode"
    return "Turn on Allow Online Access in the preferences"


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
            # preferences can be open in more than one window, so redraw them
            # all. the unwrap panels show progress too
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type in {"PREFERENCES", "VIEW_3D", "IMAGE_EDITOR"}:
                        area.tag_redraw()
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        # imported late: the engines package imports this module while loading
        from . import invalidate_engine_caches

        invalidate_engine_caches()
        for area in context.screen.areas:
            area.tag_redraw()
        if task_state["error"] is not None:
            self.report({"ERROR"}, f"{self.bl_label} failed: {task_state['error']}")
            return {"CANCELLED"}
        self.report({"INFO"}, self.done_message)
        return {"FINISHED"}


def draw_update_row(layout, owner, default_phase, pending):
    """Shared body for Engine.draw_update_notice: progress while this engine's
    task runs, else the update label. Returns the row the engine puts its
    update button on, or None when nothing should draw."""
    if task_state["running"] and task_state["owner"] == owner:
        draw_progress(layout, default_phase)
        return None
    if not pending:
        return None
    row = layout.row()
    row.label(text="Engine update available", icon="FILE_REFRESH")
    return row


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
