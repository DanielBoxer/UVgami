import importlib.util

from ..utils.paths import get_preferences


class Engine:
    id = ""
    label = ""
    description = ""
    icon = "TOOL_SETTINGS"
    # the engine enum value saved in blend files. must be unique and must never
    # change or be reused, and it can't be derived from ENGINES because not
    # every build ships every engine
    enum_value = 0
    # every bpy class the engine needs (its property group plus any operators)
    property_group = None
    classes = ()
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    supports_import_uvs = False
    # the proxy finish flattens with optcuts
    supports_proxy = False

    def describe(self):
        """Engine name and version for the log."""
        return self.label

    def is_available(self):
        """Whether this engine can run on the current platform."""
        return True

    def is_installed(self, prefs):
        """Whether the engine is ready to run, so the dropdown lists it."""
        return self.validate(prefs)[1] is None

    def validate(self, prefs):
        """Return (ctx, None) if usable, else (None, error_message). ctx is an
        engine-defined run context passed back to the build_* and stop methods."""
        raise NotImplementedError

    def invalidate_caches(self):
        """Drop anything the engine cached about its own install."""

    def draw_settings(self, layout, props):
        """Draw this engine's settings rows in the main panel."""

    def active_settings(self, props):
        """This engine's non-default settings, as (icon, label, path) entries
        for the panel's active strip."""
        return []

    def import_uvs_ignored(self, props):
        """Whether a setting of this engine replaces the mesh's uv map, which
        leaves Import UVs with nothing to do."""
        return False

    def uses_import_uvs(self, props):
        return (
            props.import_uvs
            and self.supports_import_uvs
            and not self.import_uvs_ignored(props)
        )

    def uses_proxy(self, props):
        return props.use_proxy and self.supports_proxy

    def prepare_uvs(self, obj, props):
        """Return whether to export obj's uv map, building one first if the
        engine wants seams of its own. obj is a temp copy, safe to edit."""
        return self.uses_import_uvs(props)

    def preseed_work(self, obj, props, mirrors=None):
        """Split prepare_uvs for the start operator's worker thread: a
        (compute, apply) pair where compute touches no bpy data and runs off
        the main thread, and apply(compute()) writes the result back and
        returns whether the preseed applied uvs. None means there is no slow
        work and prepare_uvs is used directly. mirrors are the symmetry
        vertex maps the seams should close under, for engines that seam."""
        return None

    def piece_uses_uvs(self, obj, props, has_uvs):
        """Per separated piece, whether its uv map goes to the engine.
        has_uvs is what prepare_uvs returned for the whole object."""
        return has_uvs

    def draw_prefs(self, layout, prefs):
        """Draw this engine's section in the addon preferences."""

    def draw_update_notice(self, layout):
        """Draw a row in the unwrap panels when a newer engine is pinned."""

    def batches_queue(self, props):
        """Whether queued meshes share one engine process. Batching and running
        several processes at once are mutually exclusive."""
        return False

    def build_args(self, ctx, input_path, props):
        """Return the subprocess argv that unwraps input_path."""
        raise NotImplementedError

    def build_batch_args(self, ctx, input_paths, props):
        """Return the argv unwrapping all input_paths in one process. Must be
        implemented when batches_queue can return True."""
        raise NotImplementedError

    def build_shared_args(self, ctx, input_path, props, threads):
        """Return the argv of a process that takes input_path and later meshes
        over stdin, one at a time, or None when input_path needs a process of
        its own. The processes are shared, so apart from threads (a cap on
        the process, 0 for every core, speed only) every non-None value in a
        session must be the same argv."""
        return None

    def build_env(self, ctx):
        """Return the subprocess env, or None to inherit."""

    def describe_failure(self, code):
        """Map an engine exit code to (message, move_to_invalid), or None if the
        engine does not recognize it (caller shows a generic unknown-error)."""
        # windows access violation (0xC0000005): the engine process crashed
        if code == -1073741819:
            return ("Engine crashed", True)
        return None

    def request_early_stop(self, process):
        """Ask a running process to stop and finish with its current result.
        Returns True if delivered. Only called when supports_early_stop is set."""
        raise NotImplementedError

    def request_snapshot(self, process):
        """Ask a running process to emit a uv snapshot for the live viewer. Only
        called when supports_viewer is set."""
        raise NotImplementedError

    def stop(self, process, ctx):
        """Stop a running unwrap process."""
        process.kill()


# imported after Engine because each module subclasses it
from . import optcuts, xatlas  # noqa: E402

# order sets the enum/ui order
_engines = [optcuts.ENGINE, xatlas.ENGINE]

# partuv is optional, some builds ship without its folder
if importlib.util.find_spec(f"{__name__}.partuv") is not None:
    from . import partuv  # noqa: E402

    _engines.append(partuv.ENGINE)

ENGINES = {e.id: e for e in _engines}


def get_engine(engine_id):
    return ENGINES[engine_id]


# panel polls and the engine enum call installed_engines on every redraw, and
# is_installed stats the filesystem, so the result is cached until an install
# task ends
_installed_cache = None


def invalidate_engine_caches():
    global _installed_cache
    _installed_cache = None
    for engine in ENGINES.values():
        engine.invalidate_caches()


def installed_engines():
    global _installed_cache
    if _installed_cache is None:
        prefs = get_preferences()
        _installed_cache = [
            e for e in ENGINES.values() if e.is_available() and e.is_installed(prefs)
        ]
    return _installed_cache


def active_engine(engine_id):
    """The installed engine an unwrap will run, or None when none is installed.
    The scene enum's getter clamps to an installed engine, so the id read back
    only misses when nothing is installed and the enum is the placeholder."""
    engine = ENGINES.get(engine_id)
    if engine is not None and engine in installed_engines():
        return engine
    return None
