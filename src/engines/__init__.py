import importlib.util

from ..utils.paths import get_preferences


class Engine:
    id = ""
    label = ""
    description = ""
    icon = "TOOL_SETTINGS"
    # saved in blend files, never change or reuse one
    enum_value = 0
    property_group = None
    # every bpy class the engine needs, its property group plus any operators
    classes = ()
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    supports_import_uvs = False
    # the proxy finish flattens with optcuts
    supports_proxy = False

    def describe(self):
        return self.label

    def is_available(self):
        return True

    def is_installed(self, prefs):
        return self.validate(prefs)[1] is None

    # returns (ctx, None) or (None, error_message). ctx comes back to build_* and stop
    def validate(self, prefs):
        raise NotImplementedError

    def invalidate_caches(self):
        pass

    def draw_settings(self, layout, props):
        pass

    # (icon, label, path) entries for the panel's active strip
    def active_settings(self, props):
        return []

    def import_uvs_ignored(self, props):
        return False

    def uses_import_uvs(self, props):
        return (
            props.import_uvs
            and self.supports_import_uvs
            and not self.import_uvs_ignored(props)
        )

    def uses_proxy(self, props):
        return props.use_proxy and self.supports_proxy

    # obj is a temp copy, safe to edit
    def prepare_uvs(self, obj, props):
        return self.uses_import_uvs(props)

    # None means no slow work and prepare_uvs is used directly
    def preseed_work(self, obj, props, mirrors=None):
        return None

    # has_uvs is what prepare_uvs returned for the whole object
    def piece_uses_uvs(self, obj, props, has_uvs):
        return has_uvs

    def draw_prefs(self, layout, prefs):
        pass

    def draw_update_notice(self, layout):
        pass

    # batching and running several processes at once are mutually exclusive
    def batches_queue(self, props):
        return False

    def build_args(self, ctx, input_path, props):
        raise NotImplementedError

    # required when batches_queue can return True
    def build_batch_args(self, ctx, input_paths, props):
        raise NotImplementedError

    # every non-None argv in a session must match apart from threads
    def build_shared_args(self, ctx, input_path, props, threads):
        return None

    # None inherits
    def build_env(self, ctx):
        pass

    # returns (message, move_to_invalid), or None when the code is unrecognized
    def describe_failure(self, code):
        # windows access violation (0xC0000005)
        if code == -1073741819:
            return ("Engine crashed", True)
        return None

    # required when supports_early_stop is set
    def request_early_stop(self, process):
        raise NotImplementedError

    # required when supports_viewer is set
    def request_snapshot(self, process):
        raise NotImplementedError

    # False means the engine has no cancel command and the solve runs to the end
    def request_cancel(self, process):
        return False

    def stop(self, process, ctx):
        process.kill()


# imported after Engine because each module subclasses it
from . import optcuts, xatlas  # noqa: E402

# order sets the enum and ui order
_engines = [optcuts.ENGINE, xatlas.ENGINE]

# partuv is optional, some builds ship without its folder
if importlib.util.find_spec(f"{__name__}.partuv") is not None:
    from . import partuv  # noqa: E402

    _engines.append(partuv.ENGINE)

ENGINES = {e.id: e for e in _engines}


def get_engine(engine_id):
    return ENGINES[engine_id]


# panel polls and the engine enum hit this on every redraw
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


# the scene enum's getter clamps to an installed engine
def active_engine(engine_id):
    engine = ENGINES.get(engine_id)
    if engine is not None and engine in installed_engines():
        return engine
    return None
