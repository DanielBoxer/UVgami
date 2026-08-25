import bpy

from ..binary_engine import EngineRelease, InstallEngineTask

# must match engine/xatlas/VERSION (check-engine-versions.yml fails on drift)
XATLAS_VERSION = "0.2.3"
# see OPTCUTS_MINIMUM_VERSION for when this has to go up
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS = EngineRelease(
    "xatlas", "xatlas", XATLAS_VERSION, XATLAS_MINIMUM_VERSION, "300 KB"
)


class UVGAMI_OT_install_xatlas(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_xatlas"
    bl_label = "Download xatlas Engine"
    owner = "xatlas"
    release = XATLAS
