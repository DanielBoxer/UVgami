import bpy

from ..binary_engine import EngineRelease, InstallEngineTask

# must match engine/xatlas/VERSION (check-engine-versions.yml fails on drift)
XATLAS_VERSION = "0.2.3"
XATLAS = EngineRelease("xatlas", "xatlas", XATLAS_VERSION, "300 KB")


class UVGAMI_OT_install_xatlas(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_xatlas"
    bl_label = "Download xatlas Engine"
    owner = "xatlas"
    release = XATLAS
