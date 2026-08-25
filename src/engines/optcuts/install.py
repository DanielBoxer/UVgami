import bpy

from ..binary_engine import EngineRelease, InstallEngineTask

# must match engine/optcuts/VERSION (check-engine-versions.yml fails on drift)
OPTCUTS_VERSION = "1.20.2"
# raise this to the release that adds a flag, a stdin command or an input file
# the addon then sends
OPTCUTS_MINIMUM_VERSION = "1.20.0"
OPTCUTS = EngineRelease(
    "optcuts", "Optcuts", OPTCUTS_VERSION, OPTCUTS_MINIMUM_VERSION, "2 MB"
)


class UVGAMI_OT_install_optcuts(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_optcuts"
    bl_label = "Download Optcuts Engine"
    owner = "optcuts"
    release = OPTCUTS
