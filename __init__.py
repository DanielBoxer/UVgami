# Copyright (C) 2022 Daniel Boxer
#
# UVgami is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UVgami is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with UVgami. If not, see <https://www.gnu.org/licenses/>.

import bpy
from .src.manager import manager
from .src.ops.start import UVGAMI_OT_start
from .src.ops.stop import (
    UVGAMI_OT_stop,
    UVGAMI_OT_cancel,
    UVGAMI_OT_cancel_all,
)
from .src.ops.guides import (
    UVGAMI_OT_draw_guides,
    UVGAMI_OT_exit_draw,
    UVGAMI_OT_clear_draw,
)
from .src.ops.uv import (
    UVGAMI_OT_show_seams,
    UVGAMI_OT_unwrap_sharp,
    UVGAMI_OT_mark_seams_sharp,
    UVGAMI_OT_pack,
)
from .src.ops.misc import (
    UVGAMI_OT_expand,
    UVGAMI_OT_reset_settings,
    UVGAMI_OT_open_preferences,
    UVGAMI_OT_preview_symmetry,
    UVGAMI_OT_setup_wsl,
)
from .src.ops.grid import (
    UVGAMI_OT_add_grid,
    UVGAMI_OT_remove_grid,
)
from .src.ops.viewer import (
    UVGAMI_OT_view_unwrap,
    UVGAMI_OT_view_uvs,
)
from .src.ops.info import (
    UVGAMI_OT_clear_logs,
    UVGAMI_OT_copy_logs,
)
from .src.ui.panels import (
    UVGAMI_PT_main,
    UVGAMI_PT_speed,
    UVGAMI_PT_guides,
    UVGAMI_PT_symmetry,
    UVGAMI_PT_grid,
    UVGAMI_PT_pack,
    UVGAMI_PT_uv,
    UVGAMI_PT_misc,
    UVGAMI_PT_info,
)
from .src.ui.props import (
    UVGAMI_PG_properties,
    UVGAMI_AP_preferences,
)


bl_info = {
    "name": "UVgami",
    "author": "Daniel Boxer",
    "description": "Automatic UV unwrapping",
    "blender": (2, 90, 0),
    "version": (1, 1, 6),
    "location": "View3D > Sidebar > UVgami",
    "category": "UV",
    "doc_url": "https://github.com/DanielBoxer/UVgami/blob/master/docs/docs.md",
    "tracker_url": "https://github.com/DanielBoxer/UVgami/issues",
}


classes = (
    UVGAMI_OT_start,
    UVGAMI_OT_stop,
    UVGAMI_OT_cancel_all,
    UVGAMI_OT_expand,
    UVGAMI_OT_open_preferences,
    UVGAMI_OT_add_grid,
    UVGAMI_OT_draw_guides,
    UVGAMI_OT_show_seams,
    UVGAMI_OT_exit_draw,
    UVGAMI_OT_clear_draw,
    UVGAMI_OT_pack,
    UVGAMI_OT_cancel,
    UVGAMI_OT_remove_grid,
    UVGAMI_OT_view_unwrap,
    UVGAMI_OT_reset_settings,
    UVGAMI_OT_preview_symmetry,
    UVGAMI_OT_unwrap_sharp,
    UVGAMI_OT_mark_seams_sharp,
    UVGAMI_OT_clear_logs,
    UVGAMI_OT_copy_logs,
    UVGAMI_OT_setup_wsl,
    UVGAMI_OT_view_uvs,
    UVGAMI_PT_main,
    UVGAMI_PT_guides,
    UVGAMI_PT_symmetry,
    UVGAMI_PT_speed,
    UVGAMI_PT_grid,
    UVGAMI_PT_pack,
    UVGAMI_PT_uv,
    UVGAMI_PT_info,
    UVGAMI_PT_misc,
    UVGAMI_PG_properties,
    UVGAMI_AP_preferences,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.uvgami = bpy.props.PointerProperty(type=UVGAMI_PG_properties)


def unregister():
    manager.stop_all()
    del bpy.types.Scene.uvgami
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
