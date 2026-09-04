# Copyright (C) 2022-2026 Daniel Boxer
# See LICENSE for more information
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

from .src.engines import ENGINES
from .src.engines.binary_engine import UVGAMI_OT_delete_engine
from .src.manager import manager
from .src.ops.grid import (
    UVGAMI_OT_add_grid,
    UVGAMI_OT_remove_grid,
)
from .src.ops.guides import (
    UVGAMI_OT_clear_draw,
    UVGAMI_OT_draw_guides,
    UVGAMI_OT_exit_draw,
    UVGAMI_OT_seed_restrictions,
)
from .src.ops.info import (
    UVGAMI_OT_clear_summary,
    UVGAMI_OT_open_logs,
)
from .src.ops.island import (
    UVGAMI_OT_combine_islands,
    UVGAMI_OT_relax_area,
    UVGAMI_OT_relax_island,
    UVGAMI_OT_unwrap_area,
    UVGAMI_OT_unwrap_island,
)
from .src.ops.misc import (
    UVGAMI_OT_expand,
    UVGAMI_OT_open_preferences,
    # start_symmetry_draw,
    # stop_symmetry_draw,
    UVGAMI_OT_reset_setting,
    UVGAMI_OT_reset_settings,
)
from .src.ops.proxy import UVGAMI_OT_preview_proxy
from .src.ops.start import UVGAMI_OT_start
from .src.ops.stop import (
    UVGAMI_OT_cancel,
    UVGAMI_OT_cancel_all,
    UVGAMI_OT_cancel_background,
    UVGAMI_OT_stop,
)
from .src.ops.uv import UVGAMI_OT_pack
from .src.ops.viewer import UVGAMI_OT_view_unwrap
from .src.ui.panels import (
    UVGAMI_PT_grid,
    UVGAMI_PT_island_settings,
    # UVGAMI_PT_symmetry,
    UVGAMI_PT_island_uv,
    UVGAMI_PT_main,
    UVGAMI_PT_misc,
    UVGAMI_PT_pack,
    UVGAMI_PT_speed,
    UVGAMI_PT_weights,
)
from .src.ui.props import (
    UVGAMI_AP_preferences,
    UVGAMI_PG_properties,
)

# every bpy class each engine needs registered (property groups and operators)
engine_classes = tuple(cls for engine in ENGINES.values() for cls in engine.classes)


classes = (
    UVGAMI_OT_start,
    UVGAMI_OT_stop,
    UVGAMI_OT_cancel_all,
    UVGAMI_OT_cancel_background,
    UVGAMI_OT_expand,
    UVGAMI_OT_open_preferences,
    UVGAMI_OT_add_grid,
    UVGAMI_OT_draw_guides,
    UVGAMI_OT_seed_restrictions,
    UVGAMI_OT_exit_draw,
    UVGAMI_OT_clear_draw,
    UVGAMI_OT_pack,
    UVGAMI_OT_preview_proxy,
    UVGAMI_OT_unwrap_island,
    UVGAMI_OT_relax_island,
    UVGAMI_OT_combine_islands,
    UVGAMI_OT_unwrap_area,
    UVGAMI_OT_relax_area,
    UVGAMI_OT_cancel,
    UVGAMI_OT_remove_grid,
    UVGAMI_OT_view_unwrap,
    UVGAMI_OT_reset_setting,
    UVGAMI_OT_reset_settings,
    UVGAMI_OT_clear_summary,
    UVGAMI_OT_open_logs,
    UVGAMI_PT_main,
    UVGAMI_PT_weights,
    # UVGAMI_PT_symmetry,
    UVGAMI_PT_island_uv,
    UVGAMI_PT_island_settings,
    UVGAMI_PT_speed,
    UVGAMI_PT_grid,
    UVGAMI_PT_pack,
    UVGAMI_PT_misc,
    # shared by every binary engine, so it isn't in one engine's classes
    UVGAMI_OT_delete_engine,
    # engine groups must register before the main group that points to them
    *engine_classes,
    UVGAMI_PG_properties,
    UVGAMI_AP_preferences,
)


@bpy.app.handlers.persistent
def _on_load_pre(*args):
    # load_pre, not post, so cleanup can still touch the objects it made
    manager.shutdown()


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.uvgami = bpy.props.PointerProperty(type=UVGAMI_PG_properties)
    bpy.app.handlers.load_pre.append(_on_load_pre)
    # start_symmetry_draw()


def unregister():
    manager.shutdown()
    # stop_symmetry_draw()
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    del bpy.types.Scene.uvgami
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
