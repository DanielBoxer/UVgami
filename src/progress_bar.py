# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import gpu
from gpu_extras.batch import batch_for_shader


COLOUR = [(0.0, 0.0, 0.7355, 1), (0.6, 1.0, 0.6, 1), (1.0, 0.0, 0.0, 1)]
X = 25
Y = 25
WIDTH = 150
TOP = Y + 5
SHADER = gpu.shader.from_builtin("2D_UNIFORM_COLOR")
batch = [None, None, None]
handle = [None, None, None]
is_active = False


def draw(index):
    SHADER.bind()
    SHADER.uniform_float("color", COLOUR[index])
    batch[index].draw(SHADER)


def start():
    global is_active
    is_active = True
    update((0, 0, 1))
    for idx in range(3):
        handle[idx] = bpy.types.SpaceView3D.draw_handler_add(
            draw, (idx,), "WINDOW", "POST_PIXEL"
        )


def update(percentages):
    start = X
    vertices = []
    for idx in range(3):
        end = (WIDTH * percentages[idx]) + start
        vertices.append(
            (
                (start, Y),
                (end, Y),
                (start, TOP),
                (end, TOP),
            )
        )
        start = end

    for idx in range(3):
        batch[idx] = batch_for_shader(
            SHADER,
            "TRIS",
            {"pos": vertices[idx]},
            indices=((0, 1, 2), (2, 1, 3)),
        )


def remove():
    global is_active
    if is_active:
        is_active = False
        for idx in range(3):
            bpy.types.SpaceView3D.draw_handler_remove(handle[idx], "WINDOW")
