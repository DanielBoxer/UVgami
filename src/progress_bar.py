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
SHADER_NAME = "UNIFORM_COLOR" if bpy.app.version >= (4, 0, 0) else "2D_UNIFORM_COLOR"
SHADER = gpu.shader.from_builtin(SHADER_NAME)


class ProgressBar:
    def __init__(self):
        self._batch = [None, None, None]
        self._handle = [None, None, None]
        self.is_active = False

    def _draw(self, index):
        SHADER.bind()
        SHADER.uniform_float("color", COLOUR[index])
        self._batch[index].draw(SHADER)

    def start(self):
        self.is_active = True
        self.update((0, 0, 1))
        for idx in range(3):
            self._handle[idx] = bpy.types.SpaceView3D.draw_handler_add(
                self._draw, (idx,), "WINDOW", "POST_PIXEL"
            )

    def update(self, percentages):
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
            self._batch[idx] = batch_for_shader(
                SHADER,
                "TRIS",
                {"pos": vertices[idx]},
                indices=((0, 1, 2), (2, 1, 3)),
            )

    def remove(self):
        if self.is_active:
            self.is_active = False
            for idx in range(3):
                bpy.types.SpaceView3D.draw_handler_remove(self._handle[idx], "WINDOW")


progress_bar = ProgressBar()
