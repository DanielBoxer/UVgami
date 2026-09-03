import bpy
import gpu
from gpu_extras.batch import batch_for_shader

COLOUR = [(0.0, 0.0, 0.7355, 1), (0.6, 1.0, 0.6, 1), (1.0, 0.0, 0.0, 1)]
X = 25
Y = 25
WIDTH = 150
TOP = Y + 5
# created on first use, gpu is unavailable at import time in background mode
SHADER = None


def _get_shader():
    global SHADER
    if SHADER is None:
        SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return SHADER


class ProgressBar:
    def __init__(self):
        self._batch = [None, None, None]
        self._handle = [None, None, None]
        self._space = None
        self.is_active = False

    def _draw(self, index):
        space = bpy.context.space_data
        # the image editor space is also the image viewer and the paint modes
        if space.type == "IMAGE_EDITOR" and space.mode != "UV":
            return
        shader = _get_shader()
        shader.bind()
        shader.uniform_float("color", COLOUR[index])
        self._batch[index].draw(shader)

    # draws in the editor the run was started from
    def start(self, uv_editor=False):
        # re-registering would leak the old draw handlers
        if self.is_active:
            return
        self.is_active = True
        self._space = bpy.types.SpaceImageEditor if uv_editor else bpy.types.SpaceView3D
        self.update((0, 0, 1))
        for idx in range(3):
            self._handle[idx] = self._space.draw_handler_add(
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
                _get_shader(),
                "TRIS",
                {"pos": vertices[idx]},
                indices=((0, 1, 2), (2, 1, 3)),
            )

    def remove(self):
        if self.is_active:
            self.is_active = False
            for idx in range(3):
                self._space.draw_handler_remove(self._handle[idx], "WINDOW")


progress_bar = ProgressBar()
