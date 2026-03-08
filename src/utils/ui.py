import bpy


def newline_label(label, layout):
    for line in label:
        layout.label(text=line)


def popup(msg, title, icon):
    def draw(self, context):
        newline_label(msg, self.layout)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def switch_shading(type):
    for area in bpy.context.screen.areas:
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = type
                break
