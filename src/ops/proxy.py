import bpy

from ..proxy import make_proxy, triangle_count
from ..utils.mesh import deselect_all


def _decimated_copy(obj, target_faces):
    copy_object = obj.copy()
    copy_object.data = obj.data.copy()
    copy_object.animation_data_clear()
    copy_object.name = f"{obj.name} Proxy"
    # an unwrap drops modifiers before it decimates
    copy_object.modifiers.clear()
    obj.users_collection[0].objects.link(copy_object)

    if not make_proxy(copy_object, target_faces):
        unused = copy_object.data
        bpy.data.objects.remove(copy_object)
        bpy.data.meshes.remove(unused)
        return None
    return copy_object


class UVGAMI_OT_preview_proxy(bpy.types.Operator):
    bl_idname = "uvgami.preview_proxy"
    bl_label = "Preview Proxy"
    bl_description = "Preview the proxy decimated mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            obj.type == "MESH" for obj in context.selected_objects
        )

    def execute(self, context):
        props = context.scene.uvgami
        proxies = []
        for obj in [obj for obj in context.selected_objects if obj.type == "MESH"]:
            proxy = _decimated_copy(obj, props.proxy_faces)
            if proxy is None:
                continue
            obj.hide_set(True)
            proxies.append(proxy)

        if not proxies:
            self.report({"WARNING"}, "Selection is already under Proxy Faces")
            return {"CANCELLED"}

        deselect_all()
        for proxy in proxies:
            proxy.select_set(True)
        context.view_layer.objects.active = proxies[-1]

        triangles = sum(triangle_count(proxy) for proxy in proxies)
        self.report({"INFO"}, f"Proxy: {triangles} triangles")
        return {"FINISHED"}
