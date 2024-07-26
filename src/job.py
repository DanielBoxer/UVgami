# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import bmesh
import re
from .utils import new_bmesh, set_bmesh, check_exists
from .ui.panels import expand
from .logger import logger


class Job:
    def __init__(self, count):
        self.count = count
        self.unwrapped = []

    def is_completed(self):
        return len(self.unwrapped) == self.count


class Preserve(Job):
    def __init__(self, count):
        super().__init__(count)
        self.type = "PRESERVE"

    def finish(self, unwrap, output, added_edges):
        # return mesh to original state
        bm = new_bmesh(output)

        e_dict = {}
        for edge in bm.edges:
            e_dict[(edge.verts[0].index, edge.verts[1].index)] = edge

        # check if the edges are set already
        if not added_edges:
            added_edges = unwrap.added_edges

        if bpy.context.scene.uvgami.maintain_mode == "PARTIAL":
            # get seams so they can be avoided
            uvs = []
            uv_idcs = []
            uvvert_to_meshvert = {}
            mesh_verts = []
            uv_count = 0
            uv_layer = bm.loops.layers.uv.active

            # get uv data
            for face in bm.faces:
                uv_i = []
                for loop in face.loops:
                    # get uv coordinate
                    uv = loop[uv_layer].uv
                    uvs.append((uv.x, uv.y, 0))
                    # all face points are added, duplicates are removed later
                    # that means the index is new each time
                    uv_i.append(uv_count)
                    uv_count += 1
                    # store the original mesh vertex so it can be accessed using the uvs
                    mesh_verts.append(loop.vert)
                uv_idcs.append(uv_i)

            # make uv bmesh out of mesh data
            mesh_data = bpy.data.meshes.new("")
            mesh_data.from_pydata(uvs, [], uv_idcs)
            uvbm = bmesh.new()
            uvbm.from_mesh(mesh_data)

            # make lookup table between mesh and uv mesh
            for uv_v_idx, uv_v in enumerate(uvbm.verts):
                uvvert_to_meshvert[uv_v] = mesh_verts[uv_v_idx]

            # the faces will all be separate, so merging by distance joins them
            bmesh.ops.remove_doubles(uvbm, verts=uvbm.verts, dist=0.0001)

            # find boundary edges of uv bmesh which are seams of original bmesh
            seams = []
            for e in uvbm.edges:
                if e.is_boundary:
                    uv_v1 = e.verts[0]
                    uv_v2 = e.verts[1]

                    m_v1 = uvvert_to_meshvert[uv_v1]
                    m_v2 = uvvert_to_meshvert[uv_v2]

                    # get edge from vertices
                    for edge in m_v1.link_edges:
                        if edge.other_vert(m_v1) is m_v2:
                            seams.append(edge)
            uvbm.free()

        dissolve_edges = []
        for e in added_edges:
            bm_edge = None

            if e in e_dict:
                bm_edge = e_dict[e]
            elif (e[1], e[0]) in e_dict:
                bm_edge = e_dict[(e[1], e[0])]
            else:
                # this shouldn't happen, edge not found
                if (
                    logger.get_latest().errors
                    and logger.get_latest().errors[-1]
                    == "    Error removing added edge"
                ):
                    # don't add duplicate errors
                    continue
                logger.add_data("errors", "Error removing added edge")
                # skip removing edge
                continue

            if bpy.context.scene.uvgami.maintain_mode == "PARTIAL":
                if bm_edge not in seams:
                    dissolve_edges.append(bm_edge)
            else:
                dissolve_edges.append(bm_edge)

        bmesh.ops.dissolve_edges(bm, edges=dissolve_edges)
        set_bmesh(bm, output)


class Join(Job):
    def __init__(self, count):
        super().__init__(count)
        self.type = "JOIN"

    def finish(self, unwrap):
        paths = [u.path.parents[1] / "output" / u.path.name for u in self.unwrapped]
        edge_path = unwrap.edge_path

        # set the current path to the first obj in the job
        # this is the file that will be imported
        path = paths[0]

        prev_v = 0
        prev_vt = 0
        # go through first obj file to get the starting size
        with paths[0].open() as f:
            for line in f:
                if line.startswith("v "):
                    prev_v += 1
                elif line.startswith("vt "):
                    prev_vt += 1
                elif line.startswith("f "):
                    break

        new_v = prev_v
        new_vt = prev_vt
        with paths[0].open("a") as f:
            # since there are multiple obj files combined, the size of the
            # previous ones must be added to the index numbers of the next
            for obj_path in paths[1:]:
                with obj_path.open() as f2:
                    for line in f2:
                        new_line = line
                        if line.startswith("v "):
                            new_v += 1
                        elif line.startswith("vt "):
                            new_vt += 1
                        elif line.startswith("f "):
                            line = line[2:]
                            line = re.split(r"[ /]", line)
                            new_line = "f "

                            count = 0
                            for num in line:
                                count += 1
                                if count == 1:
                                    new_line += str(int(num) + prev_v)
                                    new_line += "/"
                                elif count == 2:
                                    new_line += str(int(num) + prev_vt)
                                    new_line += " "
                                    count = 0
                            new_line += "\n"

                        f.write(new_line)
                    prev_v = new_v
                    prev_vt = new_vt

        added_edges = []
        if unwrap.preserve_job is not None:
            # combine all added edges in the group
            v_count = 0
            for e_idx, edges in enumerate([u.added_edges for u in self.unwrapped]):
                for v1, v2 in edges:
                    added_edges.append((v1 + v_count, v2 + v_count))
                v_count += self.unwrapped[e_idx].vertex_count

            # combine all edge files
            unwraps = self.unwrapped
            edge_path = unwraps[0].edge_path
            v_count = unwraps[0].vertex_count
            e_paths = [u.edge_path for u in unwraps]
            with e_paths[0].open("a") as f:

                for e_idx, e_path in enumerate(e_paths[1:], 1):
                    with e_path.open() as f2:
                        for line in f2:
                            line = line.split()
                            f.write(
                                f"{int(line[0]) + v_count} {int(line[1]) + v_count}\n"
                            )
                    v_count += unwraps[e_idx].vertex_count

        # if the last unwrap of group is cancelled and there are > 1 unwrapped in group
        # expand could be length 0 if there are no other groups
        if len(expand) > 0:
            # delete expand index 0, which will shift all values down by one
            del expand[0]

        return (path, edge_path, added_edges)


class Cleanup(Job):
    def __init__(self, count, action):
        super().__init__(count)
        self.type = "CLEANUP"
        self.action = action

    def finish(self, input_mesh):
        if check_exists(input_mesh):
            if self.action == "HIDE":
                input_mesh.hide_set(True)
            # deleting the object while editing it will crash blender
            elif self.action == "DELETE" and input_mesh.mode != "EDIT":
                bpy.data.objects.remove(input_mesh, do_unlink=True)


class Symmetrise(Job):
    def __init__(self, count, axes, center, overlap):
        super().__init__(count)
        self.type = "SYMMETRISE"
        self.x = True if "X" in axes else False
        self.y = True if "Y" in axes else False
        self.z = True if "Z" in axes else False
        self.center = center
        self.overlap = overlap

    def finish(self, output):
        mirror = output.modifiers.new("Mirror", "MIRROR")
        mirror.use_axis = (self.x, self.y, self.z)
        empty = None
        # if the object origin is not at the center, the mirror axis will be wrong
        if self.center != output.matrix_world.to_translation():
            empty = bpy.data.objects.new("Empty", None)
            empty.location.x = self.center.x
            empty.location.y = self.center.y
            empty.location.z = self.center.z
            mirror.mirror_object = empty

        if not self.overlap:
            # separate islands
            mirror.use_mirror_u = True
            mirror.use_mirror_v = True

        old_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = output
        bpy.ops.object.modifier_apply(modifier=mirror.name)
        bpy.context.view_layer.objects.active = old_active
        if empty is not None:
            bpy.data.objects.remove(empty, do_unlink=True)
