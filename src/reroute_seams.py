# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import math
import numpy
from .pyqtree import Index


class Vertex:
    def __init__(self, co):
        self.x = float(co[0])
        self.y = float(co[1])
        self.z = float(co[2])


class UV:
    def __init__(self, co):
        self.x = float(co[0])
        self.y = float(co[1])
        self.edge_v = []


class Face:
    def __init__(self):
        self.v = []
        self.vt = []


def is_ccw(a, b, c):
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


def is_collinear(a, b, c):
    return math.isclose(abs((c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)), 0)


def do_overlap(a, b, c, d):
    if is_collinear(a, c, d) or is_collinear(b, c, d):
        return False

    return is_ccw(a, c, d) != is_ccw(b, c, d) and is_ccw(a, b, c) != is_ccw(a, b, d)


def reroute_seams(path, edge_path):
    vertices = []
    uvs = []
    faces = []
    v_vt = {}
    edge_face = {}
    quadtree = Index(bbox=(0, 0, 1, 1))

    # read obj file
    for line in path.open("r"):
        if line.startswith("v "):
            line = line[2:]
            vertices.append(Vertex(line.split()))

        elif line.startswith("vt "):
            line = line[3:]
            uv = UV(line.split())
            uvs.append(uv)
            # add to quadtree
            quadtree.insert(uv, (uv.x, uv.y))

        elif line.startswith("f "):
            line = line[2:]
            face = Face()
            faces.append(face)

            # vertex indices
            v = []
            # uv coordinate indices
            vt = []
            for face_v in line.split():
                face_v = face_v.split("/")
                face.v.append(int(face_v[0]))
                v.append(int(face_v[0]))
                face.vt.append(int(face_v[1]))
                vt.append(int(face_v[1]))

            # get linked texture coordinates
            for v_idx, vertex in enumerate(v):

                # if the vertex key isn't in dictionary yet
                if vertex not in v_vt:
                    # set it to empty array (unknown size)
                    v_vt[vertex] = []

                # get the vt linked to the current vertex (v)
                face_vt = vt[v_idx]

                # don't add if already linked
                if face_vt not in v_vt[vertex]:
                    v_vt[vertex].append(face_vt)

            # get uv edges

            # format: other vt, first vt, second vt

            # edge 1
            uvs[vt[0] - 1].edge_v.append((vt[1], vt[0], vt[1]))
            uvs[vt[1] - 1].edge_v.append((vt[0], vt[0], vt[1]))

            # edge 2
            uvs[vt[1] - 1].edge_v.append((vt[2], vt[1], vt[2]))
            uvs[vt[2] - 1].edge_v.append((vt[1], vt[1], vt[2]))

            # edge 3
            uvs[vt[2] - 1].edge_v.append((vt[0], vt[2], vt[0]))
            uvs[vt[0] - 1].edge_v.append((vt[2], vt[2], vt[0]))

            # get face edges
            # edge is linked to a face
            edge_face[(v[0], v[1])] = face
            edge_face[(v[1], v[2])] = face
            edge_face[(v[2], v[0])] = face

    # go through added edges
    for line in edge_path.open("r"):
        # line format: vertex index 1, vertex index 2
        line = line.split()
        line = int(line[0]) + 1, int(line[1]) + 1

        # get linked vt
        linked_vertices = v_vt[line[0]] + v_vt[line[1]]
        # remove duplicates
        linked_vertices = list(set(linked_vertices))

        # then check uv_edges for a match (two matches needed)
        found_edges = []

        for lv in linked_vertices:
            edge_vs = uvs[lv - 1].edge_v

            for edge_v in edge_vs:
                # edge_v[0] is the other vt in the uv edge
                if edge_v[0] in linked_vertices:
                    # this means that a uv edge was formed
                    edge = (edge_v[1], edge_v[2])
                    other_combination = (edge_v[2], edge_v[1])

                    # the same edge can be found twice, so check for that
                    if found_edges:
                        if not (
                            edge in found_edges or other_combination in found_edges
                        ):
                            found_edges.append(edge)
                    else:
                        found_edges.append(edge)

            # two found edges means the seam is found
            if len(found_edges) == 2:
                # get face using the edge
                face = edge_face[(line[0], line[1])]

                # check which edge the face has
                curr_edge = None
                other_edge = None

                if found_edges[0][0] in face.vt and found_edges[0][1] in face.vt:
                    curr_edge = found_edges[0]
                    other_edge = found_edges[1]
                else:
                    curr_edge = found_edges[1]
                    other_edge = found_edges[0]

                vt_i1 = face.vt.index(curr_edge[0])
                vt_i2 = face.vt.index(curr_edge[1])

                # get point c index(point that isn't on the seam)
                tri_indices = {0, 1, 2}
                move_indices = {vt_i1, vt_i2}
                other_v = (tri_indices - move_indices).pop()

                # get points
                point_a = uvs[curr_edge[0] - 1]
                point_b = uvs[curr_edge[1] - 1]
                point_a2 = uvs[other_edge[0] - 1]
                point_b2 = uvs[other_edge[1] - 1]
                point_c = uvs[face.vt[other_v] - 1]

                # get vector from point c to point a on triangle
                c_vector = point_c.x - point_a.x, point_c.y - point_a.y

                # get a1b1 vector and a2b2 vector
                ab1_vector = point_b.x - point_a.x, point_b.y - point_a.y
                ab2_vector = point_b2.x - point_a2.x, point_b2.y - point_a2.y

                # get lengths
                ab1_len = math.hypot(ab1_vector[0], ab1_vector[1])
                ab2_len = math.hypot(ab2_vector[0], ab2_vector[1])

                # get angle between two found edges
                angle = math.acos(
                    ((numpy.dot(ab1_vector, ab2_vector)) / ab1_len) / ab2_len
                )
                # get other solution
                angle2 = math.pi - angle

                # rotate ac vector by angle
                s = math.sin(angle)
                c = math.cos(angle)
                xnew = c_vector[0] * c - c_vector[1] * s
                ynew = c_vector[0] * s + c_vector[1] * c
                new_vector = xnew, ynew

                # rotate by angle2
                s2 = math.sin(angle2)
                c2 = math.cos(angle2)
                xnew = c_vector[0] * c2 - c_vector[1] * s2
                ynew = c_vector[0] * s2 + c_vector[1] * c2
                new_vector2 = xnew, ynew

                # rotate a1b1 by angle
                xnew = ab1_vector[0] * c - ab1_vector[1] * s
                ynew = ab1_vector[0] * s + ab1_vector[1] * c
                ab_rotated = xnew, ynew

                # rotate a1b1 by angle2
                xnew = ab1_vector[0] * c2 - ab1_vector[1] * s2
                ynew = ab1_vector[0] * s2 + ab1_vector[1] * c2
                ab_rotated2 = xnew, ynew

                # check if a1==a2 and get new point c by applying vector
                # (order is switched so use b instead of a)
                t1 = point_b2.x + ab_rotated[0], point_b2.y + ab_rotated[1]
                t2 = point_b2.x + ab_rotated2[0], point_b2.y + ab_rotated2[1]
                t3 = point_b2.x - ab_rotated[0], point_b2.y - ab_rotated[1]
                t4 = point_b2.x - ab_rotated2[0], point_b2.y - ab_rotated2[1]

                dist1 = math.hypot(t1[0] - point_a2.x, t1[1] - point_a2.y)
                dist2 = math.hypot(t2[0] - point_a2.x, t2[1] - point_a2.y)
                dist3 = math.hypot(t3[0] - point_a2.x, t3[1] - point_a2.y)
                dist4 = math.hypot(t4[0] - point_a2.x, t4[1] - point_a2.y)

                point_c2 = None
                dist_l = [dist1, dist2, dist3, dist4]
                min_idx = dist_l.index(min(dist_l))

                if min_idx == 0:
                    point_c2 = point_b2.x + new_vector[0], point_b2.y + new_vector[1]
                elif min_idx == 1:
                    point_c2 = point_b2.x + new_vector2[0], point_b2.y + new_vector2[1]
                elif min_idx == 2:
                    point_c2 = point_b2.x - new_vector[0], point_b2.y - new_vector[1]
                else:
                    point_c2 = point_b2.x - new_vector2[0], point_b2.y - new_vector2[1]

                point_c2 = UV((point_c2[0], point_c2[1]))
                # the edge vertices are point a2 and point b2
                # same format (other, first, second)
                point_c2.edge_v = [
                    (other_edge[0], len(uvs) + 1, other_edge[0]),
                    (other_edge[1], len(uvs) + 1, other_edge[1]),
                ]

                # search within the distance of the new vector
                search_distance_bc = math.hypot(
                    point_c2.x - point_b2.x, point_c2.y - point_b2.y
                )

                # some extra points are found, intersect uses rectangle not circle
                found_points = quadtree.intersect(
                    (
                        point_b2.x - search_distance_bc,
                        point_b2.y - search_distance_bc,
                        point_b2.x + search_distance_bc,
                        point_b2.y + search_distance_bc,
                    )
                )

                # fix overlap
                closest_uv = None
                closest_dist = float("inf")
                # use midpoint of AB edge to determine closest
                midpoint = (
                    (point_a2.x + point_b2.x) / 2,
                    (point_a2.y + point_b2.y) / 2,
                )
                for uv in found_points:
                    if uv != point_a2 and uv != point_b2:
                        for _, v1, v2 in uv.edge_v:
                            p1 = uvs[v1 - 1]
                            p2 = uvs[v2 - 1]

                            if do_overlap(p1, p2, point_a2, point_c2) or do_overlap(
                                p1, p2, point_b2, point_c2
                            ):
                                dist_to_point = math.hypot(
                                    uv.x - midpoint[0], uv.y - midpoint[1]
                                )
                                if dist_to_point < closest_dist:
                                    closest_uv = uv
                                    closest_dist = dist_to_point

                if closest_uv is not None:
                    point_c2.x = closest_uv.x
                    point_c2.y = closest_uv.y

                # change links to other edge (order is switched)
                face.vt[vt_i1] = other_edge[1]
                face.vt[vt_i2] = other_edge[0]

                # change 3rd vt to new outer vt (make new one)
                uvs.append(point_c2)
                quadtree.insert(point_c2, (point_c2.x, point_c2.y))
                face.vt[other_v] = len(uvs)

                break

    # rewrite obj file
    with path.open("w") as file:
        for v in vertices:
            file.write(f"v {v.x} {v.y} {v.z}\n")

        for vt in uvs:
            file.write(f"vt {vt.x} {vt.y}\n")

        for f in faces:
            file.write(f"f {f.v[0]}/{f.vt[0]} {f.v[1]}/{f.vt[1]} {f.v[2]}/{f.vt[2]}\n")
