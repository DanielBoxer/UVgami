import collections
import math

# low on purpose: a bevel splits one crease into several small turns
LOW_ANGLE = 10


def cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def norm(v):
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def diagonal(verts):
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    return norm([hi[i] - lo[i] for i in range(3)])


def pair(a, b):
    return (a, b) if a < b else (b, a)


# each polygon covers a contiguous run of loops, so the totals alone place it
def split_per_face(values, totals):
    faces = []
    start = 0
    for total in totals:
        faces.append(values[start : start + total])
        start += total
    return faces


# parent maps each element to its parent, itself for a root
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


# a face's edges as sorted vertex index pairs
def face_keys(face):
    return [pair(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]


# edge -> owning faces, keyed by sorted vertex index pair
def face_edges(faces):
    edges = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for key in face_keys(face):
            edges[key].append(fi)
    return edges


# degrees the surface turns across an edge
def turn_angle(weighted, owners):
    ax, ay, az = weighted[owners[0]]
    bx, by, bz = weighted[owners[1]]
    # two roots, not one over the product: folding them flips the odd seam
    scale = math.sqrt(ax * ax + ay * ay + az * az) * math.sqrt(
        bx * bx + by * by + bz * bz
    )
    if not scale:
        return 0.0
    dot = (ax * bx + ay * by + az * bz) / scale
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


# scaled by twice the area of the first three corners
def weighted_normals(verts, faces):
    weighted = []
    for face in faces:
        a, b, c = (verts[i] for i in face[:3])
        weighted.append(
            cross([b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)])
        )
    return weighted


# per-face weighted normals and areas, plus edge -> owning faces
def build(verts, faces):
    weighted = weighted_normals(verts, faces)
    return weighted, [norm(n) / 2 for n in weighted], face_edges(faces)


def signed_area(pts):
    total = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        total += a[0] * b[1] - b[0] * a[1]
    return total / 2


# between a collapsed map's float noise and any real packed map's area
COLLAPSED_UV_AREA = 1e-8


# a uv map crushed to points, a failed flatten's signature
def uvs_collapsed(polygons):
    return sum(abs(signed_area(pts)) for pts in polygons) < COLLAPSED_UV_AREA


# joined by interior edges not on a seam
def island_groups(faces, seams, edges):
    parent = list(range(len(faces)))

    for key, owners in edges.items():
        if len(owners) == 2 and key not in seams:
            a, b = find(parent, owners[0]), find(parent, owners[1])
            if a != b:
                parent[a] = b

    members = collections.defaultdict(list)
    for fi in range(len(faces)):
        members[find(parent, fi)].append(fi)
    return list(members.values())


# edges whose faces don't share their corner uvs. a boundary edge is never one
def uv_seams(faces, uvs, edges):
    seams = set()
    for (u, v), owners in edges.items():
        if len(owners) < 2:
            continue
        first = faces[owners[0]]
        first_uv = uvs[owners[0]]
        at_u = first_uv[first.index(u)]
        at_v = first_uv[first.index(v)]
        for g in owners[1:]:
            face = faces[g]
            face_uv = uvs[g]
            if face_uv[face.index(u)] != at_u or face_uv[face.index(v)] != at_v:
                seams.add((u, v))
                break
    return seams


# follows the uv map itself, so it needs no seam marks
def uv_island_groups(faces, uvs, edges):
    return island_groups(faces, uv_seams(faces, uvs, edges), edges)


# joined by any shared vertex, what mesh.separate(type="LOOSE") splits on
def vertex_components(faces):
    parent = {}

    for face in faces:
        for v in face:
            parent.setdefault(v, v)
        for v in face[1:]:
            ra, rb = find(parent, face[0]), find(parent, v)
            if ra != rb:
                parent[ra] = rb

    members = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        members[find(parent, face[0])].append(fi)
    return list(members.values())


# boxes can touch without the boundaries crossing
def islands_overlap(boxes):
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for k, i in enumerate(order):
        for j in order[k + 1 :]:
            if boxes[j][0] >= boxes[i][2]:
                break
            if boxes[j][1] < boxes[i][3] and boxes[i][1] < boxes[j][3]:
                return True
    return False


# applied as u -> flip - u when flip is not None, then add (du, dv)
def island_layout(boxes, areas):
    gap = 0.05 * max(x1 - x0 for x0, _, x1, _ in boxes)
    transforms = []
    cursor = 0.0
    for (x0, y0, x1, _), area in zip(boxes, areas):
        flip = x0 + x1 if area < 0 else None
        transforms.append((flip, cursor - x0, -y0))
        cursor += x1 - x0 + gap
    return transforms


# keeps a repaired island inside the spot its old layout occupied
def uv_fit(points, bbox):
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    x0, y0, x1, y1 = bbox
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scales = []
    if w > 0:
        scales.append((x1 - x0) / w)
    if h > 0:
        scales.append((y1 - y0) / h)
    s = min(scales) if scales else 1.0
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ox, oy = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda uv: (ox + (uv[0] - cx) * s, oy + (uv[1] - cy) * s)


# keeps the island's texel density, which uv_fit loses
def uv_area_fit(polygons, area, bbox):
    new_area = sum(abs(signed_area(p)) for p in polygons)
    points = [uv for p in polygons for uv in p]
    if area <= 0 or new_area <= 0:
        return uv_fit(points, bbox)
    s = (area / new_area) ** 0.5
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    x0, y0, x1, y1 = bbox
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ox, oy = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda uv: (ox + (uv[0] - cx) * s, oy + (uv[1] - cy) * s)
