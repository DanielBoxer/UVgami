import re

import bpy
import numpy


# the built-in exporter rounds to 6 decimals, which flips tiny uv triangles
def export_obj(obj, path, export_uv, flip_mirrored=False, matrix=None):
    mesh = obj.data
    matrix = numpy.array(obj.matrix_world if matrix is None else matrix)
    co = numpy.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]

    loop_verts = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)

    # a mirror flips positions but not corner order, so faces bake in inside out
    loop_order = None
    if flip_mirrored and numpy.linalg.det(matrix[:3, :3]) < 0:
        starts = numpy.cumsum(totals) - totals
        within = numpy.arange(len(loop_verts)) - numpy.repeat(starts, totals)
        loop_order = numpy.repeat(starts + totals - 1, totals) - within
        loop_verts = loop_verts[loop_order]

    layer = mesh.uv_layers.active
    export_uv = export_uv and layer is not None
    vt_verts = None

    with path.open("w") as f:
        f.write(f"o {obj.name}\n")
        # a single %-format over the whole array runs in C, f-strings are 10x slower
        f.write(("v %.9f %.9f %.9f\n" * len(co)) % tuple(co.ravel().tolist()))

        if export_uv:
            uvs = numpy.empty(len(mesh.loops) * 2)
            layer.data.foreach_get("uv", uvs)
            uvs = uvs.reshape(-1, 2)
            if loop_order is not None:
                uvs = uvs[loop_order]
            # optcuts gives each vt its own vertex, so a shared uv would weld two
            scaled = numpy.rint(uvs * 1e9).astype(numpy.int64)
            order = numpy.lexsort((scaled[:, 1], scaled[:, 0], loop_verts))
            sv = loop_verts[order]
            su = scaled[order, 0]
            sw = scaled[order, 1]
            first = numpy.zeros(len(order), dtype=bool)
            first[:1] = True
            first[1:] = (sv[1:] != sv[:-1]) | (su[1:] != su[:-1]) | (sw[1:] != sw[:-1])
            loop_vts = numpy.empty(len(order), dtype=numpy.int64)
            loop_vts[order] = numpy.cumsum(first) - 1
            vt_verts = sv[first]
            values = numpy.column_stack((su[first], sw[first])) / 1e9
            f.write(("vt %.9f %.9f\n" * len(values)) % tuple(values.ravel().tolist()))
            corners = numpy.column_stack((loop_verts + 1, loop_vts + 1))
            token = "%d/%d"
        else:
            corners = loop_verts[:, None] + 1
            token = "%d"

        if len(totals) and (totals == totals[0]).all():
            size = int(totals[0])
            fmt = "f " + " ".join([token] * size) + "\n"
            f.write((fmt * len(totals)) % tuple(corners.ravel().tolist()))
        else:
            flat = corners.ravel().tolist()
            width = corners.shape[1]
            lines = []
            li = 0
            for total in totals.tolist():
                fmt = "f " + " ".join([token] * total) + "\n"
                lines.append(fmt % tuple(flat[li : li + total * width]))
                li += total * width
            f.writelines(lines)

    return vt_verts


def _block(text, prefix):
    # every writer here keeps same-type lines contiguous, merged files interleave them
    tag = "\n" + prefix + " "
    if text.startswith(prefix + " "):
        start = 0
    else:
        start = text.find(tag)
        if start == -1:
            return ""
        start += 1
    end = text.find("\n", max(text.rfind(tag), start) + 1)
    block = text[start:] + "\n" if end == -1 else text[start : end + 1]
    if block.count("\n") == block.count(tag) + 1:
        return block
    runs = re.findall(rf"(?m)^(?:{prefix} [^\n]*\n?)+", text)
    block = "".join(runs)
    return block if block.endswith("\n") else block + "\n"


def _numeric_columns(text, prefix, comps):
    block = _block(text, prefix)
    if not block:
        return numpy.empty((0, comps))
    lines = block.count("\n")
    tokens = block.replace(prefix + " ", " ").split()
    if len(tokens) % lines == 0 and len(tokens) // lines >= comps:
        stride = len(tokens) // lines
        return numpy.array(tokens, dtype=numpy.float64).reshape(-1, stride)[:, :comps]
    return numpy.array(
        [ln.split()[1 : comps + 1] for ln in block.splitlines()],
        dtype=numpy.float64,
    )


def _parse_faces(text):
    block = _block(text, "f")
    empty = numpy.empty(0, dtype=numpy.int64)
    if not block:
        return numpy.empty(0, dtype=numpy.int32), empty, empty
    lines = block.count("\n")
    first_line = block.split("\n", 1)[0][2:].split()
    size = len(first_line)
    stride = first_line[0].count("/") + 1
    # "v//vn" gets a 0 placeholder so every corner expands to the same width
    tokens = block.replace("f ", " ").replace("//", "/0/").replace("/", " ").split()
    if len(tokens) == lines * size * stride:
        nums = numpy.array(tokens, dtype=numpy.int64).reshape(-1, stride)
        totals = numpy.full(lines, size, dtype=numpy.int32)
        loop_verts = nums[:, 0] - 1
        if stride > 1:
            loop_vts = numpy.maximum(nums[:, 1] - 1, 0)
        else:
            loop_vts = numpy.zeros(len(nums), dtype=numpy.int64)
        return totals, loop_verts, loop_vts

    # mixed token forms, e.g. merged pieces with and without uvs
    totals = []
    loop_verts = []
    loop_vts = []
    for ln in block.splitlines():
        corners = ln.split()[1:]
        totals.append(len(corners))
        for corner in corners:
            parts = corner.split("/")
            loop_verts.append(int(parts[0]) - 1)
            if len(parts) > 1 and parts[1]:
                loop_vts.append(int(parts[1]) - 1)
            else:
                loop_vts.append(0)
    return (
        numpy.array(totals, dtype=numpy.int32),
        numpy.array(loop_verts, dtype=numpy.int64),
        numpy.array(loop_vts, dtype=numpy.int64),
    )


# not bpy.ops.wm.obj_import: that changes selection, the active object and undo
def import_obj(path, name=""):
    text = path.read_text()

    if not name:
        o_match = re.search(r"(?m)^o (.*)$", text)
        name = o_match.group(1).strip() if o_match else path.stem

    verts = _numeric_columns(text, "v", 3)
    uvs = _numeric_columns(text, "vt", 2)
    totals, loop_verts, loop_vts = _parse_faces(text)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(len(loop_verts))
    mesh.loops.foreach_set("vertex_index", loop_verts)
    mesh.polygons.add(len(totals))
    starts = numpy.cumsum(totals) - totals
    mesh.polygons.foreach_set("loop_start", starts)
    mesh.polygons.foreach_set("loop_total", totals)
    # some meshes have no sharp_face attribute
    mesh.polygons.foreach_set("use_smooth", numpy.zeros(len(totals), dtype=bool))
    if len(uvs):
        layer = mesh.uv_layers.new()
        layer.data.foreach_set("uv", uvs[loop_vts].ravel())
    mesh.update(calc_edges=True)
    mesh.validate()

    imported_obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(imported_obj)
    return imported_obj


def print_stdin(process, msg):
    if process.poll() is not None:
        return False
    try:
        print(msg, file=process.stdin, flush=True)
    except OSError:
        return False
    return True
