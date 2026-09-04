import importlib.util
from pathlib import Path

# loaded from file, the addon package imports bpy
spec = importlib.util.spec_from_file_location(
    "addon_objfile", Path(__file__).parents[2] / "src" / "objfile.py"
)
objfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(objfile)


def write_obj(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_merge_with_vt(tmp_path):
    a = write_obj(
        tmp_path,
        "a.obj",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
    )
    b = write_obj(
        tmp_path,
        "b.obj",
        "v 2 0 0\nv 3 0 0\nv 2 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
    )

    result = objfile.merge_obj_files([a, b])
    assert result == a
    lines = a.read_text().splitlines()

    # first obj is untouched, second obj face is offset by 3 verts and 3 vts
    assert lines[6] == "f 1/1 2/2 3/3"
    assert lines[-1] == "f 4/4 5/5 6/6"
    assert lines.count("f 4/4 5/5 6/6") == 1


def test_merge_without_vt(tmp_path):
    a = write_obj(tmp_path, "a.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    b = write_obj(tmp_path, "b.obj", "v 2 0 0\nv 3 0 0\nv 2 1 0\nf 1 2 3\n")

    objfile.merge_obj_files([a, b])
    lines = a.read_text().splitlines()

    assert lines[3] == "f 1 2 3"
    assert lines[-1] == "f 4 5 6"


def test_merge_drops_o_lines(tmp_path):
    a = write_obj(
        tmp_path,
        "a.obj",
        "# blender export\no robot.146\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )
    b = write_obj(
        tmp_path,
        "b.obj",
        "# blender export\no robot.147\nv 2 0 0\nv 3 0 0\nv 2 1 0\nf 1 2 3\n",
    )

    objfile.merge_obj_files([a, b])
    lines = a.read_text().splitlines()

    o_lines = [line for line in lines if line.startswith("o ")]
    assert len(o_lines) == 1
    assert o_lines[0] == "o robot.146"
    assert lines[-1] == "f 4 5 6"
