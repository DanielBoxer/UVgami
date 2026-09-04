import importlib.util

# nothing here can be collected without a live bpy
if importlib.util.find_spec("bpy") is None:
    collect_ignore_glob = ["test_*.py"]
else:
    from blender_fixtures import (  # noqa: F401
        empty_scene,
        face_uvs,
        invalid_objects,
        load_obj,
        make_mesh,
        outputs,
        seam_edges,
        session,
        unwrap,
    )
