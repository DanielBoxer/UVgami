import pathlib

import bpy
import pytest
from blender_fixtures import addon, manager, needs_engine, needs_partuv, needs_xatlas

pytestmark = [needs_engine, pytest.mark.smoke]

OTHER_ENGINES = [
    pytest.param("XATLAS", marks=needs_xatlas),
    pytest.param("PARTUV", marks=needs_partuv),
]


def test_unwrap_runs_to_completion(load_obj, unwrap, outputs):
    load_obj("cylinder")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert list(outputs()) == ["cylinder_unwrapped"]
    layer = outputs()["cylinder_unwrapped"].data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)

    # a many-part model runs out of file descriptors partway through
    assert manager.results
    for piece, _ in manager.results:
        if piece.process is None:
            continue
        assert piece.process.returncode is not None
        assert piece.process.stdout.closed
        assert piece.process.stderr.closed


def test_flipped_face_is_rewound_instead_of_refused(load_obj, unwrap, outputs):
    load_obj("flipped-face")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert len(outputs()) == 1
    layer = next(iter(outputs().values())).data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)


# shade_auto_smooth cancels on headless 4.3
def smooth_by_angle_node_group():
    name = addon.src.utils.mesh.AUTO_SMOOTH_MODIFIER_NAME
    assets = pathlib.Path(bpy.utils.system_resource("DATAFILES")) / "assets"
    for path in sorted(assets.rglob("*.blend")):
        with bpy.data.libraries.load(str(path)) as (source, target):
            found = name in source.node_groups
            if found:
                target.node_groups = [name]
        if found:
            return bpy.data.node_groups[name]
    raise LookupError(f"no bundled asset holds {name}")


def test_shading_modifiers_carry_over(load_obj, unwrap, outputs):
    obj = load_obj("cylinder")[0]
    weighted = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    weighted.keep_sharp = True
    angle = 0.5
    smooth = obj.modifiers.new("Smooth by Angle", "NODES")
    smooth.node_group = smooth_by_angle_node_group()
    addon.src.utils.mesh._set_node_input_values(smooth, {"Input_1": angle})
    unwrap()

    assert manager.error_messages == []
    output = outputs()["cylinder_unwrapped"]
    assert [m.type for m in output.modifiers] == ["WEIGHTED_NORMAL", "NODES"]
    assert output.modifiers[0].keep_sharp is True
    assert output.modifiers[1].node_group is not None
    records = addon.src.utils.mesh.get_shading_modifiers(output)
    assert records[1][2][1]["Input_1"] == pytest.approx(angle)


@pytest.mark.parametrize("engine", OTHER_ENGINES)
def test_other_engines_unwrap_the_cylinder(load_obj, unwrap, outputs, engine):
    load_obj("cylinder")
    if engine == "PARTUV":
        bpy.context.scene.uvgami.partuv.segmentation = "GEOMETRIC"

    unwrap(engine)

    assert manager.summary == ["UV unwrap complete!"]
    assert manager.error_messages == []
    layer = outputs()["cylinder_unwrapped"].data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)
