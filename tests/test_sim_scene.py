import pytest

pytest.importorskip("mujoco")

from robot_manipulator.sim.model import ASSET_DIR, JOINTS, load_model  # noqa: E402
from robot_manipulator.sim.scene import build  # noqa: E402


@pytest.fixture
def scene_xml():
    # beside the vendored assets (the MJCF include is relative), never over the real scene_bench.xml
    out = ASSET_DIR / "_test_scene_bench.xml"
    try:
        yield build(out=out)
    finally:
        out.unlink(missing_ok=True)


def test_bench_scene_keeps_follower_layout_and_adds_obstacles(scene_xml):
    m = load_model(scene_xml.name)
    assert m.nu == 6 and [m.actuator(i).name for i in range(6)] == list(JOINTS)
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert {"table", "column", "camera_arm"} <= names
    assert m.body("leader").id > 0
    assert any(m.camera(i).name == "workspace_cam" for i in range(m.ncam))
    # leader sits on the far side of the column (follower's left)
    assert m.body_pos[m.body("leader").id][1] > m.geom_pos[m.geom("column").id][1] > 0
