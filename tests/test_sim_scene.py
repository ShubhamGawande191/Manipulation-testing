import pytest

pytest.importorskip("mujoco")

from robot_manipulator.sim.model import JOINTS, load_model  # noqa: E402
from robot_manipulator.sim.scene import build  # noqa: E402


def test_bench_scene_keeps_follower_layout_and_adds_obstacles(tmp_path):
    out = build(out=tmp_path.parent / "unused.xml") if False else build()
    m = load_model(out.name)
    assert m.nu == 6 and [m.actuator(i).name for i in range(6)] == list(JOINTS)
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert {"table", "column", "camera_arm"} <= names
    assert m.body("leader").id > 0
    assert any(m.camera(i).name == "workspace_cam" for i in range(m.ncam))
    # leader sits on the far side of the column (follower's left)
    assert m.body_pos[m.body("leader").id][1] > m.geom_pos[m.geom("column").id][1] > 0
