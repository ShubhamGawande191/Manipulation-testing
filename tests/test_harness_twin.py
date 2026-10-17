import pytest

from robot_manipulator.harness.twin import DEFAULT_MAP, flipped
from robot_manipulator.harness.units import JointMap


def test_flipped_negates_only_named_joints():
    jm = JointMap.from_yaml(DEFAULT_MAP)
    f = flipped(jm, ["wrist_flex", "shoulder_pan"])
    assert f.body["wrist_flex"].sign == -jm.body["wrist_flex"].sign
    assert f.body["shoulder_pan"].sign == -jm.body["shoulder_pan"].sign
    assert f.body["elbow_flex"] == jm.body["elbow_flex"]
    assert f.gripper == jm.gripper and f.calibrated == jm.calibrated


def test_gripper_follows_lerobot_convention():
    jm = JointMap.from_yaml(DEFAULT_MAP)
    vals = {f"{j}.pos": 0.0 for j in jm.body} | {"gripper.pos": 100.0}
    assert jm.lerobot_to_sim(vals)[-1] == pytest.approx(jm.gripper.sim_open)  # 100 = open
    vals["gripper.pos"] = 0.0
    assert jm.lerobot_to_sim(vals)[-1] == pytest.approx(jm.gripper.sim_closed)
