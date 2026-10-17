"""Unit tests: LeRobot <-> sim joint units (harness/units.py) and config/sim2real.yaml."""

from pathlib import Path

import numpy as np
import pytest

from robot_manipulator.harness.units import BODY_JOINTS, CMD_JOINTS, JointMap

REPO_CFG = Path(__file__).resolve().parents[1] / "config" / "sim2real.yaml"

BASE_YAML = """
calibrated: {calibrated}
joints:
  shoulder_pan: {{sign: {s0}, offset_deg: {o0}}}
  shoulder_lift: {{sign: 1, offset_deg: 0.0}}
  elbow_flex: {{sign: 1, offset_deg: 0.0}}
  wrist_flex: {{sign: 1, offset_deg: 0.0}}
  wrist_roll: {{sign: 1, offset_deg: 0.0}}
gripper: {{lerobot_open: 0.0, lerobot_closed: 100.0, sim_open: 1.74533, sim_closed: -0.17453}}
"""


def make_map(tmp_path: Path, sign0: int = 1, offset0: float = 0.0, calibrated="false") -> JointMap:
    p = tmp_path / "s2r.yaml"
    p.write_text(BASE_YAML.format(calibrated=calibrated, s0=sign0, o0=offset0))
    return JointMap.from_yaml(p)


def lerobot_dict(rng: np.random.Generator) -> dict[str, float]:
    d = {f"{j}.pos": float(rng.uniform(-170, 170)) for j in BODY_JOINTS}
    d["gripper.pos"] = float(rng.uniform(0, 100))
    return d


def test_repo_config_loads():
    # calibrated: true since hardware ticket H2 (2026-10-17); the guard itself is tested with
    # explicit uncalibrated maps elsewhere
    jm = JointMap.from_yaml(REPO_CFG)
    assert isinstance(jm.calibrated, bool)
    assert set(jm.body) == set(BODY_JOINTS)


def test_cmd_order_is_stable():
    assert CMD_JOINTS == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )


@pytest.mark.parametrize(("sign", "offset"), [(1, 0.0), (-1, 0.0), (1, 7.5), (-1, -12.25)])
def test_round_trip_lerobot_sim_lerobot(tmp_path, sign, offset):
    jm = make_map(tmp_path, sign0=sign, offset0=offset)
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = lerobot_dict(rng)
        back = jm.sim_to_lerobot(jm.lerobot_to_sim(d))
        for k in d:
            assert back[k] == pytest.approx(d[k], abs=1e-9)


def test_round_trip_sim_lerobot_sim(tmp_path):
    jm = make_map(tmp_path, sign0=-1, offset0=3.0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(-1.5, 1.5, size=6)
        q[-1] = rng.uniform(-0.17453, 1.74533)  # gripper inside its sim range
        np.testing.assert_allclose(jm.lerobot_to_sim(jm.sim_to_lerobot(q)), q, atol=1e-9)


def test_sign_minus_one_flips_body_joint(tmp_path):
    jm = make_map(tmp_path, sign0=-1, offset0=0.0)
    d = {f"{j}.pos": 0.0 for j in BODY_JOINTS} | {"gripper.pos": 0.0}
    d["shoulder_pan.pos"] = 30.0
    q = jm.lerobot_to_sim(d)
    assert q[0] == pytest.approx(-np.deg2rad(30.0), abs=1e-12)
    assert q[1] == pytest.approx(0.0, abs=1e-12)


def test_offset_is_added_in_radians_space(tmp_path):
    jm = make_map(tmp_path, sign0=1, offset0=10.0)
    d = {f"{j}.pos": 0.0 for j in BODY_JOINTS} | {"gripper.pos": 0.0}
    assert jm.lerobot_to_sim(d)[0] == pytest.approx(np.deg2rad(10.0), abs=1e-12)


def test_gripper_endpoints_and_linear_midpoint(tmp_path):
    jm = make_map(tmp_path)
    base = {f"{j}.pos": 0.0 for j in BODY_JOINTS}
    open_q = jm.lerobot_to_sim(base | {"gripper.pos": 0.0})[-1]
    closed_q = jm.lerobot_to_sim(base | {"gripper.pos": 100.0})[-1]
    mid_q = jm.lerobot_to_sim(base | {"gripper.pos": 50.0})[-1]
    assert open_q == pytest.approx(1.74533)
    assert closed_q == pytest.approx(-0.17453)
    assert mid_q == pytest.approx(0.5 * (1.74533 - 0.17453))


def test_gripper_clips_out_of_range(tmp_path):
    jm = make_map(tmp_path)
    base = {f"{j}.pos": 0.0 for j in BODY_JOINTS}
    assert jm.lerobot_to_sim(base | {"gripper.pos": 150.0})[-1] == pytest.approx(-0.17453)
    assert jm.lerobot_to_sim(base | {"gripper.pos": -10.0})[-1] == pytest.approx(1.74533)
    assert jm.sim_to_lerobot(np.array([0, 0, 0, 0, 0, 9.0]))["gripper.pos"] == pytest.approx(0.0)
    assert jm.sim_to_lerobot(np.array([0, 0, 0, 0, 0, -9.0]))["gripper.pos"] == pytest.approx(100.0)


def test_missing_key_raises(tmp_path):
    jm = make_map(tmp_path)
    with pytest.raises(KeyError, match="shoulder_lift.pos"):
        jm.lerobot_to_sim({"shoulder_pan.pos": 0.0, "gripper.pos": 0.0})


def test_wrong_length_raises(tmp_path):
    jm = make_map(tmp_path)
    with pytest.raises(ValueError):
        jm.sim_to_lerobot(np.zeros(5))


def test_bad_sign_rejected(tmp_path):
    with pytest.raises(ValueError, match="sign"):
        make_map(tmp_path, sign0=2)


def test_calibrated_flag_parsed(tmp_path):
    assert make_map(tmp_path, calibrated="true").calibrated is True
    assert make_map(tmp_path, calibrated="false").calibrated is False


def test_cmd_order_matches_sim_model():
    pytest.importorskip("mujoco")
    from robot_manipulator.sim.model import JOINTS

    assert tuple(JOINTS) == CMD_JOINTS
