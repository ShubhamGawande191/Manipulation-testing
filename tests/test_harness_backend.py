"""Unit tests: fake, sim and LeRobot backends. LeRobot runs only against an injected fake."""

from pathlib import Path

import numpy as np
import pytest

from robot_manipulator.exec.scheduler import SimClock
from robot_manipulator.harness.backend import FakeBackend, LeRobotBackend, RobotBackend
from robot_manipulator.harness.units import BODY_JOINTS, JointMap

REPO_CFG = Path(__file__).resolve().parents[1] / "config" / "sim2real.yaml"
CAL_YAML = """
calibrated: true
joints:
  shoulder_pan: {sign: -1, offset_deg: 5.0}
  shoulder_lift: {sign: 1, offset_deg: 0.0}
  elbow_flex: {sign: 1, offset_deg: 0.0}
  wrist_flex: {sign: 1, offset_deg: 0.0}
  wrist_roll: {sign: 1, offset_deg: 0.0}
gripper: {lerobot_open: 0.0, lerobot_closed: 100.0, sim_open: 1.74533, sim_closed: -0.17453}
"""


class FakeRobot:
    """Stands in for SOFollower: records calls, never touches a serial port."""

    def __init__(self, obs_deg):
        self.obs = dict(obs_deg)
        self.sent: list[dict] = []
        self.disconnected = False

    def get_observation(self):
        return dict(self.obs)

    def send_action(self, action):
        self.sent.append(dict(action))
        return action

    def disconnect(self):
        self.disconnected = True


def _obs_deg(**overrides):
    d = {f"{j}.pos": 0.0 for j in BODY_JOINTS} | {"gripper.pos": 0.0}
    d.update({f"{k}.pos": v for k, v in overrides.items()})
    return d


def _cal_map(tmp_path: Path) -> JointMap:
    p = tmp_path / "cal.yaml"
    p.write_text(CAL_YAML)
    return JointMap.from_yaml(p)


# ---- FakeBackend -------------------------------------------------------------------


def test_fake_backend_satisfies_protocol():
    assert isinstance(FakeBackend(), RobotBackend)


def test_fake_backend_converges_geometrically():
    fb = FakeBackend(q0=np.zeros(6), alpha=0.3, clock=SimClock())
    target = np.array([0.5, -0.2, 0.1, 0.0, 0.3, 0.8])
    errs = []
    for _ in range(60):
        fb.command(target)
        errs.append(np.abs(fb.q - target).max())
    assert all(b < a for a, b in zip(errs, errs[1:]))
    np.testing.assert_allclose(fb.read()["q"], target, atol=1e-6)


def test_fake_backend_velocity_is_finite_difference():
    clock = SimClock()
    fb = FakeBackend(q0=np.zeros(6), alpha=0.5, clock=clock)
    fb.read()
    fb.command(np.ones(6))  # q: 0 -> 0.5
    clock.sleep(0.125)
    obs = fb.read()
    assert obs["t"] == 0.125
    np.testing.assert_allclose(obs["dq"], 0.5 / 0.125)


def test_fake_backend_rejects_wrong_shape():
    with pytest.raises(ValueError):
        FakeBackend().command(np.zeros(5))


# ---- SimBackend (MuJoCo) -----------------------------------------------------------


def test_sim_backend_substeps_and_protocol():
    pytest.importorskip("mujoco")
    from robot_manipulator.harness.backend import SimBackend

    sb = SimBackend(hz=50.0)
    assert sb.n_substeps == 4  # 1 / (50 Hz * 0.005 s)
    assert isinstance(sb, RobotBackend)
    sb.close()


def test_sim_backend_tracks_constant_target_within_2cm_rad_after_2s():
    pytest.importorskip("mujoco")
    from robot_manipulator.harness.backend import SimBackend

    sb = SimBackend(hz=50.0)
    target = np.array([0.2, -1.0, 1.0, 0.3, 0.1, -0.5])  # gripper inside the PincOpen range
    for _ in range(100):  # 2 s of control at 50 Hz
        sb.command(target)
    obs = sb.read()
    assert obs["t"] == pytest.approx(2.0, abs=1e-9)
    np.testing.assert_allclose(obs["q"], target, atol=0.02)
    sb.close()


def test_sim_backend_reset_places_pose():
    pytest.importorskip("mujoco")
    from robot_manipulator.harness.backend import SimBackend

    sb = SimBackend(hz=50.0)
    q0 = np.array([0.1, -0.3, 0.4, 0.0, 0.0, 0.2])
    sb.reset(q0)
    np.testing.assert_allclose(sb.read()["q"], q0, atol=1e-9)
    sb.close()


# ---- LeRobotBackend (fake robot only) ----------------------------------------------


def test_lerobot_refuses_live_when_uncalibrated():
    import dataclasses

    jm = dataclasses.replace(JointMap.from_yaml(REPO_CFG), calibrated=False)
    with pytest.raises(RuntimeError, match="calibrated"):
        LeRobotBackend("unused", jm, dry_run=False, robot=FakeRobot(_obs_deg()))


def test_lerobot_dry_run_never_sends_but_records_command():
    jm = JointMap.from_yaml(REPO_CFG)
    robot = FakeRobot(_obs_deg())
    lb = LeRobotBackend("unused", jm, dry_run=True, robot=robot)
    lb.command(np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0]))
    assert robot.sent == []
    assert lb.last_command is not None
    assert set(lb.last_command) == {f"{j}.pos" for j in BODY_JOINTS + ("gripper",)}


def test_lerobot_read_converts_degrees_and_gripper(tmp_path):
    jm = _cal_map(tmp_path)
    robot = FakeRobot(_obs_deg(shoulder_pan=30.0, elbow_flex=-20.0, **{"gripper": 100.0}))
    lb = LeRobotBackend("unused", jm, dry_run=True, robot=robot)
    obs = lb.read()
    # shoulder_pan: sign -1, offset 5 deg -> rad = -deg2rad(30) + deg2rad(5)
    assert obs["q"][0] == pytest.approx(np.deg2rad(-30.0 + 5.0), abs=1e-12)
    assert obs["q"][2] == pytest.approx(np.deg2rad(-20.0), abs=1e-12)
    assert obs["q"][5] == pytest.approx(-0.17453, abs=1e-9)  # gripper 100 -> sim_closed
    assert obs["dq"].shape == (6,)


def test_lerobot_live_sends_degrees_and_close_disconnects(tmp_path):
    jm = _cal_map(tmp_path)
    robot = FakeRobot(_obs_deg())
    lb = LeRobotBackend("unused", jm, dry_run=False, robot=robot)
    q = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 1.74533])
    lb.command(q)
    assert len(robot.sent) == 1
    sent = robot.sent[0]
    assert sent == pytest.approx(jm.sim_to_lerobot(q))
    assert sent["gripper.pos"] == pytest.approx(0.0, abs=1e-6)  # sim_open -> lerobot 0
    lb.close()
    assert robot.disconnected is True
