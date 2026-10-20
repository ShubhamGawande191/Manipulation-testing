from contextlib import nullcontext

import numpy as np
import pytest

import robot_manipulator.hw.follower as fol
from robot_manipulator.harness.cli import leader_clear
from robot_manipulator.hw.leader import LeaderStow


class FakeBus:
    def __init__(self, q0):
        self.q = {j: float(v) for j, v in zip(fol.JOINTS, q0)}
        self.torque = False
        self.log: list[str] = []
        self.disconnected = None

    def sync_read(self, reg, normalize=True):
        return dict(self.q)

    def sync_write(self, reg, values):
        assert self.torque or all(abs(values[j] - self.q[j]) < 1e-9 for j in values), (
            "goal changed before torque on (would jump)"
        )
        self.q.update(values)

    def write(self, reg, motor, value, num_retry=0):
        self.log.append(f"{reg}={value}")

    def enable_torque(self, motors=None, num_retry=0):
        self.torque = True
        self.log.append("torque_on")

    def disable_torque(self, motors=None, num_retry=0):
        self.torque = False
        self.log.append("torque_off")

    def disconnect(self, disable_torque=True):
        self.disconnected = disable_torque


class FakeLeader:
    def __init__(self, q0):
        self.bus = FakeBus(q0)


CFG = {
    "port": "p",
    "leader_id": "id",
    "stow_pose_deg": None,
    "default_stow_pan_deg": -60.0,
    "rest_pose_deg": [-3.1, -107.0, 96.6, 77.3, -85.1, 1.5],
    "torque_limit": 300,
    "speed_deg_s": 30.0,
    "hold_torque_during_run": True,
}


def _q(leader):
    return np.array([leader.bus.q[j] for j in fol.JOINTS])


def test_default_stow_turns_left_holds_and_returns_to_rest(monkeypatch):
    monkeypatch.setattr(fol.time, "sleep", lambda s: None)
    start = [20.6, -97.1, 97.0, 73.6, -84.7, 1.1]  # wherever the leader happens to be
    leader = FakeLeader(start)
    s = LeaderStow(dict(CFG), connect=lambda port, lid: leader)
    with s:
        q = _q(leader)
        assert q[0] == -60.0  # the leader's left (negative pan)
        np.testing.assert_allclose(q[1:], CFG["rest_pose_deg"][1:])
        assert leader.bus.torque is True  # holding during the run
        assert "Torque_Limit=300" in leader.bus.log
    np.testing.assert_allclose(_q(leader), CFG["rest_pose_deg"])  # back to REST, not to start
    assert leader.bus.torque is False and leader.bus.disconnected is False


def test_taught_stow_pose_is_used(monkeypatch):
    monkeypatch.setattr(fol.time, "sleep", lambda s: None)
    taught = [-75.0, -60.0, 70.0, 40.0, 0.0, 10.0]
    leader = FakeLeader(CFG["rest_pose_deg"])
    with LeaderStow({**CFG, "stow_pose_deg": taught}, connect=lambda port, lid: leader):
        np.testing.assert_allclose(_q(leader), taught)


def test_teach_rewrites_only_the_pose_line(tmp_path):
    from robot_manipulator.hw.leader import _save_pose

    path = tmp_path / "stow.yaml"
    path.write_text("port: x\nstow_pose_deg: null          # comment kept\nspeed_deg_s: 30.0\n")
    _save_pose("stow_pose_deg", np.array([-60.0, -100, 90, 70, -80, 2]), path)
    text = path.read_text()
    assert "stow_pose_deg: [-60.0, -100.0, 90.0, 70.0, -80.0, 2.0]  # comment kept" in text
    assert "port: x" in text and "speed_deg_s: 30.0" in text


def test_leader_clear_only_for_real_live_runs():
    assert isinstance(leader_clear("sim", True, True), nullcontext)
    assert isinstance(leader_clear("real", False, True), nullcontext)
    assert isinstance(leader_clear("real", True, False), nullcontext)
    assert isinstance(leader_clear("real", True, True), LeaderStow)


def test_torque_on_releases_the_leader_if_enabling_fails():
    from robot_manipulator.hw.leader import torque_on

    class Flaky(FakeBus):
        def enable_torque(self, motors=None, num_retry=0):
            self.torque_calls = getattr(self, "torque_calls", 0) + 1
            raise ConnectionError("no status packet")

    bus = Flaky(np.zeros(6))
    disabled = []
    bus.disable_torque = lambda motors=None, num_retry=0: disabled.append(True)
    leader = type("L", (), {"bus": bus})()
    with pytest.raises(ConnectionError):
        torque_on(leader, 300)
    assert disabled  # released before the error propagated
