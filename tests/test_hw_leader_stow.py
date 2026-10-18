from contextlib import nullcontext

import numpy as np

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

    def write(self, reg, motor, value):
        self.log.append(f"{reg}={value}")

    def enable_torque(self, motors=None):
        self.torque = True
        self.log.append("torque_on")

    def disable_torque(self, motors=None):
        self.torque = False
        self.log.append("torque_off")

    def disconnect(self, disable_torque=True):
        self.disconnected = disable_torque


class FakeLeader:
    def __init__(self, q0):
        self.bus = FakeBus(q0)


def test_stow_rotates_only_pan_releases_and_restores(monkeypatch):
    monkeypatch.setattr(fol.time, "sleep", lambda s: None)
    rest = [-3.1, -107.0, 96.6, 77.3, -85.1, 1.5]
    leader = FakeLeader(rest)
    s = LeaderStow(
        "p",
        "id",
        stow_pan_deg=60.0,
        torque_limit=300,
        speed_deg_s=30.0,
        connect=lambda port, lid: leader,
    )
    with s:
        q = np.array([leader.bus.q[j] for j in fol.JOINTS])
        assert q[0] == 60.0 and np.allclose(q[1:], rest[1:])  # only pan moved
        assert leader.bus.torque is False  # released while the follower works
        assert "Torque_Limit=300" in leader.bus.log
    q = np.array([leader.bus.q[j] for j in fol.JOINTS])
    assert q[0] == -3.1 and leader.bus.torque is False and leader.bus.disconnected is False


def test_leader_clear_only_for_real_live_runs():
    assert isinstance(leader_clear("sim", True, True), nullcontext)
    assert isinstance(leader_clear("real", False, True), nullcontext)
    assert isinstance(leader_clear("real", True, False), nullcontext)
    assert isinstance(leader_clear("real", True, True), LeaderStow)
