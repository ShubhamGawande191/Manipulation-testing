import numpy as np

import robot_manipulator.hw.follower as fol


class FakeBus:
    def __init__(self, q0):
        self.q = {j: float(v) for j, v in zip(fol.JOINTS, q0)}
        self.goals: list[np.ndarray] = []
        self.torque_off: list[str] = []

    def sync_read(self, reg, normalize=True):
        assert reg == "Present_Position"
        return dict(self.q)

    def sync_write(self, reg, values):
        assert reg == "Goal_Position"
        self.q.update(values)  # perfect tracking
        self.goals.append(np.array([values[j] for j in fol.JOINTS]))

    def disable_torque(self, motor):
        self.torque_off.append(motor)


class FakeRobot:
    def __init__(self, q0):
        self.bus = FakeBus(q0)


def test_park_goes_via_home_before_staged_fold(monkeypatch):
    monkeypatch.setattr(fol.time, "sleep", lambda s: None)
    start = np.array([-36.0, -3.0, 17.0, -5.0, -84.0, 97.0])  # end of a real reach (2026-10-18)
    home = np.array([0.0, -89.95, 89.95, 0.0, 0.0, 97.46])
    robot = FakeRobot(start)
    ticks = []
    fol.park_and_release(
        robot, np.r_[fol.REST_ARM, 97.0], speed=30.0, hz=50.0, via=home, on_tick=ticks.append
    )
    goals = np.array(robot.bus.goals)
    # the first segment moves several arm joints at once (joint-space glide to HOME) ...
    first_step = goals[0] - start
    assert np.count_nonzero(np.abs(first_step[:5]) > 1e-9) >= 4
    # ... and reaches HOME (arm joints) before the staged fold begins
    reached = np.flatnonzero(np.all(np.abs(goals[:, :5] - home[:5]) < 1e-6, axis=1))
    assert reached.size and reached[0] < len(goals) - 1
    np.testing.assert_allclose(goals[-1][:5], fol.REST_ARM, atol=1e-6)
    assert robot.bus.torque_off == list(reversed(fol.STAGE_ORDER))
    assert len(ticks) == len(goals)  # every commanded tick is reported (telemetry)
