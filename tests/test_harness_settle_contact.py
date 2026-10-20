import numpy as np
import pytest

from robot_manipulator.exec.scheduler import SimClock
from robot_manipulator.harness.policies.settle import SettleTrim
from robot_manipulator.harness.runner import run_skill
from robot_manipulator.harness.safety_layer import SafetyFilter
from robot_manipulator.harness.skill import ContactGuard, Skill, TimeLimit


class SaggingArm:
    """Stops short of every command by a fixed amount (gravity sag / deadband stand-in)."""

    joint_names = ("a", "b", "c", "d", "e", "f")

    def __init__(self, sag: np.ndarray, clock) -> None:
        self.sag, self.clock, self.q = sag, clock, np.zeros(6)

    def read(self):
        return {"t": self.clock.now(), "q": self.q.copy(), "dq": np.zeros(6)}

    def command(self, q):
        self.q = np.asarray(q, dtype=float) - self.sag

    def close(self):
        pass


def test_settle_trim_removes_steady_state_error():
    clock = SimClock()
    sag = np.radians([0.5, 2.0, -1.5, 1.0, 0.0, 0.0])
    arm = SaggingArm(sag, clock)
    target = np.radians([10.0, -30.0, 40.0, 20.0, -80.0, 0.0])
    arm.q = target - sag
    safety = SafetyFilter(
        lo=np.full(6, -3.0), hi=np.full(6, 3.0), margin=0.0, max_vel=1.0, clock=clock
    )
    skill = Skill("settle", SettleTrim(target), [TimeLimit(3.0)], hz=50.0)
    run_skill(skill, arm, safety, goal={}, clock=clock)
    assert np.abs(arm.q[:5] - target[:5]).max() < np.radians(0.2)


def test_settle_trim_is_bounded():
    clock = SimClock()
    arm = SaggingArm(np.radians([0, 10.0, 0, 0, 0, 0]), clock)  # pushing against something
    safety = SafetyFilter(
        lo=np.full(6, -3.0), hi=np.full(6, 3.0), margin=0.0, max_vel=1.0, clock=clock
    )
    pol = SettleTrim(np.zeros(6), max_trim=np.radians(4.0))
    run_skill(Skill("settle", pol, [TimeLimit(5.0)], hz=8.0), arm, safety, goal={}, clock=clock)
    assert np.abs(pol.trim).max() <= np.radians(4.0) + 1e-12


def test_contact_guard_trips_on_the_table_not_in_free_air():
    pytest.importorskip("mujoco")
    from robot_manipulator.calib.extrinsics import TipFK
    from robot_manipulator.harness.backend import SimBackend
    from robot_manipulator.harness.policies.glide import GlideTo
    from robot_manipulator.harness.skill import JointReached
    from robot_manipulator.ik.analytic import AnalyticIK
    from robot_manipulator.sim.model import JOINTS, joint_ranges, load_model

    ik = AnalyticIK(TipFK())
    m = load_model()  # scene_pinc: the floor is the table top, 2.75 cm below the base
    lo, hi = joint_ranges(m, JOINTS)
    roll = np.radians(-80)
    above = ik.solve(np.array([0.24, 0.0, 0.02]), np.pi / 2, roll)  # 4.75 cm above the table
    below = ik.solve(np.array([0.24, 0.0, -0.05]), np.pi / 2, roll)  # 2.25 cm into the table
    assert above is not None and below is not None
    for goal_q, expect in ((above, "success"), (below, "failure")):
        clock = SimClock()
        be = SimBackend(model=load_model(), hz=50.0)
        start = ik.solve(np.array([0.24, 0.0, 0.05]), np.pi / 2, roll)  # 7.75 cm above
        assert start is not None  # straight down needs wrist_flex > 95 deg much higher than this
        be.reset(start)
        safety = SafetyFilter(lo=lo, hi=hi, margin=0.05, max_vel=0.15, clock=clock)
        sk = Skill(
            "approach",
            GlideTo(goal_q, max_vel=0.15, hz=50.0),
            [
                ContactGuard(),
                JointReached(goal_q, tol=0.01, joints=(0, 1, 2, 3, 4)),
                TimeLimit(8.0),
            ],
            hz=50.0,
        )
        res = run_skill(sk, be, safety, goal={}, clock=clock)
        assert res.status == expect, (expect, res.reason)
        if expect == "failure":
            assert res.reason == "contact"
