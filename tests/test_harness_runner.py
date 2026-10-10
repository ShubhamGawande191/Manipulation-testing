"""Unit tests: skill runtime (end conditions, redo, sequences, trajectory parquet).

Everything runs on FakeBackend + SimClock at 8 Hz, so dt = 0.125 is float-exact and
tick times are exact.
"""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from robot_manipulator.core.policy_iface import Health
from robot_manipulator.exec.scheduler import SimClock
from robot_manipulator.harness.backend import FakeBackend
from robot_manipulator.harness.runner import run_sequence, run_skill
from robot_manipulator.harness.safety_layer import SafetyFilter
from robot_manipulator.harness.skill import (
    EEReached,
    JointReached,
    Skill,
    Stalled,
    TimeLimit,
)
from robot_manipulator.harness.trajlog import TrajectoryLog

HZ = 8.0


class ConstPolicy:
    name = "const"

    def __init__(self, target, chunk_len=1):
        self.target = np.asarray(target, dtype=np.float64)
        self.chunk_len = chunk_len

    def reset(self):
        pass

    def act(self, obs):
        return np.tile(self.target, (self.chunk_len, 1))

    def health(self):
        return Health()


def make_env(alpha=0.3, max_vel=1.0):
    clock = SimClock()
    backend = FakeBackend(q0=np.zeros(6), alpha=alpha, clock=clock)
    safety = SafetyFilter(
        lo=np.full(6, -2.0),
        hi=np.full(6, 2.0),
        margin=0.05,
        max_vel=max_vel,
        watchdog_timeout=0.25,
        clock=clock,
    )
    return clock, backend, safety


def test_constant_target_succeeds_via_joint_reached():
    clock, backend, safety = make_env()
    target = np.array([0.4, -0.3, 0.2, 0.1, 0.0, 0.5])
    skill = Skill(
        "reach",
        ConstPolicy(target),
        [JointReached(target, tol=0.01), TimeLimit(5.0)],
        hz=HZ,
    )
    res = run_skill(skill, backend, safety, goal={}, clock=clock)
    assert res.status == "success"
    assert res.reason == "joint_reached"
    assert 0 < res.n_steps < 40
    np.testing.assert_allclose(backend.read()["q"], target, atol=0.01)


def test_never_arriving_policy_fails_via_time_limit_at_exact_step():
    clock, backend, safety = make_env()
    unreachable = np.full(6, 1.5)
    skill = Skill(
        "hold",
        ConstPolicy(np.zeros(6)),  # command stays at 0, so the goal is never reached
        [JointReached(unreachable, tol=0.01), TimeLimit(1.0)],
        hz=HZ,
    )
    res = run_skill(skill, backend, safety, goal={}, clock=clock)
    assert res.status == "failure"
    assert res.reason == "time_limit"
    assert res.n_steps == 8  # ticks 0..7 are commanded; tick 8 sees elapsed == 1.0
    assert res.duration_s == pytest.approx(1.0)
    assert res.redo_count == 0


def test_max_redo_counts_redo_and_restarts_clock():
    clock, backend, safety = make_env()
    skill = Skill(
        "short",
        ConstPolicy(np.zeros(6)),
        [JointReached(np.full(6, 1.5), tol=0.01), TimeLimit(0.5)],
        hz=HZ,
        max_redo=1,
    )
    res = run_skill(skill, backend, safety, goal={}, clock=clock)
    assert res.status == "failure"
    assert res.redo_count == 1
    assert res.n_steps == 8  # two 4-tick attempts
    assert res.duration_s == pytest.approx(1.0)


def test_on_start_runs_before_each_attempt():
    clock, backend, safety = make_env()
    calls = []
    skill = Skill(
        "short",
        ConstPolicy(np.zeros(6)),
        [TimeLimit(0.25)],
        hz=HZ,
        max_redo=2,
        on_start=lambda b: calls.append(float(b.read()["t"])),
    )
    res = run_skill(skill, backend, safety, goal={}, clock=clock)
    assert res.redo_count == 2
    assert len(calls) == 3


def test_velocity_cap_limits_each_commanded_step(tmp_path):
    clock, backend, safety = make_env(alpha=1.0, max_vel=0.5)  # instant backend
    target = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    skill = Skill("ramp", ConstPolicy(target), [TimeLimit(0.5)], hz=HZ)
    log = TrajectoryLog(tmp_path / "run")
    run_skill(skill, backend, safety, goal={}, clock=clock, log=log)

    q_safe = pq.read_table(tmp_path / "run" / "traj_ramp_1.parquet")["q_safe_0"].to_pylist()
    steps = np.diff([0.0, *q_safe])
    assert np.all(steps <= 0.5 * (1 / HZ) + 1e-12)
    assert q_safe[-1] == pytest.approx(4 * 0.5 * (1 / HZ))


def test_sequence_stops_at_first_failure():
    clock, backend, safety = make_env()
    target = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    failing = Skill(
        "fail",
        ConstPolicy(np.zeros(6)),
        [JointReached(np.full(6, 1.5), tol=0.01), TimeLimit(0.25)],
        hz=HZ,
    )
    ok = Skill("ok", ConstPolicy(target), [JointReached(target, 0.01), TimeLimit(5.0)], hz=HZ)
    results = run_sequence([(failing, {}), (ok, {})], backend, safety, clock=clock)
    assert [r.status for r in results] == ["failure"]


def test_sequence_runs_all_on_success():
    clock, backend, safety = make_env()
    target = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ok = Skill("ok", ConstPolicy(target), [JointReached(target, 0.01), TimeLimit(5.0)], hz=HZ)
    ok2 = Skill("ok2", ConstPolicy(-target), [JointReached(-target, 0.01), TimeLimit(5.0)], hz=HZ)
    results = run_sequence([(ok, {}), (ok2, {})], backend, safety, clock=clock)
    assert [r.status for r in results] == ["success", "success"]


def test_stalled_fails_when_joints_do_not_move():
    clock, backend, safety = make_env()
    skill = Skill(
        "stuck",
        ConstPolicy(np.zeros(6)),
        [Stalled(window_s=0.5, eps=1e-3), TimeLimit(5.0)],
        hz=HZ,
    )
    res = run_skill(skill, backend, safety, goal={}, clock=clock)
    assert res.status == "failure"
    assert res.reason == "stalled"
    assert res.n_steps == 4  # first full 0.5 s window ends at tick 4


def test_ee_reached_uses_goal_target():
    clock, backend, safety = make_env()
    goal_xyz = np.array([0.1, 0.2, 0.3])
    skill = Skill(
        "ee",
        ConstPolicy(np.r_[goal_xyz, 0.0, 0.0, 0.0]),
        [EEReached(tol=0.01, ee_fn=lambda obs: obs["q"][:3]), TimeLimit(5.0)],
        hz=HZ,
    )
    res = run_skill(skill, backend, safety, goal={"target": goal_xyz}, clock=clock)
    assert res.status == "success"
    assert res.reason == "ee_reached"


def test_skill_needs_an_end_condition():
    with pytest.raises(ValueError, match="end condition"):
        Skill("none", ConstPolicy(np.zeros(6)), [], hz=HZ)


def test_trajectory_parquet_is_written_and_reloads(tmp_path):
    clock, backend, safety = make_env()
    target = np.array([0.4, -0.3, 0.2, 0.1, 0.0, 0.5])
    skill = Skill("reach", ConstPolicy(target), [JointReached(target, 0.01), TimeLimit(5.0)], hz=HZ)
    log = TrajectoryLog(tmp_path / "runs" / "r1")
    res = run_skill(skill, backend, safety, goal={}, clock=clock, log=log)

    path = tmp_path / "runs" / "r1" / "traj_reach_1.parquet"
    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == res.n_steps
    for col in ["t", "skill", "step", "q_0", "dq_5", "q_target_3", "q_safe_5", "ee_0", "extra"]:
        assert col in table.column_names
    assert table.column("ee_0").null_count == table.num_rows  # FakeBackend has no ee
    assert table.column("skill")[0].as_py() == "reach"
    assert json.loads(table.column("extra")[0].as_py())["chunk_len"] == 1


def test_second_attempt_of_same_skill_gets_next_index(tmp_path):
    clock, backend, safety = make_env()
    log = TrajectoryLog(tmp_path)
    skill = Skill("tap", ConstPolicy(np.zeros(6)), [TimeLimit(0.25)], hz=HZ)
    run_skill(skill, backend, safety, goal={}, clock=clock, log=log)
    run_skill(skill, backend, safety, goal={}, clock=clock, log=log)
    assert (tmp_path / "traj_tap_1.parquet").exists()
    assert (tmp_path / "traj_tap_2.parquet").exists()
