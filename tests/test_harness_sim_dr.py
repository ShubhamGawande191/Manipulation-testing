"""Unit tests: SimBackend domain randomization (sensor noise, bias, action delay) and GlideTo."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulator.exec.scheduler import SimClock  # noqa: E402
from robot_manipulator.harness.backend import SimBackend  # noqa: E402
from robot_manipulator.harness.policies.glide import GlideTo  # noqa: E402
from robot_manipulator.harness.runner import run_skill  # noqa: E402
from robot_manipulator.harness.safety_layer import SafetyFilter  # noqa: E402
from robot_manipulator.harness.skill import JointReached, Skill, TimeLimit  # noqa: E402
from robot_manipulator.sim.envs.reach import GRIPPER_HOLD, HOME_ARM  # noqa: E402
from robot_manipulator.sim.randomize import DRConfig  # noqa: E402

HOME6 = np.r_[HOME_ARM, GRIPPER_HOLD]  # PincOpen fully open (default model)


def _flat_dr(**kw) -> DRConfig:
    """DR with every physics range pinned to 1 and no sensor noise, so only the pipeline varies."""
    base = dict(
        enabled=True,
        kp_kv=(1.0, 1.0),
        damping=(1.0, 1.0),
        frictionloss=(1.0, 1.0),
        armature=(1.0, 1.0),
        link_mass=(1.0, 1.0),
        gripper_payload_kg=(0.0, 0.0),
        action_delay_steps=(0,),
        obs_noise_q=0.0,
        obs_noise_dq=0.0,
        joint_bias=0.0,
    )
    base.update(kw)
    return DRConfig(**base)


def test_dr_off_read_is_exact_and_repeatable():
    sb = SimBackend(hz=50.0, dr=DRConfig(enabled=False))
    sb.reset(HOME6)
    a = sb.read()
    b = sb.read()
    np.testing.assert_array_equal(a["q"], b["q"])
    np.testing.assert_array_equal(a["q"], a["q_true"])
    sb.close()


def test_dr_on_adds_bias_and_noise_to_measured_joints():
    sb = SimBackend(hz=50.0, dr=DRConfig(enabled=True), seed=3)
    sb.reset(HOME6)
    reads = [sb.read() for _ in range(20)]
    measured = np.array([r["q"] for r in reads])
    truth = np.array([r["q_true"] for r in reads])
    err = measured - truth
    assert np.max(np.abs(err[:, :5])) > 0.0  # sensor is not exact
    assert np.all(np.abs(err[:, :5]) < 0.05)  # bias (0.02) + noise (0.005), a few sigma
    np.testing.assert_array_equal(err[:, 5], 0.0)  # gripper is not perturbed
    sb.close()


def test_same_seed_gives_identical_dr_episodes():
    traces = []
    for _ in range(2):
        sb = SimBackend(hz=50.0, dr=DRConfig(enabled=True), seed=11)
        sb.reset(HOME6)
        trace = []
        for k in range(10):
            sb.command(HOME6 + 0.01 * k)
            trace.append(sb.read()["q"])
        traces.append(np.array(trace))
        sb.close()
    np.testing.assert_array_equal(traces[0], traces[1])


def test_action_delay_two_ticks_matches_the_env_pipeline():
    cfg = _flat_dr(action_delay_steps=(2,))
    sb = SimBackend(hz=50.0, dr=cfg, seed=0)
    sb.reset(HOME6)
    cmd_a = HOME6 + 0.1
    cmd_b = HOME6 + 0.2
    cmd_c = HOME6 + 0.3
    ctrl = lambda: sb.data.ctrl[sb._act].copy()  # noqa: E731

    sb.command(cmd_a)
    np.testing.assert_allclose(ctrl(), HOME6, atol=1e-12)  # held: first command not yet due
    sb.command(cmd_b)
    np.testing.assert_allclose(ctrl(), HOME6, atol=1e-12)
    sb.command(cmd_c)
    np.testing.assert_allclose(ctrl(), cmd_a, atol=1e-12)  # two ticks later, A arrives
    sb.close()


def test_glide_reaches_home_in_sim_without_dr():
    sb = SimBackend(hz=50.0, dr=DRConfig(enabled=False))
    sb.reset(np.zeros(6))
    clock = SimClock()
    safety = SafetyFilter(lo=np.full(6, -2.0), hi=np.full(6, 2.0), max_vel=1.0, clock=clock)
    skill = Skill(
        "home",
        GlideTo(HOME6, max_vel=1.0, hz=50.0),
        [JointReached(HOME6, tol=0.01), TimeLimit(5.0)],
        hz=50.0,
    )
    res = run_skill(skill, sb, safety, goal={}, clock=clock)
    sb.close()
    assert res.status == "success", res.reason
    assert res.n_steps <= 5 * 50
