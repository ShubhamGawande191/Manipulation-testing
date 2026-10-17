"""Unit tests: SO101Reach-v0 env, the shared obs/command helpers, and domain randomization."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from robot_manipulator.sim.envs.reach import (  # noqa: E402
    ACTION_SCALE,
    ENV_ID,
    EP_STEPS,
    GRIPPER_HOLD,
    OBS_DIM,
    DRConfig,
    ReachEnv,
    build_obs,
    command_box,
    integrate_command,
)
from robot_manipulator.sim.kinematics import FK  # noqa: E402
from robot_manipulator.sim.model import joint_ranges, load_model  # noqa: E402


def ik_action(obs, fk, k=0.5):
    """Jacobian pseudo-inverse toward the target. Reads only the observation, like a policy."""
    q = obs[:5].astype(np.float64)
    ee = obs[15:18].astype(np.float64)
    target = obs[18:21].astype(np.float64)
    jac = fk.jacobian(np.r_[q, GRIPPER_HOLD])[:, :5]
    dq = np.linalg.pinv(jac, rcond=1e-3) @ (target - ee)
    return np.clip(k * dq / ACTION_SCALE, -1.0, 1.0)


def test_build_obs_layout_and_dtype():
    obs = build_obs(
        q=[1, 2, 3, 4, 5],
        dq=[6, 7, 8, 9, 10],
        last_action=[11, 12, 13, 14, 15],
        ee=[16, 17, 18],
        target=[19, 20, 21],
    )
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[:5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(obs[5:10], [6, 7, 8, 9, 10])
    np.testing.assert_array_equal(obs[10:15], [11, 12, 13, 14, 15])
    np.testing.assert_array_equal(obs[15:18], [16, 17, 18])
    np.testing.assert_array_equal(obs[18:21], [19, 20, 21])
    np.testing.assert_array_equal(obs[21:24], [3, 3, 3])  # target - ee


def test_build_obs_rejects_wrong_shape():
    with pytest.raises(ValueError, match="q"):
        build_obs(np.zeros(6), np.zeros(5), np.zeros(5), np.zeros(3), np.zeros(3))


def test_integrate_command_steps_and_clips():
    lo, hi = command_box(*joint_ranges(load_model()))
    q = np.zeros(5)
    np.testing.assert_allclose(
        integrate_command(q, np.full(5, 0.5), (lo, hi)), np.full(5, 0.5 * ACTION_SCALE)
    )
    # actions outside [-1, 1] are clipped before integrating
    np.testing.assert_allclose(
        integrate_command(q, np.full(5, 7.0), (lo, hi)), np.full(5, ACTION_SCALE)
    )
    # the command never leaves the margin-shrunk joint limits
    near_top = hi - 0.001
    assert np.all(integrate_command(near_top, np.ones(5), (lo, hi)) <= hi)


def test_check_env_passes():
    env = ReachEnv(dr=DRConfig(enabled=False))
    check_env(env, skip_render_check=True)


def test_registered_id_makes_env_with_options():
    env = gym.make(ENV_ID, dr=DRConfig(enabled=False))
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert info["action_delay"] == 0
    env.close()


def test_zero_action_holds_pose_without_dr():
    env = ReachEnv(dr=DRConfig(enabled=False))
    obs, _ = env.reset(seed=0)
    q0 = obs[:5].copy()
    for _ in range(50):
        obs, *_ = env.step(np.zeros(5, dtype=np.float32))
    assert np.max(np.abs(obs[:5] - q0)) < 0.01


def test_dr_off_is_deterministic_for_a_seed():
    rollouts = []
    for _ in range(2):
        env = ReachEnv(dr=DRConfig(enabled=False))
        obs, _ = env.reset(seed=3)
        acts = np.random.default_rng(7).uniform(-1, 1, size=(30, 5))
        trace = [np.r_[obs, 0.0]]
        for a in acts:
            obs, r, *_ = env.step(a)
            trace.append(np.r_[obs, r])
        rollouts.append(np.array(trace))
    np.testing.assert_array_equal(rollouts[0], rollouts[1])


def test_dr_changes_parameters_and_disabling_restores_nominal():
    env = ReachEnv(dr=DRConfig(enabled=True))
    nominal_gain = env.model.actuator_gainprm[:, 0].copy()
    nominal_mass = env.model.body_mass.copy()
    env.reset(seed=1)
    assert not np.allclose(env.model.actuator_gainprm[:, 0], nominal_gain)
    assert not np.allclose(env.model.body_mass, nominal_mass)

    env.dr = DRConfig(enabled=False)
    env.reset(seed=2)
    np.testing.assert_array_equal(env.model.actuator_gainprm[:, 0], nominal_gain)
    np.testing.assert_array_equal(env.model.body_mass, nominal_mass)


def test_dr_adds_gripper_payload_on_gripper_body():
    env = ReachEnv(dr=DRConfig(enabled=True, link_mass=(1.0, 1.0), gripper_payload_kg=(0.2, 0.2)))
    nominal = env.model.body_mass[env._gripper_body]
    env.reset(seed=0)
    assert env.model.body_mass[env._gripper_body] == pytest.approx(nominal + 0.2)


def test_dr_reset_does_not_drift_over_many_resets():
    env = ReachEnv(dr=DRConfig(enabled=True))
    nominal_damp = env._nominal.damping.copy()
    for seed in range(20):
        env.reset(seed=seed)
    env.dr = DRConfig(enabled=False)
    env.reset(seed=99)
    np.testing.assert_array_equal(env.model.dof_damping, nominal_damp)


def test_truncates_exactly_at_episode_length():
    env = ReachEnv(dr=DRConfig(enabled=False))
    env.reset(seed=0)
    flags = []
    for _ in range(EP_STEPS):
        _, _, term, trunc, _ = env.step(np.zeros(5))
        flags.append(trunc)
        assert term is False
    assert flags[-1] is True and not any(flags[:-1])


def test_ik_policy_reaches_most_targets_without_dr():
    """Acceptance: the targets are reachable and the action pathway works (≥ 80 % of 20)."""
    fk = FK()
    successes = 0
    for seed in range(20):
        env = ReachEnv(dr=DRConfig(enabled=False))
        obs, _ = env.reset(seed=seed)
        info = {}
        for _ in range(EP_STEPS):
            obs, _, _, truncated, info = env.step(ik_action(obs, fk))
            if truncated:
                break
        successes += bool(info["success"])
    assert successes >= 16, f"only {successes}/20 targets reached"


def test_targets_stay_in_table_workspace():
    from robot_manipulator.sim.envs.reach import TARGET_BOX

    env = ReachEnv(dr=DRConfig(enabled=False))
    targets = np.array([env.reset(seed=i)[1]["target"] for i in range(50)])
    assert (targets >= TARGET_BOX[0]).all() and (targets <= TARGET_BOX[1]).all()
