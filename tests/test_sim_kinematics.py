"""Unit tests: FK / Jacobian / reachable sampling on the SO101 MJCF."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_manipulator.sim.kinematics import FK  # noqa: E402
from robot_manipulator.sim.model import joint_ranges, load_model  # noqa: E402

HOME = np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_ee_is_deterministic_for_fixed_q(model):
    a, b = FK(model), FK(model)
    q = np.array([0.3, -0.8, 0.9, 0.2, -0.4, 0.5])
    first = a.ee_pos(q)
    np.testing.assert_array_equal(first, a.ee_pos(q))
    np.testing.assert_array_equal(first, b.ee_pos(q))
    assert first.shape == (3,) and np.isfinite(first).all()


def test_sampled_configs_are_in_limits_and_above_z_min(model):
    fk = FK(model)
    lo, hi = joint_ranges(model)
    rng = np.random.default_rng(0)
    q, ee = fk.sample_reachable(rng, 50, lo[:5], hi[:5], z_min=0.05)
    assert q.shape == (50, 6) and ee.shape == (50, 3)
    assert np.all(ee[:, 2] >= 0.05)
    assert np.all(q[:, :5] >= lo[:5]) and np.all(q[:, :5] <= hi[:5])
    np.testing.assert_allclose(q[:, 5], 0.0)  # gripper fixed at its default
    for qi, eei in zip(q[:5], ee[:5]):
        np.testing.assert_allclose(fk.ee_pos(qi), eei, atol=1e-12)


def test_sample_reachable_accepts_six_entry_limits(model):
    fk = FK(model)
    lo, hi = joint_ranges(model)
    q, _ = fk.sample_reachable(np.random.default_rng(1), 5, lo, hi, gripper=0.7)
    np.testing.assert_allclose(q[:, 5], 0.7)


def test_sample_zero_returns_empty_arrays(model):
    q, ee = FK(model).sample_reachable(
        np.random.default_rng(2), 0, np.full(5, -1.0), np.full(5, 1.0)
    )
    assert q.shape == (0, 6) and ee.shape == (0, 3)


def test_jacobian_matches_finite_difference(model):
    fk = FK(model)
    q = HOME + np.array([0.1, 0.05, -0.05, 0.1, 0.0, 0.0])
    jac = fk.jacobian(q)
    assert jac.shape == (3, 6)
    eps = 1e-6
    for i in range(5):  # the gripper does not move the end effector
        dq = np.zeros(6)
        dq[i] = eps
        numeric = (fk.ee_pos(q + dq) - fk.ee_pos(q - dq)) / (2 * eps)
        np.testing.assert_allclose(jac[:, i], numeric, atol=1e-5)
    np.testing.assert_allclose(jac[:, 5], 0.0, atol=1e-9)


def test_fk_matches_sim_backend_state():
    from robot_manipulator.harness.backend import SimBackend

    sb = SimBackend(hz=50.0)
    target = np.array([0.2, -1.0, 1.0, 0.3, 0.1, 0.5])
    for _ in range(40):
        sb.command(target)
    q = sb.read()["q"]
    fk = FK(sb.model)
    np.testing.assert_allclose(fk.ee_pos(q), sb.ee_pos(), atol=1e-12)
    sb.close()
