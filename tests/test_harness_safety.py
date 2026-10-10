"""Unit tests: harness SafetyFilter (clamp, velocity cap, stale hold) with a SimClock."""

import numpy as np
import pytest

from robot_manipulator.exec.scheduler import SimClock
from robot_manipulator.harness.safety_layer import SafetyFilter

LO = np.full(6, -1.0)
HI = np.full(6, 1.0)


def make(clock=None, max_vel=1.0) -> SafetyFilter:
    return SafetyFilter(
        lo=LO,
        hi=HI,
        margin=0.05,
        max_vel=max_vel,
        watchdog_timeout=0.25,
        clock=clock or SimClock(),
    )


def test_previous_command_outside_box_is_clipped_before_stepping():
    sf = make()
    q = sf.filter(np.full(6, 2.0), np.zeros(6), dt=0.1)
    np.testing.assert_allclose(q, np.full(6, 0.85))  # 2.0 -> 0.95, then one 0.1 step


def test_clamps_out_of_range_target_to_margin_box():
    sf = make()
    q = sf.filter(np.zeros(6), np.array([5.0, -5.0, 0.0, 0.0, 0.0, 9.0]), dt=100.0)
    np.testing.assert_allclose(q, [0.95, -0.95, 0.0, 0.0, 0.0, 0.95])


def test_step_is_capped_to_max_vel_times_dt():
    sf = make()
    q = sf.filter(np.zeros(6), np.full(6, 0.9), dt=0.1)
    np.testing.assert_allclose(q, np.full(6, 0.1))  # 1.0 rad/s * 0.1 s


def test_small_move_passes_through():
    sf = make()
    target = np.array([0.02, -0.03, 0.0, 0.01, 0.0, 0.05])  # every step < 1 rad/s * 0.1 s
    np.testing.assert_allclose(sf.filter(np.zeros(6), target, dt=0.1), target)


def test_per_joint_velocity_array():
    vel = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    sf = make(max_vel=vel)
    q = sf.filter(np.zeros(6), np.full(6, 0.9), dt=0.1)
    assert q[0] == pytest.approx(0.2) and q[1] == pytest.approx(0.1)


def test_consecutive_outputs_respect_velocity_cap_and_box():
    rng = np.random.default_rng(0)
    sf = make()
    prev = np.zeros(6)
    for _ in range(200):
        target = rng.uniform(-3, 3, size=6)
        q = sf.filter(prev, target, dt=0.02)
        assert np.all(q >= LO + 0.05 - 1e-12) and np.all(q <= HI - 0.05 + 1e-12)
        assert np.all(np.abs(q - prev) <= 1.0 * 0.02 + 1e-12)
        prev = q


def test_stale_watchdog_holds_previous_command():
    clock = SimClock()
    sf = make(clock=clock)
    prev = np.full(6, 0.2)
    clock.sleep(0.3)  # > watchdog_timeout 0.25 with no beat()
    assert sf.stale
    np.testing.assert_allclose(sf.filter(prev, np.full(6, 0.9), dt=0.1), prev)
    sf.beat()
    assert not sf.stale
    moved = sf.filter(prev, np.full(6, 0.9), dt=0.1)
    np.testing.assert_allclose(moved, prev + 0.1)


def test_stale_after_exact_timeout_boundary():
    clock = SimClock()
    sf = make(clock=clock)
    clock.sleep(0.25)
    assert not sf.stale  # Watchdog uses strict '>'
    clock.sleep(0.125)
    assert sf.stale


def test_margin_larger_than_range_rejected():
    with pytest.raises(ValueError):
        SafetyFilter(lo=np.full(6, -0.01), hi=np.full(6, 0.01), margin=0.05)


def test_default_limits_from_mjcf_with_margin():
    pytest.importorskip("mujoco")
    from robot_manipulator.harness.safety_layer import default_limits

    lo, hi = default_limits()
    sf = SafetyFilter(clock=SimClock())
    np.testing.assert_allclose(sf.lo, lo + 0.05, atol=1e-12)
    np.testing.assert_allclose(sf.hi, hi - 0.05, atol=1e-12)
