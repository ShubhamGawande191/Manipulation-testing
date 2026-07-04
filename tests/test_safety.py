"""Unit tests: watchdog, workspace box, velocity cap."""

import numpy as np

from robot_manipulator.core.safety import VelocityCap, Watchdog, WorkspaceBox


def test_watchdog_staleness():
    t = {"now": 0.0}
    wd = Watchdog(timeout_s=0.1, clock=lambda: t["now"])
    assert not wd.stale
    t["now"] = 0.05
    assert not wd.stale
    t["now"] = 0.15
    assert wd.stale
    wd.beat()
    assert not wd.stale and wd.age_s == 0.0


def test_workspace_box():
    box = WorkspaceBox(lo=np.array([-0.1, -0.2, 0.0]), hi=np.array([0.3, 0.2, 0.25]))
    assert box.contains([0.0, 0.0, 0.1])
    assert not box.contains([0.0, 0.0, 0.3])
    np.testing.assert_allclose(box.clamp([0.5, -0.5, 0.1]), [0.3, -0.2, 0.1])


def test_velocity_cap():
    cap = VelocityCap(max_speed=np.array([1.0, 2.0]))
    stepped = cap.clamp_step(current=[0.0, 0.0], target=[10.0, -10.0], dt=0.1)
    np.testing.assert_allclose(stepped, [0.1, -0.2])
    # inside the limit -> passes through untouched
    np.testing.assert_allclose(cap.clamp_step([0.0, 0.0], [0.05, -0.1], 0.1), [0.05, -0.1])
