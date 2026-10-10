"""Joint-space safety filter in front of every backend.

Each command goes through three steps, in order:
1. Stale watchdog: if the loop has gone quiet for `watchdog_timeout`, hold the previous
   command instead of moving (same idea as FROZEN in `core/arbiter.py`).
2. Clamp the target to `[lo + margin, hi - margin]`.
3. Limit the step to `max_vel * dt` per joint (`core.safety.VelocityCap`).

The output stays inside the box and never steps faster than `max_vel` unless the
previous command was already outside the box. In that case it is clipped to the box first.
Limits default to the MJCF joint ranges, so importing this module does not require MuJoCo.
"""

from __future__ import annotations

import numpy as np

from robot_manipulator.core.safety import VelocityCap, Watchdog
from robot_manipulator.exec.scheduler import WallClock
from robot_manipulator.harness.clock import Clock
from robot_manipulator.harness.units import CMD_JOINTS

N = len(CMD_JOINTS)


def default_limits() -> tuple[np.ndarray, np.ndarray]:
    """Joint limits (rad, CMD order) from the vendored SO101 MJCF."""
    from robot_manipulator.sim.model import JOINTS, joint_ranges, load_model

    if tuple(JOINTS) != CMD_JOINTS:
        raise RuntimeError("sim.model.JOINTS and harness CMD order disagree")
    return joint_ranges(load_model())


class SafetyFilter:
    def __init__(
        self,
        lo: np.ndarray | None = None,
        hi: np.ndarray | None = None,
        margin: float = 0.05,
        max_vel: float | np.ndarray = 1.0,
        watchdog_timeout: float = 0.25,
        clock: Clock | None = None,
    ) -> None:
        if lo is None or hi is None:
            lo, hi = default_limits()
        self.clock = clock or WallClock()
        self.lo = np.asarray(lo, dtype=np.float64).reshape(N) + margin
        self.hi = np.asarray(hi, dtype=np.float64).reshape(N) - margin
        if np.any(self.lo >= self.hi):
            raise ValueError("margin is larger than half of some joint range")
        vel = np.broadcast_to(np.asarray(max_vel, dtype=np.float64), (N,)).copy()
        if np.any(vel <= 0):
            raise ValueError("max_vel must be positive")
        self.cap = VelocityCap(vel)
        self.watchdog = Watchdog(timeout_s=watchdog_timeout, clock=self.clock.now)

    def beat(self) -> None:
        self.watchdog.beat()

    @property
    def stale(self) -> bool:
        return self.watchdog.stale

    def filter(self, q_cmd_prev: np.ndarray, q_target: np.ndarray, dt: float) -> np.ndarray:
        """Return the joint command to send this tick."""
        prev = np.clip(np.asarray(q_cmd_prev, dtype=np.float64).reshape(N), self.lo, self.hi)
        if self.stale:
            return prev
        target = np.clip(np.asarray(q_target, dtype=np.float64).reshape(N), self.lo, self.hi)
        return np.clip(self.cap.clamp_step(prev, target, dt), self.lo, self.hi)
