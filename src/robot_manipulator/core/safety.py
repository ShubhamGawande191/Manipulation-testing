"""Safety envelope (PLAN item 10).

Heartbeat watchdog (freeze if the command stream goes stale), per-arm joint
velocity caps, and per-arm workspace boxes. Keyboard-zone region mutexes are
reserved for Phase 4 (multi-arm) and intentionally absent here.

Servo current caps live in the Feetech registers and are configured at the
driver layer, not here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class Watchdog:
    """Freeze-trigger when the command stream is stale for > timeout_s."""

    def __init__(self, timeout_s: float = 0.1, clock: Callable[[], float] = time.monotonic):
        self.timeout_s = timeout_s
        self._clock = clock
        self._last_beat = clock()

    def beat(self) -> None:
        self._last_beat = self._clock()

    @property
    def age_s(self) -> float:
        return self._clock() - self._last_beat

    @property
    def stale(self) -> bool:
        return self.age_s > self.timeout_s


@dataclass
class WorkspaceBox:
    """Axis-aligned cartesian box the end-effector must stay inside (meters)."""

    lo: np.ndarray
    hi: np.ndarray

    def contains(self, p: np.ndarray) -> bool:
        p = np.asarray(p, dtype=np.float64)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=np.float64), self.lo, self.hi)


@dataclass
class VelocityCap:
    """Per-joint velocity limit (units/s, same units as the joint commands)."""

    max_speed: np.ndarray

    def clamp_step(self, current: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """Limit target so no joint moves faster than max_speed over dt."""
        current = np.asarray(current, dtype=np.float64)
        step = np.asarray(target, dtype=np.float64) - current
        limit = np.asarray(self.max_speed, dtype=np.float64) * dt
        return current + np.clip(step, -limit, limit)
