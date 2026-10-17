"""Fake raw-unit joint bus for the auto-calibration tests. Simulation only, never hardware.

Each joint has a true encoder offset, mechanical stops in link counts, and a bounded following
speed. When a joint is pushed past a stop, its load rises. Position reads carry Gaussian
noise, clipped to +-2 counts. `read_raw` returns the position before homing (0..4095), as the real
bus will.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COUNTS = 4096


@dataclass
class FakeJoint:
    offset: int  # raw count of link position 0: raw = link + offset (mod 4096)
    lo_link: float  # lower mechanical stop, link counts
    hi_link: float  # upper mechanical stop, link counts
    start_link: float = 0.0
    max_speed: float = 12.0  # counts per tick the joint can follow
    load_gain: float = 10.0  # load counts per count the goal is pushed past a stop

    def __post_init__(self) -> None:
        self.pos = float(self.start_link)
        self.goal = self.pos
        self.load = 0

    def raw_stop(self, which: str) -> int:
        link = self.lo_link if which == "lo" else self.hi_link
        return int(round(link + self.offset)) % COUNTS


class FakeJointBus:
    def __init__(self, joints: dict[str, FakeJoint], seed: int = 0, noise: float = 1.0) -> None:
        self.joints = joints
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.touched: set[str] = set()
        self.torque_limits: dict[str, int] = {}
        self.enabled: set[str] = set()
        self.ticks = 0

    def _j(self, joint: str) -> FakeJoint:
        self.touched.add(joint)
        return self.joints[joint]

    def read_raw(self, joint: str) -> int:
        j = self._j(joint)
        n = float(np.clip(self.rng.normal(0.0, self.noise), -2.0, 2.0))
        return int(round(j.pos + j.offset + n)) % COUNTS

    def set_goal_raw(self, joint: str, raw: int) -> None:
        """Set the goal to the nearest raw position to the present one, then advance one tick."""
        j = self._j(joint)
        present_raw = int(round(j.pos + j.offset)) % COUNTS
        delta = ((int(raw) - present_raw + COUNTS // 2) % COUNTS) - COUNTS // 2
        j.goal = j.pos + delta
        self.ticks += 1
        step = max(-j.max_speed, min(j.max_speed, j.goal - j.pos))
        new = j.pos + step
        beyond = 0.0
        if new >= j.hi_link:
            new = j.hi_link
            beyond = max(0.0, j.goal - j.hi_link)
        elif new <= j.lo_link:
            new = j.lo_link
            beyond = min(0.0, j.goal - j.lo_link)
        j.pos = new
        load_noise = float(self.rng.normal(0.0, 2.0))
        j.load = int(np.clip(round(j.load_gain * beyond + load_noise), -1000, 1000))

    def read_load(self, joint: str) -> int:
        return self._j(joint).load

    def set_torque_limit(self, joint: str, permille: int) -> None:
        self._j(joint)
        self.torque_limits[joint] = permille

    def enable(self, joint: str) -> None:
        self._j(joint)
        self.enabled.add(joint)

    def disable(self, joint: str) -> None:
        self._j(joint)
        self.enabled.discard(joint)
