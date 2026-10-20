"""Settle trim: remove the steady-state error the servos leave behind (missing integral action).

The follower's position loops run P/D only (I = 0, ``hw/follower.py``), so a joint stops short of
its command by gravity sag + friction, by up to ~2.5 deg, and by an amount that depends on the
approach direction (backlash; measured 2026-10-20 with ``tip-accuracy``: 6-10 mm at the fingertip).
The encoders read the true joint angle, so an outer loop can fix it: hold the target and add
``gain * (target - measured)`` to the command every tick, bounded to ``max_trim`` so it can never
wind up into a contact, and frozen per joint inside ``deadband`` so it does not hunt on encoder
quantisation (0.088 deg).
"""

from __future__ import annotations

import numpy as np

from robot_manipulator.core.policy_iface import BasePolicy, Obs
from robot_manipulator.harness.units import CMD_JOINTS

N = len(CMD_JOINTS)


class SettleTrim(BasePolicy):
    name = "settle_trim"

    def __init__(
        self,
        target: np.ndarray,
        gain: float = 0.08,
        max_trim: float = np.radians(4.0),
        deadband: float = np.radians(0.15),
        joints: tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> None:
        super().__init__()
        self.target = np.asarray(target, dtype=np.float64).reshape(N)
        self.gain, self.max_trim, self.deadband = gain, max_trim, deadband
        self.joints = list(joints)
        self.trim = np.zeros(N)

    def reset(self) -> None:
        super().reset()
        self.trim = np.zeros(N)

    def _act(self, obs: Obs) -> np.ndarray:
        err = self.target - np.asarray(obs["q"], dtype=np.float64).reshape(N)
        active = np.zeros(N, dtype=bool)
        active[self.joints] = np.abs(err[self.joints]) > self.deadband
        self.trim[active] += self.gain * err[active]
        self.trim = np.clip(self.trim, -self.max_trim, self.max_trim)
        return (self.target + self.trim)[None, :]
