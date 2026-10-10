"""Scripted skill policy: velocity-capped glide toward a fixed joint pose (the `home` skill)."""

from __future__ import annotations

import numpy as np

from robot_manipulator.core.policy_iface import BasePolicy, Obs
from robot_manipulator.core.safety import VelocityCap
from robot_manipulator.harness.units import CMD_JOINTS

N = len(CMD_JOINTS)


class GlideTo(BasePolicy):
    """Each tick, step the command toward `target` by at most `max_vel * dt` per joint.

    The command starts from the measured pose on the first call after `reset()`. The output is
    always one row (1, 6) in CMD order.
    """

    name = "glide"

    def __init__(self, target: np.ndarray, max_vel: float, hz: float) -> None:
        super().__init__()
        self.target = np.asarray(target, dtype=np.float64).reshape(N)
        self.dt = 1.0 / hz
        self.cap = VelocityCap(np.full(N, float(max_vel)))
        self.cmd: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.cmd = None

    def _act(self, obs: Obs) -> np.ndarray:
        if self.cmd is None:
            self.cmd = np.asarray(obs["q"], dtype=np.float64).reshape(N).copy()
        self.cmd = self.cap.clamp_step(self.cmd, self.target, self.dt)
        return self.cmd[None, :]
