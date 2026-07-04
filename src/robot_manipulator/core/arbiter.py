"""Arbiter state machine over policy plugins (PLAN item 2).

Fallback rule: health degraded -> FROZEN (hold, send nothing) -> RECOVERY
(classical recovery primitive) -> retry the primary, or escalate to TELEOP
after `max_retries`. Every transition is logged with a timestamp and reason.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from robot_manipulator.core.policy_iface import Health, HealthLimits, Obs, Policy

log = logging.getLogger(__name__)


class ArbiterState(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    RECOVERY = "recovery"
    TELEOP = "teleop"


@dataclass
class Transition:
    t: float
    src: ArbiterState
    dst: ArbiterState
    reason: str


@dataclass
class Arbiter:
    primary: Policy
    recovery: Policy
    teleop: Policy | None = None
    limits: HealthLimits = field(default_factory=HealthLimits)
    max_retries: int = 2
    recovery_steps: int = 10
    clock: object = time.monotonic

    state: ArbiterState = ArbiterState.ACTIVE
    retries: int = 0
    transitions: list[Transition] = field(default_factory=list)
    _recovery_left: int = 0

    def step(self, obs: Obs) -> np.ndarray | None:
        """One arbiter tick. Returns an action chunk, or None while frozen."""
        if self.state is ArbiterState.ACTIVE:
            action = self.primary.act(obs)
            health = self.primary.health()
            if health.degraded(self.limits):
                self._to(ArbiterState.FROZEN, f"primary degraded: {self._why(health)}")
                return None
            return action

        if self.state is ArbiterState.FROZEN:
            self._to(ArbiterState.RECOVERY, "starting classical recovery primitive")
            self.recovery.reset()
            self._recovery_left = self.recovery_steps
            return self.recovery.act(obs)

        if self.state is ArbiterState.RECOVERY:
            self._recovery_left -= 1
            if self._recovery_left <= 0:
                self.retries += 1
                if self.retries > self.max_retries:
                    self._to(ArbiterState.TELEOP, f"retries exhausted ({self.max_retries})")
                else:
                    self.primary.reset()
                    self._to(ArbiterState.ACTIVE, f"recovery done, retry {self.retries}")
                return None
            return self.recovery.act(obs)

        # TELEOP: terminal until a human resets the arbiter.
        return self.teleop.act(obs) if self.teleop is not None else None

    def reset(self) -> None:
        self.retries = 0
        self.primary.reset()
        self._to(ArbiterState.ACTIVE, "manual reset")

    def _to(self, dst: ArbiterState, reason: str) -> None:
        tr = Transition(self.clock(), self.state, dst, reason)
        self.transitions.append(tr)
        log.info("arbiter: %s -> %s (%s)", tr.src.value, dst.value, reason)
        self.state = dst

    def _why(self, h: Health) -> str:
        return (
            f"latency={h.latency_ms:.1f}ms staleness={h.staleness_s:.3f}s "
            f"confidence={h.confidence:.2f}"
        )
