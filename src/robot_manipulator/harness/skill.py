"""Skill = policy + end conditions + redo + execution rate.

End conditions are checked on the measured state at the start of each control tick,
before the next command is sent. The first condition to return a verdict wins.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from robot_manipulator.core.policy_iface import Obs, Policy
from robot_manipulator.harness.backend import RobotBackend

Verdict = Literal["success", "failure"]


@dataclass
class RunContext:
    goal: dict[str, Any]
    step: int  # control tick index within the attempt
    elapsed: float  # seconds since the attempt started
    t: float  # clock time


class EndCondition(Protocol):
    name: str

    def reset(self) -> None: ...

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None: ...


class TimeLimit:
    name = "time_limit"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def reset(self) -> None:
        pass

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        return "failure" if ctx.elapsed >= self.seconds else None


class JointReached:
    """Success when every joint is within `tol` (rad, max-norm) of `target`."""

    name = "joint_reached"

    def __init__(self, target: np.ndarray, tol: float) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.tol = tol

    def reset(self) -> None:
        pass

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        err = np.max(np.abs(np.asarray(obs["q"], dtype=np.float64) - self.target))
        return "success" if err < self.tol else None


class EEReached:
    """Success when `ee_fn(obs)` is within `tol` (m) of `ctx.goal["target"]`."""

    name = "ee_reached"

    def __init__(self, tol: float, ee_fn: Callable[[Obs], np.ndarray]) -> None:
        self.tol = tol
        self.ee_fn = ee_fn

    def reset(self) -> None:
        pass

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        target = np.asarray(ctx.goal["target"], dtype=np.float64)
        dist = np.linalg.norm(np.asarray(self.ee_fn(obs), dtype=np.float64) - target)
        return "success" if dist < self.tol else None


class Stalled:
    """Failure when the joints moved less than `eps` (max-norm) over the last `window_s`."""

    name = "stalled"

    def __init__(self, window_s: float, eps: float) -> None:
        self.window_s = window_s
        self.eps = eps
        self._hist: deque[tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._hist.clear()

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        q = np.asarray(obs["q"], dtype=np.float64).copy()
        self._hist.append((ctx.t, q))
        cutoff = ctx.t - self.window_s
        # keep one sample at or before the window start so the window is fully covered
        while len(self._hist) > 1 and self._hist[1][0] <= cutoff:
            self._hist.popleft()
        t_old, q_old = self._hist[0]
        if t_old > cutoff:
            return None  # not enough history yet
        return "failure" if np.max(np.abs(q - q_old)) < self.eps else None


@dataclass
class Skill:
    name: str
    policy: Policy
    end_conditions: list[EndCondition]
    hz: float = 50.0
    max_redo: int = 0
    on_start: Callable[[RobotBackend], None] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.end_conditions:
            raise ValueError(f"skill {self.name!r} needs at least one end condition")
        if self.hz <= 0:
            raise ValueError("hz must be positive")


@dataclass
class SkillResult:
    status: Verdict
    n_steps: int
    duration_s: float
    redo_count: int
    reason: str
