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
    q_cmd: np.ndarray | None = None  # the last command sent (rad, CMD order); None before any


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
    """Success when every joint (or each of `joints`) is within `tol` (rad, max-norm) of
    `target`."""

    name = "joint_reached"

    def __init__(
        self, target: np.ndarray, tol: float, joints: tuple[int, ...] | None = None
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.tol = tol
        self.idx = slice(None) if joints is None else list(joints)

    def reset(self) -> None:
        pass

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        err = np.abs(np.asarray(obs["q"], dtype=np.float64) - self.target)[self.idx]
        return "success" if np.max(err) < self.tol else None


class ContactGuard:
    """Failure when the arm stops following its command: unexpected contact or an obstruction.

    Following error = last command - measured, per joint. It has a steady part (gravity sag,
    servo lag while moving) that depends on the pose and the speed, so the guard learns a
    baseline over ticks ``skip``..``skip + learn`` of the attempt and trips when any of ``joints``
    exceeds it by ``rise`` rad for ``sustain`` consecutive ticks, in the direction of the command
    (the joint is held back). Meant for slow final approaches; added after the fingertip pressed
    a calibration board during the first accuracy run (2026-10-20).
    """

    name = "contact"

    def __init__(
        self,
        rise: float = np.radians(1.5),
        joints: tuple[int, ...] = (1, 2, 3),
        skip: int = 5,
        learn: int = 10,
        sustain: int = 3,
    ) -> None:
        self.rise, self.joints = rise, list(joints)
        self.skip, self.learn, self.sustain = skip, learn, sustain
        self.reset()

    def reset(self) -> None:
        self._samples: list[np.ndarray] = []
        self._base: np.ndarray | None = None
        self._count = 0

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        if ctx.q_cmd is None or ctx.step < self.skip:
            return None
        err = (np.asarray(ctx.q_cmd) - np.asarray(obs["q"]))[self.joints]
        if self._base is None:
            self._samples.append(err)
            if len(self._samples) >= self.learn:
                self._base = np.median(self._samples, axis=0)
            return None
        # held back = |error| grew beyond the baseline (either sign of motion)
        grew = np.abs(err) - np.abs(self._base) > self.rise
        self._count = self._count + 1 if grew.any() else 0
        return "failure" if self._count >= self.sustain else None


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


class Oscillation:
    """Failure when an arm joint keeps reversing direction without getting anywhere (hunting).

    Over the last ``window_s``, if some arm joint's measured velocity changed sign at least
    ``reversals`` times (velocities below ``dq_min`` ignored, so noise at rest doesn't count) while
    that joint's net displacement stayed below ``travel``, the skill fails with reason
    ``oscillation``. Added after a fitted-actuator policy visibly shook the real arm (2026-10-18).
    Defaults tuned on real traces (2026-10-18): they catch 20/20 shaky episodes (0.7-2.4 s in) and
    raise 0 false alarms on 60 smooth real episodes. At ``reversals=6`` the original policy's
    normal settling corrections already trip it, so the margin is narrow; re-check on new data.
    """

    name = "oscillation"

    def __init__(
        self,
        window_s: float = 1.0,
        reversals: int = 8,
        dq_min: float = 0.05,
        travel: float = 0.05,
        n_joints: int = 5,
    ) -> None:
        self.window_s, self.reversals = window_s, reversals
        self.dq_min, self.travel, self.n = dq_min, travel, n_joints
        self._hist: deque[tuple[float, np.ndarray, np.ndarray]] = deque()

    def reset(self) -> None:
        self._hist.clear()

    def check(self, obs: Obs, ctx: RunContext) -> Verdict | None:
        q = np.asarray(obs["q"], dtype=np.float64)[: self.n].copy()
        dq = np.asarray(obs["dq"], dtype=np.float64)[: self.n]
        sign = np.where(np.abs(dq) < self.dq_min, 0.0, np.sign(dq))
        self._hist.append((ctx.t, q, sign))
        while self._hist and self._hist[0][0] < ctx.t - self.window_s:
            self._hist.popleft()
        if len(self._hist) < 3:
            return None
        signs = np.array([h[2] for h in self._hist])
        flips = np.zeros(self.n, dtype=int)
        for j in range(self.n):
            nz = signs[:, j][signs[:, j] != 0]
            flips[j] = int(np.count_nonzero(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
        travel = np.abs(self._hist[-1][1] - self._hist[0][1])
        if np.any((flips >= self.reversals) & (travel < self.travel)):
            return "failure"
        return None
