"""Skill runtime: runs one skill at its control rate, or a sequence of skills (a task graph).

Each control tick:
    read the backend -> check end conditions on the measured state -> if the action
    chunk is used up, query the policy -> safety-filter the next row -> command -> beat
    the watchdog -> log.

Redo (`max_redo`): on failure, reset the policy and end conditions, run `on_start`
again, and start a new attempt. Ticks are scheduled at `t0 + k / hz`, so timing does not
drift. Pass the same clock to `SafetyFilter` and `run_skill`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_manipulator.core.policy_iface import Obs
from robot_manipulator.exec.scheduler import WallClock
from robot_manipulator.harness.backend import RobotBackend
from robot_manipulator.harness.clock import Clock
from robot_manipulator.harness.safety_layer import SafetyFilter
from robot_manipulator.harness.skill import EndCondition, RunContext, Skill, SkillResult, Verdict
from robot_manipulator.harness.trajlog import TrajectoryLog
from robot_manipulator.harness.units import CMD_JOINTS

N = len(CMD_JOINTS)


def _as_chunk(action: Any) -> np.ndarray:
    chunk = np.atleast_2d(np.asarray(action, dtype=np.float64))
    if chunk.ndim != 2 or chunk.shape[1] != N or chunk.shape[0] == 0:
        raise ValueError(f"policy must return a non-empty (k, {N}) chunk, got {chunk.shape}")
    return chunk


def _check(conditions: list[EndCondition], obs: Obs, ctx: RunContext) -> tuple[Verdict | None, str]:
    for cond in conditions:
        verdict = cond.check(obs, ctx)
        if verdict is not None:
            return verdict, cond.name
    return None, ""


def _attempt(
    skill: Skill,
    backend: RobotBackend,
    safety: SafetyFilter,
    goal: dict[str, Any],
    clock: Clock,
    log: TrajectoryLog | None,
) -> tuple[Verdict, str, int]:
    dt = 1.0 / skill.hz
    skill.policy.reset()
    for cond in skill.end_conditions:
        cond.reset()
    if skill.on_start is not None:
        skill.on_start(backend)
    safety.beat()

    t0 = clock.now()
    q_prev = np.asarray(backend.read()["q"], dtype=np.float64).copy()
    chunk: np.ndarray | None = None
    idx = 0
    n = 0
    k = 0
    if log is not None:
        log.begin(skill.name)
    try:
        while True:
            clock.sleep_until(t0 + k * dt)
            obs = backend.read()
            obs["t"] = clock.now()
            obs["goal"] = goal
            ctx = RunContext(
                goal=goal, step=k, elapsed=clock.now() - t0, t=clock.now(), q_cmd=q_prev.copy()
            )
            verdict, reason = _check(skill.end_conditions, obs, ctx)
            if verdict is not None:
                return verdict, reason, n

            if chunk is None or idx >= len(chunk):
                chunk = _as_chunk(skill.policy.act(obs))
                idx = 0
            target = chunk[idx]
            q_safe = safety.filter(q_prev, target, dt)
            backend.command(q_safe)
            safety.beat()
            if log is not None:
                log.append(
                    {
                        "t": clock.now(),
                        "q": obs["q"],
                        "dq": obs.get("dq", np.zeros(N)),
                        "q_target": target,
                        "q_safe": q_safe,
                        "ee": obs.get("ee", obs.get("ee_true")),
                        "extra": {"chunk_idx": idx, "chunk_len": len(chunk)},
                    }
                )
            idx += 1
            q_prev = q_safe
            n += 1
            k += 1
    finally:
        if log is not None:
            log.end()


def run_skill(
    skill: Skill,
    backend: RobotBackend,
    safety: SafetyFilter,
    goal: dict[str, Any],
    clock: Clock | None = None,
    log: TrajectoryLog | None = None,
) -> SkillResult:
    clock = clock or WallClock()
    t_start = clock.now()
    total_steps = 0
    redo = 0
    while True:
        verdict, reason, n = _attempt(skill, backend, safety, goal, clock, log)
        total_steps += n
        if verdict == "success" or redo >= skill.max_redo:
            return SkillResult(
                status=verdict,
                n_steps=total_steps,
                duration_s=clock.now() - t_start,
                redo_count=redo,
                reason=reason,
            )
        redo += 1


def run_sequence(
    steps: list[tuple[Skill, dict[str, Any]]],
    backend: RobotBackend,
    safety: SafetyFilter,
    clock: Clock | None = None,
    log: TrajectoryLog | None = None,
) -> list[SkillResult]:
    """Run skills in order, stopping at the first one that does not succeed."""
    results: list[SkillResult] = []
    for skill, goal in steps:
        result = run_skill(skill, backend, safety, goal, clock=clock, log=log)
        results.append(result)
        if result.status != "success":
            break
    return results
