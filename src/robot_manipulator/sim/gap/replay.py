"""Replay a motion through the harness runner: one code path for the sim and the real arm."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_manipulator.core.policy_iface import BasePolicy, Obs
from robot_manipulator.harness.clock import Clock
from robot_manipulator.harness.runner import run_skill
from robot_manipulator.harness.safety_layer import SafetyFilter
from robot_manipulator.harness.skill import Skill, TimeLimit
from robot_manipulator.harness.trajlog import TrajectoryLog
from robot_manipulator.harness.units import CMD_JOINTS
from robot_manipulator.sim.envs.reach import HZ
from robot_manipulator.sim.gap.motions import read_csv

N = len(CMD_JOINTS)


class ReplayPolicy(BasePolicy):
    """Returns motion row k on the k-th call. The runner calls once per tick (chunk length 1)."""

    name = "replay"

    def __init__(self, motion: np.ndarray) -> None:
        super().__init__()
        self.motion = np.asarray(motion, dtype=np.float64).reshape(-1, N)
        self.k = 0

    def reset(self) -> None:
        super().reset()
        self.k = 0

    def _act(self, obs: Obs) -> np.ndarray:
        row = self.motion[min(self.k, len(self.motion) - 1)]
        self.k += 1
        return row[None, :]


@dataclass
class ReplayTrace:
    t: np.ndarray  # (n,) clock time at each tick
    q_target: np.ndarray  # (n, 6) motion row requested
    q_cmd: np.ndarray  # (n, 6) command actually sent, after the safety filter
    q: np.ndarray  # (n, 6) measured joints before each command
    dq: np.ndarray  # (n, 6) measured velocities

    def __len__(self) -> int:
        return len(self.t)


def replay(
    motion: np.ndarray | str | Path,
    backend: Any,
    safety: SafetyFilter,
    *,
    hz: float = HZ,
    clock: Clock | None = None,
) -> ReplayTrace:
    """Run every row of `motion` as one tick each, and return the per-tick trace."""
    if isinstance(motion, (str, Path)):
        motion = read_csv(motion)
    motion = np.asarray(motion, dtype=np.float64).reshape(-1, N)
    skill = Skill(
        "replay",
        ReplayPolicy(motion),
        [TimeLimit((len(motion) - 0.5) / hz)],  # ends after exactly len(motion) ticks
        hz=hz,
    )
    with tempfile.TemporaryDirectory() as tmp:
        log = TrajectoryLog(tmp)
        run_skill(skill, backend, safety, goal={}, clock=clock, log=log)
        from pyarrow import parquet as pq

        table = pq.read_table(Path(tmp) / "traj_replay_1.parquet")
        cols = table.to_pydict()

    def block(prefix: str) -> np.ndarray:
        return np.column_stack(
            [np.asarray(cols[f"{prefix}_{i}"], dtype=np.float64) for i in range(N)]
        )

    return ReplayTrace(
        t=np.asarray(cols["t"], dtype=np.float64),
        q_target=block("q_target"),
        q_cmd=block("q_safe"),
        q=block("q"),
        dq=block("dq"),
    )
