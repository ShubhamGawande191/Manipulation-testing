"""Per-tick trajectory log for skill attempts, written as one parquet file per attempt.

Files are named `traj_<skill>_<n>.parquet` under `run_dir`, where `n` counts attempts of
that skill within this log. Columns: t, skill, step, q_0..5, dq_0..5, q_target_0..5,
q_safe_0..5, ee_0..2 (null when the backend has no end-effector pose), extra (JSON).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from robot_manipulator.harness.units import CMD_JOINTS

N = len(CMD_JOINTS)


class TrajectoryLog:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self._attempts: dict[str, int] = defaultdict(int)
        self._skill: str | None = None
        self._rows: list[dict[str, Any]] = []

    def begin(self, skill: str) -> None:
        if self._skill is not None:
            raise RuntimeError("begin() called before end() of the previous attempt")
        self._skill = skill
        self._rows = []

    def append(self, row: dict[str, Any]) -> None:
        if self._skill is None:
            raise RuntimeError("append() outside begin()/end()")
        self._rows.append(row)

    def end(self) -> Path | None:
        """Write the current attempt and return its path (None if it logged no ticks)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._skill is None:
            raise RuntimeError("end() without begin()")
        skill, rows = self._skill, self._rows
        self._skill, self._rows = None, []
        if not rows:
            return None

        self._attempts[skill] += 1
        path = self.run_dir / f"traj_{skill}_{self._attempts[skill]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([_flatten(i, r, skill) for i, r in enumerate(rows)])
        pq.write_table(table, path)
        return path


def _flatten(step: int, row: dict[str, Any], skill: str) -> dict[str, Any]:
    out: dict[str, Any] = {"t": float(row["t"]), "skill": skill, "step": step}
    for prefix in ("q", "dq", "q_target", "q_safe"):
        vec = np.asarray(row[prefix], dtype=np.float64).reshape(N)
        for i in range(N):
            out[f"{prefix}_{i}"] = float(vec[i])
    ee = row.get("ee")
    ee_vec = None if ee is None else np.asarray(ee, dtype=np.float64).reshape(3)
    for i in range(3):
        out[f"ee_{i}"] = None if ee_vec is None else float(ee_vec[i])
    out["extra"] = json.dumps(row.get("extra", {}), default=str, sort_keys=True)
    return out
