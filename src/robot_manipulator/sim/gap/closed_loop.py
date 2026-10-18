"""Closed-loop twin check: score harness reach runs (sim or real) the same way.

An actuator fit can predict open-loop replay well and still be wrong closed-loop (2026-10-18:
v2 fit, held-out steps 6.6 mrad, but its policy hunted on the real arm). The test that matters
is whether a fixed policy behaves the same on the twin as on the arm. Run the same policies,
targets and flags (``harness reach --settle``) on both and compare:

- final accuracy: success within 2 cm, median and p90 final distance;
- hunting: the fraction of episodes in which the ``Oscillation`` guard would trip (replayed
  offline over the logged q/dq), and command direction reversals per second in the second half.

    uv run gap closed-loop runs/harness/<real_run> runs/harness/<sim_run> ...
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SETTLED_AFTER_S = 2.0  # reversals are counted after this, once the reach itself is over
MIN_STEP = 1e-4  # rad; smaller command changes don't count as a direction


def _episode(path: Path) -> dict:
    import pyarrow.parquet as pq

    from robot_manipulator.harness.skill import Oscillation

    t = pq.read_table(path).to_pandas()
    q = t[[f"q_{i}" for i in range(6)]].to_numpy()
    dq = t[[f"dq_{i}" for i in range(6)]].to_numpy()
    cmd = t[[f"q_safe_{i}" for i in range(5)]].to_numpy()
    time = t["t"].to_numpy() - t["t"].iloc[0]
    guard, tripped_at = Oscillation(), None
    for k in range(len(t)):
        if guard.check({"q": q[k], "dq": dq[k]}, SimpleNamespace(t=time[k])) == "failure":
            tripped_at = float(time[k])
            break
    late = time[1:] >= SETTLED_AFTER_S
    step = np.diff(cmd, axis=0)[late]
    sign = np.where(np.abs(step) < MIN_STEP, 0.0, np.sign(step))
    flips = 0
    for j in range(sign.shape[1]):
        s = sign[:, j][sign[:, j] != 0]
        flips += int(np.sum(s[1:] != s[:-1]))
    span = max(float(time[-1] - SETTLED_AFTER_S), 1e-9)
    return {"tripped_at_s": tripped_at, "cmd_flips_per_s": flips / span}


def score_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    eps = [_episode(p) for p in sorted((run_dir / "traj").glob("traj_reach_*.parquet"))]
    return {
        "run": run_dir.name,
        "backend": summary.get("backend"),
        "policy": Path(summary.get("policy", "")).parent.name,
        "actuator_fit": summary.get("actuator_fit"),
        "n": len(eps),
        "success_rate": summary.get("success_rate"),
        "median_cm": 100 * summary["median_final_dist_m"],
        "p90_cm": 100 * summary["p90_final_dist_m"],
        "guard_trip_rate": float(np.mean([e["tripped_at_s"] is not None for e in eps]))
        if eps
        else None,
        "cmd_flips_per_s": float(np.median([e["cmd_flips_per_s"] for e in eps])) if eps else None,
    }
