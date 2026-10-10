"""Compare two replay traces (sim vs real) joint by joint.

Metrics per arm joint:
- rmse_rad:        RMSE of measured q between the traces
- lag_ms:          delay of the second trace relative to the first (+ = second lags), from the
                   cross-correlation peak within +-max_lag_ms
- overshoot_rad:   largest signed excess of q past q_cmd, in the direction of commanded motion
- ss_err_rad:      mean |q - q_cmd| over the final second (steady-state error)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_manipulator.sim.gap.replay import ReplayTrace
from robot_manipulator.sim.model import JOINTS

N_ARM = 5


def _lag_samples(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Shift k that best aligns b(t) with a(t - k). Positive k: b lags a."""
    a = a - a.mean()
    b = b - b.mean()
    best_k, best = 0, -np.inf
    n = len(a)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            score = float(np.dot(a[: n - k], b[k:]))
        else:
            score = float(np.dot(a[-k:], b[: n + k]))
        if score > best:
            best, best_k = score, k
    return best_k


def _overshoot(q: np.ndarray, q_cmd: np.ndarray) -> float:
    direction = np.sign(np.gradient(q_cmd))
    excess = (q - q_cmd) * direction
    return float(max(0.0, np.max(excess)))


def compare(
    first: ReplayTrace,
    second: ReplayTrace,
    hz: float = 50.0,
    max_lag_ms: float = 200.0,
) -> dict[str, dict[str, float]]:
    """Per-joint metrics for arm joints. `first` is usually the sim and `second` the real arm."""
    n = min(len(first), len(second))
    max_lag = int(round(max_lag_ms / 1000.0 * hz))
    tail = max(1, int(round(hz)))  # final second for steady state
    report: dict[str, dict[str, float]] = {}
    for j in range(N_ARM):
        a = first.q[:n, j]
        b = second.q[:n, j]
        cmd = first.q_cmd[:n, j]
        k = _lag_samples(a, b, max_lag)
        report[JOINTS[j]] = {
            "rmse_rad": float(np.sqrt(np.mean((a - b) ** 2))),
            "lag_ms": 1000.0 * k / hz,
            "overshoot_first_rad": _overshoot(a, cmd),
            "overshoot_second_rad": _overshoot(b, second.q_cmd[:n, j]),
            "ss_err_first_rad": float(np.mean(np.abs(a[-tail:] - cmd[-tail:]))),
            "ss_err_second_rad": float(np.mean(np.abs(b[-tail:] - second.q_cmd[-tail:, j]))),
        }
    return report


def plot_overlay(
    first: ReplayTrace,
    second: ReplayTrace,
    out_path: str | Path,
    title: str = "sim vs real",
    labels: tuple[str, str] = ("sim", "real"),
) -> Path:
    """Overlay the two measured traces against the command for each arm joint (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(first), len(second))
    t = first.t[:n] - first.t[0]
    fig, axes = plt.subplots(N_ARM, 1, figsize=(9, 2.0 * N_ARM), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, first.q_cmd[:n, j], color="0.6", lw=1.0, label="command")
        ax.plot(t, first.q[:n, j], lw=1.2, label=labels[0])
        ax.plot(t, second.q[:n, j], lw=1.2, ls="--", label=labels[1])
        ax.set_ylabel(JOINTS[j], fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
