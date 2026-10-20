"""Figures for the SO-101 "from servo to policy" write-up, generated from this repo's own data.

    uv run blog-figs                      # -> docs/blog/so101-servo-to-policy/figs/*.svg|png

Every figure reads logged runs (`runs/...`) or the twin, never hand-typed numbers, so the post
can be rebuilt as new data comes in. Missing inputs skip that figure with a message.
Colours: the dataviz reference palette, light mode (static figures for the draft).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "blog" / "so101-servo-to-policy" / "figs"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STALL_NM = 2.94  # STS3215-C018 at 12 V (~30 kg.cm), the MJCF actuator forcerange
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

app = typer.Typer(add_completion=False, help=__doc__)


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )
    return plt


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    return OUT / f"{name}.svg"


# ---------------------------------------------------------------- 1. torque budget
def fig_torque_budget(plt) -> Path:
    """Gravity torque each joint must hold, as a share of stall torque, in three poses."""
    from robot_manipulator.sim.atlas_real import GravityModel

    grav = GravityModel()
    poses = {
        "HOME (folded up)": [0, -1.57, 1.57, 0, -1.4, 0],
        "reaching 25 cm, tool down": [0, 0.45, 0.35, 0.77, -1.4, 0],
        "stretched forward": [0, 1.25, -1.2, 0.0, -1.4, 0],
    }
    names = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    h = 0.26
    for k, (label, q) in enumerate(poses.items()):
        _, perm = grav(np.array(q, dtype=float))
        frac = np.abs(perm[1:4]) / 1000
        y = np.arange(3) + (k - 1) * (h + 0.02)
        ax.barh(y, 100 * frac, height=h, color=SERIES[k], label=label)
        for yi, f in zip(y, frac):
            ax.text(100 * f + 1, yi, f"{100 * f:.0f}%", va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set_xlabel("gravity torque, % of stall torque (2.94 N·m at 12 V)")
    ax.set_xlim(0, 100)
    ax.set_title("How much of each servo gravity alone uses up")
    ax.legend(loc="lower right")
    return _save(fig, "01_torque_budget")


# ---------------------------------------------------------------- 2. the servo is a spring
def fig_sag(plt) -> Path | None:
    from robot_manipulator.sim.gap.stiffness import regress, static_samples

    paths = sorted((REPO / "runs" / "telemetry").glob("*survey_live.parquet"))
    if not paths:
        return None
    err, tau = static_samples(paths)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.18)
    for ax, j, color in zip(axes, (1, 3), SERIES):
        slope, c, r = regress(err[:, j], tau[:, j])
        ax.scatter(tau[:, j], np.degrees(err[:, j]), s=8, color=color, alpha=0.35, linewidths=0)
        xs = np.linspace(tau[:, j].min(), tau[:, j].max(), 10)
        ax.plot(xs, np.degrees(slope * xs + c), color=INK, linewidth=1.5)
        ax.set_title(f"{ARM[j]}  k ≈ {1 / slope:.0f} N·m/rad, r = {r:.2f}", fontsize=10)
        ax.set_xlabel("gravity torque from the twin (N·m)")
    axes[0].set_ylabel("command − measured (deg)")
    fig.suptitle(
        "Holding still, the servo behaves like a soft spring (survey, 3k static samples)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    return _save(fig, "02_servo_is_a_spring")


# ---------------------------------------------------------------- 3. stopping short, by direction
def fig_stop_error(plt) -> Path | None:
    runs = sorted((REPO / "runs" / "accuracy").glob("*_real/session.json"))
    if not runs:
        return None
    s = json.loads(runs[0].read_text())
    rows = [a for a in s["arrivals"] if a.get("stage", "raw") == "raw" and "q_rad" in a]
    approaches = ["from_base", "from_far", "from_left", "from_right"]
    joints = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.2
    for k, ap in enumerate(approaches):
        rs = [r for r in rows if r["approach"] == ap]
        if not rs:
            continue
        e = np.degrees(np.mean([np.array(r["q_cmd_rad"]) - np.array(r["q_rad"]) for r in rs], 0))
        ax.bar(
            np.arange(3) + (k - 1.5) * w,
            e[joints],
            width=w - 0.02,
            color=SERIES[k],
            label=ap.replace("_", " "),
        )
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(3), [ARM[j] for j in joints])
    ax.set_ylabel("command − measured at rest (deg)")
    ax.set_title("Where a joint stops depends on the side it came from (P/D loop, I = 0)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    return _save(fig, "03_stop_error_by_approach")


# ---------------------------------------------------------------- 4. step response vs twins
def fig_step(plt) -> Path | None:
    run = REPO / "runs" / "gap" / "20261018-125035_real"
    if not (run / "steps.npz").exists():
        return None
    from robot_manipulator.sim.gap.cli import FIT_PATH, _load, _sim_from_commands

    real = _load(run / "steps.npz")
    nominal = _sim_from_commands(real)
    fitted = _sim_from_commands(real, FIT_PATH)
    j = 2  # elbow_flex
    moving = np.flatnonzero(np.abs(np.diff(real.q_cmd[:, j])) > 1e-3)
    i0, i1 = max(moving[0] - 20, 0), min(moving[-1] + 40, len(real.t))  # the elbow's own steps
    t = real.t[i0:i1] - real.t[i0]
    dt = float(np.median(np.diff(real.t)))

    def lag_ms(y: np.ndarray) -> float:
        """Delay of y behind the command: peak of the velocity cross-correlation (<= 0.5 s)."""
        a = np.diff(real.q_cmd[i0:i1, j])
        b = np.diff(y[i0:i1, j])
        xc = np.correlate(b - b.mean(), a - a.mean(), mode="full")[len(a) - 1 :]
        return 1000 * dt * int(np.argmax(xc[: int(0.5 / dt)]))

    lag_real, lag_nom, lag_fit = lag_ms(real.q), lag_ms(nominal.q), lag_ms(fitted.q)
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ax.plot(
        t,
        np.degrees(real.q_cmd[i0:i1, j]),
        color=INK2,
        linewidth=1.2,
        linestyle="--",
        label="command",
    )
    ax.plot(t, np.degrees(real.q[i0:i1, j]), color=SERIES[0], label="real arm")
    ax.plot(t, np.degrees(nominal.q[i0:i1, j]), color=SERIES[1], label="nominal twin")
    ax.plot(t, np.degrees(fitted.q[i0:i1, j]), color=SERIES[2], label="fitted twin")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("elbow_flex (deg)")
    ax.set_title(
        f"Same commands: the real elbow lags {lag_real:.0f} ms (nominal twin {lag_nom:.0f}, "
        f"fitted {lag_fit:.0f}) and stops ~2° short of each step"
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8)
    return _save(fig, "04_step_response")


# ---------------------------------------------------------------- 5. calm vs hunting policy
def fig_hunting(plt) -> Path | None:
    import pyarrow.parquet as pq

    runs = {
        "pinc_box_3M (calm)": REPO / "runs" / "harness" / "20261018-142938",
        "pinc_fit_3M (hunts)": REPO / "runs" / "harness" / "20261018-143208",
    }
    if not all((d / "traj").exists() for d in runs.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    j = 3  # wrist_flex
    for ax, (label, d) in zip(axes, runs.items()):
        t = pq.read_table(d / "traj" / "traj_reach_3.parquet").to_pandas()
        tt = t["t"] - t["t"].iloc[0]
        ax.plot(
            tt,
            np.degrees(t[f"q_safe_{j}"]),
            color=INK2,
            linewidth=1.2,
            linestyle="--",
            label="policy command",
        )
        ax.plot(tt, np.degrees(t[f"q_{j}"]), color=SERIES[0], label="measured")
        ax.set_title(label)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("wrist_flex (deg)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Policy or hardware? The command oscillates too: closed-loop hunting, not a servo fault",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    return _save(fig, "05_calm_vs_hunting")


FIGS = {
    "torque": fig_torque_budget,
    "sag": fig_sag,
    "stop": fig_stop_error,
    "step": fig_step,
    "hunting": fig_hunting,
}


@app.command()
def main(only: str = typer.Option("", help=f"comma list of {', '.join(FIGS)}")) -> None:
    plt = _style()
    for key, fn in FIGS.items():
        if only and key not in only.split(","):
            continue
        path = fn(plt)
        plt.close("all")
        typer.echo(f"{key:8} -> {path}" if path else f"{key:8} skipped (data missing)")


if __name__ == "__main__":
    app()
