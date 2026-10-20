"""IK benchmark in the twin: closed-form vs damped least squares vs a learned seed (+ hybrid).

    uv run ik-bench train-nn            # ~3-5 min CPU -> runs/ik/seed_mlp.pt
    uv run ik-bench run                 # table + runs/ik/bench_<ts>.json

Scenarios (fingertip position + tool pitch; roll is free and held at the natural -80 deg):

- ``random``: reachable targets from random joint vectors (the whole workspace)
- ``keys``: MiniLab-style presses, tool straight down on a grid over the table, 1 cm above it
  (only ~65 % of that grid is reachable straight down: the analytic success is the ceiling)
- ``line``: a 30 cm straight path with the tool down, inside the reachable band, solved point by
  point warm-started from the previous solution (what a Cartesian controller does); measures joint
  jumps between neighbours

Success = fingertip within 1 mm, pitch within 0.5 deg, inside the joint limits. Times are wall
clock per solve on this CPU (Python + MuJoCo FK calls; compare solvers to each other, not to C++).
All of this is in twin coordinates: on the real arm, add the kinematic calibration error (§7 of
the twin runbook) on top.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import typer

from robot_manipulator.calib.camera import REPO
from robot_manipulator.ik.analytic import AnalyticIK
from robot_manipulator.ik.numeric import dls, dls_restarts, solved

OUT = REPO / "runs" / "ik"
SEED_PATH = OUT / "seed_mlp.pt"
ROLL = np.radians(-80.0)  # the arm's natural roll (gripper upright)
TABLE_Z = -0.0275
app = typer.Typer(add_completion=False, help=__doc__)


def scenarios(ik: AnalyticIK, n: int, seed: int = 0) -> dict[str, list[tuple[np.ndarray, float]]]:
    rng = np.random.default_rng(seed)
    rand = []
    while len(rand) < n:
        q = np.r_[rng.uniform(ik.lo[:5] + 0.05, ik.hi[:5] - 0.05), 0.0]
        t = ik.task(q)
        if t[2] > TABLE_Z + 0.01:  # above the table
            rand.append((t[:3], float(t[3])))
    keys = [
        (np.array([x, y, TABLE_Z + 0.01]), np.pi / 2)
        for x in np.linspace(0.10, 0.36, 14)
        for y in np.linspace(-0.22, 0.22, 12)
    ]
    # inside the straight-down band (reachable to ~28-30 cm forward), so it tests continuity
    a, b = np.array([0.12, -0.14, TABLE_Z + 0.03]), np.array([0.27, 0.12, TABLE_Z + 0.03])
    line = [(a + s * (b - a), np.pi / 2) for s in np.linspace(0, 1, 120)]
    return {"random": rand, "keys": keys, "line": line}


def _home(ik: AnalyticIK) -> np.ndarray:
    from robot_manipulator.sim.envs.reach import HOME_ARM

    return np.clip(np.r_[HOME_ARM, 0.0], ik.lo, ik.hi)


def run_solver(name: str, ik: AnalyticIK, targets, warm: bool, rng, nn=None) -> dict:
    home = _home(ik)
    times, ok, iters, sols = [], [], [], []
    prev = home.copy()
    for p, phi in targets:
        seed = prev if warm else home
        t0 = time.perf_counter()
        if name == "analytic":
            q, it = ik.solve(p, phi, ROLL, seed=seed), 0
        elif name == "dls":
            q, it = dls(ik, p, phi, ROLL, seed)
        elif name == "dls+restarts":
            seeds = [seed] + [np.r_[rng.uniform(ik.lo[:4], ik.hi[:4]), ROLL, 0.0] for _ in range(7)]
            q, it, _ = dls_restarts(ik, p, phi, ROLL, seeds)
        elif name == "nn":
            q, it = nn(p, phi, ROLL), 0
        elif name == "nn+dls":
            q, it = dls(ik, p, phi, ROLL, nn(p, phi, ROLL), max_iter=30)
        else:
            raise ValueError(name)
        times.append(time.perf_counter() - t0)
        good = solved(ik, q, p, phi)
        ok.append(good)
        iters.append(it)
        sols.append(q if good else None)
        if good:
            prev = q
    res = {
        "success": float(np.mean(ok)),
        "time_ms_median": 1e3 * float(np.median(times)),
        "time_ms_p95": 1e3 * float(np.percentile(times, 95)),
        "iters_median": float(np.median(iters)),
    }
    if name == "nn":  # how good the raw guess is, solved or not
        err = [np.linalg.norm(ik.task(nn(p, phi, ROLL))[:3] - p) for p, phi in targets]
        res["seed_err_mm_median"] = 1e3 * float(np.median(err))
    if warm:  # largest joint jump between consecutive solved points of the path
        jumps = [
            float(np.abs(a[:4] - b[:4]).max())
            for a, b in zip(sols, sols[1:])
            if a is not None and b is not None
        ]
        res["max_joint_jump_deg"] = float(np.degrees(max(jumps))) if jumps else None
    return res


def reach_grid(ik: AnalyticIK, height: float, pitch_deg: float, xs, ys) -> np.ndarray:
    """Per grid cell: 1 = solvable, 0 = out of reach, -1 = reachable only outside joint limits."""
    out = np.zeros((len(xs), len(ys)), dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p, phi = np.array([x, y, TABLE_Z + height]), np.radians(pitch_deg)
            if ik.branches(p, phi, ROLL):
                out[i, j] = 1 if ik.solve(p, phi, ROLL) is not None else -1
    return out


@app.command("reach-map")
def reach_map(
    height: float = typer.Option(0.01, help="fingertip height above the table, m"),
    pitch: list[float] = typer.Option([90.0, 70.0], help="tool pitch, deg (90 = straight down)"),
) -> None:
    """Where on the table the fingertip can reach at a given pitch (twin, no collision check)."""
    ik = AnalyticIK()
    xs, ys = np.arange(0.04, 0.42, 0.02), np.arange(-0.24, 0.25, 0.04)
    for ph in pitch:
        g = reach_grid(ik, height, ph, xs, ys)
        typer.echo(
            f"\npitch {ph:.0f} deg, {100 * height:.0f} cm above the table "
            "('#' ok, '.' out of reach, 'L' joint limits); rows x forward, columns y (cm)"
        )
        typer.echo("x \\ y " + "".join(f"{100 * y:>4.0f}" for y in ys))
        for x, row in zip(xs, g):
            typer.echo(
                f"{100 * x:5.0f}  " + "".join({1: "   #", 0: "   .", -1: "   L"}[v] for v in row)
            )


@app.command("train-nn")
def train_nn(n: int = 120_000, epochs: int = 60, seed: int = 0) -> None:
    from robot_manipulator.ik.learned import LearnedSeed, make_dataset

    ik = AnalyticIK()
    t0 = time.monotonic()
    X, Y = make_dataset(ik, n, seed)
    typer.echo(f"{n} labelled samples in {time.monotonic() - t0:.0f}s")
    nn = LearnedSeed()
    nn.fit(X, Y, epochs=epochs, log=typer.echo)
    nn.save(SEED_PATH)
    typer.echo(f"-> {SEED_PATH}")


@app.command("run")
def run(n: int = 400, seed: int = 1) -> None:
    ik = AnalyticIK()
    nn = None
    solvers = ["analytic", "dls", "dls+restarts"]
    if SEED_PATH.exists():
        from robot_manipulator.ik.learned import LearnedSeed

        nn = LearnedSeed(SEED_PATH)
        solvers += ["nn", "nn+dls"]
    else:
        typer.echo(f"no {SEED_PATH}: run `uv run ik-bench train-nn` for the learned solvers")
    report: dict = {}
    for scen, targets in scenarios(ik, n, seed).items():
        report[scen] = {}
        for name in solvers:
            rng = np.random.default_rng(seed)
            report[scen][name] = run_solver(name, ik, targets, scen == "line", rng, nn)
    head = f"{'scenario':8} {'solver':14} {'success':>8} {'ms med':>7} {'ms p95':>7} {'iters':>6}"
    typer.echo(head + f" {'extra':>24}")
    for scen, rows in report.items():
        for name, r in rows.items():
            extra = ""
            if "seed_err_mm_median" in r:
                extra = f"seed err {r['seed_err_mm_median']:.1f} mm"
            if r.get("max_joint_jump_deg") is not None:
                extra += f" max jump {r['max_joint_jump_deg']:.1f}°"
            typer.echo(
                f"{scen:8} {name:14} {r['success']:8.3f} {r['time_ms_median']:7.2f} "
                f"{r['time_ms_p95']:7.2f} {r['iters_median']:6.0f} {extra:>24}"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"bench_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(f"-> {path}")


if __name__ == "__main__":
    app()
