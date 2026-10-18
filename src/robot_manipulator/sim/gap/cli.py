"""Actuator real2sim: replay motions on sim / the real arm, compare, fit (`uv run gap ...`).

    uv run gap run --backend real --live --max-vel 1.0          # operator present; all motions
    uv run gap run --backend real --live --motion steps         # one motion (config/gap_motions)
    uv run gap compare runs/gap/<real_run>                       # sim fed the real sent commands
    uv run gap fit runs/gap/<real_run>                           # -> config/actuator_fit.yaml
    uv run gap stiffness                                         # static k from survey sag
    uv run gap fit runs/gap/<real_run> --kp-from-sag --out <yaml> # kp pinned, rest fitted

Both backends go through the same SafetyFilter, so sim and real see identical commands
even where the velocity cap clips a motion. ``compare`` and ``fit`` take the *sent* commands of
the real run as input, which isolates the actuator response. Each motion starts with a glide to
its first pose. A glide that stops short by gravity sag (<= 0.15 rad, load <= 60 %) is accepted
as in the survey; anything worse aborts the run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer

from robot_manipulator.sim.gap.replay import ReplayTrace, replay

REPO = Path(__file__).resolve().parents[4]
MOTIONS_DIR = REPO / "config" / "gap_motions"
RUNS_DIR = REPO / "runs" / "gap"
FIT_PATH = REPO / "config" / "actuator_fit.yaml"
HZ = 50.0
MARGIN = 0.05

app = typer.Typer(add_completion=False, help=__doc__)


def _save(trace: ReplayTrace, path: Path) -> None:
    np.savez(path, t=trace.t, q_target=trace.q_target, q_cmd=trace.q_cmd, q=trace.q, dq=trace.dq)


def _load(path: Path) -> ReplayTrace:
    d = np.load(path)
    return ReplayTrace(t=d["t"], q_target=d["q_target"], q_cmd=d["q_cmd"], q=d["q"], dq=d["dq"])


def _motions(name: str) -> list[Path]:
    if name == "all":
        return sorted(MOTIONS_DIR.glob("*.csv"))
    p = MOTIONS_DIR / f"{name}.csv"
    if not p.exists():
        raise typer.BadParameter(f"{p} not found; available: {[m.stem for m in _motions('all')]}")
    return [p]


@app.command()
def run(
    backend: str = typer.Option("sim", help="sim | real"),
    motion: str = typer.Option("all", help="motion name in config/gap_motions, or 'all'"),
    live: bool = typer.Option(False, "--live/--dry-run", help="real: send commands"),
    max_vel: float = typer.Option(1.0, help="rad/s cap per joint (same for sim and real)"),
    port: str = typer.Option("/dev/so101_follower"),
    joint_map: Path = typer.Option(Path("config/sim2real.yaml")),
    stow_leader: bool = typer.Option(
        True, "--stow-leader/--no-stow-leader", help="real live: rotate the leader out of the way"
    ),
) -> None:
    """Replay motions and save one trace per motion under runs/gap/<ts>_<backend>/."""
    from robot_manipulator.harness.cli import leader_clear

    with leader_clear(backend, live, stow_leader):
        _run(backend, motion, live, max_vel, port, joint_map)


def _run(backend: str, motion: str, live: bool, max_vel: float, port: str, joint_map: Path) -> None:
    from robot_manipulator.exec.scheduler import SimClock, WallClock
    from robot_manipulator.harness.backend import LeRobotBackend, SimBackend
    from robot_manipulator.harness.policies.glide import GlideTo
    from robot_manipulator.harness.runner import run_skill
    from robot_manipulator.harness.safety_layer import SafetyFilter
    from robot_manipulator.harness.skill import JointReached, Skill, TimeLimit
    from robot_manipulator.harness.survey import SAG_TOL_RAD
    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.gap.motions import read_csv
    from robot_manipulator.sim.model import JOINTS, joint_ranges, load_model
    from robot_manipulator.sim.randomize import DRConfig

    lo, hi = joint_ranges(load_model(), JOINTS)
    out = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{backend}"
    out.mkdir(parents=True, exist_ok=True)
    if backend == "sim":
        be: Any = SimBackend(hz=HZ, dr=DRConfig(enabled=False))
        clock: Any = SimClock()
    elif backend == "real":
        be = LeRobotBackend(port, JointMap.from_yaml(joint_map), dry_run=not live)
        if be.telemetry is not None:
            be.telemetry.source = "gap_live" if live else "gap_dry"
            be.telemetry.extra_every = 2
        clock = WallClock()
    else:
        raise typer.BadParameter("backend must be sim or real")
    done: list[str] = []
    try:
        for i, path in enumerate(_motions(motion)):
            m = np.clip(read_csv(path), lo + MARGIN, hi - MARGIN)
            if backend == "sim" and i == 0:
                be.reset(m[0])
            safety = SafetyFilter(lo=lo, hi=hi, margin=MARGIN, max_vel=max_vel, clock=clock)
            glide = Skill(
                "gap_glide",
                GlideTo(m[0], max_vel=max_vel, hz=HZ),
                [JointReached(m[0], tol=0.05), TimeLimit(10.0)],
                hz=HZ,
            )
            res = run_skill(glide, be, safety, goal={}, clock=clock)
            err = np.abs(np.asarray(be.read()["q"]) - m[0])[:5].max()
            if res.status != "success" and (backend == "sim" or live) and err > SAG_TOL_RAD:
                typer.echo(f"{path.stem}: start pose not reached (err {err:.3f} rad); stopping")
                break
            trace = replay(m, be, safety, hz=HZ, clock=clock)
            _save(trace, out / f"{path.stem}.npz")
            done.append(path.stem)
            typer.echo(f"{path.stem}: {len(trace)} ticks")
    except KeyboardInterrupt:
        typer.echo("interrupted")
    finally:
        be.close()
    (out / "meta.json").write_text(
        json.dumps(
            {"backend": backend, "live": live, "max_vel": max_vel, "motions": done}, indent=2
        )
        + "\n"
    )
    typer.echo(f"-> {out}")


def _sim_from_commands(real: ReplayTrace, fit: Path | None = None) -> ReplayTrace:
    """Feed the real run's sent commands to the nominal (or fitted) sim, starting at its pose."""
    from robot_manipulator.exec.scheduler import SimClock
    from robot_manipulator.harness.backend import SimBackend
    from robot_manipulator.harness.safety_layer import SafetyFilter
    from robot_manipulator.sim.model import JOINTS, apply_actuator_fit, joint_ranges, load_model
    from robot_manipulator.sim.randomize import DRConfig

    model = load_model()
    q_cmd = np.array(real.q_cmd, dtype=np.float64)
    if fit is not None:
        delays = apply_actuator_fit(model, fit)  # returns {joint: delay_steps}
        for j, name in enumerate(JOINTS):
            k = int(delays.get(name, 0))
            if k > 0:  # command pipeline delay: joint j sees the command from k ticks earlier
                q_cmd[k:, j] = q_cmd[:-k, j].copy()
                q_cmd[:k, j] = q_cmd[0, j]
    lo, hi = joint_ranges(model, JOINTS)
    be = SimBackend(model=model, hz=HZ, dr=DRConfig(enabled=False))
    be.reset(real.q[0])
    clock = SimClock()
    safety = SafetyFilter(lo=lo, hi=hi, margin=0.0, max_vel=1e6, clock=clock)  # already filtered
    trace = replay(q_cmd, be, safety, hz=HZ, clock=clock)
    be.close()
    return trace


@app.command()
def compare(
    real_dir: Path, fitted: bool = typer.Option(False, help="use config/actuator_fit.yaml")
) -> None:
    """Per-joint sim-vs-real metrics + overlay plots for every trace in ``real_dir``."""
    from robot_manipulator.sim.gap.compare import compare as metrics
    from robot_manipulator.sim.gap.compare import plot_overlay

    report: dict[str, Any] = {}
    tag = "fitted" if fitted else "nominal"
    for path in sorted(real_dir.glob("*.npz")):
        real = _load(path)
        sim = _sim_from_commands(real, FIT_PATH if fitted else None)
        report[path.stem] = metrics(sim, real, hz=HZ)
        plot_overlay(sim, real, real_dir / f"{path.stem}_{tag}.png", title=f"{path.stem} ({tag})")
    (real_dir / f"compare_{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(f"{'motion':22} " + " ".join(f"{j[:10]:>11}" for j in report[next(iter(report))]))
    for name, joints in report.items():
        typer.echo(
            f"{name:22} " + " ".join(f"{1000 * v['rmse_rad']:9.1f}mr" for v in joints.values())
        )
    typer.echo(f"(RMSE sim vs real, mrad) -> {real_dir / f'compare_{tag}.json'}")


def _stiffness(telemetry: str):
    from robot_manipulator.sim.gap.stiffness import estimate, static_samples
    from robot_manipulator.sim.model import actuator_ids, load_model

    paths = sorted((REPO / "runs" / "telemetry").glob(telemetry))
    if not paths:
        raise typer.BadParameter(f"no telemetry matching runs/telemetry/{telemetry}")
    m = load_model()
    err, tau = static_samples(paths)
    kp_nom = m.actuator_gainprm[actuator_ids(m)[:5], 0]
    return estimate(err, tau, kp_nom), paths


@app.command()
def stiffness(
    telemetry: str = typer.Option("*survey_live.parquet", help="glob under runs/telemetry"),
) -> None:
    """Static servo stiffness per joint from held poses (sag vs twin gravity torque)."""
    est, paths = _stiffness(telemetry)
    typer.echo(f"{len(paths)} telemetry files, {est[0].n} quasi-static samples")
    for e in est:
        r = "  n/a" if e.corr is None else f"{e.corr:+.2f}"
        typer.echo(
            f"{e.joint:14} k {e.k_nm_per_rad:6.1f} N·m/rad  kp_scale {e.kp_scale:.4f}  "
            f"corr {r}  ({e.source})"
        )


@app.command()
def fit(
    real_dir: Path,
    kp_from_sag: bool = typer.Option(
        False, help="pin kp to the static-sag stiffness (`gap stiffness`), fit the rest"
    ),
    telemetry: str = typer.Option("*survey_live.parquet", help="with --kp-from-sag"),
    out: Path = typer.Option(FIT_PATH, help="output yaml"),
) -> None:
    """Fit per-joint actuator parameters from each joint's chirp (else sine_all) -> yaml."""
    from robot_manipulator.sim.gap.fit import fit_joint, write_fit_yaml
    from robot_manipulator.sim.model import JOINTS

    kp = {e.joint: e.kp_scale for e in _stiffness(telemetry)[0]} if kp_from_sag else {}
    results = []
    for j, name in enumerate(JOINTS[:5]):
        path = real_dir / f"chirp_{name}.npz"
        if not path.exists():
            path = real_dir / "sine_all.npz"
        if not path.exists():
            typer.echo(f"{name}: no chirp_{name} or sine_all trace; skipped")
            continue
        real = _load(path)
        r = fit_joint(j, real.q_cmd, real.q[:, j], hz=HZ, kp_scale=kp.get(name))
        results.append(r)
        typer.echo(f"{name:14} {r.params}  rmse {1000 * r.rmse_rad:.1f} mrad")
    if results:
        src = "sim.gap.fit (least_squares per joint, delay grid 0..6"
        src += ", kp pinned from static sag)" if kp else ")"
        typer.echo(f"-> {write_fit_yaml(results, out, hz=HZ, source=src)}")


@app.command("closed-loop")
def closed_loop(runs: list[Path]) -> None:
    """Score harness reach runs (real and sim twins) on accuracy and hunting, side by side."""
    from robot_manipulator.sim.gap.closed_loop import score_run

    typer.echo(
        f"{'run':18} {'backend':7} {'policy':12} {'fit':22} {'n':>3} {'succ':>5} {'med cm':>7} "
        f"{'p90 cm':>7} {'trips':>6} {'flips/s':>8}"
    )
    for r in map(score_run, runs):
        fit = Path(r["actuator_fit"]).name if r["actuator_fit"] else "-"
        typer.echo(
            f"{r['run']:18} {r['backend'] or '?':7} {r['policy'][:12]:12} {fit[:22]:22} "
            f"{r['n']:3d} {r['success_rate']:5.2f} {r['median_cm']:7.2f} {r['p90_cm']:7.2f} "
            f"{r['guard_trip_rate']:6.2f} {r['cmd_flips_per_s']:8.1f}"
        )


if __name__ == "__main__":
    app()
