"""Deployment harness CLI.

    harness reach --backend sim --policy runs/ppo_reach/<run>/reach.onnx --random 100 --dr

Each episode is `home` (scripted, velocity-capped glide to HOME) followed by `reach` (the ONNX
policy until the end effector is within 2 cm of the target or 4 s pass). Both run through the
same runner, safety filter, and trajectory log as the real arm. Success is scored on the
simulator's true end-effector position, matching the training env.

`--backend real` drives the LeRobot follower. It is refused unless `config/sim2real.yaml` has
`calibrated: true` and `--live` is given. Without `--live` it is a dry run, which only reads
the arm and records the commands it would send.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer

from robot_manipulator.profile import ACTIVE

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

HOME_TOL = 0.05  # rad, per joint; measured q carries sensor bias under DR
SAFETY_MARGIN = 0.05  # rad, SafetyFilter default margin inside the joint limits
GLIDE_TIMEOUT_S = 5.0
BACKENDS = ("sim", "fake", "real")


def run_reach(
    *,
    backend: str,
    policy_path: Path,
    targets: np.ndarray | None = None,
    n_episodes: int = 1,
    seed: int = 0,
    dr: bool = False,
    viewer: bool = False,
    live: bool = False,
    max_vel: float = 1.0,
    port: str = ACTIVE.port,
    joint_map_path: Path = ACTIVE.joint_map,
    out: Path | None = None,
    settle: bool = False,
    oscillation_guard: bool | None = None,
    actuator_fit: Path | None = None,
) -> dict[str, Any]:
    """Run `n_episodes` home+reach episodes and write episodes.parquet, summary.json and traj/."""
    from robot_manipulator.core.episode_log import EpisodeRecord, write_records
    from robot_manipulator.exec.scheduler import SimClock, WallClock
    from robot_manipulator.harness.backend import FakeBackend, LeRobotBackend, SimBackend
    from robot_manipulator.harness.policies.glide import GlideTo
    from robot_manipulator.harness.policies.onnx_reach import OnnxReachPolicy
    from robot_manipulator.harness.runner import run_sequence
    from robot_manipulator.harness.safety_layer import SafetyFilter
    from robot_manipulator.harness.skill import (
        EEReached,
        JointReached,
        Oscillation,
        Skill,
        TimeLimit,
    )
    from robot_manipulator.harness.trajlog import TrajectoryLog
    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.envs.reach import (
        ACTION_SCALE,
        EP_SECONDS,
        GRIPPER_HOLD,
        HOME_ARM,
        HZ,
        N_ARM,
        RESET_NOISE,
        SUCCESS_TOL,
        TARGET_BOX,
        target_box,
    )
    from robot_manipulator.sim.kinematics import FK
    from robot_manipulator.sim.model import joint_ranges, load_model
    from robot_manipulator.sim.randomize import DRConfig

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == "fake" and dr:
        typer.echo("note: --dr has no effect on the fake backend", err=True)

    model = load_model()
    fk = FK(model)
    lo6, hi6 = joint_ranges(model)
    # HOME must be a pose the SafetyFilter will actually command: GRIPPER_HOLD sits exactly on the
    # PincOpen limit and the filter keeps commands `margin` inside, so an unclipped HOME left a
    # built-in 0.05 rad gripper error = HOME_TOL and every real HOME glide timed out (2026-10-17).
    home6 = np.clip(np.r_[HOME_ARM, GRIPPER_HOLD], lo6 + SAFETY_MARGIN, hi6 - SAFETY_MARGIN)
    rng = np.random.default_rng(seed)
    if targets is None:
        # same distribution as training: joint-space samples whose EE lies in TARGET_BOX
        _, targets = fk.sample_reachable(rng, n_episodes, *target_box(lo6, hi6), box=TARGET_BOX)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)

    run_dir = (
        out
        if out is not None
        else Path("runs") / "harness" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    traj = TrajectoryLog(run_dir / "traj")
    policy = OnnxReachPolicy(policy_path, max_vel=max_vel)

    sim_be = None
    real_be = None
    if backend == "sim":
        sim_be = SimBackend(
            hz=HZ, viewer=viewer, dr=DRConfig(enabled=dr), seed=seed, actuator_fit=actuator_fit
        )
    elif backend == "real":
        joint_map = JointMap.from_yaml(joint_map_path)
        real_be = LeRobotBackend(port, joint_map, dry_run=not live)

    def ee_of(obs: dict) -> np.ndarray:
        if "ee_true" in obs:
            return np.asarray(obs["ee_true"])
        return fk.ee_pos(np.r_[np.asarray(obs["q"])[:N_ARM], GRIPPER_HOLD])

    episodes: list[dict[str, Any]] = []
    records: list[EpisodeRecord] = []
    arm_id = {"sim": "sim_so101", "fake": "fake_so101", "real": "so101_follower"}[backend]
    run_id = run_dir.name
    policy_step = ACTION_SCALE * HZ  # rad/s the policy can command at full action
    trained_caps = policy.meta.get("vel_cap_dr")  # [lo, hi] rad/s seen in training, or absent
    if max_vel < policy_step and not (trained_caps and max_vel >= trained_caps[0] - 1e-9):
        typer.echo(
            f"warning: --max-vel {max_vel} rad/s is below the policy's {policy_step:.2f} rad/s "
            f"step and outside the speed caps it was trained with ({trained_caps or 'none'}); "
            "the policy integrates at the cap, but expect degraded behaviour",
            err=True,
        )
    use_guard = (backend == "real") if oscillation_guard is None else oscillation_guard
    try:
        for i, target in enumerate(targets):
            clock = WallClock() if backend == "real" else SimClock()
            if sim_be is not None:
                start = np.clip(
                    home6 + np.r_[rng.uniform(-RESET_NOISE, RESET_NOISE, N_ARM), 0.0], lo6, hi6
                )
                sim_be.reset(start)
                be: Any = sim_be
            elif backend == "fake":
                start = np.clip(
                    home6 + np.r_[rng.uniform(-RESET_NOISE, RESET_NOISE, N_ARM), 0.0], lo6, hi6
                )
                be = FakeBackend(q0=start, alpha=0.3, clock=clock)
            else:
                be = real_be

            guard = [Oscillation()] if use_guard else []
            safety = SafetyFilter(
                lo=lo6, hi=hi6, margin=SAFETY_MARGIN, max_vel=max_vel, clock=clock
            )
            home = Skill(
                "home",
                GlideTo(home6, max_vel=max_vel, hz=HZ),
                [JointReached(home6, tol=HOME_TOL), TimeLimit(GLIDE_TIMEOUT_S)],
                hz=HZ,
            )
            reach = Skill(
                "reach",
                policy,
                # settle: no early stop, run the full episode and judge the final pose (accuracy)
                # Oscillation guard first (real arm by default): a hunting policy stops within
                # ~1-2 s instead of shaking on. Off in sim by default: the nominal twin is livelier
                # than the real arm and the same calm policy dithers there (2026-10-18).
                guard
                + (
                    [TimeLimit(EP_SECONDS)]
                    if settle
                    else [EEReached(SUCCESS_TOL, ee_fn=ee_of), TimeLimit(EP_SECONDS)]
                ),
                hz=HZ,
            )
            t_ep = time.monotonic()
            results = run_sequence(
                [(home, {}), (reach, {"target": target})], be, safety, clock=clock, log=traj
            )
            wall = time.monotonic() - t_ep
            reach_res = results[-1] if len(results) == 2 else None
            n_steps = reach_res.n_steps if reach_res else 0
            final = be.read()
            final_dist = float(np.linalg.norm(ee_of(final) - target))
            if settle:
                success = bool(reach_res is not None and final_dist < SUCCESS_TOL)
            else:
                success = bool(reach_res and reach_res.status == "success")
            if backend == "real" and not live:
                typer.echo(f"dry run, episode {i}: last command {real_be.last_command}", err=True)

            episodes.append(
                {
                    "episode": i,
                    "target_m": target.tolist(),
                    "success": success,
                    "home_ok": results[0].status == "success",
                    "n_steps": n_steps,
                    "final_dist_m": final_dist,
                    "time_to_success_s": (reach_res.duration_s if success and not settle else None),
                    "reason": reach_res.reason if reach_res else "home_failed",
                }
            )
            records.append(
                EpisodeRecord(
                    run_id=f"{run_id}-{i:03d}",
                    arm_id=arm_id,
                    policy=policy.name,
                    executor="harness",
                    chunk_len=1,
                    precision="fp32",
                    policy_latency_ms=float(policy.health().latency_ms),
                    e2e_latency_ms=1000.0 * wall / max(n_steps, 1),
                    jitter_ms=0.0,
                    success=success,
                    time_to_success_s=reach_res.duration_s if success else None,
                    task="reach",
                    backend=backend,
                    n_steps=n_steps,
                )
            )
            if sim_be is None and backend != "real":
                be.close()

    except KeyboardInterrupt:
        # stop between/within episodes; the finally below still parks the real arm
        typer.echo("interrupted — closing backends (real arm parks before torque off)", err=True)
    finally:
        if real_be is not None:
            real_be.close()
        if sim_be is not None:
            sim_be.close()

    write_records(records, run_dir / "episodes.parquet")
    succ = [e for e in episodes if e["success"]]
    summary = {
        "backend": backend,
        "dr": bool(dr) if backend == "sim" else None,
        "live": bool(live) if backend == "real" else None,
        "policy": str(policy_path),
        "n_episodes": len(episodes),
        "success_rate": len(succ) / len(episodes) if episodes else 0.0,
        "mean_final_dist_m": float(np.mean([e["final_dist_m"] for e in episodes])),
        "mean_time_to_success_s": (
            float(np.mean(times))
            if (
                times := [
                    e["time_to_success_s"] for e in succ if e["time_to_success_s"] is not None
                ]
            )
            else None
        ),
        "median_final_dist_m": float(np.median([e["final_dist_m"] for e in episodes])),
        "p90_final_dist_m": float(np.percentile([e["final_dist_m"] for e in episodes], 90)),
        "settle": bool(settle),
        "oscillation_guard": bool(use_guard),
        "actuator_fit": str(actuator_fit) if actuator_fit else None,
        "out": str(run_dir),
        "episodes": episodes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@app.callback()
def _root() -> None:
    """Deployment harness. Subcommands: `reach`."""


def leader_clear(backend: str, live: bool, stow_leader: bool):
    """Context manager: stow the leader for real live runs (no-op otherwise)."""
    from contextlib import nullcontext

    if backend == "real" and live and stow_leader:
        from robot_manipulator.hw.leader import LeaderStow

        return LeaderStow.from_config()
    return nullcontext()


@app.command("reach")
def reach_cmd(
    backend: str = typer.Option("sim", help="sim | fake | real"),
    policy: Path = typer.Option(..., exists=True, help="reach.onnx; policy_meta.json beside it"),
    target: str | None = typer.Option(None, help="x,y,z in meters, one episode"),
    random: int = typer.Option(0, "--random", help="N random reachable targets (sim or fake)"),
    seed: int = typer.Option(0),
    dr: bool = typer.Option(False, "--dr/--no-dr", help="sim: training-style domain randomization"),
    viewer: bool = typer.Option(False, help="sim: open the MuJoCo passive viewer"),
    live: bool = typer.Option(False, "--live/--dry-run", help="real: send commands to the arm"),
    max_vel: float = typer.Option(1.0, help="rad/s cap per joint"),
    port: str = typer.Option(ACTIVE.port, help="real: serial port"),
    joint_map: Path = typer.Option(ACTIVE.joint_map, help="real: joint map"),
    out: Path | None = typer.Option(None, help="output dir (default runs/harness/<ts>)"),
    settle: bool = typer.Option(
        False, "--settle", help="run the full episode; success/accuracy from the final pose"
    ),
    stow_leader: bool = typer.Option(
        True, "--stow-leader/--no-stow-leader", help="real live: rotate the leader out of the way"
    ),
    oscillation_guard: bool | None = typer.Option(
        None,
        "--oscillation-guard/--no-oscillation-guard",
        help="stop a hunting policy early (default: on for real, off for sim/fake)",
    ),
    actuator_fit: Path | None = typer.Option(
        None, exists=True, help="sim: twin with this actuator fit (e.g. config/actuator_fit.yaml)"
    ),
) -> None:
    targets = None
    n_episodes = 1
    if target is not None:
        parts = [float(x) for x in target.split(",")]
        if len(parts) != 3:
            raise typer.BadParameter("--target needs x,y,z")
        targets = np.array([parts])
    elif random > 0:
        n_episodes = random
    with leader_clear(backend, live, stow_leader):
        summary = _run_reach_cmd(
            backend,
            policy,
            targets,
            n_episodes,
            seed,
            dr,
            viewer,
            live,
            max_vel,
            port,
            joint_map,
            out,
            settle,
            oscillation_guard,
            actuator_fit,
        )
    brief = {k: v for k, v in summary.items() if k != "episodes"}
    typer.echo(json.dumps(brief, indent=2))


def _run_reach_cmd(
    backend,
    policy,
    targets,
    n_episodes,
    seed,
    dr,
    viewer,
    live,
    max_vel,
    port,
    joint_map,
    out,
    settle=False,
    oscillation_guard=None,
    actuator_fit=None,
):
    return run_reach(
        backend=backend,
        policy_path=policy,
        targets=targets,
        n_episodes=n_episodes,
        seed=seed,
        dr=dr,
        viewer=viewer,
        live=live,
        max_vel=max_vel,
        port=port,
        joint_map_path=joint_map,
        out=out,
        settle=settle,
        oscillation_guard=oscillation_guard,
        actuator_fit=actuator_fit,
    )


@app.command("survey")
def survey_cmd(
    backend: str = typer.Option("sim", help="sim | real"),
    n: int = typer.Option(20, help="number of poses to visit"),
    hold: float = typer.Option(1.5, help="seconds to hold still at each pose"),
    max_vel: float = typer.Option(0.5, help="rad/s cap per joint"),
    live: bool = typer.Option(False, "--live/--dry-run", help="real: send commands to the arm"),
    seed: int = typer.Option(0),
    port: str = typer.Option(ACTIVE.port, help="real: serial port"),
    joint_map: Path = typer.Option(ACTIVE.joint_map, help="real: joint map"),
    stow_leader: bool = typer.Option(
        True, "--stow-leader/--no-stow-leader", help="real live: rotate the leader out of the way"
    ),
) -> None:
    """Visit spread-out workspace poses and hold still (telemetry for the workspace atlas)."""
    import json

    from robot_manipulator.harness.survey import run_survey

    with leader_clear(backend, live, stow_leader):
        summary = run_survey(
            backend=backend,
            n=n,
            hold_s=hold,
            max_vel=max_vel,
            live=live,
            seed=seed,
            port=port,
            joint_map_path=joint_map,
        )
    short = {k: v for k, v in summary.items() if k != "results"}
    typer.echo(json.dumps(short, indent=2))


if __name__ == "__main__":
    app()
