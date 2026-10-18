"""Workspace survey: visit spread-out poses, hold still, log telemetry (atlas layer 2 data).

Reach runs are almost all motion, so they yield few quasi-static load samples (1 in the first
two live runs). The survey glides to each of ``n`` collision-free poses whose EE lies in the
table workspace (``TARGET_BOX``) and holds there for ``hold_s`` seconds. On the real arm the
telemetry reads load every ``EXTRA_EVERY`` steps during the survey. ``uv run atlas real`` then
calibrates measured load against the twin's gravity model and fills the atlas grid.

    uv run harness survey --backend sim --n 20                       # dry check in sim
    uv run harness survey --backend real --n 20 --live --max-vel 0.5 # operator present
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot_manipulator.sim.envs.reach import GRIPPER_HOLD, HOME_ARM, TARGET_BOX, target_box
from robot_manipulator.sim.kinematics import FK
from robot_manipulator.sim.model import JOINTS, apply_joint_couplings, joint_qpos_adr, joint_ranges

EXTRA_EVERY = 2
GLIDE_TOL = 0.05  # rad, per joint
GLIDE_TIMEOUT_S = 10.0


def collision_free(model: mujoco.MjModel, q: np.ndarray) -> bool:
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    d.qpos[joint_qpos_adr(model, JOINTS)] = q
    apply_joint_couplings(model, d)
    mujoco.mj_forward(model, d)
    return d.ncon == 0


def survey_poses(n: int, seed: int = 0, model: mujoco.MjModel | None = None) -> np.ndarray:
    """``n`` spread-out, collision-free poses (rad, CMD order, gripper open), ordered as a short
    nearest-neighbour path starting near HOME."""
    fk = FK(model)
    model = fk.model
    rng = np.random.default_rng(seed)
    lo, hi = joint_ranges(model, JOINTS)
    tlo, thi = target_box(lo, hi)
    q_cand, ee_cand = fk.sample_reachable(rng, max(20 * n, 200), tlo, thi, box=TARGET_BOX)
    q_cand[:, 5] = GRIPPER_HOLD
    ok = np.array([collision_free(model, q) for q in q_cand])
    q_cand, ee_cand = q_cand[ok], ee_cand[ok]
    if len(q_cand) < n:
        raise RuntimeError(f"only {len(q_cand)} collision-free candidates for {n} poses")
    home_ee = fk.ee_pos(np.r_[HOME_ARM, GRIPPER_HOLD])
    chosen = [int(np.argmin(np.linalg.norm(ee_cand - home_ee, axis=1)))]
    dist = np.linalg.norm(ee_cand - ee_cand[chosen[0]], axis=1)
    while len(chosen) < n:  # farthest-point sampling in EE space
        k = int(np.argmax(dist))
        chosen.append(k)
        dist = np.minimum(dist, np.linalg.norm(ee_cand - ee_cand[k], axis=1))
    order, left = [chosen[0]], set(chosen[1:])
    while left:  # nearest-neighbour tour in joint space (less travel between holds)
        last = q_cand[order[-1]]
        nxt = min(left, key=lambda i: float(np.abs(q_cand[i, :5] - last[:5]).max()))
        order.append(nxt)
        left.remove(nxt)
    return q_cand[order]


def run_survey(
    backend: str = "sim",
    n: int = 20,
    hold_s: float = 1.5,
    max_vel: float = 0.5,
    live: bool = False,
    seed: int = 0,
    port: str = "/dev/so101_follower",
    joint_map_path: Path | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    from robot_manipulator.exec.scheduler import SimClock, WallClock
    from robot_manipulator.harness.backend import LeRobotBackend, SimBackend
    from robot_manipulator.harness.policies.glide import GlideTo
    from robot_manipulator.harness.runner import run_skill
    from robot_manipulator.harness.safety_layer import SafetyFilter
    from robot_manipulator.harness.skill import JointReached, Skill, TimeLimit
    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.model import load_model

    hz = 50.0
    model = load_model()
    lo6, hi6 = joint_ranges(model, JOINTS)
    margin = 0.05
    poses = np.clip(survey_poses(n, seed, model), lo6 + margin, hi6 - margin)
    home6 = np.clip(np.r_[HOME_ARM, GRIPPER_HOLD], lo6 + margin, hi6 - margin)
    out = out or Path("runs/survey") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    if backend == "sim":
        be: Any = SimBackend(hz=hz)
        be.reset(home6)
        clock: Any = SimClock()
    elif backend == "real":
        repo = Path(__file__).resolve().parents[3]
        jm = JointMap.from_yaml(joint_map_path or repo / "config" / "sim2real.yaml")
        be = LeRobotBackend(port, jm, dry_run=not live)
        if be.telemetry is not None:
            be.telemetry.source = "survey_live" if live else "survey_dry"
            be.telemetry.extra_every = EXTRA_EVERY
        clock = WallClock()
    else:
        raise ValueError("backend must be sim or real")

    results: list[dict[str, Any]] = []
    try:
        for i, target in enumerate([home6, *poses]):
            safety = SafetyFilter(lo=lo6, hi=hi6, margin=margin, max_vel=max_vel, clock=clock)
            glide = Skill(
                "survey_glide",
                GlideTo(target, max_vel=max_vel, hz=hz),
                [JointReached(target, tol=GLIDE_TOL), TimeLimit(GLIDE_TIMEOUT_S)],
                hz=hz,
            )
            res = run_skill(glide, be, safety, goal={}, clock=clock)
            if res.status != "success":
                results.append({"pose": i, "reached": False, "reason": res.reason})
                if backend == "real" and not live:
                    continue  # dry run: nothing moves, just show we got this far
                break
            hold = Skill(
                "survey_hold", GlideTo(target, max_vel=max_vel, hz=hz), [TimeLimit(hold_s)], hz=hz
            )
            run_skill(hold, be, safety, goal={}, clock=clock)
            q = np.asarray(be.read()["q"])
            results.append(
                {
                    "pose": i,
                    "reached": True,
                    "target_rad": target.tolist(),
                    "err_deg": np.degrees(q - target).round(2).tolist(),
                }
            )
    except KeyboardInterrupt:
        results.append({"interrupted": True})
    finally:
        be.close()
    summary = {
        "backend": backend,
        "live": live if backend == "real" else None,
        "n_poses": n,
        "reached": sum(1 for r in results if r.get("reached")),
        "hold_s": hold_s,
        "max_vel": max_vel,
        "results": results,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
