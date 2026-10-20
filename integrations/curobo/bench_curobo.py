"""cuRobo IK on the SO-101 benchmark targets. Runs in cuRobo's own environment, not the repo's.

    uv run urdf-export --no-mimic                        # repo env: URDF from the MJCF
    uv run ik-bench export-targets                       # repo env: runs/ik/targets.json
    ~/tools/curobo/.venv/bin/python integrations/curobo/bench_curobo.py   # this script
    uv run ik-bench run                                  # repo env: table incl. cuRobo

Steps: build the cuRobo robot config from the URDF (RobotBuilder: collision spheres fitted to
the meshes, self-collision matrix; the gripper and the four PincOpen finger joints are locked at
the closed pose), check cuRobo's FK against the closed-form reference solutions,
then solve every scenario target one by one (latency) and all at once (batched throughput).
Success uses the same rule as the rest of the benchmark (fingertip within 1 mm; orientation within
0.5 deg, which is stricter than the pitch-only rule because cuRobo solves full 6-DOF poses).
Written against cuRobo v0.8.0 (the v2 API).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.robot_builder import RobotBuilder
from curobo.types import GoalToolPose, Pose

REPO = Path(__file__).resolve().parents[2]
POS_TOL = 1e-3
ROT_TOL = np.radians(0.5)
TOOL = "fingertip"
# the gripper joint itself drives the PincOpen coupler, which has no collision mesh and is off the
# path to the fingertip, so cuRobo never puts it in its kinematic tree (locking it -> KeyError)
LOCKED = (
    "pinc_right_arm",
    "pinc_right_finger",
    "pinc_left_arm",
    "pinc_left_finger",
)


def build_config(urdf: Path, out: Path, seed: int = 0) -> Path:
    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(urdf.parent), tool_frames=[TOOL])
    kwargs = {"compute_metrics": False}
    if _accepts_seed(builder.fit_collision_spheres):
        kwargs["seed"] = seed
    builder.fit_collision_spheres(**kwargs)
    builder.compute_collision_matrix()
    builder.save(builder.build(), str(out))
    data = yaml.safe_load(out.read_text())
    # the PincOpen finger joints (no mimic tags: cuRobo 0.8.0's builder crashes on them) are
    # locked at their closed-gripper value, 0 (the couplings have zero offset)
    data["kinematics"]["lock_joints"] = {j: 0.0 for j in LOCKED}
    # RobotBuilder.save (0.8.0) also writes runtime-only fields that RobotCfg.create passes again
    # (TypeError: multiple values for 'load_collision_spheres'): drop them
    for key in ("load_collision_spheres", "num_envs"):
        data["kinematics"].pop(key, None)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"robot config -> {out} ({builder.num_spheres} spheres)")
    return out


def _accepts_seed(fn) -> bool:
    import inspect

    return "seed" in inspect.signature(fn).parameters


def make_ik(cfg: Path, batch: int, self_collision: bool) -> InverseKinematics:
    return InverseKinematics(
        InverseKinematicsCfg.create(
            robot=str(cfg),
            num_seeds=32,
            max_batch_size=batch,
            position_tolerance=POS_TOL,
            orientation_tolerance=ROT_TOL,
            self_collision_check=self_collision,
        )
    )


def poses(rows) -> Pose:
    dev = dict(device="cuda", dtype=torch.float32)
    return Pose(
        position=torch.tensor([r["pos"] for r in rows], **dev),
        quaternion=torch.tensor([r["quat_wxyz"] for r in rows], **dev),
    )


def errors(result) -> tuple[np.ndarray, np.ndarray]:
    pos = result.position_error.detach().reshape(-1).cpu().numpy()
    rot = result.rotation_error.detach().reshape(-1).cpu().numpy()
    return pos, rot


def fk_check(ik: InverseKinematics, rows) -> float:
    """Max fingertip distance between cuRobo FK of the reference joints and the target (m)."""
    from curobo.types import JointState

    ref = [r for r in rows if r["reachable"]][:64]
    names = list(ik.joint_names)
    q = torch.tensor(
        [[dict(zip(_JOINTS, r["q_ref"])).get(n, 0.0) for n in names] for r in ref],
        device="cuda",
        dtype=torch.float32,
    )
    state = ik.compute_kinematics(JointState.from_position(q, joint_names=names))
    tip = state.tool_poses.get_link_pose(TOOL).position.detach().cpu().numpy().reshape(-1, 3)
    return float(np.linalg.norm(tip - np.array([r["pos"] for r in ref]), axis=1).max())


_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def run(args) -> dict:
    targets = json.loads(Path(args.targets).read_text())["scenarios"]
    cfg = Path(args.config)
    if args.rebuild or not cfg.exists():
        build_config(Path(args.urdf), cfg)
    report: dict = {}
    single = make_ik(cfg, 1, args.self_collision)
    first = next(iter(targets.values()))
    print(f"cuRobo FK vs closed-form reference: max {1e3 * fk_check(single, first):.3f} mm")
    tool = single.tool_frames[0]
    for scen, rows in targets.items():
        # one at a time (latency), after a warm-up solve
        single.solve_pose(GoalToolPose.from_poses({tool: poses(rows[:1])}, num_goalset=1))
        times, ok = [], []
        for r in rows:
            goal = GoalToolPose.from_poses({tool: poses([r])}, num_goalset=1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = single.solve_pose(goal)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            pe, re_ = errors(res)
            ok.append(bool(pe[0] < POS_TOL and re_[0] < ROT_TOL))
        # all targets in one call (throughput)
        batch = make_ik(cfg, len(rows), args.self_collision)
        goal = GoalToolPose.from_poses({tool: poses(rows)}, num_goalset=1)
        batch.solve_pose(goal)  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = batch.solve_pose(goal)
        torch.cuda.synchronize()
        t_batch = time.perf_counter() - t0
        pe, re_ = errors(res)
        ok_b = (pe < POS_TOL) & (re_ < ROT_TOL)
        reach = np.array([r["reachable"] for r in rows])
        report[scen] = {
            "success": float(np.mean(ok)),
            "time_ms_median": 1e3 * float(np.median(times)),
            "time_ms_p95": 1e3 * float(np.percentile(times, 95)),
            "batch_success": float(ok_b.mean()),
            "batch_ms_total": 1e3 * t_batch,
            "batch_ms_per_target": 1e3 * t_batch / len(rows),
            "solved_unreachable": int((ok_b & ~reach).sum()),  # should be 0
            "per_target_ok": [bool(x) for x in ok_b],
        }
        print(scen, {k: v for k, v in report[scen].items() if k != "per_target_ok"})
        del batch
        torch.cuda.empty_cache()
    report["_meta"] = {
        "curobo": __import__("curobo").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "self_collision": args.self_collision,
        "num_seeds": 32,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--urdf", default=str(REPO / "runs/urdf/so101_pinc/so101_pinc_nomimic.urdf")
    )  # uv run urdf-export --no-mimic
    ap.add_argument("--config", default=str(REPO / "runs/ik/curobo/so101_pinc.yml"))
    ap.add_argument("--targets", default=str(REPO / "runs/ik/targets.json"))
    ap.add_argument("--out", default=str(REPO / "runs/ik/curobo_results.json"))
    ap.add_argument("--rebuild", action="store_true", help="refit spheres / rebuild the config")
    ap.add_argument(
        "--no-self-collision",
        dest="self_collision",
        action="store_false",
        help="skip self-collision (cuRobo 0.8.0's seed stage then has no costs and fails)",
    )
    args = ap.parse_args()
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
