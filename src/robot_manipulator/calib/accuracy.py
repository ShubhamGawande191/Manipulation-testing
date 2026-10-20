"""Real fingertip accuracy and backlash: hover over board corners, measure with the wrist camera.

    uv run tip-accuracy plan                       # sim: targets, IK, approach paths, collisions
    uv run tip-accuracy run --backend sim          # rehearse the motion in the twin
    uv run tip-accuracy run --backend real --live  # operator present, ~4 min
    uv run tip-accuracy analyze runs/accuracy/<ts> # (run does this at the end too)

The calibrated overhead camera locates the board in the robot frame. For a few board corners the
closed-form IK puts the fingertip ``HOVER_M`` above the corner, and each target is reached four
times, from opposite sides (radially closer/farther and left/right, ``APPROACH_M`` away and a bit
higher), so every joint arrives once from each direction.

Measurement at each arrival uses the wrist camera only: PnP of the board in the wrist image gives
the target corner in the camera frame, and the camera mount (``config/extrinsics/wrist_mount.yaml``,
from the wrist-camera calibration) plus the twin's fingertip point give the fingertip in the same
frame. Their difference is the real fingertip error, independent of the joint encoders. Expressed
in the board frame: in-plane (x, y) and height (z, nominal ``HOVER_M``).

Backlash: per target, the spread of the four arrivals is the play at the tool. The difference of
the encoder readings between opposite arrivals is the part the servos see (deadband in the
position loop); the rest is play after the encoder (horn, bearings, link flex).

Caveats: the fingertip point comes from the PincOpen model (ruler check pending) and the mount from
the calibration; an error in either is a constant bias in the tool frame, not scatter.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import typer
import yaml

from robot_manipulator.calib.camera import REPO, detect, make_board
from robot_manipulator.calib.extrinsics import EXTRINSICS_DIR, INTRINSICS_DIR, TipFK
from robot_manipulator.calib.wristcal import _Cams, board_from_camera, board_points
from robot_manipulator.sim.model import ASSET_DIR, JOINTS, apply_joint_couplings, joint_qpos_adr

OUT_DIR = REPO / "runs" / "accuracy"
# 40 mm: the first run (20 mm) touched the board - servo sag (6-10 mm), a fingertip ~6 mm longer
# than the model, the camera-located board a few mm low; also lets the wrist camera see more
HOVER_M = 0.04
APPROACH_M = 0.03
LIFT_M = 0.02  # pre-poses this much higher than the target
SAFE_M = 0.08  # transfer height above the board between targets
PITCHES_DEG = (90, 80, 70, 60)
ROLL = np.radians(-80.0)
BOARD_SLAB_M = 0.02  # collision box under the board surface
BOARD_MARGIN_M = 0.02  # beyond the printed corners (the plate is larger)
N_TARGETS = 6
MIN_SEEN = 10  # board corners the wrist camera must see at a target
app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------- geometry
class Checker:
    """Bench scene + a box for the board, for collision checks of planned poses and legs."""

    def __init__(self, R_rb: np.ndarray, t_rb: np.ndarray, outer: np.ndarray) -> None:
        spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "scene_bench.xml"))
        lo, hi = outer.min(0), outer.max(0)
        centre = (lo + hi) / 2
        half = (hi - lo) / 2 + BOARD_MARGIN_M
        up = _board_up(R_rb)
        g = spec.worldbody.add_geom()
        g.name = "calib_board"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [half[0], half[1], BOARD_SLAB_M / 2]
        # box frame = board frame; its top face lies on the board surface
        g.pos = R_rb @ centre + t_rb - up * BOARD_SLAB_M / 2
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_rb.ravel())
        g.quat = quat
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.qadr = joint_qpos_adr(self.m, JOINTS)

    def clear(self, q: np.ndarray) -> bool:
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.qadr] = q
        apply_joint_couplings(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        return not any(c.dist < 0 for c in self.d.contact[: self.d.ncon])

    def leg_clear(self, qa: np.ndarray, qb: np.ndarray, n: int = 15) -> bool:
        return all(self.clear(qa + s * (qb - qa)) for s in np.linspace(0, 1, n))


def _board_up(R_rb: np.ndarray) -> np.ndarray:
    """Board normal pointing away from the table (the board frame's z may point either way)."""
    n = R_rb[:, 2]
    return n if n[2] > 0 else -n


def tip_in_camera(fk: TipFK) -> np.ndarray:
    """Fingertip in the wrist camera frame (OpenCV axes): rigid, so any joint vector will do."""
    mount = yaml.safe_load((EXTRINSICS_DIR / "wrist_mount.yaml").read_text())
    R_m, t_m = np.array(mount["R"]), np.array(mount["t_m"])
    q = np.zeros(6)
    tip_w = fk.tip(q)  # also runs FK for the frames below
    body = fk.m.body(mount["body"]).id
    p_b, R_b = fk.d.xpos[body].copy(), fk.d.xmat[body].reshape(3, 3).copy()
    tip_body = R_b.T @ (tip_w - p_b)
    return R_m.T @ (tip_body - t_m)


# ---------------------------------------------------------------- planning
def pick_corners(inner: np.ndarray, n: int = N_TARGETS) -> np.ndarray:
    """``n`` inner board corners spread over the board (farthest-point from the centre one)."""
    c = inner.mean(0)
    chosen = [int(np.argmin(np.linalg.norm(inner - c, axis=1)))]
    dist = np.linalg.norm(inner - inner[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(dist))
        chosen.append(k)
        dist = np.minimum(dist, np.linalg.norm(inner - inner[k], axis=1))
    return np.array(chosen)


def plan_targets(R_rb, t_rb, n: int = N_TARGETS) -> dict:
    from robot_manipulator.ik.analytic import AnalyticIK

    board, _ = make_board()
    outer, inner = board_points(board)
    ik = AnalyticIK()
    chk = Checker(R_rb, t_rb, outer)
    up = _board_up(R_rb)

    from robot_manipulator.calib.wristcal import WristRig, visible_corners

    rig = WristRig()
    mount_cfg = yaml.safe_load((EXTRINSICS_DIR / "wrist_mount.yaml").read_text())
    mount = (np.array(mount_cfg["R"]), np.array(mount_cfg["t_m"]))

    def solve(p, seed=None):
        for pitch in PITCHES_DEG:
            q = ik.solve(p, np.radians(pitch), ROLL, seed=seed, margin=0.05)
            if q is not None and chk.clear(q):
                return q, pitch
        return None, None

    targets = []
    for idx in pick_corners(inner, len(inner)):  # spread first; some won't plan or won't be seen
        corner = R_rb @ inner[idx] + t_rb
        tip = corner + HOVER_M * up
        q, pitch = solve(tip)
        if q is None:
            continue
        # the wrist camera must see enough of the board to measure (run 2: far-edge targets
        # showed 0-3 corners); predicted with the calibrated mount and intrinsics
        seen = visible_corners(rig, q, R_rb, t_rb, inner, mount)
        if seen < MIN_SEEN:
            continue
        radial = tip[:2] - ik.g.pan_xy
        radial = np.r_[radial / np.linalg.norm(radial), 0.0]
        tangent = np.cross([0, 0, 1.0], radial)
        approaches = {}
        for name, d in (
            ("from_base", -radial),
            ("from_far", radial),
            ("from_left", tangent),
            ("from_right", -tangent),
        ):
            pre_p = tip + APPROACH_M * d + LIFT_M * up
            pre, _ = solve(pre_p, seed=q)
            if pre is None or not chk.leg_clear(pre, q):
                break
            approaches[name] = pre
        safe, _ = solve(tip + SAFE_M * up, seed=q)
        if len(approaches) < 4 or safe is None:
            continue
        targets.append(
            {
                "corner_id": int(idx),
                "corner_robot": corner.tolist(),
                "tip_target": tip.tolist(),
                "pitch_deg": pitch,
                "predicted_corners_seen": int(seen),
                "q": q.tolist(),
                "safe": safe.tolist(),
                "approaches": {k: v.tolist() for k, v in approaches.items()},
            }
        )
        if len(targets) == n:
            break
    # transfers between targets go safe -> safe; check those legs and the way in from HOME
    from robot_manipulator.sim.envs.reach import HOME_ARM

    home = np.clip(np.r_[HOME_ARM, 0.0], ik.lo + 0.05, ik.hi - 0.05)
    legs = [home] + [np.array(t["safe"]) for t in targets] + [home]
    blocked = [i for i, (a, b) in enumerate(zip(legs, legs[1:])) if not chk.leg_clear(a, b)]
    return {
        "R_robot_board": np.asarray(R_rb).tolist(),
        "t_robot_board": np.asarray(t_rb).tolist(),
        "hover_m": HOVER_M,
        "targets": targets,
        "blocked_transfers": blocked,
    }


@app.command()
def plan(n: int = N_TARGETS) -> None:
    """Locate the board with the overhead camera and plan hover targets + approaches (sim)."""
    R_rb, t_rb = board_from_camera()
    p = plan_targets(R_rb, t_rb, n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plan.json").write_text(json.dumps(p, indent=1))
    typer.echo(f"{len(p['targets'])} targets planned -> {OUT_DIR / 'plan.json'}")
    for t in p["targets"]:
        typer.echo(
            f"  corner {t['corner_id']:2d} at {np.round(100 * np.array(t['corner_robot']), 1)} cm, "
            f"pitch {t['pitch_deg']} deg"
        )
    if p["blocked_transfers"]:
        typer.echo(f"WARNING: transfer legs {p['blocked_transfers']} not collision-free")


# ---------------------------------------------------------------- measuring
def measure(gray: np.ndarray, board, K, dist) -> tuple[np.ndarray, np.ndarray]:
    """Board pose in the wrist camera (R_cb, t_cb) from one image."""
    import cv2

    cc, ci = detect(board, gray, min_corners=6)
    if cc is None:
        raise ValueError("board not seen")
    P, uv = board.matchImagePoints(cc, ci)
    ok, rvec, tvec = cv2.solvePnP(P, uv, K, dist)
    if not ok:
        raise ValueError("PnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def tip_error_board(R_cb, t_cb, corner_board: np.ndarray, tip_cam: np.ndarray) -> np.ndarray:
    """Fingertip minus the target corner, in board coordinates (m). z = height above the board
    along the board normal toward the camera."""
    tip_board = R_cb.T @ (tip_cam - t_cb)
    e = tip_board - corner_board
    if (R_cb.T @ (-t_cb))[2] < 0:  # camera on the board's -z side: flip so z = height
        e[2] = -e[2]
    return e


def analyze(session: Path) -> dict:
    import cv2

    s = json.loads((session / "session.json").read_text())
    spec = session / "board.yaml"
    board, _ = make_board(spec) if spec.exists() else make_board()
    _, inner = board_points(board)
    intr = yaml.safe_load((INTRINSICS_DIR / "wrist.yaml").read_text())
    K, dist = np.array(intr["camera_matrix"]), np.array(intr["dist_coeffs"])
    fk = TipFK()
    tip_cam = tip_in_camera(fk)
    rows = []
    for a in s["arrivals"]:
        if "wrist" not in a:  # contact on approach, or no camera
            rows.append({**a, "error": "no image"})
            continue
        g = cv2.imread(str(session / a["wrist"]), cv2.IMREAD_GRAYSCALE)
        try:
            R_cb, t_cb = measure(g, board, K, dist)
        except ValueError as e:
            rows.append({**a, "error": str(e)})
            continue
        e = tip_error_board(R_cb, t_cb, inner[a["corner_id"]], tip_cam)
        rows.append({**a, "err_board_m": e.tolist()})
    report: dict[str, Any] = {"arrivals": rows, "stages": {}}
    hover = s.get("hover_m", HOVER_M)
    for stage in ("raw", "trimmed"):
        st = [r for r in rows if r.get("stage", "raw") == stage and "err_board_m" in r]
        if st:
            report["stages"][stage] = _stage_report(st, hover)
    report["contacts"] = sum(1 for r in rows if r.get("contact"))
    (session / "report.json").write_text(json.dumps(report, indent=1))
    return report


def _stage_report(rows: list[dict], hover: float) -> dict:
    """Per target: mean error, spread over approaches, play between opposite approaches (at the
    tool, from the camera; and in the encoders); plus a summary over targets."""
    targets: dict[str, Any] = {}
    errs = []
    pairs = [("from_base", "from_far"), ("from_left", "from_right")]
    for cid in sorted({r["corner_id"] for r in rows}):
        rs = [r for r in rows if r["corner_id"] == cid]
        E = np.array([r["err_board_m"] for r in rs])
        Q = np.array([r["q_rad"] for r in rs])
        ap = {r["approach"]: i for i, r in enumerate(rs)}
        ok = [(a, b) for a, b in pairs if a in ap and b in ap]
        mean = E.mean(0)
        errs.append(mean)
        targets[str(cid)] = {
            "n": len(rs),
            "mean_err_mm": (1000 * mean).round(2).tolist(),
            "in_plane_mm": float(1000 * np.linalg.norm(mean[:2])),
            "spread_mm": float(1000 * np.linalg.norm(E[:, :2] - mean[:2], axis=1).max()),
            "play_tool_mm": {
                f"{a}-{b}": float(1000 * np.linalg.norm(E[ap[a], :2] - E[ap[b], :2])) for a, b in ok
            },
            "play_encoder_deg": {
                f"{a}-{b}": np.degrees(Q[ap[a], :5] - Q[ap[b], :5]).round(3).tolist() for a, b in ok
            },
        }
    E = np.array(errs)
    inp = 1000 * np.linalg.norm(E[:, :2], axis=1)
    plays = [v for t in targets.values() for v in t["play_tool_mm"].values()]
    return {
        "targets": targets,
        "summary": {
            "targets": len(errs),
            "in_plane_mm_median": float(np.median(inp)),
            "in_plane_mm_max": float(inp.max()),
            "height_err_mm_median": float(1000 * np.median(E[:, 2] - hover)),
            "play_tool_mm_median": float(np.median(plays)) if plays else float("nan"),
        },
    }


def _print_report(rep: dict) -> None:
    for stage, st in rep["stages"].items():
        typer.echo(f"--- {stage}")
        for cid, t in st["targets"].items():
            typer.echo(
                f"corner {cid:>2}: error {t['mean_err_mm']} mm (in-plane {t['in_plane_mm']:.1f}), "
                f"spread {t['spread_mm']:.1f} mm, play at tool "
                + ", ".join(f"{k} {v:.1f}" for k, v in t["play_tool_mm"].items())
            )
        s = st["summary"]
        typer.echo(
            f"{stage}: in-plane error median {s['in_plane_mm_median']:.1f} mm "
            f"(max {s['in_plane_mm_max']:.1f}); height error {s['height_err_mm_median']:+.1f} mm; "
            f"play at tool median {s['play_tool_mm_median']:.1f} mm"
        )
    if rep.get("contacts"):
        typer.echo(f"contacts detected: {rep['contacts']}")


@app.command("analyze")
def analyze_cmd(session: Path) -> None:
    _print_report(analyze(session))


# ---------------------------------------------------------------- running
@app.command()
def run(
    backend: str = typer.Option("sim", help="sim | real"),
    live: bool = typer.Option(False, "--live/--dry-run"),
    max_vel: float = typer.Option(0.5, help="rad/s cap for transfers"),
    approach_vel: float = typer.Option(0.25, help="rad/s cap for the final approach"),
    hold_s: float = typer.Option(1.0, help="hold before each photo (also the trim time)"),
    trim: bool = typer.Option(True, help="settle trim after arrival (photos before and after)"),
    plan_path: Path = typer.Option(OUT_DIR / "plan.json"),
    port: str = typer.Option("/dev/so101_follower"),
    stow_leader: bool = typer.Option(True, "--stow-leader/--no-stow-leader"),
) -> None:
    """Visit every target from its four approaches; wrist image + joints at each arrival, before
    ("raw") and after ("trimmed") the settle trim. The final approach has a contact guard."""
    import cv2

    from robot_manipulator.calib.camera import BOARD_CFG
    from robot_manipulator.exec.scheduler import SimClock, WallClock
    from robot_manipulator.harness.backend import LeRobotBackend, SimBackend
    from robot_manipulator.harness.cli import leader_clear
    from robot_manipulator.harness.policies.glide import GlideTo
    from robot_manipulator.harness.policies.settle import SettleTrim
    from robot_manipulator.harness.runner import run_skill
    from robot_manipulator.harness.safety_layer import SafetyFilter
    from robot_manipulator.harness.skill import ContactGuard, JointReached, Skill, TimeLimit
    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.envs.reach import HOME_ARM
    from robot_manipulator.sim.model import joint_ranges, load_model

    p = json.loads(plan_path.read_text())
    model = load_model()
    lo, hi = joint_ranges(model, JOINTS)
    home = np.clip(np.r_[HOME_ARM, 0.0], lo + 0.05, hi - 0.05)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{backend}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "board.yaml").write_text(BOARD_CFG.read_text())
    if backend == "sim":
        be: Any = SimBackend(model=model, hz=50.0)
        be.reset(home)
        clock: Any = SimClock()
    elif backend == "real":
        be = LeRobotBackend(
            port, JointMap.from_yaml(REPO / "config" / "sim2real.yaml"), dry_run=not live
        )
        if be.telemetry is not None:
            be.telemetry.source = "accuracy_live" if live else "accuracy_dry"
        clock = WallClock()
    else:
        raise typer.BadParameter("backend must be sim or real")
    cams = _Cams(("wrist",)) if backend == "real" else None
    arrivals: list[dict] = []
    arm = (0, 1, 2, 3, 4)

    def safety(vel):
        return SafetyFilter(lo=lo, hi=hi, margin=0.05, max_vel=vel, clock=clock)

    def go(target, vel, tol=0.03, timeout=15.0, guard=False) -> str:
        """'ok', 'contact' or 'missed'."""
        conds = [ContactGuard()] if guard else []
        conds += [JointReached(target, tol=tol, joints=arm), TimeLimit(timeout)]
        sk = Skill("acc_glide", GlideTo(target, max_vel=vel, hz=50.0), conds, hz=50.0)
        res = run_skill(sk, be, safety(vel), goal={}, clock=clock)
        if backend == "real" and not live:
            return "ok"
        if res.reason == "contact":
            return "contact"
        err = np.abs(np.asarray(be.read()["q"]) - target)[:5].max()
        return "ok" if res.status == "success" or err < 0.1 else "missed"

    def hold_and_shoot(policy, ti: int, name: str, stage: str, q_t) -> dict:
        # no following-error guard here: the trim moves the command away from a joint that
        # friction holds, which is exactly what that guard reads as contact (run 2: 6 false
        # trips, trims aborted). The trim is bounded (+-4 deg) instead.
        sk = Skill(f"acc_{stage}", policy, [TimeLimit(hold_s)], hz=50.0)
        res = run_skill(sk, be, safety(approach_vel), goal={}, clock=clock)
        q = np.mean([np.asarray(be.read()["q"]) for _ in range(5)], axis=0)
        a = {
            "target": ti,
            "corner_id": p["targets"][ti]["corner_id"],
            "approach": name,
            "stage": stage,
            "q_rad": q.tolist(),
            "q_cmd_rad": np.asarray(q_t).tolist(),
            "contact": res.reason == "contact",
        }
        if cams is not None:
            a["wrist"] = f"wrist_t{ti}_{name}_{stage}.png"
            cv2.imwrite(str(out / a["wrist"]), cams.grab("wrist"))
        return a

    try:
        with leader_clear(backend, live, stow_leader):
            if go(home, max_vel) != "ok":
                raise RuntimeError("HOME not reached")
            for ti, t in enumerate(p["targets"]):
                q_t, safe = np.array(t["q"]), np.array(t["safe"])
                if go(safe, max_vel) != "ok":
                    typer.echo(f"target {ti}: safe pose not reached; stopping")
                    break
                for name, pre in t["approaches"].items():
                    pre = np.array(pre)
                    if go(pre, max_vel) != "ok":
                        typer.echo(f"target {ti} {name}: pre-pose not reached; skipping")
                        continue
                    status = go(q_t, approach_vel, tol=0.02, guard=True)
                    if status != "ok":
                        msg = "CONTACT on approach" if status == "contact" else "not reached"
                        typer.echo(f"target {ti} {name}: {msg}; backing out")
                        arrivals.append(
                            {
                                "target": ti,
                                "corner_id": t["corner_id"],
                                "approach": name,
                                "stage": "approach",
                                "contact": status == "contact",
                            }
                        )
                        go(pre, approach_vel)
                        continue
                    glide = GlideTo(q_t, max_vel=approach_vel, hz=50.0)
                    arrivals.append(hold_and_shoot(glide, ti, name, "raw", q_t))
                    if trim and not arrivals[-1]["contact"]:
                        arrivals.append(hold_and_shoot(SettleTrim(q_t), ti, name, "trimmed", q_t))
                    go(pre, approach_vel)  # back out the way it came
                go(safe, max_vel)
                typer.echo(f"target {ti + 1}/{len(p['targets'])} done")
            go(home, max_vel)
    except KeyboardInterrupt:
        typer.echo("interrupted; parking")
    finally:
        be.close()
        if cams is not None:
            cams.close()
    (out / "session.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "live": live,
                "hover_m": p["hover_m"],
                "plan": str(plan_path),
                "arrivals": arrivals,
            },
            indent=1,
        )
    )
    typer.echo(f"{len(arrivals)} arrivals -> {out}")
    if backend == "real" and live and arrivals:
        _print_report(analyze(out))


if __name__ == "__main__":
    app()
