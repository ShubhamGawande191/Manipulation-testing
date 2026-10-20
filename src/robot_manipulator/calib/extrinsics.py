"""Camera extrinsics in the robot frame: ChArUco board seen by the camera + touched by the arm.

    uv run camcal-extrinsics workspace          # interactive, operator at the arm (torque OFF)

1. The camera image gives the board pose in the camera frame (``solvePnP`` with
   ``config/camera_intrinsics/<cam>.yaml`` and ``config/charuco_board.yaml``). The 4 outer grid
   corners are drawn numbered 1-4 in ``runs/extrinsics/<cam>_corners.png``.
2. The operator guides the follower's **closed** fingertips onto corners 1..4, ``--per-corner``
   times each at different wrist angles (torque off; the tool offers to close the gripper with its
   own motor first, since the PincOpen isn't back-drivable). The calibrated joint map + twin FK
   give each touch. ``fit_touches`` solves for the board pose in the robot frame **and** the
   fingertip point (the touches at different wrist angles make it identifiable). The raw joints
   are saved, so ``--refit runs/extrinsics/<cam>.json`` re-runs the fit offline.
3. Chaining the two gives the camera pose in the robot frame. It's written to
   ``config/extrinsics/<cam>.yaml`` (``pos`` / ``xyaxes`` / ``fovy_deg``), which ``scene build``
   uses over the placeholder in ``config/scene_bench.yaml``. Then rebuild
   the scene (``uv run scene build``) and compare views (``uv run scene render``).

Checks printed: residual per touch, the board's side lengths from the touches vs the real board
(a pattern here points at a joint calibration error, e.g. only the pan-swept sides short), the
fingertip correction, and the board height vs the table top. Above ``--max-residual-mm`` the camera
pose is only written after confirmation.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import typer
import yaml

from robot_manipulator.calib.camera import REPO, detect, make_board
from robot_manipulator.cameras import apply_controls, load_camera_config
from robot_manipulator.profile import ACTIVE

INTRINSICS_DIR = REPO / "config" / "camera_intrinsics"
SCENE_CFG = REPO / "config" / "scene_bench.yaml"
EXTRINSICS_DIR = REPO / "config" / "extrinsics"
OUT_DIR = REPO / "runs" / "extrinsics"
# closed PincOpen: the finger meshes reach 6.1 mm past `gripperframe` along the tool axis (model,
# 2026-10-18). Touching with the tips puts the surface there, not at the site.
TIP_OFFSET_M = 0.0061
GRIPPER_OPEN_ABOVE = 15.0  # LeRobot 0..100; above this the fingertips are apart
app = typer.Typer(add_completion=False, help=__doc__)


def board_outer_corners(board) -> np.ndarray:
    """(4, 3) outer corners of the square grid in board coordinates, in the numbering order."""
    sx, sy = board.getChessboardSize()
    L = board.getSquareLength()
    return np.array([[0, 0, 0], [sx * L, 0, 0], [sx * L, sy * L, 0], [0, sy * L, 0]], dtype=float)


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid R, t minimising |R src + t - dst|; returns (R, t, per-point residual norms)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    res = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    return R, t, res


def camera_in_robot(R_cb, t_cb, R_rb, t_rb) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and rotation (columns = OpenCV camera axes) in the robot frame.
    X_cam = R_cb X_board + t_cb ; X_robot = R_rb X_board + t_rb."""
    R_rc = R_rb @ R_cb.T
    p = t_rb - R_rc @ t_cb
    return p, R_rc


def mujoco_xyaxes(R_rc: np.ndarray) -> list[float]:
    """MuJoCo camera looks along its -z with +y up; OpenCV looks along +z with +y down."""
    return [*R_rc[:, 0].tolist(), *(-R_rc[:, 1]).tolist()]


class TipFK:
    """Fingertip tip position in the robot frame, with the tip as a point fixed in the EE site's
    body (``tip_local``). The nominal tip is ``TIP_OFFSET_M`` past ``gripperframe`` along the tool
    axis; ``fit_touches`` can refine it from touches made at several wrist angles."""

    def __init__(self, model=None) -> None:
        import mujoco

        from robot_manipulator.sim.model import EE_SITE, load_model

        self.m = model if model is not None else load_model()
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self.body = int(self.m.site_bodyid[self.site])
        self._pose(np.zeros(6))
        tool_x = self.d.body("pinc_base").xmat.reshape(3, 3)[:, 0]  # palm -> fingertips
        tip = self.d.site_xpos[self.site] + TIP_OFFSET_M * tool_x
        p, R = self.d.xpos[self.body], self.d.xmat[self.body].reshape(3, 3)
        self.tip_local_nominal = R.T @ (tip - p)
        self.tool_axis_local = R.T @ tool_x

    def _pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        from robot_manipulator.sim.model import JOINTS, apply_joint_couplings, joint_qpos_adr

        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[joint_qpos_adr(self.m, JOINTS)] = q
        apply_joint_couplings(self.m, self.d)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.body].copy(), self.d.xmat[self.body].reshape(3, 3).copy()

    def poses(self, qs: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self._pose(np.asarray(q, dtype=float)) for q in qs]

    def tip(self, q: np.ndarray, tip_local: np.ndarray | None = None) -> np.ndarray:
        p, R = self._pose(np.asarray(q, dtype=float))
        return p + R @ (self.tip_local_nominal if tip_local is None else tip_local)


def tip_point(model, data, q: np.ndarray) -> np.ndarray:
    """Nominal closed-fingertip tip in the robot frame for ``q`` (rad, CMD order)."""
    return TipFK(model).tip(q)


TIP_PRIOR_M = 0.01  # prior sigma of the 3-D tip correction vs the 1 mm touch noise
JOINT_NAMES = ("pan", "lift", "elbow", "wflex", "wroll")
# fit variants (key: joint offsets fitted, tip model); `diagnose` prints all of them
FITS = {
    "nominal": ((), "none"),
    "tip": ((), "axis"),
    "tip3d": ((), "3d"),
    "joints": ((1, 2, 3), "none"),
    "joints+tip": ((1, 2, 3), "axis"),
    "joints4+tip": ((0, 1, 2, 3), "axis"),
}


def fit_touches(
    fk: TipFK,
    qs: np.ndarray,
    corner_ids: np.ndarray,
    corners: np.ndarray,
    offsets: tuple[int, ...] = (),
    tip: str = "3d",
) -> dict:
    """Board pose in the robot frame from corner touches, optionally with a kinematic correction.

    ``qs`` (n, 6) joint vectors, ``corner_ids`` (n,) index into ``corners`` (4, 3, board frame).
    ``offsets``: joint indices whose encoder offset is fitted (rad, added to the reading).
    ``tip``: "none" (model fingertip), "axis" (fingertip length along the tool axis), or "3d"
    (free tip point, regularized toward the model with sigma ``TIP_PRIOR_M``). Touches of each
    corner from clearly different arm configurations make these identifiable.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    qs = np.asarray(qs, dtype=float)
    tip0, ax = fk.tip_local_nominal, fk.tool_axis_local
    n_off, n_tip = len(offsets), {"none": 0, "axis": 1, "3d": 3}[tip]

    def unpack(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        q = qs.copy()
        for k, j in enumerate(offsets):
            q[:, j] += x[6 + k]
        tx = x[6 + n_off :]
        tip_local = tip0 + (tx[0] * ax if tip == "axis" else tx if tip == "3d" else 0.0)
        return R, x[3:6], q, tip_local

    def touched(q, tip_local):
        return np.array([p + Rb @ tip_local for p, Rb in fk.poses(q)])

    def residual(x):
        R, t, q, tip_local = unpack(x)
        r = (touched(q, tip_local) - (corners[corner_ids] @ R.T + t)).ravel()
        if tip == "3d":
            r = np.r_[r, 0.001 * x[6 + n_off :] / TIP_PRIOR_M]
        return r

    R0, t0, _ = kabsch(corners[corner_ids], touched(qs, tip0))
    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), t0, np.zeros(n_off + n_tip)]
    sol = least_squares(residual, x0)
    R, t, q, tip_local = unpack(sol.x)
    pts = touched(q, tip_local)
    res = np.linalg.norm(pts - (corners[corner_ids] @ R.T + t), axis=1)
    return {
        "R": R,
        "t": t,
        "tip_local": tip_local,
        "tip_shift_m": tip_local - tip0,
        "offsets_rad": {JOINT_NAMES[j]: float(sol.x[6 + k]) for k, j in enumerate(offsets)},
        "res": res,
        "pts": pts,
    }


def diagnose(fk: TipFK, qs, corner_ids, corners) -> list[tuple[str, dict]]:
    """Fit every variant in ``FITS``; a big drop in residual names the likely error source."""
    return [(k, fit_touches(fk, qs, corner_ids, corners, *v)) for k, v in FITS.items()]


def side_lengths(pts_by_corner: np.ndarray) -> list[float]:
    """|1-2|, |2-3|, |3-4|, |4-1| of the (4, 3) mean touch per corner."""
    return [float(np.linalg.norm(pts_by_corner[k] - pts_by_corner[(k + 1) % 4])) for k in range(4)]


def _close_gripper(robot, closed: float) -> None:
    """Close the gripper with its own motor only (arm torques stay off), then release it."""
    import time

    bus = robot.bus
    bus.write("Goal_Position", "gripper", bus.sync_read("Present_Position")["gripper"])
    bus.write("Torque_Enable", "gripper", 1)
    try:
        bus.write("Goal_Position", "gripper", closed)
        time.sleep(1.5)
    finally:
        bus.write("Torque_Enable", "gripper", 0)  # not back-drivable: it stays closed


def _grab(cam_name: str):
    import time

    import cv2

    cam = load_camera_config()[cam_name]
    apply_controls(cam["path"], cam.get("controls", {}))
    cap = cv2.VideoCapture(cam["path"])
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.5:
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"camera {cam_name} returned no frame")
    return frame


def _board_in_camera(cam_name: str, board, K, dist) -> tuple[np.ndarray, np.ndarray, float]:
    """Grab frames until the board is seen; returns (R_cam_board, t_cam_board, PnP rms px)."""
    import cv2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(30):  # ~1 min: give the operator time to place the board
        frame = _grab(cam_name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci = detect(board, gray)
        if cc is not None and len(ci) >= 12:
            break
        if attempt == 0:
            typer.echo("waiting for the board (flat on the table, fully visible)…")
    else:
        cv2.imwrite(str(OUT_DIR / f"{cam_name}_last.png"), frame)
        raise RuntimeError(
            f"board not detected; camera view saved to {OUT_DIR / f'{cam_name}_last.png'}"
        )
    obj, img = board.matchImagePoints(cc, ci)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    pnp_err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, 1))))
    px, _ = cv2.projectPoints(board_outer_corners(board), rvec, tvec, K, dist)
    show = frame.copy()
    for k, (u, v) in enumerate(px.reshape(-1, 2), start=1):
        cv2.circle(show, (int(u), int(v)), 8, (0, 0, 255), 2)
        cv2.putText(show, str(k), (int(u) + 10, int(v) - 10), 0, 0.9, (0, 0, 255), 2)
    img_path = OUT_DIR / f"{cam_name}_corners.png"
    cv2.imwrite(str(img_path), show)
    typer.echo(
        f"board detected ({len(ci)} corners, PnP rms {pnp_err:.2f} px); "
        f"corners labelled in {img_path}"
    )
    R_cb, _ = cv2.Rodrigues(rvec)
    return R_cb, tvec.ravel(), pnp_err


def _touch(port: str, robot_id: str, per_corner: int) -> tuple[np.ndarray, np.ndarray]:
    """Operator touches each outer corner ``per_corner`` times; returns (qs rad, corner ids)."""
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config
    from robot_manipulator.harness.units import JointMap

    jm = JointMap.from_yaml(ACTIVE.joint_map)
    robot = SOFollower(build_follower_config(port=port, robot_id=robot_id, cameras=False))
    robot.bus.connect()
    qs, ids = [], []
    hints = (
        "tool pointing straight down",
        "wrist tilted toward the robot, elbow high",
        "wrist tilted away from the robot, elbow low",
        "tool tilted sideways, arm reaching differently",
    )
    try:
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
        if any(v for k, v in torque.items() if k != "gripper"):
            raise RuntimeError("follower torque is on: run `uv run teleop --park-only` first")
        grip = robot.bus.sync_read("Present_Position")["gripper"]
        if grip > GRIPPER_OPEN_ABOVE:
            input(f"Gripper is open ({grip:.0f}). Enter closes it (gripper motor only)… ")
            _close_gripper(robot, jm.gripper.lerobot_closed)
            typer.echo(f"  gripper now {robot.bus.sync_read('Present_Position')['gripper']:.0f}")
        for k in range(4):
            for n in range(per_corner):
                hint = hints[n % len(hints)]
                input(f"Corner {k + 1}, touch {n + 1}/{per_corner} ({hint}); Enter… ")
                pos = robot.bus.sync_read("Present_Position")
                qs.append(jm.lerobot_to_sim({f"{j}.pos": float(v) for j, v in pos.items()}))
                ids.append(k)
    finally:
        robot.bus.disconnect(disable_torque=False)
    return np.array(qs), np.array(ids)


@app.command()
def main(
    cam_name: str = typer.Argument("workspace"),
    port: str = typer.Option(ACTIVE.port),
    robot_id: str = typer.Option(ACTIVE.robot_id),
    per_corner: int = typer.Option(4, help="touches per corner, each in a different arm pose"),
    fit: str | None = typer.Option(
        None,
        help=f"model for the camera pose: {', '.join(FITS)} (default: joints+tip from "
        "12 touches, else tip3d)",
    ),
    max_residual_mm: float = typer.Option(8.0, help="ask before writing the scene above this"),
    refit: Path | None = typer.Option(
        None, help="re-fit a saved session (runs/extrinsics/<cam>.json); no camera, no arm"
    ),
) -> None:
    from robot_manipulator.sim.model import load_model

    board, bcfg = make_board()
    corners = board_outer_corners(board)
    intr_path = INTRINSICS_DIR / f"{cam_name}.yaml"
    if refit is not None:
        saved = json.loads(refit.read_text())
        R_cb, t_cb = np.array(saved["R_cam_board"]), np.array(saved["t_cam_board"])
        pnp_err, fovy = saved["pnp_rms_px"], saved["fovy_deg"]
        qs, ids = np.array(saved["touch_q_rad"]), np.array(saved["touch_corner"])
    else:
        if not intr_path.exists():
            raise typer.BadParameter(f"{intr_path} missing: run `uv run camcal intrinsics`")
        intr = yaml.safe_load(intr_path.read_text())
        K, dist = np.array(intr["camera_matrix"]), np.array(intr["dist_coeffs"])
        fovy = float(np.degrees(2 * np.arctan(intr["image_size"][1] / 2 / K[1, 1])))
        R_cb, t_cb, pnp_err = _board_in_camera(cam_name, board, K, dist)  # 1. camera sees board
        qs, ids = _touch(port, robot_id, per_corner)  # 2. arm touches it

    # 3. fit the board in the robot frame, chain to the camera
    fk = TipFK(load_model())
    fit_key = fit or ("joints+tip" if len(qs) >= 12 else "tip3d")
    if fit_key not in FITS:
        raise typer.BadParameter(f"--fit must be one of {list(FITS)}")
    typer.echo(f"{'model':14} {'max mm':>7} {'rms mm':>7}  parameters")
    fits = dict(diagnose(fk, qs, ids, corners))
    for key, f in fits.items():
        r = 1000 * f["res"]
        params = [f"{j} {np.degrees(v):+.1f}°" for j, v in f["offsets_rad"].items()]
        if FITS[key][1] == "axis":
            params.append(f"tip {1000 * float(f['tip_shift_m'] @ fk.tool_axis_local):+.1f} mm")
        elif FITS[key][1] == "3d":
            params.append(f"tip {np.round(1000 * f['tip_shift_m'], 1).tolist()} mm")
        mark = "*" if key == fit_key else " "
        typer.echo(
            f"{mark}{key:13} {r.max():7.1f} {np.sqrt(np.mean(r**2)):7.1f}  {', '.join(params)}"
        )
    fit = fits[fit_key]
    p, R_rc = camera_in_robot(R_cb, t_cb, fit["R"], fit["t"])
    mean_pts = np.array([fit["pts"][ids == k].mean(0) for k in range(4)])
    truth = side_lengths(corners)
    table_z = yaml.safe_load(SCENE_CFG.read_text())["table"]["top_z"]
    res_mm = 1000 * fit["res"]
    typer.echo(f"touch residuals (mm): {np.round(res_mm, 1).tolist()}")
    typer.echo(
        "board sides from touches vs board (cm): "
        + ", ".join(
            f"{k + 1}-{(k + 1) % 4 + 1} {100 * a:.1f}/{100 * b:.1f}"
            for k, (a, b) in enumerate(zip(side_lengths(mean_pts), truth))
        )
    )
    if fit["offsets_rad"]:
        typer.echo(
            "joint offsets suggested by this session (validate on a second session before "
            "touching config/sim2real.yaml): "
            + ", ".join(f"{j} {np.degrees(v):+.1f}°" for j, v in fit["offsets_rad"].items())
        )
    typer.echo(f"board height {mean_pts[:, 2].mean() * 100:.1f} cm vs table {table_z * 100:.1f} cm")
    typer.echo(f"camera position (robot frame) {np.round(p, 4).tolist()} m, fovy {fovy:.1f} deg")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{cam_name}.json"
    record = {
        "R_cam_board": R_cb.tolist(),
        "t_cam_board": t_cb.tolist(),
        "pnp_rms_px": pnp_err,
        "fovy_deg": fovy,
        "touch_q_rad": qs.tolist(),
        "touch_corner": ids.tolist(),
        "R_robot_board": fit["R"].tolist(),
        "t_robot_board": fit["t"].tolist(),
        "tip_local": fit["tip_local"].tolist(),
        "fit": fit_key,
        "joint_offsets_rad": fit["offsets_rad"],
        "camera_pos": p.tolist(),
        "R_robot_cam": R_rc.tolist(),
        "touch_residual_m": fit["res"].tolist(),
        "board_square_m": bcfg["square_length_m"],
    }
    if refit is None:  # keep raw sessions; refits don't overwrite them
        out_json.write_text(json.dumps(record, indent=2) + "\n")
        typer.echo(f"session saved to {out_json} (re-fit offline with --refit)")
    if res_mm.max() > max_residual_mm and not typer.confirm(
        f"max residual {res_mm.max():.1f} mm > {max_residual_mm} mm; write the camera pose anyway?",
        default=False,
    ):
        typer.echo("not written")
        return
    out_yaml = EXTRINSICS_DIR / f"{cam_name}.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    pose = {
        "pos": [round(float(v), 4) for v in p],
        "xyaxes": [round(float(v), 5) for v in mujoco_xyaxes(R_rc)],
        "fovy_deg": round(fovy, 2),
        "source": f"camcal-extrinsics {date.today()} (touch residual max {res_mm.max():.1f} mm)",
    }
    out_yaml.write_text(
        "# Camera pose in the follower base frame (MuJoCo camera convention). GENERATED by\n"
        "# `uv run camcal-extrinsics`; `uv run scene build` uses it over the scene_bench.yaml\n"
        "# placeholder.\n" + yaml.safe_dump(pose, sort_keys=False)
    )
    typer.echo(f"written to {out_yaml} -> run `uv run scene build` and `uv run scene render`")


if __name__ == "__main__":
    app()
