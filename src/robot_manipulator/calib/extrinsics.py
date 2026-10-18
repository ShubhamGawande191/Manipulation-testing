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


TIP_PRIOR_M = 0.01  # prior sigma of the tip-offset correction vs the 1 mm touch noise


def fit_touches(
    fk: TipFK, qs: np.ndarray, corner_ids: np.ndarray, corners: np.ndarray, fit_tip: bool = True
) -> dict:
    """Board pose in the robot frame (and optionally the fingertip point) from corner touches.

    ``qs`` (n, 6) joint vectors, ``corner_ids`` (n,) index into ``corners`` (4, 3, board frame).
    With touches at several wrist angles per corner the tip offset is identifiable; it is
    regularized toward the model's nominal tip (sigma ``TIP_PRIOR_M``).
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    poses = fk.poses(qs)
    tip0 = fk.tip_local_nominal
    dst = np.array([p + R @ tip0 for p, R in poses])
    R0, t0, _ = kabsch(corners[corner_ids], dst)

    def unpack(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        tip = tip0 + (x[6:9] if fit_tip else 0.0)
        return R, x[3:6], tip

    def residual(x):
        R, t, tip = unpack(x)
        pred = corners[corner_ids] @ R.T + t
        meas = np.array([p + Rb @ tip for p, Rb in poses])
        r = (meas - pred).ravel()
        if fit_tip:
            r = np.r_[r, 0.001 * x[6:9] / TIP_PRIOR_M]
        return r

    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), t0, np.zeros(3 if fit_tip else 0)]
    sol = least_squares(residual, x0)
    R, t, tip = unpack(sol.x)
    meas = np.array([p + Rb @ tip for p, Rb in poses])
    res = np.linalg.norm(meas - (corners[corner_ids] @ R.T + t), axis=1)
    return {"R": R, "t": t, "tip_local": tip, "tip_shift_m": tip - tip0, "res": res, "pts": meas}


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

    jm = JointMap.from_yaml(REPO / "config" / "sim2real.yaml")
    robot = SOFollower(build_follower_config(port=port, robot_id=robot_id, cameras=False))
    robot.bus.connect()
    qs, ids = [], []
    hints = ("tool pointing straight down", "wrist tilted, arm reaching differently")
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
    port: str = typer.Option("/dev/so101_follower"),
    robot_id: str = typer.Option("my_so101_follower"),
    per_corner: int = typer.Option(2, help="touches per corner (vary the wrist between them)"),
    fit_tip: bool = typer.Option(True, help="also fit the fingertip point (needs per_corner >= 2)"),
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
    fit = fit_touches(fk, qs, ids, corners, fit_tip=fit_tip and per_corner >= 2)
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
    typer.echo(f"fingertip correction vs model (mm): {np.round(1000 * fit['tip_shift_m'], 1)}")
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
