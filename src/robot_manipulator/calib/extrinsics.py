"""Camera extrinsics in the robot frame: ChArUco board seen by the camera + touched by the arm.

    uv run camcal-extrinsics workspace          # interactive, operator at the arm (torque OFF)

1. The camera image gives the board pose in the camera frame (``solvePnP`` with
   ``config/camera_intrinsics/<cam>.yaml`` and ``config/charuco_board.yaml``). The 4 outer grid
   corners are drawn numbered 1-4 in ``runs/extrinsics/<cam>_corners.png``.
2. The operator guides the follower's **closed** fingertips onto corners 1..4 and presses Enter
   at each. The tool offers to close the gripper first (gripper motor only; the PincOpen isn't
   back-drivable). The calibrated joint map + twin FK give each touch point, and Kabsch fits the
   board pose in the robot frame. The touch point is the fingertip tip, ``TIP_OFFSET_M`` beyond
   the EE site ``gripperframe`` along the tool axis.
3. Chaining the two gives the camera pose in the robot frame. It's written to
   ``config/scene_bench.yaml`` (``workspace_camera.pos`` / ``xyaxes`` / ``fovy_deg``). Then rebuild
   the scene (``uv run scene build``) and compare views (``uv run scene render``).

Checks printed: Kabsch residual per corner (touch accuracy) and board height vs the table top.
"""

from __future__ import annotations

import json

import numpy as np
import typer
import yaml

from robot_manipulator.calib.camera import REPO, detect, make_board
from robot_manipulator.cameras import apply_controls, load_camera_config

INTRINSICS_DIR = REPO / "config" / "camera_intrinsics"
SCENE_CFG = REPO / "config" / "scene_bench.yaml"
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


def tip_point(model, data, q: np.ndarray) -> np.ndarray:
    """Closed-fingertip tip in the robot frame for joint vector ``q`` (rad, CMD order)."""
    import mujoco

    from robot_manipulator.sim.model import EE_SITE, JOINTS, apply_joint_couplings, joint_qpos_adr

    data.qpos[:] = model.qpos0
    data.qpos[joint_qpos_adr(model, JOINTS)] = q
    apply_joint_couplings(model, data)
    mujoco.mj_kinematics(model, data)
    tool_x = data.body("pinc_base").xmat.reshape(3, 3)[:, 0]  # palm -> fingertips
    return data.site(EE_SITE).xpos + TIP_OFFSET_M * tool_x


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


@app.command()
def main(
    cam_name: str = typer.Argument("workspace"),
    port: str = typer.Option("/dev/so101_follower"),
    robot_id: str = typer.Option("my_so101_follower"),
) -> None:
    import cv2
    import mujoco

    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.model import load_model

    intr_path = INTRINSICS_DIR / f"{cam_name}.yaml"
    if not intr_path.exists():
        raise typer.BadParameter(f"{intr_path} missing: run `uv run camcal intrinsics {cam_name}`")
    intr = yaml.safe_load(intr_path.read_text())
    K = np.array(intr["camera_matrix"])
    dist = np.array(intr["dist_coeffs"])
    board, bcfg = make_board()

    # 1. board in the camera frame
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
    R_cb, _ = cv2.Rodrigues(rvec)
    t_cb = tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    pnp_err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, 1))))
    corners = board_outer_corners(board)
    px, _ = cv2.projectPoints(corners, rvec, tvec, K, dist)
    show = frame.copy()
    for k, (u, v) in enumerate(px.reshape(-1, 2), start=1):
        cv2.circle(show, (int(u), int(v)), 8, (0, 0, 255), 2)
        cv2.putText(
            show, str(k), (int(u) + 10, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
    img_path = OUT_DIR / f"{cam_name}_corners.png"
    cv2.imwrite(str(img_path), show)
    typer.echo(
        f"board detected ({len(ci)} corners, PnP rms {pnp_err:.2f} px); "
        f"corners labelled in {img_path}"
    )

    # 2. touch the corners with the follower (torque must be off)
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config

    jm = JointMap.from_yaml(REPO / "config" / "sim2real.yaml")
    model = load_model()
    mdata = mujoco.MjData(model)
    robot = SOFollower(build_follower_config(port=port, robot_id=robot_id, cameras=False))
    robot.bus.connect()
    touched = []
    try:
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
        if any(v for k, v in torque.items() if k != "gripper"):
            raise RuntimeError("follower torque is on: run `uv run teleop --park-only` first")
        grip = robot.bus.sync_read("Present_Position")["gripper"]
        if grip > GRIPPER_OPEN_ABOVE:
            input(f"Gripper is open ({grip:.0f}). Enter closes it (gripper motor only)… ")
            _close_gripper(robot, jm.gripper.lerobot_closed)
            typer.echo(f"  gripper now {robot.bus.sync_read('Present_Position')['gripper']:.0f}")
        for k in range(1, 5):
            input(f"Put the closed fingertips on corner {k} (see the image), then press Enter… ")
            pos = robot.bus.sync_read("Present_Position")
            q = jm.lerobot_to_sim({f"{j}.pos": float(v) for j, v in pos.items()})
            touched.append(tip_point(model, mdata, q))
            typer.echo(f"  corner {k}: robot frame {np.round(touched[-1], 4).tolist()} m")
    finally:
        robot.bus.disconnect(disable_torque=False)
    touched = np.array(touched)

    # 3. chain
    R_rb, t_rb, res = kabsch(corners, touched)
    p, R_rc = camera_in_robot(R_cb, t_cb, R_rb, t_rb)
    table_z = yaml.safe_load(SCENE_CFG.read_text())["table"]["top_z"]
    board_z = float(touched[:, 2].mean())
    fovy = float(np.degrees(2 * np.arctan(intr["image_size"][1] / 2 / K[1, 1])))
    typer.echo(f"touch residuals per corner (mm): {np.round(1000 * res, 1).tolist()}")
    typer.echo(f"board height {board_z * 100:.1f} cm vs table top {table_z * 100:.1f} cm")
    typer.echo(f"camera position (robot frame) {np.round(p, 4).tolist()} m, fovy {fovy:.1f} deg")

    cfg = yaml.safe_load(SCENE_CFG.read_text())
    cfg["workspace_camera"].update(
        {
            "pos": [round(float(v), 4) for v in p],
            "xyaxes": [round(float(v), 5) for v in mujoco_xyaxes(R_rc)],
            "fovy_deg": round(fovy, 2),
            "source": f"camcal-extrinsics (touch residual max {1000 * res.max():.1f} mm)",
        }
    )
    SCENE_CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT_DIR / f"{cam_name}.json").write_text(
        json.dumps(
            {
                "R_cam_board": R_cb.tolist(),
                "t_cam_board": t_cb.tolist(),
                "R_robot_board": R_rb.tolist(),
                "t_robot_board": t_rb.tolist(),
                "camera_pos": p.tolist(),
                "R_robot_cam": R_rc.tolist(),
                "touch_residual_m": res.tolist(),
                "pnp_rms_px": pnp_err,
                "board_square_m": bcfg["square_length_m"],
            },
            indent=2,
        )
        + "\n"
    )
    typer.echo(
        "written to config/scene_bench.yaml -> run `uv run scene build` and `uv run scene render`"
    )


if __name__ == "__main__":
    app()
