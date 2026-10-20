"""Automatic calibration with the wrist camera: no touching, no manual board moves.

The ChArUco board lies flat on the table. The follower moves its wrist camera through planned
viewpoints over the board (comfortable reach, never stretched) and at each one stores the wrist
image, the workspace image and the joint readings. One bundle adjustment then solves for

- the board pose in the robot frame,
- where the wrist camera sits on the gripper (hand-eye),
- encoder offsets of shoulder_lift, elbow_flex and wrist_flex (pan and wrist_roll offsets are not
  observable this way: pan rotates everything rigidly, roll is absorbed by the camera mount),

by minimising the reprojection error of every detected board corner in every wrist view. The
wrist intrinsics come from the same views. The workspace camera pose then follows from its own
view of the board.

    uv run camcal-wrist plan                    # sim: pick viewpoints, check reach/collisions
    uv run camcal-wrist run --backend sim       # dry rehearsal of the motion in the twin
    uv run camcal-wrist run --backend real --live   # operator present; ~3 min
    uv run camcal-wrist solve runs/wristcal/<ts> # -> report; --write to save poses/intrinsics

The first board-pose guess comes from a touch session (``runs/extrinsics/*.json``); a few cm off
is fine, the planner only needs to know roughly where to look.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import typer
import yaml

from robot_manipulator.calib.camera import REPO, detect, make_board
from robot_manipulator.sim.model import (
    JOINTS,
    apply_joint_couplings,
    joint_qpos_adr,
    joint_ranges,
    load_model,
)

OUT_DIR = REPO / "runs" / "wristcal"
WRIST_CAM = "wrist_cam"
IMG_W, IMG_H = 640, 480
TABLE_Z = -0.0275
OFFSET_JOINTS = (1, 2, 3)  # shoulder_lift, elbow_flex, wrist_flex
MARGIN = 0.1  # rad inside the joint limits (the harness safety box is 0.05)
MIN_PX = 20  # projected corners at least this far inside the image
DIST_RANGE = (0.16, 0.46)  # camera to board centre, m
MAX_VIEW_DEG = 55.0  # angle between the camera axis and the board normal
MAX_REACH = 0.21  # shoulder_lift axis to wrist_flex axis, m (stretched = 0.251)
MIN_CLEAR = 0.07  # fingertip / camera height above the table, m
FINGER_V = 0.62 * IMG_H  # closed fingers cover the bottom-centre of the wrist image below this
MIN_CORNERS = 14  # of the 24 inner ChArUco corners, visible per planned view
BA_MIN_CORNERS = 8  # detected corners for a view to enter the bundle adjustment
LEADER_Y = 0.17  # nothing of the follower beyond this (the leader stows at y ~ 0.23+)
# wrist_roll kept within +-60 deg of the arm's natural roll (-80 deg here: gripper upright), so the
# gripper and camera cables are never wound round the wrist
ROLL_RANGE = (np.radians(-140.0), np.radians(-20.0))

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------- geometry
def _rot(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(v).as_matrix()


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


MJ_TO_CV = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera looks along -z, y up; OpenCV +z, y down


class WristRig:
    """FK of the wrist camera (OpenCV convention) for joint vectors, with optional corrections."""

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        self.m = model if model is not None else load_model("scene_bench.xml")
        self.d = mujoco.MjData(self.m)
        self.qadr = joint_qpos_adr(self.m, JOINTS)
        cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)
        self.body = int(self.m.cam_bodyid[cam])
        R_mj = np.zeros(9)
        mujoco.mju_quat2Mat(R_mj, self.m.cam_quat[cam])
        self.R_mount0 = R_mj.reshape(3, 3) @ MJ_TO_CV  # camera axes in its body frame
        self.t_mount0 = self.m.cam_pos[cam].copy()
        # planning intrinsics: the wrist camera's own calibration if present, else the workspace
        # camera's (same eMeet C950 model; the MJCF fovy of 61 deg is wider than the real lens)
        for name in ("wrist", "workspace"):
            f = REPO / "config" / "camera_intrinsics" / f"{name}.yaml"
            if f.exists():
                self.K0 = np.array(yaml.safe_load(f.read_text())["camera_matrix"])
                break
        else:
            fy = (IMG_H / 2) / np.tan(np.radians(float(self.m.cam_fovy[cam])) / 2)
            self.K0 = np.array([[fy, 0, IMG_W / 2], [0, fy, IMG_H / 2], [0, 0, 1.0]])
        self._lift = self.m.joint("shoulder_lift").id
        self._wflex = self.m.joint("wrist_flex").id
        self._site = self.m.site("gripperframe").id
        root = self.m.body_rootid[self.m.body("gripper").id]
        self._arm = np.flatnonzero(self.m.body_rootid == root)  # not the leader's static body

    def set(self, q: np.ndarray) -> None:
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.qadr] = q
        apply_joint_couplings(self.m, self.d)
        mujoco.mj_kinematics(self.m, self.d)

    def camera(self, q, mount: tuple[np.ndarray, np.ndarray] | None = None):
        """(R_robot_cam, p_robot_cam), OpenCV axes; ``mount`` = camera (R, t) in its body."""
        self.set(np.asarray(q, dtype=float))
        R_b = self.d.xmat[self.body].reshape(3, 3)
        p_b = self.d.xpos[self.body]
        R_m, t_m = mount if mount is not None else (self.R_mount0, self.t_mount0)
        return R_b @ R_m, p_b + R_b @ t_m

    def reach(self) -> float:
        return float(np.linalg.norm(self.d.xanchor[self._wflex] - self.d.xanchor[self._lift]))

    def clear_of_obstacles(self) -> bool:
        mujoco.mj_forward(self.m, self.d)
        if any(c.dist < 0 for c in self.d.contact[: self.d.ncon]):  # any penetration
            return False
        body_pts = self.d.xpos[self._arm]
        return bool(
            self.d.site_xpos[self._site][2] > TABLE_Z + MIN_CLEAR
            and body_pts[:, 2].min() > TABLE_Z + 0.02
            and body_pts[:, 1].max() < LEADER_Y
        )


def project(K, R_rc, p_rc, X_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection (no distortion) of robot-frame points; returns (uv, depth)."""
    Xc = (X_robot - p_rc) @ R_rc
    z = Xc[:, 2]
    uv = (Xc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z


# ---------------------------------------------------------------- planning
def board_points(board) -> tuple[np.ndarray, np.ndarray]:
    """(outer 4 corners, inner ChArUco corners) in the board frame."""
    from robot_manipulator.calib.extrinsics import board_outer_corners

    return board_outer_corners(board), np.asarray(board.getChessboardCorners(), dtype=float)


def visible_corners(rig: WristRig, q, R_rb, t_rb, inner) -> int:
    """Inner board corners the wrist camera would see at ``q``: in front, inside the image with a
    margin, and not behind the closed fingers (bottom-centre of the wrist image)."""
    R_rc, p_rc = rig.camera(q)
    uv, z = project(rig.K0, R_rc, p_rc, inner @ R_rb.T + t_rb)
    ok = (z > 0.05) & (uv[:, 0] > MIN_PX) & (uv[:, 0] < IMG_W - MIN_PX) & (uv[:, 1] > MIN_PX)
    ok &= uv[:, 1] < IMG_H - MIN_PX
    fingers = (uv[:, 0] > 0.25 * IMG_W) & (uv[:, 0] < 0.75 * IMG_W) & (uv[:, 1] > FINGER_V)
    return int((ok & ~fingers).sum())


def view_ok(rig: WristRig, q, R_rb, t_rb, inner, centre) -> bool:
    if visible_corners(rig, q, R_rb, t_rb, inner) < MIN_CORNERS:
        return False
    R_rc, p_rc = rig.camera(q)
    c = centre @ R_rb.T + t_rb
    dist = np.linalg.norm(c - p_rc)
    view = np.degrees(np.arccos(abs(float(R_rc[:, 2] @ R_rb[:, 2]))))
    return DIST_RANGE[0] < dist < DIST_RANGE[1] and view < MAX_VIEW_DEG and rig.reach() < MAX_REACH


def path_ok(rig: WristRig, qa, qb, n: int = 12) -> bool:
    for s in np.linspace(0, 1, n):
        rig.set(qa + s * (qb - qa))
        if not rig.clear_of_obstacles():
            return False
    return True


def aim_camera(rig: WristRig, look_at: np.ndarray, rng, lo, hi) -> np.ndarray | None:
    """Joints putting the wrist camera at a random spot above ``look_at`` (distance, elevation and
    azimuth drawn at random), looking at a point within 3 cm of it. None if no solution."""
    from scipy.optimize import least_squares

    d = rng.uniform(0.24, 0.44)
    el, az = np.radians(rng.uniform(40, 85)), rng.uniform(-np.pi, np.pi)
    cam = look_at + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    aim = look_at + rng.normal(0, 0.015, 3) * np.array([1, 1, 0])
    axis = (aim - cam) / np.linalg.norm(aim - cam)
    q0 = rng.uniform(lo[:5] + MARGIN, hi[:5] - MARGIN)
    q0[0] = np.clip(np.arctan2(cam[1], cam[0]), lo[0] + MARGIN, hi[0] - MARGIN)  # face it

    def res(x):
        R_rc, p = rig.camera(np.r_[x, 0.0])
        return np.r_[p - cam, 0.2 * (R_rc[:, 2] - axis)]

    sol = least_squares(res, q0, bounds=(lo[:5] + MARGIN, hi[:5] - MARGIN), max_nfev=60)
    if np.linalg.norm(sol.fun[:3]) > 0.01 or np.linalg.norm(sol.fun[3:]) > 0.2 * 0.1:
        return None
    return np.r_[sol.x, 0.0]  # gripper closed


def plan_views(
    R_rb, t_rb, n: int = 24, seed: int = 0, rig: WristRig | None = None, n_samples: int = 2000
) -> np.ndarray:
    """``n`` diverse, reachable, collision-free joint vectors whose wrist view frames the board,
    ordered as a short path from HOME with every leg checked for clearance."""
    from robot_manipulator.sim.envs.reach import HOME_ARM

    rig = rig or WristRig()
    board, _ = make_board()
    outer, inner = board_points(board)
    centre = outer.mean(0)
    lo, hi = joint_ranges(rig.m, JOINTS)
    lo[4], hi[4] = max(lo[4], ROLL_RANGE[0] - MARGIN), min(hi[4], ROLL_RANGE[1] + MARGIN)
    rng = np.random.default_rng(seed)
    c_world = centre @ R_rb.T + t_rb
    cands = []
    for _ in range(n_samples):
        q = aim_camera(rig, c_world, rng, lo, hi)
        if q is None or not view_ok(rig, q, R_rb, t_rb, inner, centre):
            continue
        rig.set(q)
        if rig.clear_of_obstacles():
            cands.append(q)
    if len(cands) < n:
        raise RuntimeError(f"only {len(cands)} viewpoints found; is the board guess right?")
    C = np.array(cands)
    feat = C[:, :5] / (hi[:5] - lo[:5])
    home = np.r_[HOME_ARM, 0.0]
    chosen = [int(np.argmin(np.abs(C[:, :5] - home[:5]).sum(1)))]
    dist = np.linalg.norm(feat - feat[chosen[0]], axis=1)
    while len(chosen) < n:  # farthest-point: spread the joint angles (identifiability)
        k = int(np.argmax(dist))
        chosen.append(k)
        dist = np.minimum(dist, np.linalg.norm(feat - feat[k], axis=1))
    order, left, last = [], set(chosen), home
    while left:  # nearest-neighbour tour with clear straight joint-space legs
        ranked = sorted(left, key=lambda i: float(np.abs(C[i, :5] - last[:5]).max()))
        nxt = next((i for i in ranked if path_ok(rig, last, C[i])), None)
        if nxt is None:
            break  # the rest is unreachable from here without a detour; drop it
        order.append(nxt)
        left.remove(nxt)
        last = C[nxt]
    return C[order]


def planar_board_pose(corners: np.ndarray, touched_xy: np.ndarray, z: float):
    """Board lying flat at height ``z``: fit only x, y and yaw (2-D Procrustes, reflection allowed
    because the board's printed side faces up) of ``corners`` (n, 3) onto ``touched_xy`` (n, 2)."""
    src, dst = corners[:, :2], touched_xy
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    M = Vt.T @ U.T  # orthogonal 2x2; det -1 = the board frame's z points into the table
    R = np.eye(3)
    R[:2, :2] = M
    R[2, 2] = np.linalg.det(M)
    t = np.r_[cd - M @ cs, z]
    return R, t


def board_guess(session: Path) -> tuple[np.ndarray, np.ndarray]:
    """Board pose from a touch session, using that it lies flat on the table: the nominal
    fingertip xy of each touch (mean per corner), planar fit. Robust to the few-cm FK errors that
    make a full 6-DOF touch fit tilt the board."""
    from robot_manipulator.calib.extrinsics import TipFK, board_outer_corners

    s = json.loads(Path(session).read_text())
    board, _ = make_board()
    corners = board_outer_corners(board)
    fk = TipFK(load_model())
    qs, ids = np.array(s["touch_q_rad"]), np.array(s["touch_corner"])
    tips = np.array([fk.tip(q) for q in qs])
    xy = np.array([tips[ids == k, :2].mean(0) for k in range(4)])
    return planar_board_pose(corners, xy, TABLE_Z + 0.001)


def board_from_camera(cam_name: str = "workspace") -> tuple[np.ndarray, np.ndarray]:
    """Board pose in the robot frame from the calibrated overhead camera: PnP of the board in the
    camera, chained with ``config/extrinsics/<cam>.yaml``. Good to a few mm, enough to plan."""
    import cv2

    from robot_manipulator.calib.extrinsics import EXTRINSICS_DIR, INTRINSICS_DIR, _grab

    intr = yaml.safe_load((INTRINSICS_DIR / f"{cam_name}.yaml").read_text())
    ext = yaml.safe_load((EXTRINSICS_DIR / f"{cam_name}.yaml").read_text())
    K, dist = np.array(intr["camera_matrix"]), np.array(intr["dist_coeffs"])
    board, _ = make_board()
    cc, ci = None, None
    for _ in range(10):
        cc, ci = detect(board, cv2.cvtColor(_grab(cam_name), cv2.COLOR_BGR2GRAY), min_corners=6)
        if cc is not None:
            break
    if cc is None:
        raise RuntimeError(f"{cam_name} camera does not see the board (fold the arm away?)")
    P, uv = board.matchImagePoints(cc, ci)
    ok, rvec, tvec = cv2.solvePnP(P, uv, K, dist)
    R_cb, _ = cv2.Rodrigues(rvec)
    x, y_up = np.array(ext["xyaxes"][:3]), np.array(ext["xyaxes"][3:])
    R_rc = np.c_[x, -y_up, np.cross(x, -y_up)]  # OpenCV camera axes in the robot frame
    return R_rc @ R_cb, R_rc @ tvec.ravel() + np.array(ext["pos"])


# ---------------------------------------------------------------- solving
def solve(
    rig: WristRig,
    qs: np.ndarray,
    obs: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    R_rb0: np.ndarray,
    t_rb0: np.ndarray,
    offsets: tuple[int, ...] = OFFSET_JOINTS,
) -> dict:
    """Bundle adjustment over board pose, camera mount and joint offsets.

    ``obs[i]`` = (board-frame corner points (m, 3), undistorted pixel coordinates (m, 2)) seen in
    view ``i`` taken at joints ``qs[i]``. ``K`` is the camera matrix the pixels refer to.
    """
    from scipy.optimize import least_squares

    n_off = len(offsets)

    def unpack(x):
        R_rb, t_rb = _rot(x[0:3]) @ R_rb0, t_rb0 + x[3:6]
        R_m, t_m = _rot(x[6:9]) @ rig.R_mount0, rig.t_mount0 + x[9:12]
        off = np.zeros(6)
        off[list(offsets)] = x[12 : 12 + n_off]
        return R_rb, t_rb, (R_m, t_m), off

    def residual(x):
        R_rb, t_rb, mount, off = unpack(x)
        out = []
        for q, (P, uv) in zip(qs, obs):
            R_rc, p_rc = rig.camera(q + off, mount)
            pred, z = project(K, R_rc, p_rc, P @ R_rb.T + t_rb)
            out.append(np.where(z[:, None] > 0, pred - uv, 1e3).ravel())
        return np.concatenate(out)

    x0 = np.zeros(12 + n_off)
    sol = least_squares(residual, x0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    R_rb, t_rb, mount, off = unpack(sol.x)
    r = sol.fun.reshape(-1, 2)
    per_view, k = [], 0
    for _, uv in obs:
        per_view.append(float(np.sqrt(np.mean(np.sum(r[k : k + len(uv)] ** 2, 1)))))
        k += len(uv)
    # 1-sigma parameter uncertainty from the Jacobian at the solution
    J = sol.jac
    sigma2 = float(np.sum(sol.fun**2) / max(len(sol.fun) - len(sol.x), 1))
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(sol.x), np.nan)
    return {
        "R_robot_board": R_rb,
        "t_robot_board": t_rb,
        "mount": mount,
        "mount_shift_m": float(np.linalg.norm(sol.x[9:12])),
        "mount_rot_deg": float(np.degrees(np.linalg.norm(sol.x[6:9]))),
        "offsets_rad": {JOINTS[j]: float(off[j]) for j in offsets},
        "offsets_std_rad": {JOINTS[j]: float(std[12 + i]) for i, j in enumerate(offsets)},
        "rms_px": float(np.sqrt(np.mean(np.sum(r**2, 1)))),
        "per_view_rms_px": per_view,
    }


def solve_joint(
    rig: WristRig, sessions: list[dict], offsets=OFFSET_JOINTS, fit_scale: bool = False
) -> dict:
    """Bundle adjustment over several sessions: one camera mount and one set of joint offsets,
    one board pose per session (the board moves between sessions, the camera mount does not).
    ``sessions[k]`` = ``load_session`` output with ``R0``/``t0`` (initial board pose).
    ``fit_scale``: also fit each board's print scale (its square size is a hand measurement; the
    arm's link lengths give the absolute scale)."""
    from scipy.optimize import least_squares

    n_s, n_off = len(sessions), len(offsets)
    m0 = 6 * n_s
    sc0 = m0 + 6 + n_off  # per-session log scale, after the offsets

    def unpack(x):
        boards = [
            (_rot(x[6 * k : 6 * k + 3]) @ s["R0"], s["t0"] + x[6 * k + 3 : 6 * k + 6])
            for k, s in enumerate(sessions)
        ]
        mount = (_rot(x[m0 : m0 + 3]) @ rig.R_mount0, rig.t_mount0 + x[m0 + 3 : m0 + 6])
        off = np.zeros(6)
        off[list(offsets)] = x[m0 + 6 : m0 + 6 + n_off]
        scales = np.exp(x[sc0 : sc0 + n_s]) if fit_scale else np.ones(n_s)
        return boards, mount, off, scales

    def residual(x):
        boards, mount, off, scales = unpack(x)
        out = []
        for (R_rb, t_rb), s, sc in zip(boards, sessions, scales):
            for q, (P, uv) in zip(s["qs"], s["obs"]):
                R_rc, p_rc = rig.camera(q + off, mount)
                pred, z = project(s["K"], R_rc, p_rc, (sc * P) @ R_rb.T + t_rb)
                out.append(np.where(z[:, None] > 0, pred - uv, 1e3).ravel())
        return np.concatenate(out)

    sol = least_squares(
        residual,
        np.zeros(sc0 + (n_s if fit_scale else 0)),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
    )
    boards, mount, off, scales = unpack(sol.x)
    r = sol.fun.reshape(-1, 2)
    per_session, k = [], 0
    for s in sessions:
        n = sum(len(uv) for _, uv in s["obs"])
        per_session.append(float(np.sqrt(np.mean(np.sum(r[k : k + n] ** 2, 1)))))
        k += n
    sigma2 = float(np.sum(sol.fun**2) / max(len(sol.fun) - len(sol.x), 1))
    try:
        std = np.sqrt(np.clip(np.diag(np.linalg.inv(sol.jac.T @ sol.jac) * sigma2), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(sol.x), np.nan)
    return {
        "boards": boards,
        "mount": mount,
        "mount_shift_m": float(np.linalg.norm(sol.x[m0 + 3 : m0 + 6])),
        "mount_rot_deg": float(np.degrees(np.linalg.norm(sol.x[m0 : m0 + 3]))),
        "offsets_rad": {JOINTS[j]: float(off[j]) for j in offsets},
        "offsets_std_rad": {JOINTS[j]: float(std[m0 + 6 + i]) for i, j in enumerate(offsets)},
        "rms_px_per_session": per_session,
        "board_scales": scales.tolist(),
    }


def load_session(session_dir: Path) -> dict:
    """Wrist views of a session: joints, undistorted corner observations, own intrinsics. The
    board spec is the session's ``board.yaml`` (saved by ``run``), else the current config."""
    import cv2

    from robot_manipulator.calib.camera import calibrate

    s = json.loads((session_dir / "session.json").read_text())
    spec = session_dir / "board.yaml"
    board, _ = make_board(spec) if spec.exists() else make_board()
    views, qs = [], []
    for v in s["views"]:
        g = cv2.imread(str(session_dir / v["wrist"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        # the closed fingers hide the board's near rows: 8 corners per view are plenty for the BA
        cc, ci = detect(board, g, min_corners=BA_MIN_CORNERS)
        if cc is not None:
            views.append(g)
            qs.append(np.array(v["q_rad"]))
    if len(views) < 8:
        raise ValueError(f"{session_dir}: only {len(views)} wrist views see the board")
    intr = calibrate(board, views, (IMG_W, IMG_H))  # uses views with >= 12 corners
    K, dist = np.array(intr["camera_matrix"]), np.array(intr["dist_coeffs"])
    obs = []
    for g in views:
        cc, ci = detect(board, g, min_corners=BA_MIN_CORNERS)
        P, uv = board.matchImagePoints(cc, ci)
        und = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        obs.append((P.reshape(-1, 3).astype(float), und))
    return {
        "dir": session_dir,
        "n_views": len(s["views"]),
        "qs": np.array(qs),
        "obs": obs,
        "K": K,
        "intr": intr,
        "board": board,
    }


def _initial_board(rig, qs, obs, K) -> tuple[np.ndarray, np.ndarray]:
    """Board pose from the median single-view PnP with the nominal mount and FK."""
    import cv2

    cands = []
    for q, (P, uv) in zip(qs, obs):
        ok, rvec, tvec = cv2.solvePnP(P, uv, K, None)
        if not ok:
            continue
        R_cb, _ = cv2.Rodrigues(rvec)
        R_rc, p_rc = rig.camera(q)
        cands.append((R_rc @ R_cb, R_rc @ tvec.ravel() + p_rc))
    ts = np.array([t for _, t in cands])
    k = int(np.argmin(np.linalg.norm(ts - np.median(ts, 0), axis=1)))
    return cands[k]


# ---------------------------------------------------------------- CLI
def _latest_touch_session() -> Path:
    files = sorted((REPO / "runs" / "extrinsics").glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise typer.BadParameter("no touch session in runs/extrinsics to guess the board pose from")
    return files[-1]


@app.command()
def plan(
    n: int = typer.Option(24, help="number of viewpoints"),
    seed: int = 0,
    board_from: Path | None = typer.Option(None, help="touch session json (default: latest)"),
    from_camera: bool = typer.Option(
        False,
        "--board-from-camera",
        help="locate the board with the calibrated overhead camera instead of touches",
    ),
) -> None:
    """Pick viewpoints in the twin and save them to runs/wristcal/plan.json."""
    if from_camera:
        session = "workspace camera"
        R_rb, t_rb = board_from_camera()
    else:
        session = board_from or _latest_touch_session()
        R_rb, t_rb = board_guess(session)
    t0 = time.monotonic()
    poses = plan_views(R_rb, t_rb, n=n, seed=seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "plan.json"
    out.write_text(
        json.dumps(
            {
                "board_from": str(session),
                "R_robot_board": R_rb.tolist(),
                "t_robot_board": t_rb.tolist(),
                "poses_rad": poses.tolist(),
                "seed": seed,
            },
            indent=1,
        )
    )
    deg = np.degrees(poses[:, :5])
    typer.echo(
        f"{len(poses)} viewpoints in {time.monotonic() - t0:.0f}s -> {out}\n"
        f"board centre guess {np.round(t_rb + R_rb @ np.array([0.1, 0.14, 0]), 3).tolist()} m\n"
        f"joint spread (deg, min..max): "
        + ", ".join(f"{j} {a:.0f}..{b:.0f}" for j, a, b in zip(JOINTS, deg.min(0), deg.max(0)))
    )


class _Cams:
    """Both cameras kept open for the whole run (no 1.5 s warm-up per shot)."""

    def __init__(self, names=("wrist", "workspace")) -> None:
        import cv2

        from robot_manipulator.cameras import apply_controls, load_camera_config

        cfg = load_camera_config()
        self.caps = {}
        for name in names:
            apply_controls(cfg[name]["path"], cfg[name].get("controls", {}))
            self.caps[name] = cv2.VideoCapture(cfg[name]["path"])
        time.sleep(1.5)

    def grab(self, name: str):
        cap = self.caps[name]
        for _ in range(5):  # drop buffered frames: the image must be from *after* the arm settled
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"camera {name} returned no frame")
        return frame

    def close(self) -> None:
        for c in self.caps.values():
            c.release()


@app.command()
def run(
    backend: str = typer.Option("sim", help="sim | real"),
    live: bool = typer.Option(False, "--live/--dry-run", help="real: move the arm"),
    max_vel: float = typer.Option(0.4, help="rad/s cap per joint"),
    hold_s: float = typer.Option(1.0, help="settle time at each viewpoint"),
    plan_path: Path = typer.Option(OUT_DIR / "plan.json"),
    port: str = typer.Option("/dev/so101_follower"),
    stow_leader: bool = typer.Option(True, "--stow-leader/--no-stow-leader"),
) -> None:
    """Visit the planned viewpoints; save joints + wrist/workspace images to runs/wristcal/<ts>/."""
    import cv2

    from robot_manipulator.exec.scheduler import SimClock, WallClock
    from robot_manipulator.harness.backend import LeRobotBackend, SimBackend
    from robot_manipulator.harness.cli import leader_clear
    from robot_manipulator.harness.policies.glide import GlideTo
    from robot_manipulator.harness.runner import run_skill
    from robot_manipulator.harness.safety_layer import SafetyFilter
    from robot_manipulator.harness.skill import JointReached, Skill, TimeLimit
    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.envs.reach import HOME_ARM

    p = json.loads(plan_path.read_text())
    poses = np.array(p["poses_rad"])
    model = load_model()
    lo, hi = joint_ranges(model, JOINTS)
    home = np.clip(np.r_[HOME_ARM, 0.0], lo + 0.05, hi - 0.05)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{backend}"
    out.mkdir(parents=True, exist_ok=True)
    from robot_manipulator.calib.camera import BOARD_CFG

    (out / "board.yaml").write_text(BOARD_CFG.read_text())  # the solve needs this session's board
    capture = backend == "real"
    if backend == "sim":
        be: Any = SimBackend(model=model, hz=50.0)
        be.reset(home)
        clock: Any = SimClock()
    elif backend == "real":
        be = LeRobotBackend(
            port, JointMap.from_yaml(REPO / "config" / "sim2real.yaml"), dry_run=not live
        )
        if be.telemetry is not None:
            be.telemetry.source = "wristcal_live" if live else "wristcal_dry"
        clock = WallClock()
    else:
        raise typer.BadParameter("backend must be sim or real")
    cams = _Cams() if capture else None
    views: list[dict] = []
    try:
        with leader_clear(backend, live, stow_leader):
            if cams is not None:  # arm still folded: a clear overhead view of the board
                cv2.imwrite(str(out / "ws_start.png"), cams.grab("workspace"))
            for i, target in enumerate([home, *poses]):
                safety = SafetyFilter(lo=lo, hi=hi, margin=0.05, max_vel=max_vel, clock=clock)
                glide = Skill(
                    "wristcal_glide",
                    GlideTo(target, max_vel=max_vel, hz=50.0),
                    [JointReached(target, tol=0.05), TimeLimit(12.0)],
                    hz=50.0,
                )
                res = run_skill(glide, be, safety, goal={}, clock=clock)
                if backend == "real" and not live:
                    continue  # dry run: nothing moves
                err = np.abs(np.asarray(be.read()["q"]) - target)[:5].max()
                if res.status != "success" and err > 0.15:
                    typer.echo(f"view {i}: target not reached (err {err:.3f} rad); stopping")
                    break
                hold = Skill(
                    "wristcal_hold",
                    GlideTo(target, max_vel=max_vel, hz=50.0),
                    [TimeLimit(hold_s)],
                    hz=50.0,
                )
                run_skill(hold, be, safety, goal={}, clock=clock)
                if i == 0:
                    continue  # HOME is only the way in
                q = np.mean([np.asarray(be.read()["q"]) for _ in range(5)], axis=0)
                v = {"i": i, "q_rad": q.tolist(), "target_rad": target.tolist()}
                if cams is not None:
                    v["wrist"] = f"wrist_{i:02d}.png"
                    v["workspace"] = f"ws_{i:02d}.png"
                    cv2.imwrite(str(out / v["wrist"]), cams.grab("wrist"))
                    cv2.imwrite(str(out / v["workspace"]), cams.grab("workspace"))
                views.append(v)
                typer.echo(f"view {i}/{len(poses)} ok")
    except KeyboardInterrupt:
        typer.echo("interrupted; parking")
    finally:
        be.close()
        if cams is not None:
            cams.close()
    (out / "session.json").write_text(
        json.dumps(
            {"backend": backend, "live": live, "plan": str(plan_path), "views": views}, indent=1
        )
    )
    typer.echo(f"{len(views)} views -> {out}")


@app.command("solve")
def solve_cmd(
    session_dir: Path,
    write: bool = typer.Option(False, help="save wrist intrinsics + workspace camera pose"),
    holdout: float = typer.Option(0.2, help="fraction of views held out to validate the fit"),
) -> None:
    """Bundle-adjust board pose, wrist-camera mount and joint offsets; report and validate."""
    ses = load_session(session_dir)
    qs, obs, K, intr, board = ses["qs"], ses["obs"], ses["K"], ses["intr"], ses["board"]
    typer.echo(f"{len(qs)}/{ses['n_views']} wrist views with the board")
    typer.echo(f"wrist intrinsics: rms {intr['rms_px']:.2f} px, fx {K[0, 0]:.0f}")
    rig = WristRig()
    R0, t0 = _initial_board(rig, qs, obs, K)
    rng = np.random.default_rng(0)
    test = rng.random(len(obs)) < holdout
    rows = []
    for name, off in (("no offsets", ()), ("lift/elbow/wflex offsets", OFFSET_JOINTS)):
        fit = solve(rig, qs[~test], [o for o, t in zip(obs, test) if not t], K, R0, t0, off)
        held = solve_eval(rig, qs[test], [o for o, t in zip(obs, test) if t], K, fit)
        rows.append((name, fit, held))
        offs = ", ".join(
            f"{j} {np.degrees(v):+.2f}±{np.degrees(fit['offsets_std_rad'][j]):.2f}°"
            for j, v in fit["offsets_rad"].items()
        )
        typer.echo(
            f"{name:26} fit rms {fit['rms_px']:5.2f} px | held-out {held:5.2f} px | "
            f"mount Δ {1000 * fit['mount_shift_m']:.0f} mm / {fit['mount_rot_deg']:.1f}° | {offs}"
        )
    best = rows[-1][1]
    if write:
        _write_results(session_dir, best, intr, board)


@app.command("solve-joint")
def solve_joint_cmd(
    session_dirs: list[Path],
    fit_scale: bool = typer.Option(False, help="also fit each board's print scale"),
) -> None:
    """One fit over several sessions (shared mount + joint offsets); compare with each alone."""
    rig = WristRig()
    sessions = []
    for d in session_dirs:
        ses = load_session(d)
        ses["R0"], ses["t0"] = _initial_board(rig, ses["qs"], ses["obs"], ses["K"])
        alone = solve(rig, ses["qs"], ses["obs"], ses["K"], ses["R0"], ses["t0"])
        ses["alone"] = alone
        sessions.append(ses)
        offs = ", ".join(f"{j} {np.degrees(v):+.2f}°" for j, v in alone["offsets_rad"].items())
        typer.echo(f"{d.name}: {len(ses['qs'])} views, alone rms {alone['rms_px']:.2f} px | {offs}")
    fit = solve_joint(rig, sessions, fit_scale=fit_scale)
    typer.echo(
        "joint fit, rms per session (px): "
        + ", ".join(f"{r:.2f}" for r in fit["rms_px_per_session"])
        + f" | mount Δ {1000 * fit['mount_shift_m']:.0f} mm / {fit['mount_rot_deg']:.1f}°"
    )
    if fit_scale:
        typer.echo(
            "board scale vs its config: "
            + ", ".join(f"{d.name} {sc:.4f}" for d, sc in zip(session_dirs, fit["board_scales"]))
        )
    typer.echo(
        "joint offsets: "
        + ", ".join(
            f"{j} {np.degrees(v):+.2f}±{np.degrees(fit['offsets_std_rad'][j]):.2f}°"
            for j, v in fit["offsets_rad"].items()
        )
    )
    out = OUT_DIR / f"joint_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "sessions": [str(d) for d in session_dirs],
                "offsets_deg": {j: float(np.degrees(v)) for j, v in fit["offsets_rad"].items()},
                "offsets_std_deg": {
                    j: float(np.degrees(v)) for j, v in fit["offsets_std_rad"].items()
                },
                "rms_px_per_session": fit["rms_px_per_session"],
                "board_scales": fit["board_scales"],
            },
            indent=2,
        )
        + "\n"
    )
    typer.echo(f"-> {out}")


def solve_eval(rig, qs, obs, K, fit) -> float:
    """Reprojection RMS (px) of held-out views under a fitted model."""
    if not len(obs):
        return float("nan")
    off = np.zeros(6)
    for j, v in fit["offsets_rad"].items():
        off[JOINTS.index(j)] = v
    errs = []
    for q, (P, uv) in zip(qs, obs):
        R_rc, p_rc = rig.camera(q + off, fit["mount"])
        pred, _ = project(K, R_rc, p_rc, P @ fit["R_robot_board"].T + fit["t_robot_board"])
        errs.append(np.sum((pred - uv) ** 2, 1))
    return float(np.sqrt(np.mean(np.concatenate(errs))))


def _write_results(session_dir: Path, fit: dict, intr: dict, board) -> None:
    import cv2

    from robot_manipulator.calib.extrinsics import EXTRINSICS_DIR, camera_in_robot, mujoco_xyaxes

    (REPO / "config" / "camera_intrinsics" / "wrist.yaml").write_text(
        yaml.safe_dump(intr, sort_keys=False)
    )
    ws_intr = yaml.safe_load((REPO / "config" / "camera_intrinsics" / "workspace.yaml").read_text())
    Kw, dw = np.array(ws_intr["camera_matrix"]), np.array(ws_intr["dist_coeffs"])
    # the board never moves during a session, but the arm hides a different part of it in each
    # overhead frame: pool every frame's corners (median pixel per corner id) for one PnP
    seen: dict[int, list[np.ndarray]] = {}
    for f in sorted(session_dir.glob("ws_*.png")):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        cc, ci = detect(board, g, min_corners=4)
        if cc is not None:
            for c, i in zip(cc.reshape(-1, 2), ci.ravel()):
                seen.setdefault(int(i), []).append(c)
    if len(seen) < 8:
        typer.echo(f"workspace camera saw only {len(seen)} board corners; pose not written")
        return
    ids = np.array(sorted(seen))
    uv = np.array([np.median(seen[i], axis=0) for i in ids], dtype=np.float32)
    P, uv = board.matchImagePoints(uv.reshape(-1, 1, 2), ids.reshape(-1, 1))
    ok, rvec, tvec = cv2.solvePnP(P, uv, Kw, dw)
    proj, _ = cv2.projectPoints(P, rvec, tvec, Kw, dw)
    pnp_rms = float(np.sqrt(np.mean(np.sum((proj - uv) ** 2, axis=-1))))
    best = (len(ids), None, None, f"{len(ids)} pooled corners, PnP {pnp_rms:.2f} px")
    R_cb, _ = cv2.Rodrigues(rvec)
    p, R_rc = camera_in_robot(R_cb, tvec.ravel(), fit["R_robot_board"], fit["t_robot_board"])
    fovy = float(np.degrees(2 * np.arctan(ws_intr["image_size"][1] / 2 / Kw[1, 1])))
    pose = {
        "pos": [round(float(v), 4) for v in p],
        "xyaxes": [round(float(v), 5) for v in mujoco_xyaxes(R_rc)],
        "fovy_deg": round(fovy, 2),
        "source": f"camcal-wrist {datetime.now():%Y-%m-%d} ({best[3]}; "
        f"wrist BA rms {fit['rms_px']:.2f} px; joint offsets fitted, not applied)",
    }
    EXTRINSICS_DIR.mkdir(parents=True, exist_ok=True)
    (EXTRINSICS_DIR / "workspace.yaml").write_text(
        "# Camera pose in the follower base frame (MuJoCo camera convention). GENERATED by\n"
        "# `uv run camcal-wrist solve --write`; `uv run scene build` uses it.\n"
        + yaml.safe_dump(pose, sort_keys=False)
    )
    typer.echo(f"workspace camera at {pose['pos']} -> {EXTRINSICS_DIR / 'workspace.yaml'}")
    typer.echo(
        "joint offsets are NOT written to config/sim2real.yaml; apply them by hand once a second "
        "session agrees: "
        + ", ".join(f"{j} {np.degrees(v):+.2f}°" for j, v in fit["offsets_rad"].items())
    )


if __name__ == "__main__":
    app()
