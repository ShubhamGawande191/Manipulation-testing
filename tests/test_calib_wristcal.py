import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulator.calib.camera import make_board  # noqa: E402
from robot_manipulator.calib.wristcal import (  # noqa: E402
    WristRig,
    _initial_board,
    _rot,
    board_points,
    planar_board_pose,
    project,
    solve,
)


def test_planar_board_pose_recovers_flat_board_with_reflection():
    corners = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.28, 0], [0, 0.28, 0]], dtype=float)
    ang = np.radians(93.0)
    M = np.array([[np.cos(ang), np.sin(ang)], [np.sin(ang), -np.cos(ang)]])  # reflection
    xy = corners[:, :2] @ M.T + [0.4, -0.13]
    R, t = planar_board_pose(corners, xy, -0.0265)
    assert np.allclose(corners @ R.T + t, np.c_[xy, np.full(4, -0.0265)], atol=1e-9)
    assert abs(np.linalg.det(R) - 1) < 1e-9


def test_bundle_adjustment_recovers_joint_offsets_mount_and_board():
    rig = WristRig()
    board, _ = make_board()
    _, inner = board_points(board)
    rng = np.random.default_rng(1)
    # board flat on the table in front of the arm; views from varied joint configurations
    R_true, t_true = planar_board_pose(
        np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.28, 0], [0, 0.28, 0]], dtype=float),
        np.array([[0.40, -0.14], [0.20, -0.14], [0.20, 0.14], [0.40, 0.14]]),
        -0.0265,
    )
    off_true = np.radians([0, 3.0, -8.0, -4.0, 0, 0])
    R_m = _rot(np.radians([4.0, -3.0, 2.0])) @ rig.R_mount0
    t_m = rig.t_mount0 + np.array([0.01, -0.005, 0.008])
    from robot_manipulator.calib.wristcal import aim_camera
    from robot_manipulator.sim.model import JOINTS, joint_ranges

    lo, hi = joint_ranges(rig.m, JOINTS)
    centre = inner.mean(0) @ R_true.T + t_true
    qs, obs = [], []
    while len(qs) < 16:
        q = aim_camera(rig, centre, rng, lo, hi)
        if q is None:
            continue
        R_rc, p_rc = rig.camera(q + off_true, (R_m, t_m))
        uv, z = project(rig.K0, R_rc, p_rc, inner @ R_true.T + t_true)
        ok = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
        if ok.sum() < 12:
            continue
        qs.append(q)
        obs.append((inner[ok], uv[ok] + rng.normal(0, 0.5, (ok.sum(), 2))))
    qs = np.array(qs)
    R0, t0 = _initial_board(rig, qs, obs, rig.K0)
    fit = solve(rig, qs, obs, rig.K0, R0, t0)
    assert fit["rms_px"] < 1.0
    for j, name in ((1, "shoulder_lift"), (2, "elbow_flex"), (3, "wrist_flex")):
        assert abs(fit["offsets_rad"][name] - off_true[j]) < np.radians(0.5), name
    assert np.allclose(fit["t_robot_board"], t_true, atol=0.005)
