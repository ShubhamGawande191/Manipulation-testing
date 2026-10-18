import numpy as np
import pytest

from robot_manipulator.calib.extrinsics import camera_in_robot, kabsch, mujoco_xyaxes


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def test_kabsch_recovers_rigid_transform():
    src = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.28, 0], [0, 0.28, 0]], float)
    R = _rot([0.1, 0.2, 1.0], 0.7)
    t = np.array([0.15, -0.1, -0.0275])
    R2, t2, res = kabsch(src, src @ R.T + t)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)
    assert res.max() < 1e-9


def test_camera_chain_maps_board_points_consistently():
    R_cb, t_cb = _rot([1, 0.3, 0], 2.5), np.array([0.02, -0.05, 0.6])
    R_rb, t_rb = _rot([0, 0, 1], 0.4), np.array([0.2, -0.05, -0.0275])
    p, R_rc = camera_in_robot(R_cb, t_cb, R_rb, t_rb)
    X_board = np.array([0.1, 0.07, 0.0])
    X_cam = R_cb @ X_board + t_cb
    X_robot = R_rb @ X_board + t_rb
    np.testing.assert_allclose(R_rc @ X_cam + p, X_robot, atol=1e-12)


def test_mujoco_axes_convention():
    # OpenCV camera looking straight down (+z_cam = -z_robot), image right = -y_robot
    R_rc = np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).astype(float)
    xy = np.array(mujoco_xyaxes(R_rc))
    x, y = xy[:3], xy[3:]
    z = np.cross(x, y)
    assert z == pytest.approx([0, 0, 1])  # MuJoCo looks along -z = straight down


def test_tip_point_is_past_the_ee_site_along_the_tool():
    mujoco = pytest.importorskip("mujoco")
    from robot_manipulator.calib.extrinsics import TIP_OFFSET_M, tip_point
    from robot_manipulator.sim.kinematics import FK
    from robot_manipulator.sim.model import load_model

    m = load_model()
    q = np.array([0.0, -1.0, 1.2, 1.0, 0.0, 0.0])  # tool pointing roughly down
    tip = tip_point(m, mujoco.MjData(m), q)
    site = FK(m).ee_pos(q)
    assert abs(np.linalg.norm(tip - site) - TIP_OFFSET_M) < 1e-9


def test_fit_touches_recovers_board_pose_and_tip_offset():
    pytest.importorskip("mujoco")
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from robot_manipulator.calib.extrinsics import TipFK, fit_touches

    fk = TipFK()
    corners = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.28, 0], [0, 0.28, 0]], dtype=float)
    R_true = Rotation.from_euler("z", 5, degrees=True).as_matrix()
    t_true = np.array([0.12, -0.14, -0.027])  # board in front of the arm, like the bench
    tip_true = fk.tip_local_nominal + np.array([0.004, -0.003, 0.005])
    world = corners @ R_true.T + t_true

    def ik(target, q0):  # 5 arm joints, gripper closed; pan seeded from both signs
        f = lambda x: fk.tip(np.r_[x, 0.0], tip_true) - target  # noqa: E731
        sols = []
        for pan in (np.arctan2(target[1], target[0]), -np.arctan2(target[1], target[0])):
            sols.append(least_squares(f, np.r_[pan, q0[1:]], bounds=(-1.6, 1.6)))
        return np.r_[min(sols, key=lambda r: r.cost).x, 0.0]

    starts = (np.array([0.0, 0.3, 0.6, 1.2, 0.0]), np.array([0.0, 0.1, 0.9, 0.7, 1.0]))
    qs, ids = [], []
    for k, w in enumerate(world):
        for q0 in starts:
            q = ik(w, q0)
            assert np.linalg.norm(fk.tip(q, tip_true) - w) < 1e-4
            qs.append(q)
            ids.append(k)
    fit = fit_touches(fk, np.array(qs), np.array(ids), corners)
    assert fit["res"].max() < 1e-3
    assert np.allclose(fit["t"], t_true, atol=2e-3)
    assert np.linalg.norm(fit["tip_local"] - tip_true) < 2e-3
