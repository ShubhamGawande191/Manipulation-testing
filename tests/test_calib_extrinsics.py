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
