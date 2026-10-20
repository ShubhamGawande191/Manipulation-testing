import numpy as np
import pytest

pytest.importorskip("mujoco")

from scipy.spatial.transform import Rotation  # noqa: E402

from robot_manipulator.calib.accuracy import pick_corners, tip_error_board  # noqa: E402


@pytest.mark.parametrize("flip", [False, True])
def test_tip_error_is_in_board_coordinates_with_height_up(flip):
    corner = np.array([0.033, 0.066, 0.0])
    # camera ~15 cm above the board looking down at it (tilted); the board z may point either way
    R_bc = Rotation.from_euler("xyz", [170, 10, 30], degrees=True).as_matrix()  # camera in board
    if flip:
        R_bc = np.diag([1, -1, -1]) @ R_bc
    p_cam_board = corner + np.array([0.01, -0.02, 0.15 if not flip else -0.15])
    R_cb, t_cb = R_bc.T, -R_bc.T @ p_cam_board  # board -> camera
    up = 1.0 if not flip else -1.0
    tip_board = corner + np.array([0.003, -0.002, up * 0.02])  # 3 / -2 mm off, 20 mm above
    tip_cam = R_cb @ tip_board + t_cb
    e = tip_error_board(R_cb, t_cb, corner, tip_cam)
    assert np.allclose(e, [0.003, -0.002, 0.02], atol=1e-9)


def test_pick_corners_spreads_over_the_board():
    grid = np.array([[x, y, 0.0] for y in range(6) for x in range(4)], dtype=float)
    idx = pick_corners(grid, 4)
    pts = grid[idx]
    assert len(set(idx)) == 4
    assert np.ptp(pts[:, 0]) >= 2 and np.ptp(pts[:, 1]) >= 4  # not clustered
