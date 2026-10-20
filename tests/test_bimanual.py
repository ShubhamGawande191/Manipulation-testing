import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_manipulator.sim import bimanual as bm  # noqa: E402
from robot_manipulator.sim.model import JOINTS  # noqa: E402


def _cfg(y_right=-0.4, yaw=0.0):
    return {
        "arms": {
            "left": {"pos": [0, 0, 0], "yaw_deg": 0, "gripper": "pinc"},
            "right": {"pos": [0, y_right, 0], "yaw_deg": yaw, "gripper": "pinc"},
        }
    }


def test_two_prefixed_arms_compile_with_arm_options():
    m = bm.build(_cfg())
    assert m.nu == 12
    names = [m.actuator(i).name for i in range(m.nu)]
    assert names == list(bm.prefixed(JOINTS, "left_")) + list(bm.prefixed(JOINTS, "right_"))
    assert {m.camera(i).name for i in range(m.ncam)} == {"left_wrist_cam", "right_wrist_cam"}
    assert m.opt.timestep == pytest.approx(0.005)  # taken from the arm MJCF, not MuJoCo's default


def test_clearance_shrinks_as_arms_move_together():
    far, near = bm.build(_cfg(-0.5)), bm.build(_cfg(-0.12))
    out = []
    for m in (far, near):
        d = mujoco.MjData(m)
        bm.set_both(m, d, bm.HOME, bm.HOME)
        out.append(bm.inter_arm_clearance(m, d))
    assert out[0] > out[1]


def test_arm_to_arm_recovers_known_transform():
    rng = np.random.default_rng(0)
    yaw = np.radians(30.0)
    R_lr = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    t_lr = np.array([0.05, -0.42, 0.0])
    R_rb, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    t_rb = np.array([0.25, 0.1, 0.0])
    # same board in the left frame: p_l = R_lr (R_rb p + t_rb) + t_lr
    left = {"R": (R_lr @ R_rb).tolist(), "t_m": (R_lr @ t_rb + t_lr).tolist()}
    right = {"R": R_rb.tolist(), "t_m": t_rb.tolist()}
    R, t = bm.arm_to_arm(left, right)
    assert np.allclose(R, R_lr, atol=1e-9) and np.allclose(t, t_lr, atol=1e-9)
