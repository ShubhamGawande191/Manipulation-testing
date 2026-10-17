"""Unit tests: vendored SO101 MJCF loads, joint/actuator order and limits match the plan."""

import pytest

mujoco = pytest.importorskip("mujoco")

import numpy as np  # noqa: E402

from robot_manipulator.sim.model import (  # noqa: E402
    ARM_JOINTS,
    EE_SITE,
    JOINTS,
    actuator_ids,
    joint_dof_adr,
    joint_qpos_adr,
    joint_ranges,
    load_model,
)


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_actuators_and_joints_follow_cmd_order(model):
    assert model.nu == 6
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(6)]
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert tuple(act_names) == JOINTS
    assert tuple(jnt_names[:6]) == JOINTS
    np.testing.assert_array_equal(actuator_ids(model), np.arange(6))


def test_joint_limits_match_menagerie_numbers():
    # the stock Menagerie arm; the default PincOpen variant has a different gripper joint
    lo, hi = joint_ranges(load_model("scene.xml"))
    np.testing.assert_allclose(
        lo, [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453], atol=1e-4
    )
    np.testing.assert_allclose(hi, [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533], atol=1e-4)


def test_addresses_are_one_dof_hinges(model):
    np.testing.assert_array_equal(joint_qpos_adr(model), np.arange(6))
    np.testing.assert_array_equal(joint_dof_adr(model), np.arange(6))
    np.testing.assert_array_equal(joint_qpos_adr(model, ARM_JOINTS), np.arange(5))


def test_ee_site_exists(model):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE) >= 0


def test_zero_ctrl_steps_without_nan(model):
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
