import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_manipulator.sim.model import (  # noqa: E402
    JOINTS,
    actuator_ids,
    joint_qpos_adr,
    load_model,
)

HOME = np.array([0.0, -1.57, 1.57, 0.0, 0.0])
COUPLED = {
    "pinc_right_arm": 0.5,
    "pinc_left_arm": -0.5,
    "pinc_right_finger": -0.5,
    "pinc_left_finger": 0.5,
}


@pytest.fixture(scope="module")
def model():
    return load_model("scene_pinc.xml")


def _settle(model, q_grip, seconds=1.0):
    d = mujoco.MjData(model)
    adr, act = joint_qpos_adr(model, JOINTS), actuator_ids(model, JOINTS)
    d.qpos[adr[:5]] = HOME
    d.qpos[adr[5]] = q_grip
    d.ctrl[act[:5]] = HOME
    d.ctrl[act[5]] = q_grip
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, d)
    return d


def test_same_joint_interface_as_stock(model):
    stock = load_model("scene.xml")
    assert model.nu == stock.nu == 6
    assert [model.actuator(i).name for i in range(6)] == list(JOINTS)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe") >= 0


def test_mass_past_wrist(model):
    root = model.body("gripper").id
    total = sum(model.body_mass[b] for b in range(model.nbody) if _descends(model, b, root))
    assert total == pytest.approx(0.212, abs=0.005)


def _descends(model, b, root):
    while b > 0:
        if b == root:
            return True
        b = model.body_parentid[b]
    return False


def test_fingers_coupled_parallel_and_open_monotonically(model):
    seps = []
    for g in (0.0, -0.6, -1.2, -1.833):
        d = _settle(model, g)
        for j, k in COUPLED.items():
            assert d.qpos[model.joint(j).qposadr[0]] == pytest.approx(k * g, abs=2e-3)
        r, b = (
            d.body("pinc_right_finger").xmat.reshape(3, 3),
            d.body("pinc_base").xmat.reshape(3, 3),
        )
        assert np.dot(r[:, 0], b[:, 0]) == pytest.approx(1.0, abs=1e-4)  # parallel jaw
        seps.append(
            np.linalg.norm(d.body("pinc_right_finger").xpos - d.body("pinc_left_finger").xpos)
        )
    assert seps == sorted(seps) and seps[0] < 0.03 < 0.09 < seps[-1]


def test_no_self_contact_at_home(model):
    d = _settle(model, 0.0, seconds=0.2)
    assert d.ncon == 0


def test_kinematic_coupling_moves_fingers_without_stepping(model):
    # the twin / FK only call mj_forward; fingers must still follow the servo
    from robot_manipulator.sim.model import apply_joint_couplings

    d = mujoco.MjData(model)
    adr = joint_qpos_adr(model, JOINTS)
    seps = []
    for g in (0.0, -1.833):
        d.qpos[:] = 0
        d.qpos[adr[5]] = g
        apply_joint_couplings(model, d)
        mujoco.mj_kinematics(model, d)
        seps.append(
            np.linalg.norm(d.body("pinc_right_finger").xpos - d.body("pinc_left_finger").xpos)
        )
    assert seps[0] == pytest.approx(0.025, abs=1e-3) and seps[1] > 0.09


def test_camera_bracket_points_away_from_base_at_rest(model):
    # user photos 2026-10-17: at the folded rest pose the bracket sticks out away from the base
    d = mujoco.MjData(model)
    adr = joint_qpos_adr(model, JOINTS)
    d.qpos[adr[:5]] = np.deg2rad([-3.3, -108.0, 96.6, 87.9, -84.8])
    mujoco.mj_forward(model, d)
    assert d.geom("camera_lens").xpos[0] > d.body("pinc_base").xpos[0] + 0.05


def test_training_gripper_hold_matches_joint_map_and_model():
    # deploy sends GRIPPER_HOLD through config/sim2real.yaml; training must use the same value
    from robot_manipulator.harness.twin import DEFAULT_MAP
    from robot_manipulator.harness.units import JointMap
    from robot_manipulator.sim.envs.reach import GRIPPER_HOLD
    from robot_manipulator.sim.model import DEFAULT_SCENE, joint_ranges

    assert DEFAULT_SCENE == "scene_pinc.xml"
    assert GRIPPER_HOLD == pytest.approx(JointMap.from_yaml(DEFAULT_MAP).gripper.sim_open)
    lo, hi = joint_ranges(load_model(), JOINTS)
    assert lo[5] <= GRIPPER_HOLD <= hi[5]
