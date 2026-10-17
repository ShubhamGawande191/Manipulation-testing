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
    assert total == pytest.approx(0.247, abs=0.005)


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


def test_reproduces_observed_camera_mount_stall(model):
    # config/so101_limits.yaml: real wrist_flex stalled at ~59 deg at this folded pose
    d = mujoco.MjData(model)
    adr = joint_qpos_adr(model, JOINTS)
    first = None
    for wf in np.arange(40.0, 80.0, 0.5):
        d.qpos[:] = 0
        d.qpos[adr[:5]] = np.deg2rad([-3.0, -108.0, 95.5, wf, 110.0])
        mujoco.mj_forward(model, d)
        if any(
            model.geom(c.geom1).name == "camera_box" or model.geom(c.geom2).name == "camera_box"
            for c in d.contact[: d.ncon]
        ):
            first = wf
            break
    assert first is not None and 55.0 <= first <= 65.0
