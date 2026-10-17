"""Vendored SO101 MJCF (Menagerie `robotstudio_so101`) and joint bookkeeping.

The XML lives in `sim/assets/so101/` at a pinned upstream commit; see its
`SOURCE.md`. `JOINTS` is the CMD order used everywhere in the harness
(radians), matching the actuator order in the MJCF.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import mujoco
import numpy as np
import yaml

JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS: tuple[str, ...] = JOINTS[:5]
EE_SITE = "gripperframe"
MENAGERIE_COMMIT = "3abca2e302625cec9429c49806121ea2855cef81"  # the vendored MJCF's upstream commit

ASSET_DIR = Path(str(resources.files(__package__).joinpath("assets", "so101")))


def load_model(scene: str = "scene.xml") -> mujoco.MjModel:
    """Load the SO101 model. `scene.xml` adds a floor; `so101.xml` is the bare arm."""
    return mujoco.MjModel.from_xml_path(str(ASSET_DIR / scene))


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint {name!r} not in model")
    return jid


def joint_qpos_adr(model: mujoco.MjModel, joints: tuple[str, ...] = JOINTS) -> np.ndarray:
    """qpos index of each named joint (all joints here are 1-DoF hinges)."""
    return np.array([model.jnt_qposadr[_joint_id(model, j)] for j in joints], dtype=np.intp)


def joint_dof_adr(model: mujoco.MjModel, joints: tuple[str, ...] = JOINTS) -> np.ndarray:
    """qvel index of each named joint."""
    return np.array([model.jnt_dofadr[_joint_id(model, j)] for j in joints], dtype=np.intp)


def joint_ranges(
    model: mujoco.MjModel, joints: tuple[str, ...] = JOINTS
) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) joint limits in radians, in the given joint order."""
    ids = [_joint_id(model, j) for j in joints]
    rng = np.asarray(model.jnt_range[ids], dtype=np.float64)
    return rng[:, 0].copy(), rng[:, 1].copy()


def actuator_ids(model: mujoco.MjModel, joints: tuple[str, ...] = JOINTS) -> np.ndarray:
    """ctrl index for each named actuator (actuators are named after their joint)."""
    ids = []
    for j in joints:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if aid < 0:
            raise KeyError(f"actuator {j!r} not in model")
        ids.append(aid)
    return np.array(ids, dtype=np.intp)


def apply_actuator_fit(model: mujoco.MjModel, fit: str | Path | dict) -> dict[str, int]:
    """Scale a freshly loaded model by a fit from `sim.gap.fit` (config/actuator_fit.yaml).

    Multiplies the current values, so call it on a model at its nominal values. Returns
    {joint: delay_steps}. The delay is a command-pipeline property, so the caller applies it
    to the backend, not to the model.
    """
    cfg = fit if isinstance(fit, dict) else yaml.safe_load(Path(fit).read_text())
    act = actuator_ids(model, JOINTS)
    dof = joint_dof_adr(model, JOINTS)
    delays: dict[str, int] = {}
    for joint, p in cfg["joints"].items():
        j = JOINTS.index(joint)
        a, d = act[j], dof[j]
        model.actuator_gainprm[a, 0] *= float(p["kp_scale"])
        model.actuator_biasprm[a, 1] *= float(p["kp_scale"])
        model.dof_damping[d] *= float(p["damping_scale"])
        model.dof_frictionloss[d] *= float(p["frictionloss_scale"])
        model.dof_armature[d] *= float(p["armature_scale"])
        delays[joint] = int(p["delay_steps"])
    return delays


def apply_joint_couplings(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Propagate active joint-equality couplings into ``data.qpos``.

    MuJoCo only enforces equality constraints while stepping. Kinematic-only callers (set qpos,
    then ``mj_forward``/``mj_kinematics``) must do this themselves, or coupled joints — e.g. the
    PincOpen arms/fingers that follow the ``gripper`` servo — stay at their reference pose.
    Joint equality: ``q1 - q1ref = poly(q2 - q2ref)`` with ``polycoef`` in ``eq_data[:5]``.
    """
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[i]:
            continue
        j1, j2 = model.eq_obj1id[i], model.eq_obj2id[i]
        a1, a2 = model.jnt_qposadr[j1], model.jnt_qposadr[j2]
        x = data.qpos[a2] - model.qpos0[a2]
        c = model.eq_data[i][:5]
        data.qpos[a1] = model.qpos0[a1] + c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
