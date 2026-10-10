"""Domain randomization of MuJoCo physics, shared by the training env and the harness sim backend.

Physics-only parameters live here: actuator gains, joint damping, friction, armature, link mass,
and the gripper payload. Sensor noise, per-episode measurement bias, and action delay are applied
by the env (training) or the sim backend (harness), using the same `DRConfig` fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class DRConfig:
    """Domain randomization ranges. Applied fresh on every reset from nominal values."""

    enabled: bool = True
    kp_kv: tuple[float, float] = (0.8, 1.2)  # one factor per joint, applied to kp and kv together
    damping: tuple[float, float] = (0.7, 1.3)
    frictionloss: tuple[float, float] = (0.5, 2.0)
    armature: tuple[float, float] = (0.8, 1.2)
    link_mass: tuple[float, float] = (0.9, 1.1)  # per body, mass and inertia together
    gripper_payload_kg: tuple[float, float] = (0.0, 0.25)  # extra mass on body `gripper` (H1)
    action_delay_steps: tuple[int, ...] = (0, 1, 2)
    obs_noise_q: float = 0.005  # rad
    obs_noise_dq: float = 0.05  # rad/s
    joint_bias: float = 0.02  # rad, per-episode offset on measured q


@dataclass
class NominalParams:
    """Parameter values from the vendored MJCF, captured once so DR never drifts."""

    gainprm: np.ndarray
    biasprm: np.ndarray
    damping: np.ndarray
    frictionloss: np.ndarray
    armature: np.ndarray
    body_mass: np.ndarray
    body_inertia: np.ndarray

    @classmethod
    def snapshot(cls, model: mujoco.MjModel) -> NominalParams:
        return cls(
            gainprm=model.actuator_gainprm.copy(),
            biasprm=model.actuator_biasprm.copy(),
            damping=model.dof_damping.copy(),
            frictionloss=model.dof_frictionloss.copy(),
            armature=model.dof_armature.copy(),
            body_mass=model.body_mass.copy(),
            body_inertia=model.body_inertia.copy(),
        )

    def restore(self, model: mujoco.MjModel) -> None:
        model.actuator_gainprm[:] = self.gainprm
        model.actuator_biasprm[:] = self.biasprm
        model.dof_damping[:] = self.damping
        model.dof_frictionloss[:] = self.frictionloss
        model.dof_armature[:] = self.armature
        model.body_mass[:] = self.body_mass
        model.body_inertia[:] = self.body_inertia


def apply_dr(
    model: mujoco.MjModel,
    nominal: NominalParams,
    dr: DRConfig,
    rng: np.random.Generator,
    gripper_body: int,
) -> None:
    """Restore nominal values, then draw one physics sample. Call `mj_setConst` afterwards."""
    nominal.restore(model)
    if not dr.enabled:
        return
    s = rng.uniform(*dr.kp_kv, size=model.nu)
    model.actuator_gainprm[:, 0] = nominal.gainprm[:, 0] * s
    model.actuator_biasprm[:, 1:3] = nominal.biasprm[:, 1:3] * s[:, None]
    model.dof_damping[:] = nominal.damping * rng.uniform(*dr.damping, size=model.nv)
    model.dof_frictionloss[:] = nominal.frictionloss * rng.uniform(*dr.frictionloss, size=model.nv)
    model.dof_armature[:] = nominal.armature * rng.uniform(*dr.armature, size=model.nv)
    f = rng.uniform(*dr.link_mass, size=model.nbody)
    f[0] = 1.0  # world body
    model.body_mass[:] = nominal.body_mass * f
    model.body_inertia[:] = nominal.body_inertia * f[:, None]
    model.body_mass[gripper_body] += rng.uniform(*dr.gripper_payload_kg)
