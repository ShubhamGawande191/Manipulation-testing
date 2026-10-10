"""SO101 reach task: the single source of truth for the policy observation and command integration.

The deploy-side policy (`harness/policies/onnx_reach.py`) imports `build_obs`,
`command_box`, and `integrate_command` from this module, so training and deployment
cannot drift apart.

Observation (24 float32, see `build_obs`): q[5], dq[5], last_action[5], ee[3], target[3],
target - ee[3]. Action: 5 values in [-1, 1]. Each action moves the arm joint command by
`ACTION_SCALE * a` rad per 50 Hz tick, clipped to the joint limits shrunk by `JOINT_MARGIN`.
The gripper is held at `GRIPPER_HOLD`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_manipulator.sim.kinematics import FK
from robot_manipulator.sim.model import (
    JOINTS,
    actuator_ids,
    joint_dof_adr,
    joint_qpos_adr,
    joint_ranges,
    load_model,
)
from robot_manipulator.sim.randomize import DRConfig, NominalParams, apply_dr

__all__ = [
    "ACTION_SCALE",
    "DRConfig",
    "EP_STEPS",
    "GRIPPER_HOLD",
    "HOME_ARM",
    "HZ",
    "JOINT_MARGIN",
    "N_ARM",
    "OBS_DIM",
    "ReachEnv",
    "SUCCESS_TOL",
    "build_obs",
    "command_box",
    "integrate_command",
    "target_box",
]

HZ = 50.0
EP_SECONDS = 4.0
EP_STEPS = int(EP_SECONDS * HZ)  # 200
ACTION_SCALE = 0.03  # rad per tick at a = 1. 0.02 (plan) reached 55 %; 0.03 reached 82 % (docs)
JOINT_MARGIN = 0.05  # rad, same as the harness SafetyFilter default
TARGET_SHRINK = 0.8  # targets are sampled from 80 % of each joint range
HOME_ARM = np.array([0.0, -1.57, 1.57, 0.0, 0.0])  # matches so101_bridge.HOME, CMD order
RESET_NOISE = 0.1  # rad, uniform per arm joint
GRIPPER_HOLD = 1.74533  # rad, sim_open in config/sim2real.yaml
SUCCESS_TOL = 0.02  # m
N_ARM = 5
OBS_DIM = 24
ENV_ID = "SO101Reach-v0"


def _vec(x: Any, n: int, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got shape {v.shape}")
    return v


def build_obs(q, dq, last_action, ee, target) -> np.ndarray:
    """Policy observation, float32[24]: q(5), dq(5), last_action(5), ee(3), target(3), target-ee(3).

    `q` and `dq` are the five arm joints, in radians and rad/s, CMD order. `ee` and `target` are
    meters in the world frame. Pure function, so the deploy policy can import it.
    """
    q_v = _vec(q, N_ARM, "q")
    dq_v = _vec(dq, N_ARM, "dq")
    a_v = _vec(last_action, N_ARM, "last_action")
    ee_v = _vec(ee, 3, "ee")
    tgt_v = _vec(target, 3, "target")
    return np.concatenate([q_v, dq_v, a_v, ee_v, tgt_v, tgt_v - ee_v]).astype(np.float32)


def command_box(lo, hi, margin: float = JOINT_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """Arm command limits: the joint limits (first five entries) shrunk by `margin`."""
    lo_v = _vec(np.asarray(lo, dtype=np.float64)[:N_ARM], N_ARM, "lo")
    hi_v = _vec(np.asarray(hi, dtype=np.float64)[:N_ARM], N_ARM, "hi")
    return lo_v + margin, hi_v - margin


def target_box(lo, hi, shrink: float = TARGET_SHRINK) -> tuple[np.ndarray, np.ndarray]:
    """Box that reach targets are sampled from: each arm range shrunk about its midpoint."""
    lo_v = _vec(np.asarray(lo, dtype=np.float64)[:N_ARM], N_ARM, "lo")
    hi_v = _vec(np.asarray(hi, dtype=np.float64)[:N_ARM], N_ARM, "hi")
    mid = 0.5 * (lo_v + hi_v)
    half = 0.5 * (hi_v - lo_v) * shrink
    return mid - half, mid + half


def integrate_command(q_cmd, action, box) -> np.ndarray:
    """q_cmd' = clip(q_cmd + ACTION_SCALE * clip(a, -1, 1), lo, hi). Shared by env and deploy."""
    lo, hi = box
    a = np.clip(_vec(action, N_ARM, "action"), -1.0, 1.0)
    return np.clip(_vec(q_cmd, N_ARM, "q_cmd") + ACTION_SCALE * a, lo, hi)


class ReachEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": HZ}

    def __init__(self, dr: DRConfig | None = None, model: mujoco.MjModel | None = None) -> None:
        super().__init__()
        self.dr = dr if dr is not None else DRConfig()
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        self.fk = FK(self.model)
        self._qadr = joint_qpos_adr(self.model, JOINTS)
        self._dof = joint_dof_adr(self.model, JOINTS)
        self._act = actuator_ids(self.model, JOINTS)
        self._gripper_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        if self._gripper_body < 0:
            raise KeyError("body 'gripper' not in model")
        lo, hi = joint_ranges(self.model, JOINTS)
        self.joint_lo, self.joint_hi = lo[:N_ARM], hi[:N_ARM]
        self.box = command_box(lo, hi)
        self.target_lo, self.target_hi = target_box(lo, hi)
        self.n_substeps = max(1, round(1.0 / (HZ * self.model.opt.timestep)))
        self._nominal = NominalParams.snapshot(self.model)

        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_ARM,), np.float32)

        self.target = np.zeros(3)
        self._q_cmd = HOME_ARM.copy()
        self._prev_a = np.zeros(N_ARM)
        self._pending: deque[np.ndarray] = deque()
        self._delay = 0
        self._bias = np.zeros(N_ARM)
        self._step = 0

    # ---- observation -----------------------------------------------------------------

    def _full_q(self, q_arm: np.ndarray) -> np.ndarray:
        return np.r_[q_arm, GRIPPER_HOLD]

    def _measure(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[self._qadr[:N_ARM]].copy()
        dq = self.data.qvel[self._dof[:N_ARM]].copy()
        if self.dr.enabled:
            q = q + self._bias + rng.normal(0.0, self.dr.obs_noise_q, N_ARM)
            dq = dq + rng.normal(0.0, self.dr.obs_noise_dq, N_ARM)
        return q, dq

    # ---- gymnasium API ---------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        apply_dr(self.model, self._nominal, self.dr, rng, self._gripper_body)
        if self.dr.enabled:
            self._delay = int(rng.choice(self.dr.action_delay_steps))
            self._bias = rng.uniform(-self.dr.joint_bias, self.dr.joint_bias, N_ARM)
        else:
            self._delay = 0
            self._bias = np.zeros(N_ARM)
        mujoco.mj_setConst(self.model, self.data)

        _, ee_target = self.fk.sample_reachable(rng, 1, self.target_lo, self.target_hi)
        self.target = ee_target[0].copy()

        q0 = np.clip(
            HOME_ARM + rng.uniform(-RESET_NOISE, RESET_NOISE, N_ARM), self.box[0], self.box[1]
        )
        self.data.qpos[self._qadr[:N_ARM]] = q0
        self.data.qpos[self._qadr[N_ARM]] = GRIPPER_HOLD
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        q_meas, dq_meas = self._measure(rng)
        self._q_cmd = q_meas.copy()  # command starts from what the sensor reports
        self.data.ctrl[self._act[:N_ARM]] = self._q_cmd
        self.data.ctrl[self._act[N_ARM]] = GRIPPER_HOLD

        self._pending = deque(np.zeros(N_ARM) for _ in range(self._delay))
        self._prev_a = np.zeros(N_ARM)
        self._step = 0
        ee = self.fk.ee_pos(self._full_q(q_meas))
        obs = build_obs(q_meas, dq_meas, np.zeros(N_ARM), ee, self.target)
        return obs, {"target": self.target.copy(), "action_delay": self._delay}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(N_ARM), -1.0, 1.0)
        if self._delay > 0:
            self._pending.append(a)
            a_exec = self._pending.popleft()
        else:
            a_exec = a
        self._q_cmd = integrate_command(self._q_cmd, a_exec, self.box)
        self.data.ctrl[self._act[:N_ARM]] = self._q_cmd
        self.data.ctrl[self._act[N_ARM]] = GRIPPER_HOLD
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        q_true = self.data.qpos[self._qadr[:N_ARM]].copy()
        dq_true = self.data.qvel[self._dof[:N_ARM]].copy()
        q_meas, dq_meas = self._measure(self.np_random)
        ee_true = self.fk.ee_pos(self._full_q(q_true))
        ee_meas = self.fk.ee_pos(self._full_q(q_meas))
        dist = float(np.linalg.norm(ee_true - self.target))

        reward = (
            -dist
            - 0.01 * float(np.sum((a - self._prev_a) ** 2))
            - 0.001 * float(np.sum(dq_true**2))
            + (0.5 if dist < SUCCESS_TOL else 0.0)
        )
        self._prev_a = a
        obs = build_obs(q_meas, dq_meas, a, ee_meas, self.target)
        truncated = self._step >= EP_STEPS
        info = {"dist": dist, "success": bool(truncated and dist < SUCCESS_TOL)}
        return obs, float(reward), False, truncated, info


ENV_ENTRY_POINT = "robot_manipulator.sim.envs.reach:ReachEnv"


def register() -> None:
    """Register `SO101Reach-v0` with gymnasium. Safe to call more than once."""
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=ENV_ENTRY_POINT, max_episode_steps=EP_STEPS)


register()
