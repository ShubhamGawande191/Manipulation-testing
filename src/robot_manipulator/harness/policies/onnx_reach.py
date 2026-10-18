"""ONNX reach policy for the harness: the deploy half of the train/deploy pair.

Uses the same code path as training: `build_obs`, `command_box`, and `integrate_command` from
`sim/envs/reach.py`, plus `normalize_obs` from `sim/export.py`. Constants are checked against
`policy_meta.json` at load time, so a policy trained with different constants is refused.

Commands are radians in CMD order, returned as one row (1, 6). The gripper is held at GRIPPER_HOLD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_manipulator.core.policy_iface import BasePolicy, Obs
from robot_manipulator.harness.units import CMD_JOINTS
from robot_manipulator.sim.envs.reach import (
    ACTION_SCALE,
    GRIPPER_HOLD,
    HZ,
    N_ARM,
    OBS_DIM,
    build_obs,
    command_box,
    integrate_command,
)
from robot_manipulator.sim.export import OBS_LAYOUT, normalize_obs
from robot_manipulator.sim.kinematics import FK
from robot_manipulator.sim.model import JOINTS, joint_ranges, load_model


class OnnxReachPolicy(BasePolicy):
    name = "onnx_reach"

    def __init__(
        self,
        onnx_path: str | Path,
        meta_path: str | Path | None = None,
        max_vel: float | None = None,
    ) -> None:
        """``max_vel`` [rad/s]: the SafetyFilter cap commands are sent at; the policy integrates at
        it (see ``integrate_command``). None = uncapped (ACTION_SCALE * HZ)."""
        super().__init__()
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        meta_path = Path(meta_path) if meta_path else onnx_path.parent / "policy_meta.json"
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        self._check_meta(self.meta, meta_path)
        self.mean = np.asarray(self.meta["obs_mean"], dtype=np.float64)
        self.var = np.asarray(self.meta["obs_var"], dtype=np.float64)
        self.clip_obs = float(self.meta["clip_obs"])
        self.epsilon = float(self.meta["epsilon"])
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        lo, hi = joint_ranges(load_model(), JOINTS)
        self.box = command_box(lo, hi)
        self.fk = FK()
        self.q_cmd: np.ndarray | None = None
        self.max_step = None if max_vel is None else float(max_vel) / HZ
        self.last_action = np.zeros(N_ARM)
        self.last_obs: np.ndarray | None = None

    @staticmethod
    def _check_meta(meta: dict[str, Any], path: Path) -> None:
        problems = []
        if meta.get("units") != "rad":
            problems.append(f"units {meta.get('units')!r}, expected 'rad'")
        if float(meta.get("action_scale", -1)) != ACTION_SCALE:
            problems.append(f"action_scale {meta.get('action_scale')} != {ACTION_SCALE}")
        if float(meta.get("hz", -1)) != HZ:
            problems.append(f"hz {meta.get('hz')} != {HZ}")
        if float(meta.get("gripper_hold", -1)) != GRIPPER_HOLD:
            problems.append(f"gripper_hold {meta.get('gripper_hold')} != {GRIPPER_HOLD}")
        if list(meta.get("joints", [])) != list(JOINTS[:N_ARM]):
            problems.append("joint order differs from sim.model.JOINTS[:5]")
        if list(meta.get("obs_layout", [])) != OBS_LAYOUT:
            problems.append("obs_layout differs from the training layout")
        if len(meta.get("obs_mean", [])) != OBS_DIM or len(meta.get("obs_var", [])) != OBS_DIM:
            problems.append(f"obs statistics must have {OBS_DIM} entries")
        if problems:
            raise ValueError(f"{path} does not match this build: " + "; ".join(problems))

    def reset(self) -> None:
        super().reset()
        self.q_cmd = None
        self.last_action = np.zeros(N_ARM)
        self.last_obs = None

    def _act(self, obs: Obs) -> np.ndarray:
        q_all = np.asarray(obs["q"], dtype=np.float64).reshape(-1)
        dq_all = np.asarray(obs["dq"], dtype=np.float64).reshape(-1)
        if q_all.shape[0] != len(CMD_JOINTS) or dq_all.shape[0] != len(CMD_JOINTS):
            raise ValueError("obs needs q and dq with six entries in CMD order")
        q = q_all[:N_ARM]
        dq = dq_all[:N_ARM]
        if self.q_cmd is None:
            self.q_cmd = q.copy()  # the command starts from what the sensor reports
        ee = self.fk.ee_pos(np.r_[q, GRIPPER_HOLD])
        target = np.asarray(obs["goal"]["target"], dtype=np.float64)

        raw = build_obs(q, dq, self.last_action, ee, target)
        self.last_obs = raw
        x = normalize_obs(raw, self.mean, self.var, self.clip_obs, self.epsilon)
        mean_action = np.asarray(self.session.run(None, {"obs": x[None]})[0], dtype=np.float64)
        a = np.clip(mean_action.reshape(N_ARM), -1.0, 1.0)

        self.q_cmd = integrate_command(self.q_cmd, a, self.box, self.max_step)
        self.last_action = a
        return np.r_[self.q_cmd, GRIPPER_HOLD][None, :]
