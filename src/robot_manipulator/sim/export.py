"""Export a trained SO101 reach policy to ONNX plus a metadata JSON for the harness.

`reach.onnx` takes the NORMALIZED observation float32[1, 24] and returns the pre-clip mean
action float32[1, 5]. The deploy side must normalize with `policy_meta.json` first:

    x = clip((obs - obs_mean) / sqrt(obs_var + epsilon), -clip_obs, clip_obs)

`normalize_obs` implements that formula, and the deploy policy imports it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_manipulator.sim.envs.reach import (
    ACTION_SCALE,
    GRIPPER_HOLD,
    HZ,
    N_ARM,
    OBS_DIM,
)
from robot_manipulator.sim.model import JOINTS, MENAGERIE_COMMIT

OBS_LAYOUT = [
    *[f"q_{j}" for j in JOINTS[:N_ARM]],
    *[f"dq_{j}" for j in JOINTS[:N_ARM]],
    *[f"last_action_{j}" for j in JOINTS[:N_ARM]],
    "ee_x",
    "ee_y",
    "ee_z",
    "target_x",
    "target_y",
    "target_z",
    "err_x",
    "err_y",
    "err_z",
]
EPSILON = 1e-8


def normalize_obs(
    obs: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    clip_obs: float = 10.0,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """VecNormalize's observation transform, for one obs or a batch (last axis = features)."""
    x = (np.asarray(obs, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / np.sqrt(
        np.asarray(var, dtype=np.float64) + epsilon
    )
    return np.clip(x, -clip_obs, clip_obs).astype(np.float32)


def _actor_module(policy: Any):
    """SB3 actor path only (policy MLP -> action_net). The value head is not exported."""
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.mlp = inner.mlp_extractor.policy_net
            self.head = inner.action_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.head(self.mlp(obs))

    return Actor(policy).eval().cpu()


def export_onnx(policy: Any, path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            _actor_module(policy),
            dummy,
            str(path),
            input_names=["obs"],
            output_names=["action"],
            opset_version=17,
            dynamo=False,  # torch>=2.9 defaults to the dynamo exporter, which needs onnxscript
        )
    return path


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_policy(
    model_path: str | Path,
    vecnorm_path: str | Path,
    out_dir: str | Path,
    *,
    dr: bool,
    train_steps: int,
    eval_success: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write `<out_dir>/reach.onnx` and `<out_dir>/policy_meta.json`. Returns both paths."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from robot_manipulator.sim.envs.reach import ReachEnv

    out_dir = Path(out_dir)
    model = PPO.load(str(model_path), device="cpu")
    vn = VecNormalize.load(str(vecnorm_path), DummyVecEnv([lambda: ReachEnv()]))
    onnx_path = export_onnx(model.policy, out_dir / "reach.onnx")

    meta: dict[str, Any] = {
        "obs_mean": vn.obs_rms.mean.astype(float).tolist(),
        "obs_var": vn.obs_rms.var.astype(float).tolist(),
        "clip_obs": float(vn.clip_obs),
        "epsilon": EPSILON,
        "action_scale": ACTION_SCALE,
        "hz": HZ,
        "joints": list(JOINTS[:N_ARM]),
        "gripper_hold": GRIPPER_HOLD,
        "obs_layout": OBS_LAYOUT,
        "units": "rad",
        "sim_commit": MENAGERIE_COMMIT,
        "repo_commit": _git_head(),
        "dr": bool(dr),
        "train_steps": int(train_steps),
        "eval_success": eval_success,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        meta.update(extra)
    meta_path = out_dir / "policy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return onnx_path, meta_path
