"""Shared fixtures."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    """A deployable reach policy: random PPO weights with non-trivial normalization stats.

    Returns the directory holding `reach.onnx` and `policy_meta.json`. No training run is needed.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnxruntime")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from robot_manipulator.sim.envs.reach import DRConfig, ReachEnv
    from robot_manipulator.sim.export import export_policy

    out = tmp_path_factory.mktemp("artifact")
    venv = DummyVecEnv([lambda: ReachEnv(dr=DRConfig(enabled=False))])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    rng = np.random.default_rng(0)
    vn.obs_rms.mean = rng.normal(0.0, 0.2, 24)
    vn.obs_rms.var = rng.uniform(0.5, 2.0, 24)
    model = PPO(
        "MlpPolicy",
        vn,
        n_steps=64,
        batch_size=64,
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    )
    model.save(str(out / "model.zip"))
    vn.save(str(out / "vn.pkl"))
    export_policy(
        out / "model.zip",
        out / "vn.pkl",
        out / "reach_artifact",
        dr=False,
        train_steps=0,
        eval_success={},
    )
    return out / "reach_artifact"
