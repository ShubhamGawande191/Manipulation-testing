"""Smoke tests: 2 048-step PPO run -> ONNX export; ONNX matches SB3 deterministic predict.

Full-length training is an acceptance run, not a test (docs/harness-sim2real.md).
"""

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
ort = pytest.importorskip("onnxruntime")

from robot_manipulator.sim.envs.reach import ACTION_SCALE, OBS_DIM, DRConfig, ReachEnv  # noqa: E402
from robot_manipulator.sim.export import normalize_obs  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    from robot_manipulator.sim.train_reach import train

    out = tmp_path_factory.mktemp("smoke") / "run"
    summary = train(
        total_steps=2048,
        n_envs=2,
        seed=0,
        dr=True,
        out=out,
        eval_every=1024,
        eval_episodes=2,
        final_episodes=2,
        dummy=True,
    )
    return out, summary


def test_training_writes_checkpoints_and_export(smoke_run):
    out, summary = smoke_run
    for name in [
        "best_model.zip",
        "vecnormalize.pkl",
        "final_model.zip",
        "vecnormalize_final.pkl",
        "reach.onnx",
        "policy_meta.json",
        "eval_final.json",
        "progress.csv",
    ]:
        assert (out / name).exists(), name
    assert set(summary["final_eval"]) == {"dr", "nodr"}


def test_meta_has_harness_contract(smoke_run):
    out, _ = smoke_run
    meta = json.loads((out / "policy_meta.json").read_text())
    assert len(meta["obs_mean"]) == OBS_DIM and len(meta["obs_var"]) == OBS_DIM
    assert meta["clip_obs"] == 10.0
    assert meta["units"] == "rad"
    assert meta["joints"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    assert len(meta["obs_layout"]) == OBS_DIM
    assert meta["hz"] == 50.0 and meta["action_scale"] == pytest.approx(ACTION_SCALE)
    assert meta["sim_commit"] == "3abca2e302625cec9429c49806121ea2855cef81"


def test_onnx_matches_sb3_deterministic_predict(smoke_run):
    from stable_baselines3 import PPO

    out, _ = smoke_run
    meta = json.loads((out / "policy_meta.json").read_text())
    mean = np.asarray(meta["obs_mean"])
    var = np.asarray(meta["obs_var"])
    model = PPO.load(str(out / "best_model.zip"), device="cpu")
    sess = ort.InferenceSession(str(out / "reach.onnx"), providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].shape == [1, OBS_DIM]

    env = ReachEnv(dr=DRConfig(enabled=True))
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    raw_obs = [obs]
    for _ in range(9):
        obs, *_ = env.step(rng.uniform(-1, 1, size=5))
        raw_obs.append(obs)

    for raw in raw_obs:
        x = normalize_obs(raw, mean, var, meta["clip_obs"], meta["epsilon"])
        sb3_action, _ = model.predict(x[None], deterministic=True)
        onnx_mean = sess.run(None, {"obs": x[None]})[0]
        np.testing.assert_allclose(np.clip(onnx_mean, -1.0, 1.0), sb3_action, atol=1e-4)
