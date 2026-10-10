"""Train/deploy parity and runner integration for the ONNX reach policy.

The artifact is a randomly initialized PPO with non-trivial normalization statistics, so the
tests need no training run. The parity test is the most important test in the plan: the deploy
policy must build the same observation and produce the same commands as the training env.
"""

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
pytest.importorskip("onnxruntime")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from robot_manipulator.exec.scheduler import SimClock  # noqa: E402
from robot_manipulator.harness.backend import SimBackend  # noqa: E402
from robot_manipulator.harness.policies.onnx_reach import OnnxReachPolicy  # noqa: E402
from robot_manipulator.harness.runner import run_skill  # noqa: E402
from robot_manipulator.harness.safety_layer import SafetyFilter  # noqa: E402
from robot_manipulator.harness.skill import EEReached, Skill, TimeLimit  # noqa: E402
from robot_manipulator.harness.trajlog import TrajectoryLog  # noqa: E402
from robot_manipulator.sim.envs.reach import (  # noqa: E402
    GRIPPER_HOLD,
    HOME_ARM,
    DRConfig,
    ReachEnv,
)
from robot_manipulator.sim.export import export_policy, normalize_obs  # noqa: E402
from robot_manipulator.sim.kinematics import FK  # noqa: E402


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
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


def _env_joint_obs(env: ReachEnv) -> dict:
    """What the deploy side sees from a backend: six CMD-order joints plus the goal."""
    q6 = np.r_[env.data.qpos[env._qadr[:5]], GRIPPER_HOLD]
    dq6 = np.r_[env.data.qvel[env._dof[:5]], 0.0]
    return {"q": q6, "dq": dq6, "goal": {"target": env.target.copy()}}


def test_meta_mismatch_is_refused(artifact, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "reach.onnx").write_bytes((artifact / "reach.onnx").read_bytes())
    meta = json.loads((artifact / "policy_meta.json").read_text())
    meta["action_scale"] = 0.03
    (bad / "policy_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="action_scale"):
        OnnxReachPolicy(bad / "reach.onnx")


def test_train_and_deploy_build_the_same_obs_and_commands(artifact):
    """Deploy obs == env obs, and deploy commands == env commands, on an identical rollout."""
    import onnxruntime as ort

    meta = json.loads((artifact / "policy_meta.json").read_text())
    sess = ort.InferenceSession(str(artifact / "reach.onnx"), providers=["CPUExecutionProvider"])
    deploy = OnnxReachPolicy(artifact / "reach.onnx")

    env = ReachEnv(dr=DRConfig(enabled=False))
    obs_env, _ = env.reset(seed=1)
    max_obs_err = 0.0
    max_cmd_err = 0.0
    for _ in range(60):
        row = deploy.act(_env_joint_obs(env))  # one (1, 6) command row
        max_obs_err = max(max_obs_err, float(np.max(np.abs(deploy.last_obs - obs_env))))

        x = normalize_obs(obs_env, meta["obs_mean"], meta["obs_var"], meta["clip_obs"])
        a = np.clip(sess.run(None, {"obs": x[None]})[0][0], -1.0, 1.0)
        obs_env, _, _, _, _ = env.step(a)
        max_cmd_err = max(max_cmd_err, float(np.max(np.abs(row[0, :5] - env._q_cmd))))

    assert max_obs_err < 1e-6, f"observation drift {max_obs_err}"
    assert max_cmd_err < 1e-9, f"command drift {max_cmd_err}"


def test_gripper_row_is_held_open(artifact):
    deploy = OnnxReachPolicy(artifact / "reach.onnx")
    env = ReachEnv(dr=DRConfig(enabled=False))
    env.reset(seed=2)
    row = deploy.act(_env_joint_obs(env))
    assert row.shape == (1, 6)
    assert row[0, 5] == pytest.approx(GRIPPER_HOLD)


def test_reset_reinitializes_the_command(artifact):
    deploy = OnnxReachPolicy(artifact / "reach.onnx")
    env = ReachEnv(dr=DRConfig(enabled=False))
    env.reset(seed=3)
    deploy.act(_env_joint_obs(env))
    assert deploy.q_cmd is not None
    deploy.reset()
    assert deploy.q_cmd is None and np.all(deploy.last_action == 0)


def test_runs_under_runner_on_sim_backend(artifact, tmp_path):
    fk = FK()
    lo = np.full(6, -2.0)
    hi = np.full(6, 2.0)
    clock = SimClock()
    backend = SimBackend(hz=50.0)
    backend.reset(np.r_[HOME_ARM, GRIPPER_HOLD])
    safety = SafetyFilter(lo=lo, hi=hi, max_vel=1.0, clock=clock)
    target = fk.sample_reachable(np.random.default_rng(4), 1, lo[:5], hi[:5])[1][0]

    sim_fk = FK(backend.model)
    skill = Skill(
        "reach",
        OnnxReachPolicy(artifact / "reach.onnx"),
        [EEReached(0.02, ee_fn=lambda obs: sim_fk.ee_pos(obs["q"])), TimeLimit(4.0)],
        hz=50.0,
    )
    log = TrajectoryLog(tmp_path / "run")
    res = run_skill(skill, backend, safety, {"target": target}, clock=clock, log=log)
    backend.close()
    assert res.status in ("success", "failure")
    assert res.n_steps > 0
    assert (tmp_path / "run" / "traj_reach_1.parquet").exists()
