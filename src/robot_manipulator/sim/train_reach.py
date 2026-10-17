"""Train the SO101 reach policy with PPO (Stable-Baselines3) on CPU, then export it for the harness.

    uv run sim-train-reach --total-steps 3000000 --n-envs 12 --dr --seed 0

Outputs under `--out` (default `runs/ppo_reach/<timestamp>`):
    best_model.zip, vecnormalize.pkl     best checkpoint by DR success rate, with its stats
    final_model.zip, vecnormalize_final.pkl
    progress.csv, eval_history.json      training logs and periodic success rates
    eval_final.json                      final success with and without DR, from the best model
    reach.onnx, policy_meta.json         deployable policy (see sim/export.py)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _linear(start: float):
    """SB3 schedule: decays linearly from `start` to 0 over training."""
    return lambda progress_remaining: progress_remaining * start


# Rollout = n_steps * n_envs (12 288 at 12 envs) split into minibatches of 2 048 -> 60 gradient
# steps per update. The original plan had batch_size (4 096) > rollout (3 072), i.e. a single
# full-batch step per epoch, which stalled learning (docs/harness-sim2real.md B1).
PPO_KWARGS: dict[str, Any] = dict(
    n_steps=1024,
    batch_size=2048,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=_linear(3e-4),
    ent_coef=0.0,
    clip_range=0.2,
    target_kl=0.02,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)


def _make_env(dr_enabled: bool):
    from robot_manipulator.sim.envs.reach import DRConfig, ReachEnv

    return ReachEnv(dr=DRConfig(enabled=dr_enabled))


def _vec(dr_enabled: bool, n_envs: int, dummy: bool):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [partial(_make_env, dr_enabled) for _ in range(n_envs)]
    return DummyVecEnv(fns) if (dummy or n_envs == 1) else SubprocVecEnv(fns)


def evaluate(model, eval_vn, n_episodes: int) -> dict[str, float]:
    """Deterministic rollouts; success is `info["success"]` on each episode's last step."""
    obs = eval_vn.reset()
    done_n = 0
    succ = 0
    final_dist: list[float] = []
    while done_n < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = eval_vn.step(action)
        for i, done in enumerate(dones):
            if done:
                done_n += 1
                succ += int(bool(infos[i].get("success", False)))
                final_dist.append(float(infos[i].get("dist", np.nan)))
    return {
        "success_rate": succ / done_n,
        "mean_final_dist": float(np.nanmean(final_dist)),
        "episodes": done_n,
    }


def _success_callback(train_vn, eval_vn, out: Path, every: int, n_episodes: int):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import sync_envs_normalization

    class SuccessCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=1)
            self.best = -1.0
            self.history: list[dict[str, Any]] = []
            self._last = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last < every:
                return True
            self._last = self.num_timesteps
            sync_envs_normalization(train_vn, eval_vn)
            res = evaluate(self.model, eval_vn, n_episodes)
            for key, value in res.items():
                self.logger.record(f"eval/{key}", value)
            self.history.append({"timesteps": int(self.num_timesteps), **res})
            print(f"[eval @ {self.num_timesteps}] {res}", flush=True)
            if res["success_rate"] > self.best:
                self.best = res["success_rate"]
                self.model.save(str(out / "best_model"))
                train_vn.save(str(out / "vecnormalize.pkl"))
            return True

    return SuccessCallback()


def train(
    total_steps: int,
    n_envs: int,
    seed: int,
    dr: bool,
    out: Path,
    eval_every: int = 100_000,
    eval_episodes: int = 50,
    final_episodes: int = 100,
    dummy: bool = False,
    batch_size: int = PPO_KWARGS["batch_size"],
) -> dict[str, Any]:
    """Run PPO, keep the best checkpoint, evaluate it, and export. Returns the summary dict."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)
    t0 = time.monotonic()

    venv = _vec(dr, n_envs, dummy)
    venv.seed(seed)
    train_vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True)
    eval_vn = VecNormalize(
        DummyVecEnv([partial(_make_env, dr)]), training=False, norm_reward=False, clip_obs=10.0
    )
    eval_vn.seed(seed + 1000)

    ppo_kwargs = {**PPO_KWARGS, "batch_size": batch_size}
    model = PPO("MlpPolicy", train_vn, seed=seed, device="cpu", verbose=1, **ppo_kwargs)
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    cb = _success_callback(train_vn, eval_vn, out, eval_every, eval_episodes)
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)

    model.save(str(out / "final_model"))
    train_vn.save(str(out / "vecnormalize_final.pkl"))
    (out / "eval_history.json").write_text(json.dumps(cb.history, indent=2) + "\n")

    # Final evaluation of the best checkpoint, with and without DR.
    best = PPO.load(str(out / "best_model.zip"), device="cpu")
    final: dict[str, dict[str, float]] = {}
    for label, dr_flag in (("dr", True), ("nodr", False)):
        vn = VecNormalize.load(str(out / "vecnormalize.pkl"), _vec(dr_flag, 1, True))
        vn.training = False
        vn.norm_reward = False
        vn.seed(seed + (2000 if dr_flag else 3000))
        final[label] = evaluate(best, vn, final_episodes)
    (out / "eval_final.json").write_text(json.dumps(final, indent=2) + "\n")

    from robot_manipulator.sim.export import export_policy

    wall = time.monotonic() - t0
    onnx_path, meta_path = export_policy(
        out / "best_model.zip",
        out / "vecnormalize.pkl",
        out,
        dr=dr,
        train_steps=total_steps,
        eval_success={k: v["success_rate"] for k, v in final.items()},
        extra={
            "n_envs": n_envs,
            "seed": seed,
            "wall_time_s": round(wall, 1),
            "best_train_eval_success": cb.best,
            "ppo": {
                k: (f"linear({v(1.0)})" if callable(v) else v)
                for k, v in ppo_kwargs.items()
                if k != "policy_kwargs"
            },
        },
    )
    summary = {
        "out": str(out),
        "wall_time_s": round(wall, 1),
        "total_steps": total_steps,
        "final_eval": final,
        "onnx": str(onnx_path),
        "meta": str(meta_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@app.command()
def main(
    total_steps: int = typer.Option(3_000_000, help="PPO environment steps"),
    n_envs: int = typer.Option(12, help="parallel environments"),
    seed: int = typer.Option(0),
    dr: bool = typer.Option(True, "--dr/--no-dr", help="domain randomization in training"),
    out: Path | None = typer.Option(None, help="output dir (default runs/ppo_reach/<ts>)"),
    eval_every: int = typer.Option(100_000, help="env steps between success evaluations"),
    eval_episodes: int = typer.Option(50),
    final_episodes: int = typer.Option(100),
    dummy: bool = typer.Option(False, help="single-process vec env (smoke tests)"),
    batch_size: int = typer.Option(PPO_KWARGS["batch_size"], help="PPO minibatch size"),
) -> None:
    if out is None:
        out = Path("runs") / "ppo_reach" / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = train(
        total_steps=total_steps,
        n_envs=n_envs,
        seed=seed,
        dr=dr,
        out=out,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        final_episodes=final_episodes,
        dummy=dummy,
        batch_size=batch_size,
    )
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
