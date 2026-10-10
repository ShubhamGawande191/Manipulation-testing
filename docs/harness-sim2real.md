# Harness + sim2real — execution log

Branch `harness-sim2real` (from `minilab-phase0`). Work order: [HARNESS-SIM2REAL-PLAN.md](HARNESS-SIM2REAL-PLAN.md).
Software only — no hardware was connected or torqued (plan hard rule 1). Hardware tickets H1–H4 are pending for the user.

## Environment notes

- **`ruff check .` baseline is not clean on `minilab-phase0`.** 13 errors exist in files this work does not touch
  (`scripts/launch_bridge.py`, `scripts/teleoperate.py`, `tasks/calculate_world_coords.py`,
  `tasks/calibrate_follower.py`, `tasks/calibrate_intrinsics.py`). They were left alone. The gate used for every ticket is
  **zero new ruff findings**, with the total held at 13.
- **`openclaw` pin (T0).** PyPI now lists only `openclaw` 2.0.1 and 2.0.2 (no 2026.x), so any fresh `uv lock` fails,
  even on the original `pyproject.toml`. `pyproject.toml` pins the exact wheel the previous lock used through
  `[tool.uv.sources]`. The sha256 (`16b84601…60dfb`) matches the old lock. Revert this pin once you decide what
  `openclaw` should be (it is imported only by `scripts/pick_and_place.py`).
- **Install command.** Use `uv sync --extra sim --extra dev --extra midi --extra audio`. Leaving out `midi`/`audio` uninstalls
  `python-rtmidi` and `sounddevice`.

## Ticket status

| Ticket | Status | Commit | Result |
|---|---|---|---|
| T0 Branch, deps, ignores | done | cd4df6c | `import mujoco, stable_baselines3, onnxruntime` OK via `uv run`. pytest 42 passed. ruff: 0 new findings. |
| T1 Vendor SO101 MJCF | done | 0069616 | Menagerie `robotstudio_so101` @ `3abca2e` (Apache-2.0), 22 files, 18 MB. Six actuators and joints in CMD order; limits match plan to 1e-4. pytest 47 passed (5 new). |
| T2 Units / joint map | done | 3b2cd02 | `harness/units.py` is the only LeRobot↔sim converter; config is uncalibrated placeholders (TODO-H2). Round-trips ±1e-9 for sign ±1 and offsets. Gripper clip and missing-key errors covered. pytest 63 passed (16 new). |
| T3 Backends | done | c65f21d | Fake, Sim (MuJoCo, 4 substeps @ 50 Hz) and LeRobot backends. Sim tracks a constant target within 0.02 rad after 2 s. LeRobot runs only with an injected fake robot. `calibrated` guard blocks live use. pytest 74 passed (11 new). |
| T4 Safety filter | done | 68a6013 | Clamp to limits ± margin, then velocity cap. Stale watchdog holds the previous command (matches Arbiter FROZEN). Limits default to MJCF ranges. pytest 84 passed (10 new). |
| T5 Skill runtime | done | e4a5a63 | `Skill`/`SkillResult`, end conditions (TimeLimit, JointReached, EEReached, Stalled), `run_skill` with redo, `run_sequence` stops at first failure, `TrajectoryLog` → `traj_<skill>_<n>.parquet`. `EpisodeRecord` gains 4 defaulted fields. Tests at 8 Hz with SimClock. pytest 96 passed (12 new + episode log unchanged). |

| T6 Kinematics helper | done | e7be2e2 | `sim/kinematics.FK`: ee_pos, 3×6 Jacobian (matches finite differences to 1e-5), `sample_reachable` (z ≥ z_min, in limits). `SimBackend.ee_pos` now recomputes kinematics. `site_xpos` otherwise lags qpos by one substep. FK equals SimBackend ee to 1e-12. pytest 102 passed (6 new). |

| T7 Reach env | done | 036d184 | `SO101Reach-v0`: 24-d obs via `build_obs`, action → `integrate_command` (ACTION_SCALE 0.02 at T7, now 0.03 — see c6e46b5, margin 0.05, gripper held at 1.74533). Targets from 80 % of limits. DR covers kp/kv, damping, friction, armature, link mass, gripper payload 0–0.25 kg, delay {0,1,2}, obs noise, per-episode bias. Nominal restored each reset (no drift). `check_env` passes. Acceptance: Jacobian-pinv policy with DR off reaches **18/20** (threshold 16). pytest 114 passed (12 new). |

| T8 PPO train + export | **acceptance NOT met**: best 82 % (DR, 100 eps) vs 90 % (see B1) | f0c150b (code), c6e46b5 (ACTION_SCALE 0.03) | Code: `sim/train_reach.py` (SB3 PPO, plan hyperparameters, success-rate callback, best-by-DR checkpoint, final DR/no-DR eval, export) and `sim/export.py` (ONNX actor on normalized obs + `policy_meta.json`). Smoke test: 2 048 steps → ONNX matches SB3 deterministic predict to 1e-4 on 10 obs. Full-length runs: see Blockers. |
| T9 ONNX reach policy | done | d771518 | `harness/policies/onnx_reach.py`. Imports `build_obs`, `command_box`, `integrate_command`, `normalize_obs` from the training code, and refuses `policy_meta.json` whose constants differ (action_scale, hz, gripper_hold, joint order, obs layout). Parity over a 60-step rollout (DR off): observation max |Δ| < 1e-6, command max |Δ| < 1e-9. pytest passes (5 new). |
| T10 Harness CLI | done; **acceptance met** (≥ 85 %): 89 % at the plan's default `--max-vel 1.0`, 90 % at `--max-vel 1.5` (see B2) | 28d1ac7 (code), c6e46b5 (warning) | `harness reach` runs `home` (GlideTo, velocity-capped) then `reach` (ONNX policy) per episode, through `run_sequence`, the safety filter, and trajectory logs. Success is scored on the sim's true ee, like the env. `--backend real` is refused unless `calibrated: true` and `--live`; the guard runs before any serial connection. Tests use sim and fake only. |
| T11 Gap toolkit | done | 3803724 | `sim/gap/`: motions (CSVs committed under `config/gap_motions/`), replay through the runner, compare (RMSE, lag by cross-correlation, overshoot, steady-state error, PNG overlay), and fit (`least_squares` per joint, delay grid 0–3, writes `config/actuator_fit.yaml`). Synthetic-real test (kp×0.8, damping×1.5, delay 1) recovers kp and damping within 15 % and delay exactly. |

Deviation (T11): motions are centred on the mid-range arm pose, not on HOME. HOME sits within 0.3 rad of the shoulder_lift and elbow limits, and the ±0.3 rad chirp clipped 35–45 % of its samples. With the centred base, no motion is clipped. The chirp peaks at about 3.8 rad/s, so it needs slower motions for hardware replay at `--max-vel 0.5` (H3).

Caveat for T5 (resolved in T5): `BasePolicy.act` computes staleness against `time.monotonic()`. `run_skill`
now overwrites `obs["t"]` with the runner clock, so health is correct only with `WallClock`. With `SimClock`
it still reads as stale. Nothing gates on health yet, so this does not affect results so far.

## Blockers

### B1 — T8 acceptance: not met. Best env-level success 82 % (DR, 100 eps) against 90 %

Plan settings (`n_steps` 256, `batch_size` 4096, 12 envs, DR on, seed 0), 3 M steps:

| Attempt | Steps | Wall time | Final success, DR (100 eps) | Final success, no DR (100 eps) | Notes |
|---|---|---|---|---|---|
| 1 (plan) | 3 M | 802 s | 2 % (mean final dist 0.134 m) | 2 % (0.128 m) | Best-by-DR checkpoint (6 % on 50 eval eps, at 1.8 M) was exported. Eval success stayed at 0–6 % from 0.8 M onward. |
| Diagnostic (not in plan) | 1 M | 353 s | 55 % (mean final dist 0.050 m) | 61 % (0.058 m) | `--batch-size 256`, otherwise identical. Training-time DR eval reached 70 % at 1 M. Run1 was at 0 % at 1 M. This supports the batch-size hypothesis below. |
| Deviation (not in plan) | 3 M | 983 s | 55 % (same as the diagnostic: same seed, same best checkpoint at 1 M) | 61 % | `--batch-size 256`, 3 M steps. Training eval peaked at 70 % at 1000 k steps, then fell to 20–40 % by 2.8–3 M. The best-checkpoint rule keeps the 1 M model. Labelled a deviation because the plan fixes `batch_size` at 4096. |
| 2 (plan escalation) | 6 M | 1763 s | 54 % (mean final dist 0.067 m) | 51 % (0.069 m) | Plan settings. Training-time DR eval climbed slowly: 2 % at 1.7 M, 50 % at 5.3 M. Best-by-DR checkpoint was exported, with a training-time success of 74 %. |
| 3 (plan escalation) | 6 M | 1601 s | **82 %** (mean final dist 0.028 m) | 80 % (0.034 m) |  `ACTION_SCALE = 0.03` set in `sim/envs/reach.py` for this run only, then reverted to 0.02. A policy from this run would need `--max-vel` ≥ 1.5 at deploy, because the safety filter caps steps at max_vel × dt. | `ACTION_SCALE = 0.03`, otherwise plan settings. Best-by-DR checkpoint: 96 % on 50 eps at 5.5 M, which is the one exported (96 % on its selection eval). That is selection optimism: 82 % on 100 fresh eps. The DR-off figure is 80 %. |

**Escalation outcome.** Steps alone (6 M) did not reach the bar, and `ACTION_SCALE` 0.03 did the most. The batch-256 diagnostic learned fastest early (55 % at 1 M), then decayed. Neither changed the bottom line: no run reached 90 % on 100 fresh episodes. The plan's escalation is now exhausted.

**Decision taken (for you to review):** `ACTION_SCALE` is 0.03 in the code (c6e46b5), because it gave the best artifact. Reverting to 0.02 would make the shipped metadata refuse that policy. Shipping 0.02 would also make a 0.03 policy's commands get clipped at deploy time if `--max-vel` is left at 1.0.

**Likely cause of the early plateau (hypothesis):** `n_steps 256 × 12 envs = 3 072` samples per update, which is smaller than `batch_size 4096`. Each PPO epoch is therefore one full-batch gradient step, about 10 per update and roughly 10 k in total over 3 M steps. Standard PPO runs use minibatches of 64–512 and far more gradient steps. The plan's hyperparameters were kept for attempts 1–3. The diagnostic row is a separate experiment and does not replace them.

**Effect on the rest of the plan:** T10's acceptance (≥ 85 % in the harness) needs a policy that meets the bar. T9 (ONNX policy and parity) and T11 (gap toolkit) do not depend on policy quality and are complete. The harness CLI runs on any artifact, so the T10 code is done. Only its acceptance number is blocked.

### B2 — T10 acceptance: met (≥ 85 %), with two caveats

Command (plan's exact form): `uv run harness reach --backend sim --policy <attempt-3 reach.onnx> --random 100 --dr`. Policy: the 6 M-step `ACTION_SCALE` 0.03 artifact (`runs/ppo_reach/attempt3_6M_as003/`, not committed; the ONNX is `.gitignore`d).

| Run | Success (100 eps, DR) | Mean final dist | Mean time to success |
|---|---|---|---|
| `--max-vel 1.0` (plan default) | 89 % | 0.027 m | 1.83 s |
| `--max-vel 1.5` (matches the policy's 1.5 rad/s step) | 90 % | 0.025 m | 1.10 s |

Caveats:
1. **Policy scale.** The T10 threshold passes, but only with a policy trained at `ACTION_SCALE` 0.03. The plan's constant was 0.02, and that policy plateaued at about 55 %.
2. **Harness versus env.** The harness run uses seed-0 targets through the full path (home glide, safety filter, runner). It scores 90 %, while the env eval gave 82 % on different seeds. The gap is within sampling noise for n = 100 (standard error about 4 points), but the two are not the same test.

### B3 — Hardware tickets H1–H4 (user, not started)

Out of scope for this run by design. `calibrated: false` in `config/sim2real.yaml` blocks live LeRobot runs.
