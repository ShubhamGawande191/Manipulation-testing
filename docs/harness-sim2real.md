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

| T7 Reach env | done | 036d184 | `SO101Reach-v0`: 24-d obs via `build_obs`, action → `integrate_command` (ACTION_SCALE 0.02, margin 0.05, gripper held at 1.74533). Targets from 80 % of limits. DR covers kp/kv, damping, friction, armature, link mass, gripper payload 0–0.25 kg, delay {0,1,2}, obs noise, per-episode bias. Nominal restored each reset (no drift). `check_env` passes. Acceptance: Jacobian-pinv policy with DR off reaches **18/20** (threshold 16). pytest 114 passed (12 new). |

| T8 PPO train + export | **acceptance NOT met** (see Blockers) | (pending) | Code: `sim/train_reach.py` (SB3 PPO, plan hyperparameters, success-rate callback, best-by-DR checkpoint, final DR/no-DR eval, export) and `sim/export.py` (ONNX actor on normalized obs + `policy_meta.json`). Smoke test: 2 048 steps → ONNX matches SB3 deterministic predict to 1e-4 on 10 obs. Full-length runs: see Blockers. |
| T9 ONNX reach policy | done | (pending) | `harness/policies/onnx_reach.py`. Imports `build_obs`, `command_box`, `integrate_command`, `normalize_obs` from the training code, and refuses `policy_meta.json` whose constants differ (action_scale, hz, gripper_hold, joint order, obs layout). Parity over a 60-step rollout (DR off): observation max |Δ| < 1e-6, command max |Δ| < 1e-9. pytest passes (5 new). |
| T10 Harness CLI | code done; **acceptance not met** (needs a policy ≥ 85 %) | (pending) | `harness reach` runs `home` (GlideTo, velocity-capped) then `reach` (ONNX policy) per episode, through `run_sequence`, the safety filter, and trajectory logs. Success is scored on the sim's true ee, like the env. `--backend real` is refused unless `calibrated: true` and `--live`; the guard runs before any serial connection. Tests use sim and fake only. |
| T11 Gap toolkit | done | (pending) | `sim/gap/`: motions (CSVs committed under `config/gap_motions/`), replay through the runner, compare (RMSE, lag by cross-correlation, overshoot, steady-state error, PNG overlay), and fit (`least_squares` per joint, delay grid 0–3, writes `config/actuator_fit.yaml`). Synthetic-real test (kp×0.8, damping×1.5, delay 1) recovers kp and damping within 15 % and delay exactly. |

Deviation (T11): motions are centred on the mid-range arm pose, not on HOME. HOME sits within 0.3 rad of the shoulder_lift and elbow limits, and the ±0.3 rad chirp clipped 35–45 % of its samples. With the centred base, no motion is clipped. The chirp peaks at about 3.8 rad/s, so it needs slower motions for hardware replay at `--max-vel 0.5` (H3).

Caveat for T5 (resolved in T5): `BasePolicy.act` computes staleness against `time.monotonic()`. `run_skill`
now overwrites `obs["t"]` with the runner clock, so health is correct only with `WallClock`. With `SimClock`
it still reads as stale. Nothing gates on health yet, so this does not affect results so far.

## Blockers

### B1 — T8 acceptance: reach policy far below the 90 % target (open)

Plan settings (`n_steps` 256, `batch_size` 4096, 12 envs, DR on, seed 0), 3 M steps:

| Attempt | Steps | Wall time | Final success, DR (100 eps) | Final success, no DR (100 eps) | Notes |
|---|---|---|---|---|---|
| 1 (plan) | 3 M | 802 s | 2 % (mean final dist 0.134 m) | 2 % (0.128 m) | Best-by-DR checkpoint (6 % on 50 eval eps, at 1.8 M) was exported. Eval success stayed at 0–6 % from 0.8 M onward. |
| Diagnostic (not in plan) | 1 M | — | pending | — | `--batch-size 256`, otherwise identical. Tests whether the minibatch size is the bottleneck. |
| 2 (plan escalation) | 6 M | — | pending | — | Plan settings. |
| 3 (plan escalation) | 6 M | — | pending | — | `ACTION_SCALE = 0.03`. |

**Likely cause (hypothesis, not yet confirmed):** `n_steps 256 × 12 envs = 3 072` samples per update, which is smaller than `batch_size 4096`. Each PPO epoch is therefore one full-batch gradient step, about 10 per update and roughly 10 k in total over 3 M steps. Standard PPO runs use minibatches of 64–512 and far more gradient steps. The plan's hyperparameters were kept for attempts 1–3. The diagnostic row is a separate experiment and does not replace them.

**Effect on the rest of the plan:** T10's acceptance (≥ 85 % in the harness) needs a policy that meets the bar. T9 (ONNX policy and parity) and T11 (gap toolkit) do not depend on policy quality and are complete. The harness CLI runs on any artifact, so the T10 code is done. Only its acceptance number is blocked.

### B2 — T10 acceptance: harness success ≥ 85 % (blocked by B1)

Command: `uv run harness reach --backend sim --policy runs/ppo_reach/<best>/reach.onnx --random 100 --dr`. Not run as acceptance until a policy from B1 meets the bar. The CLI was checked with the random-weight artifact from the tests (`tests/test_harness_cli.py`, `tests/test_harness_sim_dr.py`).

### B3 — Hardware tickets H1–H4 (user, not started)

Out of scope for this run by design. `calibrated: false` in `config/sim2real.yaml` blocks live LeRobot runs.
