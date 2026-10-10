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
| T5 Skill runtime | done | (this commit) | `Skill`/`SkillResult`, end conditions (TimeLimit, JointReached, EEReached, Stalled), `run_skill` with redo, `run_sequence` stops at first failure, `TrajectoryLog` → `traj_<skill>_<n>.parquet`. `EpisodeRecord` gains 4 defaulted fields. Tests at 8 Hz with SimClock. pytest 96 passed (12 new + episode log unchanged). |

Caveat for T5: `BasePolicy.act` computes staleness against `time.monotonic()`, but `SimBackend` reports
simulated time in `obs["t"]`. Health readings are wrong for sim obs until the runner sets `t` from its own clock.
Nothing gates on health yet, so this does not affect results so far.

## Blockers

None yet.
