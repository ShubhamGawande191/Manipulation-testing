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
| T0 Branch, deps, ignores | done | (this commit) | `import mujoco, stable_baselines3, onnxruntime` OK via `uv run`. pytest 42 passed. ruff: 0 new findings. |

## Blockers

None yet.
