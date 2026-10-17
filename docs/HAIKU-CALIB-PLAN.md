# Haiku work order — calibration check + auto-calibration (≈30 min, software only)

Written 2026-10-17 by Opus. The user is away. Branch: `harness-sim2real`.

## Hard rules
1. **No hardware.** Both arms are plugged in and powered. Never open `/dev/so101_*`, never run `uv run teleop`, never
   run `--live` on anything, and never write to `~/.cache/huggingface/lerobot/calibration/`. Tests use fakes only.
2. Each ticket ends with: `uv run pytest` green, `uv run ruff check <touched files>` clean, then one commit
   (`feat(calib): …`). The commit footer is `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` if you are
   Opus; use your own model name otherwise.
3. Reuse; don't re-derive. Read `lerobot/motors/motors_bus.py`, `set_half_turn_homings` / `_get_half_turn_homings`
   and `_normalize` / `_unnormalize` (in `.venv/lib/python3.12/site-packages/lerobot/`). Mirror LeRobot's homing-offset
   sign convention exactly, and prove it with a round-trip test against LeRobot's own math.

## Why (context)
Teleop showed the follower's `shoulder_pan` and `wrist_roll` sitting at different angles from the leader.
LeRobot `DEGREES` mode puts each joint's zero at the **midpoint of the calibrated range** (`(range_min+range_max)/2`).
For full-turn joints (`wrist_roll`, range 0..4095) the zero comes from the homing offset alone.

Our calibration files (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_so101_follower.json` and
`teleoperators/so_leader/my_so101_leader.json`) show two problems:
- The follower's `shoulder_pan` range is 1905 counts. The leader's is 2712, and the NVIDIA fleet mean is 2725 ± 32
  (15 arms). So the follower sweep was incomplete, and its zero shifted.
- The follower's `wrist_roll` homing is −2 against the leader's −1253, about 110°, left over from a July re-zero.
- The follower's `elbow_flex` is 2128 against a fleet mean of 2222 ± 9.

The user wants (a) a checker that catches this, and (b) **auto-calibration**: calibrate one arm by hand once, store a
shareable *profile*, and have any SO101 calibrate itself from it by creeping to its mechanical stops. The fleet stats
show that the stops are consistent across arms.

## Step 0 — commit pending work (2 min)
These files are uncommitted. Commit them as-is in one commit, `feat(teleop): staged teleop tool + research notes`:
- `src/robot_manipulator/tasks/teleop_staged.py`
- `tests/test_teleop_staged.py`
- `docs/RESEARCH-deployment-harness.md`
- `pyproject.toml` (the `teleop` entry point)
- this file

Run pytest first.

## C1 — Calibration checker (≈10 min)
- Vendor the NVIDIA fleet stats to `config/calibration_stats_so101.json`, fetched with
  `curl -s https://raw.githubusercontent.com/isaac-sim/Sim-to-Real-SO-101-Workshop/main/docker/real/scripts/calibration_stats.json`.
  Then add a top-level key `"_source"` with the URL and "Apache-2.0, isaac-sim/Sim-to-Real-SO-101-Workshop, n=15
  follower arms". The gripper stats don't apply to the PincOpen, so mark `"gripper": {"skip": true, ...}` or drop it.
- New package `src/robot_manipulator/calib/` (`__init__.py`, `check.py`):
  - `load_calibration(path) -> dict[str, dict]` (plain JSON; no draccus needed).
  - `range_counts(cal) -> dict[str, int]`.
  - `fleet_check(cal, stats, n_std=2.0) -> list[Issue]`: an issue when the range deviates > `n_std` σ, or
    `|homing_offset| > 2048`. Skip joints marked skip.
  - `frame_mismatch(cal_a, cal_b) -> dict[str, float]`: an **estimated degree-frame offset** between two arms' zeros.
    For range-defined joints it's the difference in where each arm's midpoint sits relative to its own range. It is
    only *indicative*, since the calibration files can't know the physical pose. For full-turn joints, use
    `(homing_a − homing_b) * 360/4096`. Document clearly that this is a hint and that physical alignment (C2 `align`)
    is the ground truth.
  - CLI `calib-check` (typer, entry point in pyproject): `uv run calib-check [--follower PATH] [--leader PATH]`,
    defaulting to the two cache files above (read-only). Prints a rich table:
    joint | range | fleet mean±σ | σ-dev | homing | flags. Exits 1 on issues.
- Tests `tests/test_calib_check.py`, using inline fixture dicts copied from the real numbers above:
  - the follower pan is flagged (|σ| > 2);
  - the leader pan is not flagged;
  - wrist_roll homing difference → ≈110° (`(−2 − (−1253)) * 360/4096 ≈ 110.0`);
  - a skipped gripper never flags;
  - the CLI exit code via `typer.testing.CliRunner` with `--follower`/`--leader` pointing at tmp files.

## C2 — Auto-calibration core, fake-bus only (≈18 min)
`src/robot_manipulator/calib/autocal.py`:
- `class JointBus(Protocol)`: `read_raw(joint) -> int` (0..4095, *before* homing), `set_goal_raw(joint, raw)`,
  `read_load(joint) -> int` (signed, ±1000), `set_torque_limit(joint, permille)`, `enable(joint)`, `disable(joint)`.
  Raw-unit level only. The real adapter over LeRobot's `FeetechMotorsBus` comes later with Opus and the user: write a
  stub class `LeRobotJointBus` that raises `NotImplementedError("live autocal is reviewed with the user first")`.
- `@dataclass JointProfile`: `name`, `method: Literal["hard_stop", "manual_zero"]`,
  `expected_range: int | None`, `range_tol: int` (counts), `torque_permille: int = 250`, `step_counts: int = 4`
  (per tick; about 17°/s at 50 Hz), `stall_err: int = 40`, `stall_ticks: int = 10`, `load_abort: int = 400`,
  `backoff: int = 15`.
  `@dataclass CalibProfile`: `robot: str`, `joints: dict[str, JointProfile]`, `zero_offsets_deg: dict[str, float]`
  (optional per-joint shift of zero from the stop midpoint, measured once on the reference arm; default 0).
  Plus `save_yaml` / `load_yaml`.
- `find_stop(bus, joint, direction: ±1, prof, max_ticks) -> int`: creep the goal by `direction*step_counts` per tick,
  **unwrapping** raw across 4095→0. A stall is `|goal − present| > stall_err` for `stall_ticks` ticks, or
  `|load| > load_abort`. On a stall, set goal = present ∓ backoff and return the unwrapped present at the stall.
  Raise if `max_ticks` is exceeded.
- `calibrate_joint(bus, prof) -> dict` (LeRobot calibration entry: `id` left to the caller, `drive_mode`,
  `homing_offset`, `range_min`, `range_max`):
  - `hard_stop`: find both stops → unwrapped `lo`, `hi` → `mid = (lo+hi)/2` → homing chosen with LeRobot's convention
    so that mid reads 2047 (or 2047 + `zero_offsets_deg * 4096/360`) → `range_min/max` = stops in homed units.
    Validate `0 ≤ range_min < range_max ≤ 4095`, `|homing| ≤ 2047`, and, when `expected_range` is set,
    `|range − expected| ≤ range_tol`; raise `CalibError` with a clear message otherwise.
  - `manual_zero` (wrist_roll; there are no stops): takes a `zero_raw` argument (the raw position read while the user
    holds the reference pose), so homing makes that pose read 2047. Range is 0..4095.
- `autocalibrate(bus, profile, order, manual_zero_raw: dict) -> dict[str, dict]`: one joint at a time in `order`, torque
  limited, returning the full calibration dict. Never touches the gripper unless it's in `order`.
- `fleet_profile(stats_json) -> CalibProfile`: builds a default profile with `expected_range = mean` and
  `range_tol = max(3σ, 40)`, `wrist_roll → manual_zero`.
- `FakeJointBus` (put it in `tests/fakes_calib.py`, not in src): per joint a true raw offset (including one joint whose
  range **wraps across 0**), hard stops in link angle, `present` following the goal at a bounded speed, load rising
  when pushed against a stop, Gaussian noise ±2 counts.
- Tests `tests/test_calib_autocal.py`:
  - recovers range within ±(backoff+5) counts for 3 joints, including the wrapping one;
  - homing convention round-trip: build a LeRobot `MotorCalibration` from the result and check with LeRobot's
    `_normalize` math (or a faithful copy, cited) that the stop midpoint reads 0.0°;
  - a stall triggers on load before position error when `load_abort` is low;
  - `CalibError` when the range is out of tolerance;
  - `manual_zero` puts `zero_raw` at 2047;
  - `max_ticks` raises;
  - the profile YAML round-trips.
- **No CLI with live hardware in this ticket.** A `calib-auto --dry-run` that runs against `FakeJointBus` and prints
  the resulting JSON is fine.

## C3 — Notes (≈2 min)
Append a "Calibration" section to `docs/harness-sim2real.md`:
- the diagnosis table (pan / wrist_roll / elbow vs fleet);
- what C1/C2 provide;
- the live steps left for Opus + user:
  1. `LeRobotJointBus` adapter;
  2. safe joint order and folded start pose;
  3. reference-arm profile capture;
  4. follower re-calibration;
  5. leader;
  6. then re-test teleop alignment.

## Stop conditions
If a ticket can't go green in its time box, commit what's green with a `WIP:` note in `docs/harness-sim2real.md` and
stop. Don't improvise hardware steps.
