# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A lab for testing control methods on the **SO101 12V** 6-DOF arm (Feetech STS3215 servos × 6, IDs 1–6). It spans three control layers that coexist in the repo: direct LeRobot servo control, a ROS2 + WebSocket bridge, and vision-guided pick-and-place tasks. The README's "Progress Tracker" is the source of truth for what currently works vs. abandoned (e.g. leader arm is uncalibrated — motor ID 2 has a broken cable; the OpenClaw/CMDOP agent route was abandoned in favor of ROS2).

## Commands

```bash
uv sync                                   # install deps + dev extras into .venv
uv run pytest                             # run smoke tests (tests/, import-level, no hardware)
uv run pytest tests/test_smoke.py::test_scripts_importable   # single test
uv run ruff check .                       # lint (E, F, I, UP; line-length 100, py312)
uv run ruff format .                      # format

uv run calibrate --follower-port /dev/so101_follower   # wraps lerobot-calibrate
uv run teleoperate                        # wraps lerobot-teleoperate
uv run pick-and-place                     # scripts/pick_and_place:app entry point
```

ROS2 bridge requires Ubuntu 24.04 + ROS2 Jazzy and a separate `~/ros2_ws/` workspace (not in this repo — see README "Project Layout"). Launch sequence is three terminals: (1) `ros2 launch so101_bringup follower.launch.py`, (2) `uv run python -m robot_manipulator.scripts.launch_bridge`, (3) a WebSocket client. **Always `source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash` before ROS2 commands**, and `unset VIRTUAL_ENV` before `colcon build` (the uv venv leaks into the ROS2 build otherwise).

## Two control paths — do not confuse them

**1. Direct LeRobot (`src/robot_manipulator/arm.py`):** `build_follower_config()` → `SOFollower` from LeRobot 0.5. Talks to the servos over USB serial directly. **Joint commands are in DEGREES** via `robot.send_action({"shoulder_pan.pos": ..., "gripper.pos": ...})`. `robot_id` must match the calibration file under `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`. Pass `cameras=False` for fast joint-only ops. The `skills/so101.md` skill file documents this path for agent use (and notes the CMDOP shell is `sh`, so it invokes `.venv/bin/python` directly rather than activating the venv).

**2. ROS2 bridge (`src/robot_manipulator/so101_bridge.py`):** `SO101Plugin`/`SO101Controller` for agent-ros-bridge. Subscribes to `/follower/joint_states`, publishes to `/follower/forward_controller/commands`. Exposed over WebSocket (port 8765) as `arm.get_state`, `arm.home`, `arm.move_joints`, `arm.gripper`, `arm.stop`, `arm.camera`. **Joint commands here are in RADIANS.** `HOME = [0, -1.57, 1.57, 0, 0, -0.4]`.

The vision tasks (`tasks/pick_and_place.py`, `tasks/pick_black_cube.py`) drive path 2 via `BridgeClient` (a WebSocket client), so they speak radians.

**3. Harness + MuJoCo sim (`src/robot_manipulator/harness/`, `src/robot_manipulator/sim/`):** one runtime for skills (policy + end conditions + redo) over three backends: `FakeBackend`, `SimBackend` (MuJoCo, vendored SO101 MJCF in `sim/assets/so101/`, Menagerie @ `3abca2e`), and `LeRobotBackend` (path 1 underneath). Everything inside the harness and sim is **radians in CMD order**. Degrees and the 0..100 gripper scale are converted only in `harness/units.py`. The joint map is `config/sim2real.yaml`. Keep `calibrated: false` until hardware ticket H2 is done, since live LeRobot runs are refused otherwise. Train/deploy parity: `sim/envs/reach.py` (`build_obs`, `command_box`, `integrate_command`) is shared by the training env and `harness/policies/onnx_reach.py`, and the policy metadata is checked at load time. Commands: `uv sync --extra sim --extra dev --extra midi --extra audio`, `uv run sim-train-reach --out runs/ppo_reach/<ts>`, `uv run harness reach --backend sim --policy runs/ppo_reach/<run>/reach.onnx --random 100 --dr --max-vel 1.5`. The policy's step is `ACTION_SCALE * 50` = 1.5 rad/s, and the safety cap must be at least that or commands get clipped (the CLI warns). Actuation-gap tooling lives in `sim/gap/` (motions in `config/gap_motions/`, fits in `config/actuator_fit.yaml`).

### Joint ordering is the #1 footgun

The forward controller command order and the `joint_states` topic order differ. `so101_bridge.py` defines both (`CMD_JOINT_ORDER`, `STATE_JOINT_ORDER`) and remaps between them (`_STATE_TO_CMD` / `_CMD_TO_STATE`). When you touch joint arrays, confirm which ordering you're in:

| index | CMD order (publish) | STATE order (`joint_states`, alphabetical) |
|---|---|---|
| 0 | shoulder_pan | elbow_flex |
| 1 | shoulder_lift | gripper |
| 2 | elbow_flex | shoulder_lift |
| 3 | wrist_flex | shoulder_pan |
| 4 | wrist_roll | wrist_flex |
| 5 | gripper | wrist_roll |

The vision tasks use CMD order throughout, with `JOINT_LIMITS`/`_clamp()` in `tasks/pick_and_place.py` as the canonical joint-limit clamp.

## Pick-and-place strategy

`tasks/pick_and_place.py` is the shared library (Detector, `BridgeClient`, `_clamp`, `make_scan_grid`); `tasks/pick_black_cube.py` imports from it. The approach is **scan + closed-loop visual servo**, not open-loop IK: scan the workspace overhead camera to roughly locate the object, jump the arm above it, then servo the **wrist** camera over the target by nudging `shoulder_pan` (PAN_GAIN) and `wrist_flex` (WF_GAIN) until the object centers, then lower/grip/lift. The separate AprilTag/ChArUco intrinsics pipeline (`tasks/detect_apriltag.py`, `find_world_tag.py`, `calibrate_intrinsics.py`, `calculate_world_coords.py` + `anchor_config.json`, `camera_intrinsics.json`) is an alternative geometric-projection path; note `pick_black_cube.py` deliberately ignores `camera_intrinsics.json` (it was overfit, rms_error ~5px) and hardcodes nominal 640×480 webcam intrinsics.

## MiniLab 3 project (current focus)

`minilab/PLAN.md` is the locked project plan: the arm(s) play an Arturia MiniLab 3 MIDI keyboard, with MIDI as automatic ground-truth scoring (robopianist-style). The Phase 0 harness lives under `src/robot_manipulator/`:

- `core/` — `policy_iface.py` (Policy protocol: `reset`/`act`/`health`), `arbiter.py` (fallback state machine ACTIVE→FROZEN→RECOVERY→TELEOP), `safety.py` (watchdog, velocity caps, workspace boxes), `episode_log.py` (parquet episode schema — the single source for all blog tables)
- `io/` — `midi.py` (MidiEvent codec, EventLog, 256-dim MidiState vector, live MidiListener), `audio.py` (per-episode wav, episode-clock aligned)
- `exec/scheduler.py` — chunk executors (`sync_replan`, `temporal_ensemble`, `async_client`, `rtc`) behind one interface; unit-tested against `SimClock` + mock bus with exact deterministic traces (tests use 8 Hz so dt=0.125 is float-exact)
- `eval/scorer.py` — task suite v0 scoring (press key, sequence, hold, fader→CC, encoder→ΔCC); the only scoring path

Hardware tests are marked `@pytest.mark.hw` and deselected by default (`addopts = -m 'not hw'`); run them with `uv run pytest -m hw`. Live MIDI needs `uv sync --extra midi` (rtmidi backend); mic capture needs `--extra audio`. MiniLab 3 shows up as ALSA card `Minilab3`, mido port `Minilab3:Minilab3 MIDI 20:0`.

## Repo hygiene note

The ~26 ad-hoc root experiment scripts (`test_*.py`, `patch*.py`, `req*.py`, `print_*.py`, `verify_wrist.py`) were deleted 2026-07-04 (they were throwaway probes for tuning gains/HSV/servo math; their tuned values already live in `tasks/`). Don't recreate probes in the repo root — put real functionality under `src/robot_manipulator/` and quick experiments outside the repo. `uv run pytest` only runs `tests/` (via `testpaths`).

## Hardware specifics

- Follower: `/dev/so101_follower` (udev symlink, serial 5AA9017934). Leader: `/dev/so101_leader` (serial 5AA9017169, currently broken).
- Cameras (eMeet C950): workspace = OpenCV index 0, wrist = index 2 (odd `/dev/video*` nodes are metadata). Indices shift on reboot — verify with `uv run lerobot-find-cameras opencv`.
- Serial permission errors: `sudo usermod -aG dialout $USER` then re-login.
- `config/so101.yaml` holds ports/camera indices; secrets (OPENCLAW_API_KEY, HF_TOKEN) go in `.env`.
