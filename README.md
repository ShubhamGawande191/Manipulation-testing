# SO101 Robot Manipulation Lab

Testing various control methods on the **SO101 12V** robot arm — from basic calibration through ROS2 control to AI-agent-driven pick-and-place.

---

## Progress Tracker

### Done

- [x] **Project scaffolding** — Git repo, uv venv, pyproject.toml, directory structure
- [x] **LeRobot 0.5.0 API integration** — arm configs, calibration/teleoperation wrappers
- [x] **Hardware discovery** — USB serial ports, cameras, motor IDs
- [x] **Follower arm calibrated** — saved to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_so101_follower.json`
- [x] **Smoke tests passing** — 2/2 (import-level, no hardware required)
- [x] **Pushed to GitHub** — `git@github.com:ShubhamGawande191/Manipulation-testing.git`
- [x] **ROS2 Jazzy workspace built** — 8 packages in `~/ros2_ws/` (so101_bringup, feetech_ros2_driver, etc.)
- [x] **Udev rules** — `/dev/so101_follower` (serial: 5AA9017934), `/dev/so101_leader` (serial: 5AA9017169)
- [x] **Follower bringup verified** — joint states streaming on `/follower/joint_states`, `forward_controller` + `joint_state_broadcaster` both active
- [x] **agent-ros-bridge v0.6.1** — cloned to `~/ros2_ws/src/agent-ros-bridge/`, installed in `.venv`
- [x] **SO101 bridge plugin created** — `so101_bridge.py` with `SO101Controller` + `SO101Plugin`
- [x] **Bridge launcher created** — `launch_bridge.py` with WebSocket transport on port 8765
- [x] **MiniLab Phase 0 harness (software)** — 2026-07-04: `core/` (policy interface, arbiter, safety, episode log), `io/` (MIDI event log + state vector, audio), `exec/` (sync_replan / temporal_ensemble / async_client / rtc executors), `eval/` (MIDI scorer, task suite v0); 33 new unit tests green (`uv run pytest`), MiniLab 3 verified live over `mido`/rtmidi (`uv run pytest -m hw`)

### Next Up

- [ ] **Test bridge end-to-end** — launch bridge, send `arm.get_state` / `arm.home` via WebSocket
- [ ] **Camera integration** — subscribe to ROS2 image topics, test `arm.camera` command
- [ ] **Pick & place** — vision-guided: detect black cube or colored ball → place into pink container
- [ ] **MiniLab Phase 0 exit criterion** — scripted classical primitive presses a key on the MiniLab 3 and passes task suite v0 with a logged episode record (see `minilab/PLAN.md`, `docs/phase0-mvp-harness.md`)

### Paused / Abandoned

- ⏸️ **Leader arm calibration** — motor ID 2 has a loose cable, needs physical fix
- ❌ **OpenClaw/CMDOP agent** — `terminal.execute()` timeouts, `ToolsUsed: 0` issue; abandoned in favor of ROS2 route

---

## Hardware

| Component | Details |
|---|---|
| Follower arm | SO101 12V — Feetech STS3215 servos × 6 (IDs 1-6), `/dev/so101_follower` |
| Leader arm | SO101 12V — Feetech STS3215 servos × 6 (IDs 1-6), `/dev/so101_leader` (motor 2 broken) |
| Workspace camera | eMeet C950 (USB3, index 0) |
| Wrist camera | eMeet C950 (USB1, index 2) |

### Joint ordering (important!)

The forward controller and joint_states use different orderings:

| Forward controller (command order) | joint_states (alphabetical) |
|---|---|
| 0: shoulder_pan | 0: elbow_flex |
| 1: shoulder_lift | 1: gripper |
| 2: elbow_flex | 2: shoulder_lift |
| 3: wrist_flex | 3: shoulder_pan |
| 4: wrist_roll | 4: wrist_flex |
| 5: gripper | 5: wrist_roll |

The `SO101Controller` in `so101_bridge.py` handles this mapping automatically.

---

## Architecture

```
┌─────────────────────┐     WebSocket      ┌──────────────────────┐
│   AI Agent / CLI    │ ◄──── :8765 ─────► │  agent-ros-bridge    │
│   (wscat, Python)   │                     │  (Bridge + Plugins)  │
└─────────────────────┘                     └──────────┬───────────┘
                                                       │
                                            ┌──────────▼───────────┐
                                            │   SO101Plugin        │
                                            │   SO101Controller    │
                                            └──────────┬───────────┘
                                                       │ rclpy
                                    ┌──────────────────┼─────────────────┐
                                    │                  │                  │
                        ┌───────────▼──┐   ┌───────────▼──┐   ┌─────────▼────┐
                        │ /follower/   │   │ /follower/   │   │ /follower/   │
                        │ joint_states │   │ forward_     │   │ cam_*/       │
                        │ (subscribe)  │   │ controller/  │   │ image_raw    │
                        │              │   │ commands     │   │ (subscribe)  │
                        └──────────────┘   │ (publish)    │   └──────────────┘
                                           └──────────────┘
                                                  │
                                     ┌────────────▼────────────┐
                                     │  ros2_control +         │
                                     │  feetech_ros2_driver    │
                                     │  (controller_manager)   │
                                     └────────────┬────────────┘
                                                  │ USB serial
                                     ┌────────────▼────────────┐
                                     │  SO101 Follower Arm     │
                                     │  /dev/so101_follower    │
                                     └─────────────────────────┘
```

### WebSocket API (via SO101Plugin)

| Command | Parameters | Description |
|---|---|---|
| `arm.get_state` | — | Returns joint positions (rad + deg) |
| `arm.home` | — | Move to home: `[0, -1.57, 1.57, 0, 0, -0.4]` |
| `arm.move_joints` | `{"joints": [6 floats]}` | Move to target joint positions (radians) |
| `arm.gripper` | `{"position": 0.0-1.0}` | 0 = open, 1 = closed |
| `arm.stop` | — | Emergency stop (hold current position) |
| `arm.camera` | `{"camera": "workspace"}` | Get latest camera image |

Example (with wscat):
```bash
wscat -c ws://localhost:8765
> {"command": {"action": "arm.get_state"}}
> {"command": {"action": "arm.home"}}
> {"command": {"action": "arm.gripper", "parameters": {"position": 0.0}}}
```

---

## Quickstart

### Prerequisites

- Ubuntu 24.04 with ROS2 Jazzy (`sudo apt install ros-jazzy-desktop`)
- `ros-jazzy-ros2-control` + `ros-jazzy-ros2-controllers`
- Python 3.12 + [uv](https://docs.astral.sh/uv/)

### 1 — Clone and install

```bash
git clone git@github.com:ShubhamGawande191/Manipulation-testing.git
cd Manipulation-testing
uv sync
```

### 2 — Build ROS2 workspace (first time only)

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-skip rosbag_to_lerobot so101_inference
colcon build --packages-select rosbag_to_lerobot so101_inference
```

### 3 — Launch follower arm

```bash
# Terminal 1
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch so101_bringup follower.launch.py \
    hardware_type:=real \
    usb_port:=/dev/so101_follower \
    use_rviz:=false
```

### 4 — Launch agent-ros-bridge

```bash
# Terminal 2
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
cd /home/beelzebub/Desktop/RobotManipulator
uv run python -m robot_manipulator.scripts.launch_bridge
```

### 5 — Send commands

```bash
# Terminal 3
wscat -c ws://localhost:8765
> {"command": {"action": "arm.get_state"}}
> {"command": {"action": "arm.home"}}
```

---

## Project Layout

```
config/
    so101.yaml              ← arm ports and camera indices
src/robot_manipulator/
    __init__.py
    arm.py                  ← LeRobot config helpers (follower + leader)
    cameras.py              ← OpenCVCamera helpers
    so101_bridge.py         ← SO101Controller + SO101Plugin (agent-ros-bridge)
    scripts/
        calibrate.py        ← uv run calibrate
        teleoperate.py      ← uv run teleoperate
        launch_bridge.py    ← launch agent-ros-bridge with SO101 plugin
        pick_and_place.py   ← pick-and-place task (to be updated for ROS2)
    tasks/                  ← future autonomous task modules
    utils/                  ← shared helpers
tests/
    test_smoke.py           ← import-level smoke tests (2/2 passing)
test_controller.py          ← quick SO101Controller test (no bridge needed)
```

### ROS2 workspace (`~/ros2_ws/src/`)

| Package | Description |
|---|---|
| `so101-ros-physical-ai/` | SO101 bringup, description, teleop, MoveIt config (from legalaspro) |
| `feetech_ros2_driver/` | ros2_control hardware interface for STS3215 servos |
| `agent-ros-bridge/` | AI agent ↔ ROS2 bridge framework |

---

## Stack

| Library | Version | Purpose |
|---|---|---|
| [LeRobot](https://github.com/huggingface/lerobot) `[feetech]` | 0.5.0 | Servo control, calibration, recording |
| [agent-ros-bridge](https://github.com/legalaspro/agent-ros-bridge) | 0.6.1 | AI agent ↔ ROS2 WebSocket bridge |
| [ROS2 Jazzy](https://docs.ros.org/en/jazzy/) | — | Robot middleware, ros2_control |
| [uv](https://docs.astral.sh/uv/) | 0.9.0 | Python package/venv manager |

---

## Key Files & Calibration

- **Follower calibration**: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_so101_follower.json`
- **ROS2 follower joints config**: `~/ros2_ws/src/so101-ros-physical-ai/so101_bringup/config/hardware/follower_joints.yaml` (updated with real calibration values)
- **Udev rules**: `/etc/udev/rules.d/99-so101.rules`

---

## Troubleshooting

**Permission denied on `/dev/ttyUSB*` or `/dev/ttyACM*`**
```bash
sudo usermod -aG dialout $USER   # log out and back in
```

**Camera indices shifted after reboot**
```bash
uv run bench-check                # full bench check (arm + cameras + MIDI)
uv run lerobot-find-cameras opencv
```

**ROS2 build fails with catkin_pkg / venv leaking**
```bash
unset VIRTUAL_ENV
# Remove .venv from PATH, then:
cd ~/ros2_ws && rm -rf build install log && colcon build --symlink-install
```

**Check USB devices**
```bash
v4l2-ctl --list-devices
lsusb
ls -la /dev/so101_*
```

---

## Credits

- **[legalaspro/so101-ros-physical-ai](https://github.com/legalaspro/so101-ros-physical-ai)** — the ROS2 packages this project's `~/ros2_ws/` workspace is built from (`so101_bringup`, hardware configs). If any of it ships in or with this repo, its license and attribution travel with it.
- **[google-research/robopianist](https://github.com/google-research/robopianist)** — reference for the MIDI-scored piano-playing task design (`minilab/PLAN.md`).
- **[Hugging Face LeRobot](https://github.com/huggingface/lerobot)** and the **SO101 / TheRobotStudio** ecosystem.

## Security & dependency policy

- Secrets live in `.env` only (gitignored). Never commit tokens; rotate anything that leaks.
- `.github/dependabot.yml` opens grouped weekly update PRs (uv lockfile + actions). Merge only with green `uv run pytest`. Full-tree refresh: `uv lock --upgrade` (last done 2026-07-04).
- Treat Dependabot *alerts* (Security tab) as the queue; the weekly PR usually clears them.
