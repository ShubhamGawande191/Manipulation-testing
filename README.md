# SO101 Lab

Personal sandbox for working with the **SO101 12 V** robot arm.
Hardware, experiments and notes all live here.

## Hardware

| Component | Details |
|---|---|
| Follower arm | SO101 12 V — Feetech STS3215 servos, 6-DOF |
| Leader arm | SO101 12 V — mixed-ratio Feetech servos for comfortable teleoperation |
| Workspace camera | INNO-MAKER U20CAM-1080P-S1 (USB UVC, 1080p/30 fps) |
| Wrist camera | INNO-MAKER U20CAM-1080P-S1 |

Both cameras are plug-and-play UVC devices — no drivers needed on Linux.

---

## Quickstart (new machine)

### 1 — System prerequisites

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y \
    git curl ffmpeg v4l-utils usbutils \
    cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
```

> **ffmpeg must be installed system-wide.** LeRobot uses `torchcodec` which
> dynamically links to the system ffmpeg.

### 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart terminal
```

### 3 — Clone and set up the environment

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
cd YOUR_REPO_NAME

uv sync            # creates .venv, pins Python 3.12, installs all deps
uv sync --extra dev  # adds Jupyter, pytest, ruff
```

### 4 — Configure secrets

```bash
cp .env.example .env
# Edit .env — add OPENCLAW_API_KEY from https://cmdop.com
# Add HF_TOKEN for Hugging Face model/dataset access (optional)
```

---

## First-time arm setup

Run these **once** when motors are brand new or re-purposed.

### Find USB ports

Plug each arm in one at a time and run:

```bash
uv run lerobot-find-port
```

Update `config/so101.yaml` with the ports you find.

### Configure motor IDs and baudrates

```bash
# Follower
uv run lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB1

# Leader
uv run lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB0
```

### Calibrate

```bash
# Follower
uv run calibrate --device follower --follower-port /dev/ttyUSB1

# Leader
uv run calibrate --device leader --leader-port /dev/ttyUSB0

# Both in sequence
uv run calibrate --device both
```

Calibration files are stored automatically by LeRobot at
`~/.cache/huggingface/lerobot/calibration/`.

---

## Everyday workflows

### Find cameras

```bash
uv run lerobot-find-cameras opencv
```

Update `config/so101.yaml` with the correct `index_or_path` values.

### Teleoperate

```bash
# Basic (no camera display)
uv run teleoperate

# With cameras + Rerun visualisation
uv run teleoperate \
    --follower-port /dev/ttyUSB1 \
    --leader-port /dev/ttyUSB0 \
    --workspace-cam 0 \
    --wrist-cam 1 \
    --display
```

### Pick-and-place (OpenClaw-orchestrated)

```bash
uv run pick-and-place
```

### Record a dataset

```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB1 \
    --robot.id=so101_follower \
    --robot.cameras="{workspace: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=so101_leader \
    --repo-id=YOUR_HF_USERNAME/so101_pick_place \
    --num-episodes=50
```

---

## Project layout

```
config/
    so101.yaml          ← arm ports and camera indices
src/robot_manipulator/
    arm.py              ← LeRobot config helpers (follower + leader)
    cameras.py          ← OpenCVCamera helpers + lerobot-find-cameras wrapper
    scripts/
        calibrate.py    ← uv run calibrate
        teleoperate.py  ← uv run teleoperate
        pick_and_place.py ← uv run pick-and-place  (OpenClaw-orchestrated)
    tasks/              ← future autonomous task modules
    utils/              ← shared helpers
tests/
    test_smoke.py       ← import-level smoke tests (no hardware required)
```

---

## Stack

| Library | Purpose |
|---|---|
| [LeRobot](https://github.com/huggingface/lerobot) `[feetech]` | Servo control, calibration, recording, imitation learning |
| [OpenClaw](https://www.openclawrobotics.com/) / CMDOP | AI agent orchestration — zero-code task pipelines |
| [uv](https://docs.astral.sh/uv/) | Fast Python package and venv manager |

---

## Troubleshooting

**Permission denied on `/dev/ttyUSB*`**
```bash
sudo usermod -aG dialout $USER   # log out and back in
```

**Camera indices shifted after reboot**
```bash
uv run lerobot-find-cameras opencv
```

**Check what's on the USB bus**
```bash
v4l2-ctl --list-devices
lsusb
```
