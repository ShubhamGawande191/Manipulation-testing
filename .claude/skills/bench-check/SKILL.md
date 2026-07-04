---
name: bench-check
description: Verify SO101 bench hardware (arm serial, both cameras, MiniLab 3 MIDI) after a desktop reboot, USB replug, or whenever a device seems missing or camera indices look wrong. Run this before any task that touches hardware.
---

# Bench check

Camera indices and tty numbers shift across reboots on this machine. Run:

```bash
uv run bench-check
```

It prints a status table, writes `config/bench_state.json`, and exits nonzero
with a diagnosis per problem.

## Interpreting failures

- **"follower arm board … NOT enumerating on USB"** — physical problem, not
  software: the CH343 chip (1a86:55d3) is absent from the USB bus. Ask the
  user to reseat the cable, use a direct machine port (not the ThinkPad
  dock/hub), and confirm board power. No software fix exists for this.
- **"CH343 enumerated but /dev/so101_follower missing"** — udev didn't fire:
  `sudo udevadm trigger`, or replug.
- **camera problems** — working indices are listed in the table; if they
  moved, update `config/so101.yaml` (workspace/wrist indices) to match, and
  confirm with `uv run lerobot-find-cameras opencv`.
- **MiniLab on USB but no MIDI port** — rtmidi backend missing:
  `uv sync --extra midi`.

After hardware is green, `uv run pytest -m hw` is the full smoke suite.
