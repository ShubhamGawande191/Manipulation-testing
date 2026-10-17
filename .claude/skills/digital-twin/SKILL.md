---
name: digital-twin
description: Build or validate a MuJoCo digital twin of a real SO-101 setup (single or bimanual) - arm calibration vs fleet stats, partial recalibration, staged teleop check, end-effector MJCF swap, sim2real joint map with the live twin, stable camera paths + locked exposure, ChArUco intrinsics, camera extrinsics, scene fixtures, actuator fit. Use when setting up a new SO-101 bench, after recalibrating an arm, swapping a gripper, moving a camera, or when sim and real disagree.
---

# Digital twin

Follow `docs/DIGITAL-TWIN-RUNBOOK.md` stage by stage (§0 conventions first). Each stage has commands,
a pass check and lessons; don't skip the pass checks, later stages assume the earlier ones hold.

Hard rules:
- Hardware motion/torque only with the operator present and agreeing; read-only probes are fine.
- One process per serial port (teleop / twin / calib tools cannot run together).
- Radians + CMD order inside sim/harness; convert only via `harness/units.py` + `config/sim2real.yaml`.
- Record new evidence and lessons back into the runbook's status table and the relevant section.

Quick entry points:

```bash
uv run bench-check            # §1
uv run calib-check            # §2
uv run teleop --max-speed 90  # §3 (Ctrl+C once)
uv run twin                   # §5 (torque off)
uv run cam-setup --snapshot   # §6
uv run camcal capture <cam>   # §6
```
