# Hardware, servos, calibration and joint map (layers 1-3)

## 1. USB, power and ports

| Symptom | Check | Action |
|---|---|---|
| `/dev/so101_follower` missing, CH343 (1a86:55d3) absent from `lsusb` | `uv run bench-check` | Physical: reseat cable, use a direct machine port (not dock/hub), check board power. No software fix. |
| CH343 present but no `/dev/so101_*` symlink | udev rule didn't fire | `sudo udevadm trigger` or replug |
| Permission denied on the port | user not in `dialout` | `sudo usermod -aG dialout $USER`, re-login |
| "Port busy" / garbage reads | another tool holds the port | stop teleop / twin / harness / calib tool; one process per port |
| One servo misses the broadcast-ping handshake | per-servo voltage (runbook §1 read-only probe) | Leader should read ~5 V, follower ~12 V. A 12 V brick on the 5 V leader made ID 2 drop out. |
| "no status packet" right after enabling torque | all joints torqued at once from a slump | brown-out of the 12 V bus: torque on joint by joint (teleop and the harness do this) |
| Leader servo 4 "no status packet" during torque-on | intermittent bus write | `hw/leader.py` retries writes (`BUS_RETRY = 3`) and releases on failure; if it persists, check voltage/cable |

## 2. Servo registers (STS3215, firmware 3.10)

Read before writing; registers and what they mean:

| Register | Value here | Effect |
|---|---|---|
| P | 16 (24 on shoulder_lift, elbow_flex) | stiffness; P=16 left the elbow 3.7 deg short, P=24 0.14 deg |
| D | 32 | damping |
| I | 0 | no steady-state correction -> stops ~2 deg short by approach direction |
| Dead zone | +-1 step (0.088 deg) | not the cause of the 2 deg stop (that is I = 0 + friction) |
| Acceleration | 254 (max) | no motion profile; every 20 ms step goes to the motor |
| Torque limit | 1000 (gripper 500) | |

Guardrails: change one register at a time, log before/after, restore on exit (`teleop --pid joint=P,I,D` is
the supported override path). PincOpen gripper (ID 6) is not back-drivable: never sweep it during calibration;
use >= 20 % torque for motion; open 2005 / closed 3200 steps; closed fingers read ~6.8, not 0.

## 3. Calibration and joint map

| Symptom | Check | Action |
|---|---|---|
| Constant offset leader vs follower on one joint | `uv run calib-check` vs the 15-arm fleet | An incomplete sweep shifts the zero (pan swept 1905 vs fleet 2725 +- 32): `uv run calib-recal --joints <j>` |
| Twin mirrors a joint (moves the other way) | `uv run twin` moving one joint by hand | sign problem: confirm with `--flip <joint>`, set `sign: -1` |
| Twin off by a constant angle | same | offset problem: `offset_deg` in `config/sim2real.yaml` (wrist-camera BA gives it: lift +4.61, elbow -4.75, wflex -4.58) |
| Gripper shows a gap when closed | gripper map | map "closed" to the reading at finger contact (~7), not 0 |
| Commands clipped near a limit after applying offsets | offset moved the sim box onto the servo travel | expected: `LeRobotBackend` clips to calibrated travel - 1 deg (`TRAVEL_MARGIN_DEG`) |
| Gripper direction inverted | follower calibration JSON | `"drive_mode": 1` for the gripper |
| Load ~100 % with a fixed error at one pose | self-collision | camera bracket hit the arm before; record in `config/so101_limits.yaml` |
| Harness refuses live | `calibrated: false` | finish runbook §5, then set `calibrated: true` |

Calibration files: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json` (and
`teleoperators/so_leader/<id>.json`). `robot_id` must match.

## Circuit breakers

- Same servo error twice: stop, read voltages, then cable/port. Max 3 bus retries.
- Calibration: if a recal is rejected by the fleet check twice, inspect the sweep (end stops reached?) before a third.
