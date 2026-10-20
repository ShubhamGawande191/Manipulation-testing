---
name: so101-robot-motion
description: >-
  Commanding the real SO-101 safely through the harness, teleop and calibration tools: dry run before live,
  joint-by-joint torque, speed caps, travel clipping, contact guard, settle trim, parking via HOME, leader stow.
  Triggers: run a policy or skill on the real arm, harness reach --backend real, tip-accuracy run, camcal-wrist run,
  teleop, park the arm, any live motion, servo errors during motion.
  Subsystems: harness (backend, safety_layer, skill, runner), hw/follower.py, hw/leader.py, calib/accuracy.py.
  Dependencies: /dev/so101_follower, /dev/so101_leader, config/sim2real.yaml (calibrated: true).
  Anti-keywords: sim-only experiments (use so101-experiments), building the twin (use digital-twin).
---

# Moving the real SO-101

Prerequisite: [so101-concepts](../so101-concepts/SKILL.md) (units, hard rules). Hardware green via
[bench-check](../bench-check/SKILL.md).

## How a live command reaches the motor

```
policy / skill (rad, CMD order)
  -> SafetyFilter: clamp to joint box (limits - 0.05 rad), velocity cap (--max-vel), watchdog
  -> LeRobotBackend: rad -> deg via units.py, clip to calibrated travel - 1 deg (TRAVEL_MARGIN_DEG)
  -> STS3215: P/D loop, I = 0, no motion profile
```

What the servo does with it: stops ~2 deg short on the approach side, sags under gravity (spring
~18 N.m/rad lift), lags ~68 ms. Accuracy features that compensate: `SettleTrim` (outer integral loop,
+-4 deg, 0.15 deg deadband) and, for contact, `ContactGuard` (following error vs an early baseline).

## Protocol

1. **Operator present and agreeing.** Say what will move, how fast, and how to stop (Ctrl+C parks).
2. **Port free:** nothing else on `/dev/so101_follower` (teleop, twin, calib tools).
3. **Dry run first.** It is read-only (opens the bus; no register writes, no torque change):
   ```bash
   uv run harness reach --backend real --policy runs/ppo_reach/<run>/reach.onnx --target 0.25,0,0.10 --dry-run
   ```
4. **Live at a low cap**, then raise:
   ```bash
   uv run harness reach --backend real --policy runs/ppo_reach/<run>/reach.onnx --random 20 --live --max-vel 0.5 --settle
   ```
   Live refuses to start unless `config/sim2real.yaml` has `calibrated: true`.
5. **Torque comes on joint by joint** with Goal = Present and P=24 on lift/elbow (all-at-once browns out
   the 12 V bus). Never call `SOFollower.connect()` directly for live work.
6. **Park via HOME** (close and Ctrl+C do this), then torque off joint by joint.

Other live tools follow the same pattern (`--dry-run` default, `--live` explicit):
`uv run tip-accuracy run --backend real --live`, `uv run camcal-wrist run ...`, `uv run teleop`,
`uv run leader-stow --stow|--rest`, `uv run teleop --park-only`.

## Paired guardrails

1. Do not send a scripted target straight to the arm; clip it into the SafetyFilter box first (HOME failed
   5/5 live when the gripper target sat on the limit).
2. Do not hover the fingertip at 20 mm over a surface on trust; hover at >= 40 mm with the contact guard on,
   because the servo stops 6-10 mm short and the tool model can be ~6 mm off.
3. Do not run the contact guard during the settle trim; the trim's own corrections look like contact.
4. Do not fold or park from wherever the task ended; go via HOME (a staged fold swept the tool into the table).
5. Do not retry a failing bus write in a loop; stop after 2-3 attempts and diagnose (see circuit breaker).

## Input-aware decision tree during motion

```
Symptom while moving
├── "no status packet" / servo N not answering
│   ├── Check: just turned torque on many joints? -> brown-out: torque joint by joint (teleop does)
│   ├── Check: voltage per servo (runbook §1 probe): leader ~5 V, follower ~12 V
│   └── Action: retry at most 3 writes (leader torque_on does), else release torque and stop
├── Arm oscillates at the goal (hunting)
│   ├── Check: does the COMMAND oscillate too? (trajectory parquet in runs/harness/<ts>/)
│   │   ├── yes -> closed loop / policy; the oscillation guard stops it (real default)
│   │   └── no  -> servo gains; compare P register values
│   └── Mitigation to test: lower policy rate (Tune to Learn: 21.8 % jitter at 100 Hz -> 5 % at 10 Hz)
├── Stops short on the same side every time  -> expected (I = 0 + friction): enable settle trim
├── Load near 100 % with a fixed error at one pose -> self-collision (camera bracket hit the arm before)
├── Command clipped / target unreachable near a limit -> encoder offset moved the box; travel clip at -1 deg
└── Contact guard trips with no contact -> baseline learned while still moving; skip/learn windows (5/10 ticks)
```

## Reflection checkpoint (before `--live`)

1. Operator present? Port free? Dry run passed with sensible joint values?
2. `calibrated: true`, offsets current (`config/sim2real.yaml`), right gripper/scene in the policy metadata?
3. Speed cap set (`--max-vel`)? Hover height and contact guard right for anything near a surface?
4. Leader out of the way (`--stow-leader`)?
5. Output dir known, so the run can be analysed and quoted later?

## Circuit breaker

- Same servo error twice in a row: stop commanding, park/release, read voltages and the runbook §1 lessons.
- Two consecutive failed episodes with the same symptom: stop the batch; inspect the trajectory log before
  changing gains, targets or code.
- Never "try again faster" on the real arm.

## Completion criteria

- [ ] Ran dry first; live run had the operator's go-ahead.
- [ ] Arm parked via HOME, torque released joint by joint.
- [ ] Run directory recorded; numbers (n, medians, intervals) go to the notebook/runbook, board task updated.
