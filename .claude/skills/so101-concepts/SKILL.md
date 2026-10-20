---
name: so101-concepts
description: >-
  Hub for the SO-101 lab repo: the three control paths and their units, joint ordering, the harness/sim/real
  layers, the hard rules, and which tool answers which question. Routes to the specialised skills.
  Triggers: orient in this repo, which command/tool to use, units or joint order questions, where a result
  or config lives, before starting any task that touches the arm, the twin or an experiment.
  Subsystems: LeRobot follower (degrees), ROS2 bridge (radians), harness + MuJoCo twin (radians, CMD order),
  calibration, vision, lab board.
  Dependencies: uv, config/sim2real.yaml, docs/DIGITAL-TWIN-RUNBOOK.md, docs/LAB-NOTEBOOK.md, .colony/tasks.yaml.
  Anti-keywords: OpenClaw, CMDOP (removed for security; never reintroduce).
---

# SO-101 lab: concepts and routing

## Core mental model

One arm (SO-101, 6x Feetech STS3215, IDs 1-6, PincOpen gripper + wrist camera), three ways to drive it:

| Path | Code | Units | Use it for |
|---|---|---|---|
| Direct LeRobot | `arm.py`, `hw/follower.py` | **degrees** (zero = calibrated range midpoint), gripper 0..100 | teleop, calibration, low-level tests |
| ROS2 bridge | `so101_bridge.py`, port 8765 | **radians**, CMD order | vision pick-and-place tasks |
| Harness + twin | `harness/`, `sim/` | **radians, CMD order**; degrees only inside `harness/units.py` | skills, policies, sim2real, experiments |

CMD order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
`joint_states` (STATE order) is alphabetical. Mixing the two is the classic bug; see CLAUDE.md's table.

What the servo actually is (measured, see blog draft and runbook §9): P/D position loop with **I = 0**,
P 16 (24 on lift/elbow), acceleration 254 (no profile). At rest it is a spring (~18 N.m/rad on the lift),
stops ~2 deg short on the side it came from, lags ~68 ms. Every controller and twin decision follows from this.

## Hard rules (always)

1. **Hardware motion or torque only with the operator present and agreeing.** Read-only probes are fine.
2. **One process per serial port** (teleop, twin, calib tools, harness never run together).
3. **Units:** convert only through `harness/units.py` + `config/sim2real.yaml`; never in env/policy code.
4. **Results claims follow the experiment skill** ([so101-experiments](../so101-experiments/SKILL.md)): n,
   intervals, validity checks, negative results recorded.
5. **Public repo:** nothing from `third_party/` (private notes) is ever committed or quoted.
6. **No probe scripts in the repo root**; experiments go under `src/robot_manipulator/` or the scratchpad.

## Which tool answers which question

```
What do you need?
├── Is the bench alive (ports, cameras, MIDI)?        -> uv run bench-check            [bench-check]
├── Is the arm's calibration sane vs the fleet?       -> uv run calib-check            [digital-twin §2]
├── Does the twin match the arm right now?            -> uv run twin                   [digital-twin §5]
├── Where are the cameras (intrinsics/extrinsics)?    -> uv run camcal / camcal-wrist  [digital-twin §6-7]
├── Where does the fingertip really go?               -> uv run tip-accuracy           [digital-twin §7b]
├── How does the servo respond (lag, stiffness, fit)? -> uv run gap run|compare|fit|stiffness|closed-loop
├── Run a policy or skill (sim or real)               -> uv run harness reach ...      [so101-robot-motion]
├── What can the arm do where (load, reach)?          -> uv run atlas prior|query|real, ik-bench
├── Sim factor study (e.g. insertion clearance)       -> uv run insertion-study sweep|compare
│                                                                                      [so101-experiments]
├── Something is wrong                                -> [so101-debugging] (layer triage first)
└── Blog figures from logged runs                     -> uv run blog-figs
```

## Where things live

| Thing | Location |
|---|---|
| Procedures, status table, lessons | `docs/DIGITAL-TWIN-RUNBOOK.md` |
| Dated results, mistakes, decisions | `docs/LAB-NOTEBOOK.md` |
| What's next (agent + you) | `.colony/tasks.yaml` (keep it current; YAML must parse) |
| Joint map / offsets | `config/sim2real.yaml` |
| Camera paths, intrinsics, extrinsics | `config/cameras.yaml`, `config/camera_intrinsics/`, `config/extrinsics/` |
| Actuator fits | `config/actuator_fit.yaml` |
| Run outputs | `runs/<tool>/<timestamp>/` (gitignored; a `README.md` there marks invalid runs) |
| Blog draft + figures | `docs/blog/so101-servo-to-policy/` |
| Private notes | `third_party/` (gitignored) |

## Progressive disclosure

| Skill | Read it when |
|---|---|
| [bench-check](../bench-check/SKILL.md) | after reboot/replug, before any hardware task |
| [digital-twin](../digital-twin/SKILL.md) | building/validating the twin, calibration, cameras, actuator fit |
| [so101-robot-motion](../so101-robot-motion/SKILL.md) | before commanding the real arm (harness, teleop, accuracy runs) |
| [so101-debugging](../so101-debugging/SKILL.md) | anything misbehaves: hardware, calibration, vision, sim, policy |
| [so101-experiments](../so101-experiments/SKILL.md) | designing, running or reporting any experiment or sweep |

## Workspace search rules

Exclude heavy trees: `.venv`, `runs`, `third_party`, `assets`, `.git`. Example:
`grep -rn --exclude-dir={.venv,runs,third_party,assets,.git} "pattern" src tests docs config`.
Prefer the code over memory: verify a flag with `uv run <tool> --help` before citing it.

## Reflection checkpoint (before acting)

1. Which control path and which units am I in? Which joint order?
2. Does this touch hardware? If yes: operator present, nothing else on the port, dry run first.
3. Will this produce a number someone will quote? If yes: follow so101-experiments.
4. Which board task does this advance, and is its status updated?
