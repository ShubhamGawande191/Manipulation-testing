---
name: so101-debugging
description: >-
  Layer-first triage for anything that misbehaves on the SO-101 bench or in its twin: USB/power, servo
  registers, calibration and joint map, cameras and extrinsics, harness safety/guards, MuJoCo twin and contacts,
  policy behaviour. Routes to per-layer references with symptom -> check -> action trees and the lessons
  this repo already paid for.
  Triggers: no status packet, device missing, wrong camera, constant joint offset, twin disagrees with the arm,
  arm stops short or sags, hunting/oscillation, guard trips, sim result implausible, policy fails on the arm only.
  Subsystems: hw/, calib/, cameras.py, harness/, sim/, sim/gap/, policies.
  Dependencies: bench-check, calib-check, twin, gap, tip-accuracy, docs/DIGITAL-TWIN-RUNBOOK.md lessons.
  Anti-keywords: designing a new experiment (so101-experiments), first-time twin build (digital-twin).
---

# Debugging the SO-101 lab

## Triage hierarchy: isolate the layer before touching code

Work bottom-up; a fault in a lower layer shows up as nonsense in every layer above it.

1. **USB, power, ports**: device enumerates, udev symlink, 12 V follower / 5 V leader, one process per port.
2. **Servo registers and firmware**: P/D/I, acceleration, torque limit, EEPROM limits, brown-out.
3. **Calibration and joint map**: LeRobot calibration vs fleet, sign/offset in `config/sim2real.yaml`, travel.
4. **Vision**: stable camera paths, locked exposure, intrinsics, extrinsics, board scale.
5. **Harness and control**: SafetyFilter box, travel clip, speed cap, guards, settle trim.
6. **Twin and simulation**: actuator model (open vs closed loop), contacts, friction, timestep, geometry.
7. **Policy**: observation pipeline, sampler parity, action integration, closed-loop behaviour.

## Progressive disclosure reference hub

| Reference | Layers | Read it when |
|---|---|---|
| [references/hardware-and-calibration.md](references/hardware-and-calibration.md) | 1-3 | device missing, no status packet, brown-out, constant offset, gripper reads wrong |
| [references/vision.md](references/vision.md) | 4 | wrong/missing camera, colours drift, bad intrinsics, extrinsics disagree, board issues |
| [references/sim-and-twin.md](references/sim-and-twin.md) | 6 | sim too good / too bad, a change does nothing, twin vs arm mismatch, contact artefacts |
| [references/policy-and-harness.md](references/policy-and-harness.md) | 5, 7 | hunting, guard trips, targets unreachable, harness vs trainer disagree, real-only failures |

## First commands (read-only)

| Question | Command |
|---|---|
| Are devices there? | `uv run bench-check` |
| Is calibration sane vs the fleet? | `uv run calib-check` |
| Does the twin mirror the arm? (torque off) | `uv run twin [--flip <joint>]` |
| Where does the fingertip actually go? | `uv run tip-accuracy analyze runs/accuracy/<ts>_real` |
| How does the servo respond vs the twin? | `uv run gap compare runs/gap/<ts>_real` |
| Did a policy hunt? | `uv run gap closed-loop runs/harness/<real> runs/harness/<sim>` |

## Top-level decision tree

```
[Anomaly]
├── Device or servo not answering ("no status packet", port missing)      -> hardware-and-calibration §1-2
├── Constant offset between leader/follower or twin/arm on one joint      -> hardware-and-calibration §3
├── Arm stops short on the side it came from; sags under load            -> expected physics (I = 0, spring):
│                                                                             settle trim; see policy-and-harness
├── Arm oscillates at the goal                                            -> policy-and-harness (hunting)
├── Camera wrong, images drift, calibration residuals per view            -> vision
├── Sim result beats a hand bound, or a design change changes nothing     -> sim-and-twin (validity) + so101-experiments
└── Works in sim, fails on the arm with the same commands                 -> sim-and-twin (actuator gap)
```

## Paired guardrails

1. Do not debug a higher layer while a lower one is unverified; run the read-only command for the layer below first.
2. Do not change gains, offsets and code in the same attempt; change one thing and re-measure.
3. Do not trust a sim number that beats a bound; inspect one episode end to end first.
4. Do not "fix" a symptom by widening a threshold (guard, tolerance) before you know its cause.
5. Do not run two tools on one serial port to "check quickly"; stop the first one.

## Reflection checkpoint

1. Which layer is this, and is every layer below it verified with a read-only check?
2. Is the evidence a logged trace (parquet/json) or an impression? Get the trace.
3. Has this exact symptom a lesson in the runbook or the notebook already? (`grep -n Lesson docs/*.md`)

## Circuit breaker

- Same failure twice with identical signature: stop retrying, go one layer down.
- Two fixes that don't change the outcome: your model of the fault is wrong; return to the trace.
- Hardware: never more than 3 bus retries; release torque and park instead of looping.

## Completion criteria

- [ ] Fault layer named, with the read-only evidence that isolates it.
- [ ] Fix verified by re-running the same check that showed the fault.
- [ ] Lesson recorded (runbook section or notebook) if it cost more than a few minutes.
