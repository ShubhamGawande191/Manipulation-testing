# Harness, guards and policy behaviour (layers 5 and 7)

## Harness and control (layer 5)

Command path: policy (rad, CMD order) -> `SafetyFilter` (joint box = limits - 0.05 rad, velocity cap
`--max-vel`, watchdog) -> `LeRobotBackend` (deg, clip to calibrated travel - 1 deg) -> servo (P/D, I = 0).

| Symptom | Check | Action |
|---|---|---|
| Scripted target (HOME) never "reached" live, passes in sim | target vs SafetyFilter box | clip scripted targets into the box; test with real-sized errors (gripper target on the limit failed 5/5) |
| Arm stops ~2 deg short, side depends on approach | expected (I = 0 + friction) | `SettleTrim` (+-4 deg, 0.15 deg deadband): fingertip error 10.5 -> 4.9 mm, play 3.1 -> 1.3 mm |
| Contact guard trips without contact | baseline learned while moving / during trim | guard skips 5 and learns over 10 ticks; never run it during the trim |
| Fingertip touches the surface at 20 mm hover | sag + stop error + tool model | hover >= 40 mm with the guard; consider sag feedforward (q + tau_g/k) |
| Parking sweeps the tool into the table | fold path from the task end pose | park via HOME; validate parking paths in sim from end poses |
| Harness success much lower than training eval | target sampler differs | one sampler (`TARGET_BOX`) shared by trainer and harness (99 % -> 78 % when they differed) |

## Policy behaviour (layer 7)

| Symptom | Check | Action |
|---|---|---|
| Arm oscillates at the goal | does the command oscillate? (trajectory parquet) | yes -> closed-loop/policy; no -> servo gains. Oscillation guard (8 reversals within 1 s, < 0.05 rad) stops it in 1-2 s; 20/20 hunting caught, 0/60 calm flagged |
| Policy trained on the fitted twin hunts, nominal-DR policy is calm | twin choice | prefer broad DR around nominal; test lower policy rate (20/10 Hz) as a mitigation |
| First live runs fail at a lower speed cap than training | action integration | integrate the policy command at the deploy cap (`max_step` = harness `--max-vel`) in training too |
| Policy refuses to load | metadata mismatch (scene, gripper, action scale) | retrain or use the matching scene; don't bypass the check |
| Wastes training on folded-arm targets | target sampling in joint space | sample from the task workspace (`TARGET_BOX`) |

## Diagnose from logs, not impressions

- `runs/harness/<ts>/` holds per-episode trajectory parquet (q, dq, q_target, q_safe, ee) + `summary.json`.
- `uv run gap closed-loop <real_run> <sim_run> ...` scores accuracy and hunting side by side.
- `uv run atlas query x y z` tells what the twin expects at a pose (load, reach).
