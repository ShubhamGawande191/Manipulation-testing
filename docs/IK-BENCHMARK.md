# IK benchmark (SO-101 + PincOpen, in the twin)

Code: `src/robot_manipulator/ik/` (`analytic.py`, `numeric.py`, `learned.py`, `bench.py`).
Started 2026-10-20. Everything here is in **twin coordinates**. On the real arm the kinematic
calibration error comes on top (encoder offsets of ~−5° elbow / −4.6° wrist_flex found in §7 of
`DIGITAL-TWIN-RUNBOOK.md`, not yet applied).

```bash
uv run ik-bench train-nn         # ~3 min CPU: 120k labelled samples + 60 epochs -> runs/ik/seed_mlp.pt
uv run ik-bench run --n 400      # the table below -> runs/ik/bench_<ts>.json
uv run ik-bench reach-map        # where on the table the fingertip reaches at a given pitch
```

## Task

Fingertip position (closed PincOpen tip) + tool pitch (0 = horizontal, 90° = straight down);
wrist roll is free because the fingertip sits on the roll axis (0.2 mm off), held at the natural
−80°. Success = fingertip within 1 mm, pitch within 0.5°, inside the joint limits.

## Solvers

| Solver | How |
|---|---|
| **closed-form** (`AnalyticIK`) | lift, elbow and wrist_flex axes are parallel, so after pan it is a planar 3R chain; geometry read from the MJCF. Up to 4 branches (facing / reaching back over the pan axis × elbow up / down); the planar model is within 0.18 mm, 3 Newton steps on the full MuJoCo FK take it below 1 µm |
| **DLS** | damped least squares on the full FK (finite-difference Jacobian), from HOME, optionally with 7 random restarts |
| **network** | MLP 4→256³→4 from (xyz, pitch) to joints. Labels are the closed-form branch closest to HOME, so the target is a function (raw random (q → task) pairs would average branches) |
| **network + DLS** (hybrid) | the network's guess refined by ≤ 30 DLS steps |

## Results (2026-10-20, `runs/ik/bench_20261020-113039.json`)

| Scenario | closed-form | DLS (HOME) | DLS + restarts | network | **network + DLS** |
|---|---|---|---|---|---|
| random poses (400) | **100 %**, 1.1 ms | 70 %, p95 59 ms | 89 %, p95 466 ms | 10 % (median 2.6 mm off) | **98 %, 0.54 ms** (p95 2.7 ms) |
| keys: straight down, 1 cm above the table (168) | 65.5 % = reachable ceiling | 65.5 % | 65.5 % | 8 % | 65.5 %, 0.51 ms |
| line: 30 cm, tool down, warm-started (120) | 100 %, max jump 3.4° | 100 %, 0.35 ms | 100 % | 39 % | 100 %, 0.49 ms |

Times are Python + MuJoCo per solve on the laptop CPU: compare solvers to each other.

## What it means

- **For this arm the closed-form solver is the reference.** It solves every reachable target,
  says *why* a target fails (out of reach vs joint limits), and lets the caller pick the branch
  (elbow up/down, reaching back). Its speed is dominated by the finite-difference polish; one
  polish step would do (the planar model alone is within 0.18 mm).
- **Plain DLS is not enough**: 30 % failures from HOME (wrong basin, joint limits), and restarts
  buy success with 100× worse tail latency.
- **The learned seed earns its keep only as a seed**: alone it is a few mm off; with 2 DLS steps
  it is the fastest general solver here (98 %). The 2 % misses sit where the preferred branch
  switches (the label is discontinuous there). With an exact closed form available, the network is
  mainly worth it for things the closed form can't express: a calibrated real arm whose FK is
  learned from data, or batched planning.
- **Straight-down reach (MiniLab):** with the tool vertical the fingertip reaches the table up to
  ~30 cm forward (narrowing past 22 cm, |y| ≤ 12 cm at 28 cm), and close in (< 8 cm) joint limits
  block it. Tilting to 70° extends reach to ~36 cm. Place the keyboard ~10–28 cm in front of the
  base, or press at ~70° pitch. (`ik-bench reach-map`; no table-collision check yet.)

## Next

- **cuRobo** (NVlabs, Apache-2.0): GPU-batched IK, collision-aware motion generation and
  trajectory optimisation. Needs CUDA PyTorch (several GB) and a URDF matching our MJCF (SO-101 +
  PincOpen). The RTX 3060 (6 GB) should run it. Compare on the same scenarios plus batched
  throughput and collision-aware paths around the bench scene.
- **Real arm:** after the joint offsets are applied, measure fingertip error for each solver's
  solutions with the wrist camera or touches (the calibrated FK is what all solvers share).
- **Collision-aware key presses:** add the table/keyboard to the reach map (the wrist body, not
  just the fingertip, must clear them).
