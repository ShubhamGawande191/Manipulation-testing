# IK benchmark (SO-101 + PincOpen, in the twin)

Code: `src/robot_manipulator/ik/` (`analytic.py`, `numeric.py`, `learned.py`, `bench.py`),
`integrations/curobo/bench_curobo.py` (cuRobo), `src/robot_manipulator/sim/urdf_export.py` (URDF).
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

### cuRobo (2026-10-20)

cuRobo 0.8.0 (the v2 API) on the RTX 3060 Laptop GPU, 32 seeds, full 6-DOF fingertip pose,
self-collision on. "One at a time" = one target per call; "batch" = all targets of a scenario in
one call.

| Scenario | one at a time | batch | closed-form |
|---|---|---|---|
| random (400) | 92.5 %, 14 ms | 92.75 %, **0.73 ms/target** (400 in 291 ms) | 100 %, 1.1 ms |
| keys (168) | 64.3 % | 64.3 %, 0.77 ms/target | 65.5 % |
| line (120) | 100 % | 100 %, 0.81 ms/target | 100 % |

- **Same kinematics:** cuRobo's FK of the closed-form solutions lands within 0.001 mm of the
  targets, so the generated URDF (below) and cuRobo agree with the twin exactly.
- **Its misses are mostly right:** of the 29 random targets cuRobo missed, the closed-form solution
  self-collides in the twin in 22. Both key misses self-collide too. 36 of the 400 closed-form
  "successes" are self-colliding poses, because that solver doesn't check collisions. On
  collision-free targets cuRobo solves ~98 %. It never reported an unreachable target as solved.
- **Where it wins:** batched throughput (0.7 ms/target with collision checking, and it scales
  with batch size on the GPU), collision-aware solutions, and the same machinery does motion
  planning and MPC around a scene. **Where it loses:** single-query latency (14 ms vs ~1 ms), a
  GPU and a 6 GB environment, and it can't tell "unreachable" from "not found".

#### Setup (cuRobo in its own environment; the repo's CPU PyTorch is untouched)

```bash
git clone --branch v0.8.0 https://github.com/NVlabs/curobo ~/tools/curobo
cd ~/tools/curobo && uv venv --python 3.11 && uv pip install ".[cu13-torch]"   # ~6 GB, CUDA 13 driver
cd -   # back to the repo
uv run urdf-export --no-mimic          # URDF from the MJCF -> runs/urdf/so101_pinc/ (FK checked vs MuJoCo)
uv run ik-bench export-targets         # scenario targets as full poses -> runs/ik/targets.json
~/tools/curobo/.venv/bin/python integrations/curobo/bench_curobo.py --rebuild   # spheres + IK
uv run ik-bench run                    # one table, cuRobo included
```

**Robot description:** `sim/urdf_export.py` converts our MJCF with
[mjcf-urdf-simple-converter](https://pypi.org/project/mjcf-urdf-simple-converter/) (MIT), then adds
a `fingertip` link, mimic tags for the PincOpen four-bar, and limits for the coupled finger joints.
Its fingertip matches MuJoCo's to 0.00 µm over random poses. cuRobo's `RobotBuilder` fits collision
spheres to the meshes and computes the self-collision matrix.

**cuRobo 0.8.0 quirks hit here** (worked around in `integrations/curobo/bench_curobo.py`):
- `RobotBuilder` crashes (CUDA index assert in the joint-space setup) on URDF `<mimic>` joints;
  export with `--no-mimic` and lock the finger joints instead.
- `RobotBuilder.save` writes runtime-only keys (`load_collision_spheres`, `num_envs`) that the
  loader then passes twice; strip them.
- Joints off the path to the tool frame and without collision links aren't in cuRobo's tree, so
  `lock_joints` on them is a `KeyError` (the PincOpen `gripper` joint drives a mesh-less coupler).
- With `self_collision_check=False` and no scene, the seed stage has no costs and fails; keep it on.
- Unlimited MJCF joints come out of the converter with limits [0, 0], which cuRobo rejects;
  the export now writes the coupled range.

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

- **Collision-aware closed form:** pick the closed-form branch that is free of self-collision
  (and of the table), so it stops returning colliding poses; cuRobo shows 9 % of random targets
  are only reachable in collision.
- **cuRobo with the bench scene:** add the table, column and keyboard as obstacles and compare
  collision-free paths (motion generation), not just IK.
- **Real arm:** after the joint offsets are applied, measure fingertip error for each solver's
  solutions with the wrist camera or touches (the calibrated FK is what all solvers share).
- **Collision-aware key presses:** add the table/keyboard to the reach map (the wrist body, not
  just the fingertip, must clear them).
