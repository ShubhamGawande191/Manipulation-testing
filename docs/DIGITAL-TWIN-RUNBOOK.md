# SO-101 digital twin runbook

How to build a MuJoCo digital twin of a real SO-101 setup and validate it against the hardware: arm
calibration, robot model, joint map, cameras, scene and actuators. It is written for people and for
agents, and it should carry over to other SO-101 setups, including bimanual ones.

This file was started on 2026-10-17 on the home bench: one follower with a PincOpen gripper, one
leader, a workspace camera, a wrist camera, and an auto-reset cup. Every stage lists its commands,
the files it reads and writes, a **pass check**, and the **lessons** that cost time here.

> **Rule for agents:** never energize, move, or torque hardware without the operator present and
> agreeing. Read-only probes are OK: they read positions, voltages and torque flags, and they never
> enable torque. Only one process can own a serial port at a time, so teleop, twin and calib tools
> cannot run together.

---

## 0. Conventions (read first)

| Thing | Convention |
|---|---|
| Sim + harness joint units | **radians, CMD order**: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| LeRobot follower units | body joints in **degrees** (`use_degrees=True`), gripper **0..100** (0 = closed, 100 = open) |
| The only converter | `src/robot_manipulator/harness/units.py` + `config/sim2real.yaml` |
| LeRobot degree zero | **midpoint of the calibrated range** (`(range_min+range_max)/2`). Full-turn joints (`wrist_roll`, range 0..4095) are zeroed by the homing offset alone. |
| MJCF zero | Menagerie `robotstudio_so101` = TheRobotStudio "new calib" model: **zero = middle of each joint's range**. This matches LeRobot, so offsets should be about 0 **if the calibration sweep was complete**. |
| Feetech homing | `Present = Actual − Homing_Offset`; register is ±2047, sign-magnitude |

---

## 1. Hardware sanity (minutes)

```bash
uv run bench-check                      # follower port, cameras, MIDI
```

Read-only voltage and ping probe (no torque):

```bash
uv run python - <<'EOF'
import scservo_sdk as scs
for name in ("so101_leader", "so101_follower"):
    p = scs.PortHandler(f"/dev/{name}"); ph = scs.PacketHandler(0); p.openPort(); p.setBaudRate(1_000_000)
    print(name, {i: (ph.read1ByteTxRx(p, i, 62)[0] / 10) for i in range(1, 7)})  # volts per servo
    p.closePort()
EOF
```

- **Pass:** all six IDs answer on both arms. The **leader reads about 5 V and the follower about 12 V.**
- **Lesson (12 V on a 5 V leader):** the leader's 12 V brick was plugged into the leader board by
  mistake. The 7.4 V-rated leader servos still read fine, but LeRobot's broadcast-ping handshake kept
  missing ID 2. When a handshake fails for a single servo, check the voltage before blaming the cable.
- **Lesson (brown-out):** a follower going from a slumped pose to full torque on all six joints at
  once can brown out the 12 V bus ("no status packet"). Turn torque on joint by joint, as
  `uv run teleop` does.

## 2. Arm calibration (per arm, about 10 min)

```bash
uv run calib-check [--follower PATH] [--leader PATH]   # vs a 15-arm fleet (NVIDIA workshop stats)
uv run calib-recal [--joints shoulder_pan,elbow_flex,wrist_flex] [--wrist-roll/--no-wrist-roll]
```

- `calib-check` flags ranges more than 2σ from the fleet. **An incomplete sweep shifts the zero:**
  this follower's pan swept 1905 counts against a fleet of 2725 ± 32 (−26σ). The result was a
  constant pan offset against the leader during teleop.
- `calib-recal` re-records only the selected joints, using LeRobot's own `set_half_turn_homings` and
  `record_ranges_of_motion`, and keeps everything else in the calibration JSON:
  - It aligns `wrist_roll` (which has no end stops) **to the leader's current angle**. The two
    wrists only need to look the same; no full rotation is needed.
  - It backs up the JSON, runs the fleet check, and lets you accept or reject. On reject or Ctrl+C
    it writes the old calibration back.
- **Never sweep a non-back-drivable gripper** (PincOpen). Recalibrate its range separately (§2b).
- **Gripper direction inverted?** Set `"drive_mode": 1` for the gripper in the follower JSON.
  LeRobot doesn't check `drive_mode` against the servo, so this is safe and applies everywhere,
  including `lerobot-record`.
- **Pass:** `calib-check` is clean, or only flags explained outliers. Example: the PincOpen allows
  more `wrist_flex` travel than the stock jaw (+7.6σ, expected).
- The calibration files are `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`
  and `.../teleoperators/so_leader/<id>.json`.

### 2b. PincOpen gripper facts

The PincOpen is wired as follower servo ID 6 and is not back-drivable.

| Fact | Value |
|---|---|
| Open | 2005 steps |
| Closed | 3200 steps (including preload) |
| LeRobot reading with fingers touching | about 6.8, not 0 |

Use a torque of 20 % or more for motion. The stall guard on load and current works; see
`robot_manipulator/gripper.py`.

## 3. Teleop verification (about 10 min)

```bash
uv run teleop --max-speed 90 [--pid elbow_flex=24,0,32]   # Ctrl+C ONCE to stop
uv run teleop --park-only                                  # recover an arm left holding torque
```

What the tool does:
- Torque comes on joint by joint, with Goal = Present first, so nothing jumps.
- The follower moves to the leader's pose one joint at a time (staging prints `residual …` for
  joints that sag).
- It mirrors with a velocity cap, and parks back to the start pose before torque goes off.
- The leader then drives itself to its rest pose at 30 % torque.
- The trace is saved to `runs/teleop/*.parquet` **before** parking. A table of tracking error and
  load is printed.

- **Pass:** steady error under 1° on all joints, the gripper direction is right, and there's no
  constant offset on any joint.
- **Lesson (gravity sag):** at P=16 the gravity-loaded joints sag. With the PincOpen the elbow was
  3.7° short; at P=24 it is 0.14°. P=24 is now the default for `shoulder_lift` and `elbow_flex`.
- **Lesson (load near 100 %):** load near 100 % with a fixed error at one pose is a **self-collision**.
  Here the wrist camera bracket hit the arm. Record it in `config/so101_limits.yaml` as evidence
  for the collision model (§5).

## 4. Robot model (MuJoCo)

- **Base model:** Menagerie `robotstudio_so101` (Apache-2.0), vendored in
  `src/robot_manipulator/sim/assets/so101/`. `SOURCE.md` records the provenance and every local edit.
- **End-effector swap** (done here for the PincOpen, `so101_pinc.xml`). Keep joint and actuator
  **names and order** unchanged; the replacement driven joint is still called `gripper`.
  1. Get the meshes and kinematics: the URDF/xacro pivots, mimic ratios and limits. For the PincOpen
     that is CNURobotics/pinc_open_driver.
  2. Model linkages with `<equality><joint polycoef="0 k 0 0 0">`, which MuJoCo enforces **only while
     stepping**. Kinematic callers (twin, FK) must call `sim.model.apply_joint_couplings()`.
  3. Use **masses from a scale**, not CAD placeholders. A community PR shipped 1.1 kg with inertias
     1000× too large. Use box-approximation inertias.
  4. Set the gripper joint range from the real servo travel (here 105° → [−1.833, 0] rad, 0 = closed).
  5. Get the **mount orientation from photos and the operator** ("the camera sits above the motor").
     The first guess here was 180° off, and the gripper passed through the arm in the twin.
  6. Move the EE site (`gripperframe`) to between the fingertips.
- **Scene:** the floor is the table. The base plate here sits **2.75 cm above the table**, so the
  floor is at z = −0.0275. Match materials to the real colours, which matters for vision.
- **Known limitation:** Menagerie's arm collision geoms are coarse boxes.
  - At the real folded rest pose they already interpenetrate: shoulder_lift is −108° there, past the
    MJCF limit of ±100°.
  - So **don't trust self-collision near the fold**, and don't "validate" a collision with them.
    One false validation was withdrawn here.
  - Better collision geometry is an open item.
- **Pass:** `uv run pytest tests/test_sim_pinc_model.py` (mass, parallel fingers, coupling, no
  contacts at HOME), and renders that match photos of the real arm.

## 5. Joint map + live twin (about 10 min)

```bash
uv run twin [--flip <joint>] [--scene scene_pinc.xml]    # read-only: torque must be off
```

The twin shows the real follower in a MuJoCo viewer through `config/sim2real.yaml`:

```
rad = sign * deg2rad(deg) + deg2rad(offset_deg)
gripper: linear map, lerobot_open -> sim_open, lerobot_closed -> sim_closed
```

1. Move **one joint at a time** by hand.
   - If the twin mirrors the joint, that's a **sign** problem; confirm with `--flip <joint>`.
   - If it's off by a constant angle, that's an **offset** problem.
2. Map the gripper's "closed" to the reading **at finger contact** (about 7 here), not 0. Otherwise the
   sim shows a gap when the real gripper is closed.
3. Set `calibrated: true`. Live harness runs are refused until you do.

- **Pass:** every joint tracks with no visible offset, and the gripper opens and closes with the real
  one. Here every sign was +1 and every offset 0.

## 6. Cameras

```bash
uv run cam-setup [--snapshot] [--list]      # lock exposure/white balance; list stable paths
uv run camcal capture <name>                 # live preview, auto-saves ~25 diverse board views
uv run camcal intrinsics <name>              # -> config/camera_intrinsics/<name>.yaml
```

- **Identify cameras by USB-port path**, set in `config/cameras.yaml` (`/dev/v4l/by-path/...`).
  - Identical webcams can share a serial (here both are `SN0001`), so `by-id` can't separate them.
  - `/dev/videoN` indices shuffle on replug. Here workspace went from 0 to 2.
  - Keep each camera in the same USB port.
- **Lock the exposure.** Auto-exposure plus "dynamic framerate" made the colours and brightness
  drift (mean 85). Locked at exposure 300, gain 10 and white balance 3800 K, the mean is 114 and
  stable. The settings reset on replug, so re-run `cam-setup`.
- **The board here:** ChArUco **5×7, `DICT_4X4_50`**, reprinted 2026-10-20 on a flat plate
  (the old `DICT_7X7_50` paper board was wavy). Squares 33.2 mm, markers 21.6 mm, set in
  `config/charuco_board.yaml`; the joint wrist-camera fit (`solve-joint --fit-scale`) checks the
  print scale.
- **Pass:** intrinsics RMS under about 0.5 px with 15 or more views, fx ≈ fy, and principal point
  near the image centre.
- **Lesson (old intrinsics):** the old `camera_intrinsics.json` (fx 9697 / fy 2109, RMS 5 px) was a
  broken fit, not a bad camera. The same archived images calibrate to fx ≈ 554.

## 7. Camera extrinsics (camera → robot base) — *in progress*

**Preferred: automatic, with the wrist camera** (`calib/wristcal.py`; no touching):

```bash
uv run camcal-wrist plan                         # sim: viewpoints over the board (roll kept ±60° of natural)
uv run camcal-wrist run --backend sim            # rehearse the motion in the twin
uv run camcal-wrist run --backend real --live    # operator present, ~3 min, arm folded at the start
uv run camcal-wrist solve runs/wristcal/<ts>     # report; add --write to save poses/intrinsics
```

The follower moves the wrist camera through ~24 planned viewpoints over the board (comfortable
reach) and stores joints + wrist + workspace images at each. A bundle adjustment solves the board
pose, the wrist-camera mount and the lift/elbow/wrist_flex encoder offsets from the reprojection
of every board corner. Held-out views validate it. The workspace camera pose follows from its own
view of the board. Pan and wrist_roll offsets aren't observable this way (pan moves everything
rigidly; roll is absorbed by the camera mount).

- **First real run (2026-10-18, `runs/wristcal/20261018-204825_real`):** 24/24 viewpoints, 23
  usable. Bundle adjustment 10.5 → 5.8 px with joint offsets (held-out 5.9 px): elbow
  −5.5 ± 0.8°, wrist_flex −4.6 ± 0.5°, lift +7.9 ± 1.8°. The touch session agrees on wrist_flex
  (−4.2°) and the elbow sign (−9.5°), but not on lift, and applying lift makes the touch data
  worse. So nothing goes into `sim2real.yaml` until a second session (board moved, rigid) agrees.
  The camera mount came out 5.6 cm from the photo estimate. Workspace camera at
  (5.7, 9.4, 49.7) cm; the sim overlay lands on the real arm within a few px.
- **Second run + joint fit (2026-10-20, `runs/wristcal/20261020-142141_real`):** new flat board
  (5×7 `DICT_4X4_50`, 33.2 mm), planned from the calibrated overhead camera
  (`plan --board-from-camera`, no touches). 24/24 views, wrist intrinsics 0.12 px (the wavy paper
  board gave 0.52). `solve-joint --fit-scale` over both runs: one camera mount and one set of
  offsets, a board pose and print scale per run. **Applied to `config/sim2real.yaml`:** lift
  +4.61 ± 1.06°, elbow −4.75 ± 0.48°, wrist_flex −4.58 ± 0.29°. Overhead camera re-derived from run
  2: (4.7, 10.0, 50.1) cm, 1.2 cm / 1° from run 1's estimate.
- **Lesson (board scale):** a hand-measured square size is a hidden calibration parameter. Fitting
  each board's scale in the joint BA (the arm's link lengths give the absolute scale) found the old
  paper board 2.8 % larger than its "40 mm" and confirmed the new one (0.9996). Without the scale,
  the two runs disagreed on lift by ~1.5°.
- **Lesson (offsets vs travel):** an encoder offset shifts where the sim joint box lands on the
  servo. Elbow −4.75° put the box's end at 98.7° against a 97.5° calibrated travel. The real
  backend now clips every command to the calibrated travel (−1°), whatever the joint map says.
- **Lesson (residual is per pose):** per board corner the residuals are small and even, so the
  board is fine. Per view they're offset by ~3–15 px, i.e. ~0.5° / 2 mm of pose error per arm
  pose (joint play, link flex). That is roughly the arm's absolute accuracy floor.
- **Lesson (corner threshold):** the closed fingers hide the board's near rows in the wrist image.
  The 12-corner capture threshold rejected 13 good views; the BA uses views with ≥ 8 corners. The
  overhead PnP pools corners from every frame of the session (the board never moves; the arm hides
  a different part each time).
- **Lesson (touching at full stretch):** with the board where the overhead camera sees it whole,
  the far corners were at the edge of the arm's reach, and touching them in several poses wasn't
  possible. The wrist-camera method only needs the board visible, from wherever the arm reaches
  comfortably.

**Fallback: corner touches** (needs every corner within comfortable reach):

```bash
uv run camcal capture workspace          # 15–25 board views, varied tilt and position
uv run camcal intrinsics workspace       # -> config/camera_intrinsics/workspace.yaml
uv run camcal-extrinsics workspace       # board taped flat; touch corners 1-4 twice each (torque off)
uv run scene build && uv run scene render
```

A touch method that needs no extra hardware:
1. Lay the board flat on the table where the camera sees it.
2. With follower torque off, touch 4 or more known board corners with the **closed gripper tip**. At
   each touch, record FK of `gripperframe` through the calibrated joint map. Fit with Kabsch to get
   the board → robot-base transform.
3. The camera detects the same board with `solvePnP` and the intrinsics, which gives
   camera → board.
4. Chain them: camera → base. Write the result to `config/cameras.yaml` (pose) and the sim scene.

- **Pass:** touch residuals of a few mm, board sides from the touches within ~3 mm of the real
  board, board height ≈ table top. The tool won't write the camera pose above `--max-residual-mm`
  (8) without confirmation.
- **Touch protocol:** tape the board down. Touch each corner twice (`--per-corner 2`): once with
  the tool vertical, once tilted with the arm reaching differently. Those varied orientations let
  the fit solve for the real fingertip point. Raw joints go to `runs/extrinsics/<cam>.json`;
  re-fit offline with `--refit`.
- **Lesson (first touch session, 2026-10-18):** 4 single touches gave residuals of 7–34 mm. The
  forward–back board sides matched within 2 mm, but the left–right sides (swept by shoulder pan)
  came out 2.2 and 5.7 cm short. That's not a rigid board, and the raw joints weren't saved, so the
  cause was left open (fingertip model, a joint calibration error, or the board sliding). Always
  save the raw joints and check the side lengths.
- **Bimanual:** do it once per arm on the same board pose. That also gives the **arm-to-arm
  transform** for the shared MJCF.

### 7b. Fingertip accuracy + play (`tip-accuracy`, about 15 min)

Checks the whole chain (joint map, servo stopping error, fingertip model, board estimate) in the
world frame. The arm hovers the closed fingertip 40 mm over board corners the wrist camera can
measure, from four approach directions (base, far, left, right), and photographs each arrival
twice: raw, then after the settle trim. The difference between approach directions is the play
seen at the tool.

```bash
uv run tip-accuracy plan                        # targets the wrist camera can see (sim only)
uv run tip-accuracy run --backend real --live   # operator present; contact guard on the approach
uv run tip-accuracy analyze runs/accuracy/<ts>_real
uv run blog-figs --only trim                    # raw vs trimmed dumbbell
```

- **Run 3 (2026-10-20, offsets applied, 5 corners × 4 approaches, no contacts):**

  | | in-plane error, median (max) | play at tool, median | spread per corner |
  |---|---|---|---|
  | raw | 10.5 mm (11.2) | 3.1 mm | 1.3–4.8 mm |
  | settle trim | **4.9 mm (7.3)** | **1.3 mm** | 1.2–1.5 mm |

- **Lesson (what the trim removes):** the trim cancels the approach-dependent stopping error (play
  3.1 → 1.3 mm, spread ≤ 1.5 mm). What's left is a **bias of −2 to −7 mm along base x, the same
  sign at every corner**, growing toward the far corners. That's a model or calibration error
  (fingertip length, board estimate), not the servo: repeatability is already 1.3 mm.
- **Lesson (runs 1–2):** at 20 mm hover the fingertip touched the board (servos stop 6–10 mm
  short, fingertip ~6 mm longer than modelled). The contact guard must be off during the trim:
  the trim's own corrections look like a rise in following error.

## 8. Scene objects (fixtures, reset rig) — *in progress*

- `uv run scene build` writes `sim/assets/so101/scene_bench.xml` from `config/scene_bench.yaml`:
  table, overhead column, workspace camera (from §7 once calibrated, a placeholder before that), and
  the leader as static geoms at its stow pose. `uv run scene render` saves the sim view of
  `workspace_cam` for comparison with the real snapshot.
- Measure the layout in the robot frame (x forward, y left from the follower base). Here: column
  centre 11.45 cm to the follower's left, leader another 11.45 cm beyond it, follower centre
  22.5 cm from the right table edge.

- Source mesh: `assets/cup.stl` (mm), the auto-reset chute. It is a 100 mm cylinder with an internal
  ramp and a side opening onto a rimmed tray (~180×160 mm); overall 300×225×125 mm.
- MuJoCo collides meshes as **convex hulls**, which would fill in the cup. Use a convex
  decomposition (e.g. CoACD) or hand-built primitives for collision, and keep the mesh as the visual.
- Find the pose by touching 3 points on the rim and 1 tray corner, then fit. Or detect it in the
  calibrated workspace camera.

## 9. Actuator real2sim

```bash
uv run gap run --backend real --live --max-vel 1.0     # 7 motions, ~3 min (operator present)
uv run gap compare runs/gap/<ts>_real [--fitted]       # sim fed the real *sent* commands
uv run gap fit runs/gap/<ts>_real                      # -> config/actuator_fit.yaml (~10 min)
uv run gap stiffness                                   # static k per joint from survey sag
uv run gap fit runs/gap/<ts>_real --kp-from-sag --out runs/gap/fits/<name>.yaml
# closed-loop check: same policies, targets and flags on the arm and on each candidate twin
uv run harness reach --backend sim --policy <p>/reach.onnx --random 20 --seed 0 --max-vel 1.0 \
    --settle --actuator-fit runs/gap/fits/<name>.yaml
uv run gap closed-loop runs/harness/<real_run> runs/harness/<sim_run> ...
```

- **Static stiffness** (`gap stiffness`): at rest `k (q_cmd − q) = τ_gravity(q)`. Regress the
  held-pose error from `harness survey` telemetry on the twin's gravity torque. Joints with no
  usable gravity signal (pan, wrist_roll, or correlation < 0.5) get the best joint's k scaled by
  the servo P coefficient. Here: shoulder_lift **18 N·m/rad** (r 0.78, P=24) and wrist_flex 5.5
  (r 0.56, P=16); the nominal model's kp is 998, so the real servos are ~50–180× softer statically.
  - **Lesson (static ≠ dynamic stiffness):** pinning kp to the sag value and fitting the rest
    failed. Every other parameter went to its lower bound and chirp RMSE rose from 5–18 to
    40–100 mrad, because a spring that soft can't follow the chirps as fast as the arm does. The
    sag is most likely gear backlash, compliance or a deadband rather than the servo's P loop. It
    belongs in the twin as a separate element (e.g. a gravity-dependent offset), not in kp.
- **Closed-loop check** (`gap closed-loop`): a twin is good when a *fixed* policy behaves on it as
  it does on the arm. That covers final accuracy and hunting, measured as the Oscillation guard
  trip rate (replayed offline) and command reversals per second after 2 s.

- **Lesson (delay):** the real arm lagged the nominal sim by 80–100 ms. Make sure the delay search
  grid covers it (0..6 ticks), and **apply the fitted per-joint delay** wherever the fit is used,
  because `apply_actuator_fit` only sets model parameters and returns the delays.
- **Lesson (closed loop):** a fit that predicts open-loop replay (held-out steps 6.6 mrad) can still be
  wrong closed-loop. A policy trained on it hunted on the real arm (17/20, p90 2.3 cm, vs 20/20,
  0.83 cm). Validate actuator fits by running a fixed policy in sim vs real (`harness reach
  --settle`), and pin weakly identifiable parameters (kp) with independent data (static sag).
- **Lesson (held-out):** fit on the chirps, judge on the motions the fit didn't see (steps, sine).
  Parameters pinned on their bounds mean the fit is compensating, not identifying.

- Replay the same motion files in sim and on the real arm. Compare per joint (RMSE, lag, overshoot),
  then fit kp, damping, friction, armature and delay. Retrain with domain randomization centred on
  the fit.
- Requires §5 done, because the joint map must be right before you compare trajectories.

## 10. Policy loop

```bash
uv run sim-train-reach --out runs/ppo_reach/<name>          # ~15 min CPU, 3M steps
uv run harness reach --backend sim --policy runs/ppo_reach/<name>/reach.onnx --random 100 --dr
uv run harness reach --backend real --live --max-vel 0.5 ...   # operator present
```

### 10a. Reach policy recipe (what works on the real arm)

Zero-shot sim2real reach: PPO in MuJoCo, exported to ONNX, run by the harness. The deployed policy
is **`pinc_box_3M`**: sim 95 % with DR / 100 % without; real arm 20/20 at 0.5 rad/s, and 20/20 at
1.0 rad/s with `--settle` (median 0.42 cm, p90 0.83 cm from the target, in twin coordinates).

**Files.** Env and the shared obs/command code: `sim/envs/reach.py`. DR: `sim/randomize.py`.
Training: `sim/train_reach.py`. Export: `sim/export.py`. Deploy: `harness/policies/onnx_reach.py`.
Model: `sim/assets/so101/scene_pinc.xml` (PincOpen). Joint map: `config/sim2real.yaml`.

**Task** (`sim/envs/reach.py`):

| Item | Value |
|---|---|
| Control | 50 Hz, 4 s episodes (200 steps) |
| Action | `Box(-1, 1, (5,))`, a joint-position *delta* per arm joint: `ACTION_SCALE` = 0.02 rad/tick = **1.0 rad/s** at a = 1. The gripper is held open (−1.833 rad) |
| Command | `integrate_command(q_cmd, a, box, max_step)`: clip to the joint box (limits − 0.05 rad), then to the speed cap. **The same function runs in training and at deploy**, with `max_step` = the harness `--max-vel` |
| Observation (24) | q (5), dq (5), last action (5), EE position (3), target (3), target − EE (3). Built by `build_obs`, shared with deploy. Normalized with the training `VecNormalize` stats, which are stored in `policy_meta.json` |
| Targets | FK of random joints within 80 % of each range, kept inside `TARGET_BOX` (base frame: x 0.10–0.40, y −0.25–0.25, z 0.02–0.30 m). **One sampler for training and the harness** |
| Reset | HOME `[0, −1.57, 1.57, 0, 0]` + U(±0.1 rad) per joint |
| Reward | `−d + (1 − tanh(d / 0.05)) − 0.01‖a − a_prev‖² − 0.001‖dq‖² + 0.5·[d < 2 cm]` (d = EE–target distance) |
| Success | d < 2 cm at the end of the episode |

**Domain randomization** (`DRConfig`, applied fresh from nominal on every reset):

| Randomized | Range | Notes |
|---|---|---|
| Actuator kp and kv | × U(0.8, 1.2) | one factor per joint, kp and kv together |
| Joint damping | × U(0.7, 1.3) | |
| Joint friction loss | × U(0.5, 2.0) | |
| Armature | × U(0.8, 1.2) | |
| Link masses and inertias | × U(0.9, 1.1) | per body |
| Extra mass on the gripper | U(0, 0.09) kg | the model has 0.212 kg past the wrist; the real gripper + camera weigh ~0.2–0.3 kg |
| Action delay | {0, 1, 2} ticks | 0–40 ms |
| Measurement noise | q N(0, 0.005 rad), dq N(0, 0.05 rad/s) | |
| Per-episode joint bias | U(±0.02 rad) on measured q | calibration error |
| Speed cap | U(0.5, 1.0) rad/s per episode | **added after `pinc_box_3M`** (in `pinc_cap_3M`, `pinc_fit_3M`); see results |

**PPO** (Stable-Baselines3, `sim/train_reach.py`): 12 `SubprocVecEnv` workers, `n_steps` 1024
(rollout 12 288), `batch_size` 2048, 10 epochs, γ 0.99, λ 0.95, learning rate 3e-4 decaying
linearly, clip 0.2, entropy 0, `target_kl` 0.02, MLP 256×256 for policy and value, `VecNormalize`
(observations and rewards, clip 10). 3 M steps take ~17 min on the laptop CPU. Evaluation: every
100 k steps on 50 DR episodes; at the end 100 episodes with and without DR.

```bash
uv sync --extra sim --extra dev
uv run sim-train-reach --out runs/ppo_reach/<name>              # 3M steps, DR on
uv run sim-train-reach --out runs/ppo_reach/<name> --actuator-fit   # nominal = config/actuator_fit.yaml
# -> runs/ppo_reach/<name>/reach.onnx + policy_meta.json (obs stats, action scale, scene,
#    gripper hold, sim/repo commit, DR, evaluation); the loader refuses a scene/gripper mismatch
uv run harness reach --backend sim --policy runs/ppo_reach/<name>/reach.onnx --random 100 --dr
uv run harness reach --backend sim ... --actuator-fit config/actuator_fit.yaml   # another twin
```

**Deploy, in this order** (operator present; the leader stows to its left on its own):

```bash
uv run harness reach --backend real --policy <p>/reach.onnx --random 5                       # dry run: read-only
uv run harness reach --backend real --policy <p>/reach.onnx --random 5 --live --max-vel 0.5
uv run harness reach --backend real --policy <p>/reach.onnx --random 20 --live --max-vel 1.0 --settle
uv run gap closed-loop runs/harness/<real_run> runs/harness/<sim_run>   # does the twin agree?
```

`--max-vel` above 1.0 doesn't make these policies faster: 1.0 rad/s is their full step. The
oscillation guard is on by default for real runs.

**Results that decided the recipe** (same 20 targets, `--settle --max-vel 1.0` unless noted):

| Policy | Trained on | Sim (DR / no DR) | Real arm |
|---|---|---|---|
| `pinc_box_3M` | nominal twin + DR above (no speed-cap DR) | 95 % / 100 % | **20/20**, median 0.42 cm, calm |
| `pinc_cap_3M` | + speed-cap DR | 96 % / 96 % | 19/20 (early 20-target run, not `--settle`) |
| `pinc_fit_3M` | twin with the v2 actuator fit as nominal | 98 % / 100 % | 17/20, median 0.62 cm, **hunts** |

- **Lesson (train on the broad twin, not the fit):** the actuator fit predicts open-loop replay
  well, yet its policy hunted on the real arm. Wide DR around the nominal model transfers better.
  Check any new twin closed-loop (`gap closed-loop`) before training on it.
- **Lesson (speed cap):** integrate the policy's command at the deploy cap (`max_step`). Without
  it, the first real runs at 0.5 rad/s went 4/5.
- **Caveat:** the accuracy numbers are in twin coordinates (FK of the measured joints). The wrist
  camera calibration (§7) found encoder offsets of ~−5° (elbow) and ~−4.6° (wrist_flex), so the
  real fingertip error is larger until those are applied to `config/sim2real.yaml`.

- **Real backend safety** (`harness/backend.py` + `hw/follower.py`, shared with teleop):
  - A **dry run is read-only**: it opens the bus only, with no register writes and no torque change.
  - **Live** runs turn torque on joint by joint with Goal = Present and P=24 on lift and elbow.
  - **Close and Ctrl+C park** at the rest pose before releasing torque joint by joint.
  - **Lesson:** the first version used `SOFollower.connect()`/`disconnect()`. That turned all six
    torques on at once (even in a dry run) and dropped the arm wherever it was on close.
- The policy metadata records the scene, the gripper hold and the action scale. The loader refuses
  a mismatch, so a stock-gripper policy can't run on the PincOpen model.
- **Lesson (target sampling):** sample targets from the **task workspace** (`TARGET_BOX`), not from
  joint space alone. Otherwise training wastes effort on targets with the arm folded over itself.
- **Lesson (targets the filter will actually send):** clip every scripted target (HOME etc.) into
  the SafetyFilter command box. The first live run failed HOME 5/5. The arm itself was within 2°,
  but the gripper target sat on the joint limit and the filter's 0.05 rad margin made it unreachable
  within HOME_TOL. In sim it passed by a hair, so test with real-sized errors.
- **Lesson (parking path):** park **via HOME**. A staged fold that starts wherever the task ended
  can sweep the tool into the table: the elbow folded while the shoulder was raised and stalled 33°
  short. Validate parking paths in sim from task end poses, not just from HOME.
- **Lesson (one sampler):** the harness and the trainer must share one target sampler. When they
  differed, harness success dropped from 99 % to 78 %.
- **Oscillation guard** (`harness/skill.py: Oscillation`, on by default for `--backend real`):
  fails the skill when a joint reverses direction ≥ 8 times within 1 s while staying within
  0.05 rad. A hunting policy stops in about 1–2 s instead of shaking for the whole episode. Tuned
  on real traces: 20/20 hunting episodes caught, 0/60 calm ones flagged.
  - **Lesson (sim dithers more than the arm):** the same calm policy that never trips the guard on
    the real arm trips it in 5/30 nominal-sim settle episodes. In sim, all joints dither at the goal
    (peak-to-peak median 38°), against 1 flagged window on the real arm. The nominal twin has far
    less friction than the arm (the actuator fit puts it at 8–18×). So the guard is real-only by
    default (`--oscillation-guard` forces it in sim), and friction belongs in the next actuator fit.


## 11. Telemetry + workspace atlas (what the arm can do where)

```bash
uv run atlas prior [--n 200000]        # layer 1: twin physics, ~35 s -> runs/atlas/sim_prior.parquet + maps
uv run harness survey --backend real --n 20 --live --max-vel 0.5   # hold at spread poses (operator present)
uv run atlas real                       # layer 2: aggregate runs/telemetry -> runs/atlas/real.parquet
uv run atlas query X Y Z [--down]       # look up a point (base frame, m)
```

- **Telemetry is automatic** on every real teleop/harness/survey run: `runs/telemetry/<stamp>_<source>`
  `.parquet` + `.json`. It logs commanded vs measured joints in LeRobot units and sim radians, plus load,
  current, temperature and voltage (every 10th step, or every 2nd during a survey). The `.json` holds
  the joint map, calibration file (sha256) and git commit.
- **Layer 1 (prior):** per 2.5 cm voxel, from contact-free sampled poses: holding-torque margin, max
  payload and max downward push, overall and with the tool pointing down. These are static
  estimates. They overstate push near singular poses (e.g. arm straight up).
- **Layer 2 (real):** per voxel, median tracking error, median quasi-static load, and the twin's
  predicted gravity load at the same poses. It fits `measured ≈ gain × predicted` per joint.
  - Reach runs give almost no static samples, so **run the survey**.
  - Use medians: the servo bus occasionally misreads a register (one 75 °C sample among a steady
    24 °C).
- **Planned:** scale presses for push-force ground truth (MiniLab), and atlas queries feeding the safety
  filter and target sampling.

---

## Adapting to another setup (e.g. the bimanual SO-101s at work)

1. **Per arm:** a udev symlink per serial board (`/dev/so101_left_follower`, …), its own LeRobot
   `id`, and its own calibration JSON. Run §1–§3 per arm and pair.
2. **One MJCF with two arm instances**, each with a `prefix` (names `left_shoulder_pan`, …). Keep
   one joint map per arm (`config/sim2real_<arm>.yaml`). `uv run twin` needs `--port`, `--robot-id`
   and `--joint-map` per arm (not implemented yet).
3. **Arm-to-arm transform:** from §7 with both arms touching the same board pose.
4. **Inter-arm collision:** the coarse-box caveat in §4 applies doubly. Get accurate collision
   geometry before trusting a safety filter between arms.
5. **End-effectors:** follow §4. Weigh them, take photos, and confirm the mount orientation with the
   operator.
6. **Cameras:** follow §6, one by-path entry per camera.
7. **Calibrate with the wrist camera, not touches** (§7): `camcal-wrist` per arm, with the board
   placed where that arm reaches it comfortably. The same session gives the joint offsets.
   Both arms seeing the same board pose gives the arm-to-arm transform.
8. **Reach policy per arm** (§10a): one training run per gripper type is enough when the arms are
   identical. Retrain if the gripper (mass, tool length) or the joint map differs. Mirror
   `TARGET_BOX` in y for an arm mounted facing the other way.
9. **Order of work that paid off here:** calibration check → teleop → joint map + live twin →
   intrinsics → wrist-camera extrinsics → policy in sim → dry run → slow live run → settle run →
   closed-loop twin check.

## Status of this setup (home bench)

| Stage | Status | Evidence |
|---|---|---|
| 1 Hardware sanity | ✅ | leader 4.9 V / follower 12.0 V; all IDs answer |
| 2 Arm calibration | ✅ follower recal'd 2026-10-17 | `calib-check`: pan 2722 (fleet 2725 ± 32) |
| 3 Teleop | ✅ | elbow steady error 0.14° at P=24; trace `runs/teleop/20261017-204250.parquet` |
| 4 Robot model | ✅ PincOpen variant (default) | 0.212 kg past wrist; collision near fold untrusted |
| 5 Joint map | ✅ `calibrated: true` | live twin, all signs +1 / offsets 0, gripper closed ↔ 7 |
| 6 Cameras | ✅ stable paths, locked exposure, intrinsics 53 views / 9 regions, rms 0.52 px | `config/camera_intrinsics/workspace.yaml` |
| 7 Extrinsics | ✅ overhead camera (run 2, PnP 0.18 px) + **encoder offsets applied** (two wrist-camera runs agree; lift +4.6, elbow −4.75, wflex −4.58°) | `config/extrinsics/workspace.yaml`; `config/sim2real.yaml`; `runs/wristcal/joint_*.json` |
| 7b Fingertip accuracy | 🟡 run 3: raw 10.5 mm → trimmed **4.9 mm** median, play 1.3 mm; −x bias left (tool model / board) | `runs/accuracy/20261020-170433_real/report.json` |
| 8 Scene / reset cup | 🟡 bench scene generated (table, column, leader); camera pose placeholder until §7; cup not added | `config/scene_bench.yaml` |
| 9 Actuator fit | 🟡 open-loop OK, **closed-loop not yet**: no twin (nominal / v2 / sag-pinned v3) reproduces 'box calm, fit hunts' (`gap closed-loop`). First fit: chirp RMSE 68–78 → 8–21 mrad, lag 80–100 → 0 ms; held-out steps 18.7 → 9.8 mrad | `config/actuator_fit.yaml`; `runs/gap/20261018-125035_real` | |
| 10 Policy | 🟡 sim 95 % / harness 99 %; **real zero-shot 20/20 at 0.5 rad/s** (4/5 before the cap fix) | `runs/harness/20261018-123358`; telemetry `runs/telemetry/20261018-120145_harness_live.*`; cap-consistent retrain next |
| 11 Atlas + telemetry | 🟡 prior built; survey 21/21 poses; 1702 static samples; lift load r=0.78 | `runs/atlas/`; `runs/telemetry/` |
