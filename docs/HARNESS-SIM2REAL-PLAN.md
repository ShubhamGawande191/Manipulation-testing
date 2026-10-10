# Deployment harness + MuJoCo sim2real — work order

Written 2026-10-10 (Opus, planning). **Executed by Haiku while the user is away → software only.**
Read this whole file before starting a ticket. Do tickets in order; one commit per ticket.

## Hard rules for the implementer

1. **No hardware.** Never connect to `/dev/so101_*`, never torque servos, never run `LeRobotBackend`
   against a real port, never run `gripper.py`. Hardware tickets (H1–H4) are for the user.
2. Branch: `harness-sim2real`, created from `minilab-phase0`. Never push to `main`.
3. Every ticket ends with: `uv run pytest` green, `uv run ruff check .` clean,
   `uv run ruff format .`, then a commit (`feat(harness): …` / `feat(sim): …`).
4. Do not commit `third_party/`, `assets/*.zip`, `runs/`, model checkpoints, or `.onnx` files.
5. Units: **sim + harness internal = radians, CMD order** (`shoulder_pan, shoulder_lift, elbow_flex,
   wrist_flex, wrist_roll, gripper`). LeRobot boundary = degrees (body) + 0..100 (gripper). Convert
   ONLY in `harness/units.py`. Never in env/policy code.
6. Tests that need MuJoCo start with `mujoco = pytest.importorskip("mujoco")`.
7. Code in `src/robot_manipulator/`; no scripts in the repo root (see CLAUDE.md hygiene note).
8. If a ticket's acceptance can't be met, stop, write what failed into `docs/harness-sim2real.md`
   under "Blockers", commit, and move on only if the next ticket doesn't depend on it.

## Context (why)

- Goal: a **deployment-first harness** that runs RL policies trained in sim, BC skills, and scripted
  skills through one runtime with safety + logging; then an RL loop on the real auto-reset rig
  (objects go through a hole on the arm's right; a gravity chute returns them to a tray).
- Simulator: **MuJoCo** (user decision). Isaac Sim is out. GPU is an RTX 3060 Laptop 6 GB
  (driver 580.178, CUDA 13 OK after reboot); training starts on **CPU (SB3 PPO, vectorized MuJoCo)**
  for robustness; MJX/Playground is a later speed-up.
- Milestone 1 = **zero-shot sim2real reach** (PPO in MuJoCo → ONNX → harness → real arm).
- The NVIDIA SO-101 course (docs.nvidia.com/learning/physical-ai/sim-to-real-so-101) is really
  *teleop-in-sim + domain randomization → GR00T N1.6 VLA*, plus co-training, Cosmos augmentation and
  SAGE/GapONet actuator-gap measurement. GR00T fine-tuning does not fit 6 GB, so we mirror its
  *strategies* with MuJoCo + ACT/SmolVLA later (Phase D) and adopt SAGE's gap measurement now (T11).

### Reference numbers pulled from sources
- Menagerie `robotstudio_so101` (Apache-2.0, commit `3abca2e302625cec9429c49806121ea2855cef81`):
  joint names == LeRobot names; `timestep=0.005`, `implicitfast`; STS3215 class:
  `kp=998.22 kv=2.731 forcerange=±2.94 Nm`, joint `damping=0.60 frictionloss=0.052 armature=0.028`;
  sites `gripperframe` (EE), `baseframe`; camera `wrist_cam`. Ranges (rad): pan ±1.91986,
  lift ±1.74533, elbow ±1.69, wrist_flex ±1.65806, wrist_roll −2.74385..2.84121, gripper −0.17453..1.74533.
- NVIDIA workshop (Isaac Lab, `assets/so101.py`): implicit PD per joint kp 55/30/25/12/7/4,
  kd 0.7/0.8/0.7/0.5/0.5/0.3, effort 30; sim dt 1/120, decimation 2 (60 Hz control); SAGE motion
  replay at 50 Hz control, 200 Hz physics. Real deploy: two 640×480@30 cams, robot "slowly rises to
  initial pose" before inference and returns home on Ctrl-C. DR: lighting exposure (−4, 3), color
  temp (2500, 9500) K, HDRI, external cam pos ±2 cm (z ±1 cm) and rot ±0.05 rad, object poses.
- LeRobot 0.5.1 `SOFollower`: `use_degrees=True` default (body joints in degrees, gripper 0..100).
  The degree zero is the *calibration* midpoint, NOT necessarily the MJCF zero → per-joint
  sign/offset must be calibrated on hardware (H2). Until then `calibrated: false` blocks real runs.

## What exists and must be reused

| Need | Reuse | Where |
|---|---|---|
| Policy interface | `Policy`, `BasePolicy`, `Health` | `core/policy_iface.py` |
| Deterministic clock for tests | `SimClock`, `WallClock` | `exec/scheduler.py:33,43` |
| Velocity cap / watchdog | `VelocityCap`, `Watchdog` | `core/safety.py:20,56` |
| Fallback | `Arbiter` | `core/arbiter.py:38` (wire in later, not in T5) |
| Episode summary parquet | `EpisodeRecord`, `write_records` | `core/episode_log.py` (extend, additive only) |
| Follower config | `build_follower_config(cameras=False)` | `arm.py:23` |
| Connect/obs/send pattern | `_connect`, `_obs_deg`, `_send_deg` | `tasks/press_key.py:61-77` |

## Tickets (software — Haiku)

### T0 — Branch, deps, ignores
- `git checkout -b harness-sim2real` (from `minilab-phase0`).
- `.gitignore`: add `third_party/`, `assets/*.zip`, `runs/`, `*.onnx`.
- `pyproject.toml`: new extra `sim = ["mujoco>=3.3", "stable-baselines3>=2.4", "onnx>=1.16",
  "onnxruntime>=1.18", "matplotlib>=3.8"]`; add `pyyaml` to main deps if not already importable.
  Entry points (fill in as tickets land): `harness = "robot_manipulator.harness.cli:app"`,
  `sim-train-reach = "robot_manipulator.sim.train_reach:app"`.
- `uv sync --extra sim --extra dev`. Accept: `uv run python -c "import mujoco, stable_baselines3, onnxruntime"`.

### T1 — Vendor the SO101 MJCF
- Download Menagerie `robotstudio_so101/` at commit `3abca2e3…` (raw.githubusercontent URLs; files:
  `so101.xml`, `scene.xml`, `LICENSE`, `assets/*`) into `src/robot_manipulator/sim/assets/so101/`.
  Write `SOURCE.md` there (URL, commit SHA, license, list of local edits — none yet).
  Make sure the package data is included (hatch/setuptools config in pyproject) so `importlib.resources` finds it.
- `sim/model.py`: `ASSET_DIR`, `JOINTS = (...)` (CMD order), `load_model(scene="scene.xml") -> MjModel`,
  `joint_qpos_adr(model) -> np.ndarray`, `joint_ranges(model) -> (lo, hi)`, `EE_SITE = "gripperframe"`.
- `tests/test_sim_model.py`: loads; `model.nu == 6`; actuator & joint names == `JOINTS` in order;
  ranges match the numbers above (atol 1e-4); 200 `mj_step`s from `keyframe`/zero → no NaN.

### T2 — Units / joint map
- `config/sim2real.yaml`: `calibrated: false`; per joint `sign: 1`, `offset_deg: 0.0`;
  `gripper: {lerobot_open: 0.0, lerobot_closed: 100.0, sim_open: <rad>, sim_closed: <rad>}`
  (defaults: sim_open = 1.74533, sim_closed = −0.17453 — mark TODO-H2).
- `harness/units.py`: `JointMap.from_yaml(path)`; `lerobot_to_sim(dict[str,float]) -> np.ndarray[6]`;
  `sim_to_lerobot(np.ndarray[6]) -> dict[str,float]` (keys `"<joint>.pos"`). Formula body joints:
  `rad = sign * deg2rad(deg) + deg2rad(offset_deg)`; gripper linear map, clipped.
- Tests: round-trip both ways (atol 1e-9), sign −1 case, gripper clip, missing key raises.

### T3 — Backends
`harness/backend.py`:
- `class RobotBackend(Protocol)`: `joint_names: tuple[str,...]`; `read() -> Obs` with keys
  `t` (float), `q` (rad, 6), `dq` (rad/s, 6); `command(q_target: np.ndarray) -> None`; `close()`.
- `FakeBackend(q0, alpha=0.3, clock)`: first-order lag `q += alpha*(target-q)` on each `command`;
  `dq` finite-diff.
- `SimBackend(model=None, hz=50, viewer=False, randomizer=None)`: `n_substeps = round(1/(hz*timestep))`;
  `command` sets `data.ctrl` and steps `n_substeps`; `read` returns qpos/qvel at `JOINTS` adr;
  `ee_pos()` via site `gripperframe`.
- `LeRobotBackend(port, robot_id="my_so101_follower", joint_map, dry_run=True)`: lazy import of
  `SOFollower`; refuses construction if `joint_map.calibrated is False and not dry_run`;
  `read` = `get_observation()` → `lerobot_to_sim`; `command` = `sim_to_lerobot` → `send_action`
  (skipped + logged when `dry_run`). **Unit-test only with a mock robot object** (monkeypatch); never connect.
- Tests: FakeBackend converges; SimBackend tracks a constant target within 0.02 rad after 2 s;
  LeRobotBackend with a fake robot converts units and honors dry_run / calibrated guard.

### T4 — Safety filter
`harness/safety_layer.py`: `SafetyFilter(lo, hi, margin=0.05, max_vel=1.0, watchdog_timeout=0.25, clock)`
→ `filter(q_cmd_prev, q_target, dt) -> np.ndarray` = clamp to `[lo+margin, hi-margin]` then
`VelocityCap.clamp_step`; `beat()`; `stale` property. Limits default from `sim.model.joint_ranges`.
Tests: clamps out-of-range, caps step size to `max_vel*dt`, stale after timeout with `SimClock`.

### T5 — Skill runtime
`harness/skill.py`:
- `EndCondition` protocol: `name`, `reset()`, `check(obs, ctx) -> Literal["success","failure"] | None`.
- Built-ins: `TimeLimit(seconds)` → failure; `JointReached(target, tol)` → success;
  `EEReached(tol, ee_fn)` → success when `‖ee_fn(obs) - ctx.goal["target"]‖ < tol`;
  `Stalled(window_s, eps)` → failure if `q` moved < eps over window.
- `@dataclass Skill`: `name`, `policy: Policy`, `end_conditions: list[EndCondition]`, `hz=50.0`,
  `max_redo=0`, `on_start: Callable | None` (e.g. glide home), `meta: dict`.
- `@dataclass SkillResult`: `status`, `n_steps`, `duration_s`, `redo_count`, `reason`.

`harness/runner.py`:
- `run_skill(skill, backend, safety, goal: dict, clock=None, log=None) -> SkillResult`. Loop at
  `skill.hz` with `clock.sleep_until`: `obs = backend.read()`; `obs["goal"] = goal`;
  chunk = `policy.act(obs)` (2-D); execute rows one per tick (re-query when exhausted);
  each tick: `q_safe = safety.filter(q_cmd_prev, row, dt)`, `backend.command(q_safe)`,
  `safety.beat()`, log row, check end conditions (first non-None wins). Redo: on failure and
  `redo_count < max_redo` → `policy.reset()`, end-condition reset, `on_start`, loop again.
- `run_sequence(steps: list[tuple[Skill, dict]], ...) -> list[SkillResult]` — stops at first failure
  (a scripted task graph).
- `harness/trajlog.py`: `TrajectoryLog` collecting per-step rows
  (`t, skill, q[6], dq[6], q_target[6], q_safe[6], ee[3]|None, extra`) → parquet
  `runs/<run_id>/traj_<skill>_<n>.parquet`.
- Extend `EpisodeRecord` **additively** with defaults: `backend: str = ""`, `n_steps: int = 0`,
  `return_: float | None = None`, `reset_ok: bool | None = None`. Existing tests must still pass.
- Tests (`FakeBackend` + `SimClock`, 8 Hz for float-exact dt): scripted constant-target policy
  succeeds via `JointReached`; never-arriving policy fails via `TimeLimit` at the exact step;
  `max_redo=1` → `redo_count == 1`; velocity cap visible in `q_safe` diffs; sequence stops on failure;
  parquet written and reloadable.

### T6 — Kinematics helper
`sim/kinematics.py`: `FK(model=None)` holding its own `MjData`; `ee_pos(q) -> np.ndarray[3]`
(`mj_kinematics` + site `gripperframe`), `sample_reachable(rng, n, lo, hi, z_min=0.05) -> (q, ee)`.
Tests: deterministic for fixed q; sampled ee all have z ≥ z_min; FK(SimBackend state) == SimBackend.ee_pos.

### T7 — Reach env (single source of truth for obs)
`sim/envs/reach.py`, `gymnasium.Env`:
- Constants: `HZ=50`, `EP_SECONDS=4` (200 steps), `ACTION_SCALE=0.02` rad/step on the 5 arm joints,
  gripper held at `sim_open`. Action space `Box(-1,1,(5,))`.
- **Command integration (must match deploy):** `q_cmd = clip(q_cmd + ACTION_SCALE*a, lo+m, hi-m)`,
  `q_cmd` initialized from measured `q` at reset.
- **`build_obs(q, dq, last_action, ee, target) -> np.float32[24]`** as a module-level pure function =
  `[q_arm(5), dq_arm(5), last_action(5), ee(3), target(3), target-ee(3)]`. The deploy policy (T9)
  imports this exact function.
- Target: `FK.sample_reachable` with joint limits shrunk 20 %; reset q = home
  `[0, -1.57, 1.57, 0, 0]` + U(±0.1) noise. (Home matches `so101_bridge.HOME`, CMD order; verify sign
  in sim visually is a TODO-H2 note, not a blocker.)
- Reward: `-d - 0.01‖a−a_prev‖² - 0.001‖dq‖² + 0.5·[d<0.02]`; `info["success"] = d < 0.02` at last step;
  `info["dist"]`.
- **Domain randomization** (`DRConfig` dataclass, `enabled=True`, applied on reset to `model` copies):
  kp & kv × U(0.8,1.2) (keep kv/kp consistent per joint), damping × U(0.7,1.3), frictionloss ×
  U(0.5,2.0), armature × U(0.8,1.2), link masses × U(0.9,1.1), **extra gripper payload U(0, 0.25) kg
  on body `gripper`** (PincOpen mass unknown until H1), action delay ∈ {0,1,2} steps, obs noise
  q N(0,0.005) dq N(0,0.05), per-episode joint offset bias U(±0.02) rad. Store nominal values once;
  re-derive from nominal each reset (no drift).
- Register `gymnasium.register("SO101Reach-v0", ...)`.
- Tests: `gymnasium.utils.env_checker.check_env`; obs shape/dtype; zero action holds pose; DR off is
  deterministic for a seed; DR on changes `model.actuator_gainprm` between resets but returns to nominal
  when disabled; a scripted IK-ish policy (Jacobian pseudo-inverse via `mj_jacSite`) reaches
  ≥ 80 % of 20 targets with DR off (proves targets are reachable and the action pathway works).

### T8 — PPO training + export
`sim/train_reach.py` (typer): `--total-steps 3_000_000 --n-envs 12 --seed 0 --dr/--no-dr --out runs/ppo_reach/<ts>`.
SB3 `PPO("MlpPolicy", SubprocVecEnv, net_arch=dict(pi=[256,256], vf=[256,256]), n_steps=256,
batch_size=4096, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.0, clip_range=0.2)`,
`VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10)`; `EvalCallback` every 100k steps on
50 DR episodes logging success rate; save best + `vecnormalize.pkl`.
`sim/export.py`: deterministic actor → `reach.onnx` (input `obs` float32[1,24], output `action` [1,5]
pre-clip mean), plus `policy_meta.json`: `{obs_mean, obs_var, clip_obs, action_scale, hz, joints,
obs_layout, units:"rad", sim_commit, dr, train_steps, eval_success}`.
- Tests: 2 048-step smoke train (`--n-envs 2`, DummyVecEnv) → export → onnxruntime output matches SB3
  `predict(deterministic=True)` (atol 1e-4) on 10 random obs.
- **Run acceptance (not in pytest):** full run; record in `docs/harness-sim2real.md`: wall time, final
  eval success with DR (target ≥ 90 %) and without DR. If < 90 %: try 6 M steps, then
  `ACTION_SCALE=0.03`; log every attempt.

### T9 — ONNX reach policy for the harness
`harness/policies/onnx_reach.py`: `OnnxReachPolicy(BasePolicy)` loading `reach.onnx` + meta.
Keeps `q_cmd`, `last_action`, and an `FK` for `ee`; `_act(obs)` → `build_obs` (imported from
`sim/envs/reach.py`) → normalize with meta → onnx → clip [−1,1] → integrate exactly like T7 → returns
`(1,6)` with gripper = `sim_open`. `reset()` re-inits `q_cmd` from the next obs.
- Tests: on SimBackend + `run_skill` with `EEReached(0.02)` + `TimeLimit(4)`: policy from T8 smoke
  model runs without error; obs fed to ONNX equals env obs for an identical rollout (train/deploy
  parity test — the most important test in this plan).

### T10 — Harness CLI
`harness/cli.py` (typer, entry `harness`):
`harness reach --backend sim|fake|real --policy PATH --target x,y,z | --random N --seed 0
 --dr/--no-dr --viewer --dry-run/--live --max-vel 0.5`.
Sim: runs N episodes via `run_sequence([home_skill, reach_skill])`, prints success rate + mean final
distance, writes `EpisodeRecord`s + trajectories under `runs/harness/<ts>/`. `home_skill` = scripted
`GlideTo(HOME)` policy (velocity-capped linear interpolation, reuse `VelocityCap`).
Real: requires `config/sim2real.yaml calibrated: true` **and** `--live`, else prints what it would
send (dry run). Haiku tests only `sim` and `fake`.
- Accept: `uv run harness reach --backend sim --policy runs/ppo_reach/<best>/reach.onnx --random 100 --dr`
  ≥ 85 % success; numbers recorded in `docs/harness-sim2real.md`.

### T11 — Actuation-gap toolkit (SAGE-lite)
`sim/gap/`:
- `motions.py`: generate motion CSVs (header = joint names, rows = rad at 50 Hz): per-joint chirp
  (0.1→2 Hz, ±0.3 rad around HOME), steps (±0.2 rad, 1 s hold), slow sine all joints. Write to
  `config/gap_motions/*.csv` (small, commit them).
- `replay.py`: `replay(motion_csv, backend, safety) -> DataFrame(t, q_cmd, q, dq)` via `run_skill`
  with a `ReplayPolicy`. Works with SimBackend now; LeRobotBackend later (H3).
- `compare.py`: per-joint RMSE, lag via cross-correlation (ms), overshoot, steady-state error; matplotlib
  overlay plot to `runs/gap/<ts>/*.png`.
- `fit.py`: `scipy.optimize.least_squares` per joint over `(kp_scale, damping, frictionloss,
  armature, delay_steps∈{0..3} grid)` minimizing sim-vs-“real” tracking error; writes
  `config/actuator_fit.yaml`; `sim/model.py` gains `apply_actuator_fit(model, yaml)`.
- Tests: synthetic “real” = sim with known params (kp×0.8, damping×1.5, delay 1) → fit recovers each
  within 15 % and the delay exactly; compare() on identical traces gives RMSE 0, lag 0.

### T12 — Docs
- `docs/harness-sim2real.md`: evidence page (what ran, numbers, plots paths, blockers).
- `CLAUDE.md`: add a short "Harness + sim" section: third control path, units rule, `config/sim2real.yaml`
  guard, commands (`uv sync --extra sim`, `uv run sim-train-reach`, `uv run harness reach …`).
- README progress tracker rows for harness / sim reach / gap toolkit.

## Hardware tickets (user, later)

- **H1 PincOpen payload:** weigh the gripper; hold poses (incl. fully extended) and log
  load/current/temperature for shoulder_lift / elbow / wrist_flex; stage torque joint-by-joint
  (brown-out lesson). Put measured mass into the MJCF `gripper` body (record in `SOURCE.md`) and
  narrow the DR payload range around it. If shoulder_lift > ~70 % load when extended, shrink the
  reach target workspace.
- **H2 Joint map calibration:** move the arm (torque off) to match the sim at q=0 and HOME; read
  LeRobot degrees; fill `sign/offset_deg` + gripper open/closed in `config/sim2real.yaml`; verify
  FK(sim) vs a ruler on 3 poses; set `calibrated: true`.
- **H3 Gap capture:** `replay` the T11 motions on the real arm (`--live`, `--max-vel 0.5`), run
  `compare` + `fit`, retrain T8 with the fitted actuator model (DR centered on it).
- **H4 Real reach:** dry run → live at `--max-vel 0.5` → full speed; targets marked on the table;
  measure error with a ruler; log via the harness.

## Phase D (after M1) — insertion + auto-reset loop + hierarchical skills

- Scene: tray + hole box on the right (measure the 3D-printed rig), chute modeled as "object respawns
  in tray with pose noise". Objects: the real random items approximated by primitives.
- Strategy 1 (NVIDIA) in MuJoCo: leader arm drives the sim arm; record LeRobot datasets with visual DR
  (lighting, textures, camera pose ±2 cm/±0.05 rad) using MuJoCo offscreen rendering of `wrist_cam` +
  an external camera at 640×480. Strategy 2: co-train with real teleop demos.
- Policies: ACT (fits 6 GB) per skill — `pick_from_tray`, `insert_through_hole`; `return_home`
  scripted. GR00T N1.6 is out of reach on 6 GB.
- Success sensor on the chute (break-beam/limit switch → microcontroller → USB serial) = sparse reward
  + `reset_ok`; real RL fine-tuning of one skill at a time (residual RL on the BC skill or HIL-SERL).
- Patterns to adopt at that point: grid-index goal features, classifier/end-condition strategies
  (gripper load history from `PincGripper.status()`), action cache for repeatable segments, redo logic,
  auto-save frames for detector retraining.

## Verification summary

- `uv run pytest` (all new tests CPU-only, < 2 min), `uv run ruff check .`
- `uv run sim-train-reach` → eval success ≥ 90 % with DR (recorded).
- `uv run harness reach --backend sim … --random 100 --dr` ≥ 85 % (recorded).
- Train/deploy obs parity test (T9) passes.
- Gap fit recovers synthetic params (T11).

---

## Appendix B — MiniLab 3 asset (noted, no work)
`assets/Arturia MiniLab 3.zip` → Isaac Sim USDC (`PXR-USDC`) with articulated keys (`Joint_Key_*`),
pads, encoders, PD drives (`stiffnessNmPerRad`) and `mass_kg`. Inspecting/converting needs `usd-core`
(`uv run --with usd-core`); MJCF conversion deferred to the MiniLab sim phase.
