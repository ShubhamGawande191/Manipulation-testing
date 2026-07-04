# Active Planner & Todo List

## Goals
- [x] **AprilTag World Anchor**: Detect AprilTag on the table to establish origin (0,0,0).
- [x] **Camera Calibration**: Calculate camera intrinsics/extrinsics using ChArUco board later.
- [x] **Coordinate Projection Script**: Convert 2D pixel to 3D world relative to anchor.
- [x] **Fix servo convergence issue**: Updated proportional gains statically derived from optics focal length (fx/fy) instead of hardcoding.
- [x] **Add pre-grasp verification**: Implemented pre-grasp geometric validation utilizing millimeter error check.
- [x] **Retry pick & place black cube**: Created `pick_black_cube.py` to leverage the new intrinsics logic.
- [ ] **Data Logging**: Save object coordinates to the local Skill Registry to prevent recalculations.

## MiniLab 3 project (see minilab/PLAN.md + docs/EXECUTION-ROADMAP.md)
- [x] Phase 0 software harness (core/io/exec/eval + tests) — 2026-07-04
- [x] `uv run bench-check` hardware verifier + Claude skill — 2026-07-04
- [ ] Reconnect follower arm on a **direct USB port** (board not enumerating behind dock)
- [ ] Phase 0 exit run: scripted key-press primitive + scorer + parquet episode record
- [ ] LeRobot v3 API migration pass (record/replay against the new dataset format)
- [ ] MuJoCo sim (PLAN Phase 5): MJCF MiniLab 3 (keys=hinges+springs, faders=sliders),
      Menagerie `trs_so_arm100` for the arm; earlier smoke-value: collision-free primitive
      testing while the bench is occupied. Do after Phase 1, not before.
- [ ] Evaluate pixi (pixi.prefix.dev) as env manager — wins if we adopt RoboStack ROS2
      (conda-packaged ROS2 + Python in ONE env, no more `unset VIRTUAL_ENV` dance);
      keep uv until that decision. pixi reuses uv for PyPI deps, so migration is cheap.
- [ ] GitHub Projects board + prioritized issues (blocked on `gh auth login`; then run
      `scripts/github_backlog.sh`)

## Long-term
- [ ] Migrate to OOP SkillBase architecture
- [ ] Explore IL/RL policies with "synthetic depth"
