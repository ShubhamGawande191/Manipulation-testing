# SO101 MJCF — vendored source

- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/3abca2e302625cec9429c49806121ea2855cef81/robotstudio_so101
- Commit: `3abca2e302625cec9429c49806121ea2855cef81`
- License: Apache-2.0 (`LICENSE` in this directory, unmodified)
- Fetched from `raw.githubusercontent.com` at the commit above: `so101.xml`, `scene.xml`, `LICENSE`, and every `assets/*.stl` (19 meshes). `scene_box.xml` and the upstream README/PNG were not vendored.
- Local edits: **none yet.**
  - TODO-H1: add the measured PincOpen gripper mass to body `gripper` and record it here.
  - TODO-H2: joint sign/offset vs. the real arm lives in `config/sim2real.yaml`, not in the XML.

## PincOpen variant (`so101_pinc.xml`, `scene_pinc.xml`) — added 2026-10-17

Generated from `so101.xml`: the stock fixed jaw, gripper servo, camera mount and moving jaw under the
`gripper` (wrist_roll) body are replaced by a PincOpen parallel gripper. Joint/actuator names and order are
unchanged (`gripper` is the PincOpen servo/coupler joint).

- Meshes `gripper_{hand_w_motor,left_arm,right_arm,left_finger,right_finger}_simple.stl` and the kinematics
  (pivot offsets, ±0.5 mimic ratios) come from https://github.com/CNURobotics/pinc_open_driver (Apache-2.0,
  commit 09fbd74). The gripper design is Pollen Robotics' PincOpen (CC BY-SA 4.0,
  https://pollen-robotics.github.io/PincOpen/). The four-bar coupling uses equality constraints, as in the
  community MuJoCo PR pollen-robotics/PincOpen#6 (whose placeholder masses/inertias were NOT used).
- Gripper joint range [-1.833, 0] rad: 0 = closed, 105° servo travel = the measured LeRobot calibration
  (steps 3200 closed → 2005 open).
- Masses: total 0.212 kg past the wrist (user estimate 200-300 g incl. camera): interface 0.015,
  palm+motor 0.13, arms 2×0.008, fingers 2×0.010, camera bracket + board camera 0.03; inertias from boxes.
- Interface ~5 mm (palm nearly flush with the wrist) and the camera L-bracket (foot on the palm face,
  strut toward the wrist, horizontal plate, lens looking along the fingers, tilted ~30°) are estimated
  from user photos 2026-10-17. **TODO-measure** both.
- `scene_pinc.xml`: the floor is the table, 2.75 cm below the robot base plate (user estimate).
- Known limitation: the Menagerie arm's collision geoms are coarse boxes; at the real folded rest pose
  (shoulder_lift ≈ -108°, past the MJCF ±100° limit) they already interpenetrate, so self-collision near
  the fold is not trustworthy. An earlier note that the camera box "reproduced" the real wrist_flex stall
  was an artifact of that (it hit the oversized shoulder box) and was withdrawn.
