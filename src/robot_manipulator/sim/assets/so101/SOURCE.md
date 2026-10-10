# SO101 MJCF — vendored source

- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/3abca2e302625cec9429c49806121ea2855cef81/robotstudio_so101
- Commit: `3abca2e302625cec9429c49806121ea2855cef81`
- License: Apache-2.0 (`LICENSE` in this directory, unmodified)
- Fetched from `raw.githubusercontent.com` at the commit above: `so101.xml`, `scene.xml`, `LICENSE`, and every `assets/*.stl` (19 meshes). `scene_box.xml` and the upstream README/PNG were not vendored.
- Local edits: **none yet.**
  - TODO-H1: add the measured PincOpen gripper mass to body `gripper` and record it here.
  - TODO-H2: joint sign/offset vs. the real arm lives in `config/sim2real.yaml`, not in the XML.
