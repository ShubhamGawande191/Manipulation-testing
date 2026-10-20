# Upstream bugs found while building this lab

Bugs in third-party frameworks we hit, each checked against the **latest upstream** before
being called a bug, with a minimal reproduction, the workaround we use, and a draft issue. Ready to
file ones are on the lab board (`.colony/tasks.yaml`, `owner: you`). Add new ones here as they
come up: library + version, repro, workaround, candidate fix, upstream status.

| # | Library | Bug | Upstream status (2026-10-20) | Ready to file |
|---|---|---|---|---|
| 1 | cuRobo | `RobotBuilder` CUDA device assert when mimic followers' leader joint has no geometry | **present on main** (`78fd485`), not reported | ✅ issue (+ PR possible) |
| 2 | mjcf-urdf-simple-converter | unlimited MJCF hinges exported as `revolute` with limits `[0, 0]` | **present on main**, not reported | ✅ issue + small PR |
| 3 | cuRobo | config saved by `RobotBuilder.save` can't be loaded (`load_collision_spheres` twice) | fixed on main after v0.8.0 | — (ask for a release?) |
| 4 | cuRobo | `lock_joints` with a joint not in the tree: bare `KeyError` | fixed on main (clear `ValueError`) | — |
| 5 | cuRobo | IK with `self_collision_check=False` and no scene: seed stage has no costs (`torch.cat` of empty list) | seen with our SO-101 config only; **not reproduced with Franka** | needs a minimal repro |

Repro script for the cuRobo items: `integrations/curobo/upstream_repros.py` (uses only cuRobo's
bundled Franka). Run it against the installed release and against main:

```bash
~/tools/curobo/.venv/bin/python integrations/curobo/upstream_repros.py
git clone --depth 1 https://github.com/NVlabs/curobo ~/tools/curobo-main   # pure Python + Warp
PYTHONPATH=~/tools/curobo-main ~/tools/curobo/.venv/bin/python integrations/curobo/upstream_repros.py
```

---

## 1. cuRobo: `RobotBuilder` crashes when a mimic leader joint has no geometry

**Where it bites:** four-bar / linkage grippers (PincOpen, many parallel grippers). The fingers
have meshes and mimic the motor joint, but the motor's own link (coupler) has no geometry and is
off the path to the tool frame, so cuRobo never adds it to the kinematic tree.

**Repro** (main `78fd485` and v0.8.0): case `builder_mimic_leader_without_geometry`. It takes the
bundled Franka URDF, adds a mesh-less `coupler_joint` on `panda_hand`, and makes both finger joints
mimic it. `RobotBuilder(...).fit_collision_spheres(); .compute_collision_matrix()` then fails with
`CUDA error: device-side assert triggered`. The same Franka with a normal mimic
(`finger_joint2` mimics `finger_joint1`) works.

**Root cause:** a mimic follower body carries its leader's joint name, so `_add_body_to_tree`
puts `coupler_joint` into `self.joint_names` (8 names). Joint limits come from
`_get_joint_position_velocity_limits()`, which only iterates `self._active_joints` (bodies whose
own joint is not a mimic), and the leader's body is not in the tree, so there are 7 rows.
`CSpaceParams.load_from_joint_limits(upper[7], lower[7], names[8])` builds mismatched tensors and
`cspace.inplace_reindex(self.joint_names)` indexes row 7 on the GPU → device assert.

**Suggested fix:** in `KinematicsLoader.initialize_tensors`, add the child link of every mimic
leader joint referenced by a link in the tree to `other_links` before building the chain, so the
leader is an active joint with its own limits. At minimum, check
`len(joint_names) == limits.shape[-1]` before creating the cspace and raise a clear error naming
the missing joint.

**Workaround here:** export the URDF without `<mimic>` (`uv run urdf-export --no-mimic`) and lock
the follower joints at the closed pose (`integrations/curobo/bench_curobo.py`).

**Draft issue title:** "RobotBuilder: CUDA device assert when mimic joints follow a leader joint
whose link has no geometry (joint_names / joint-limit length mismatch)"

## 2. mjcf-urdf-simple-converter: unlimited hinges get limits `[0, 0]`

**Repro** (0.7.3 and main): any MJCF hinge without `range`/`limited` (MuJoCo reports
`jnt_limited = 0`, `jnt_range = [0, 0]`):

```python
import mujoco, tempfile, pathlib
from mjcf_urdf_simple_converter import convert
xml = """<mujoco><worldbody><body name="a"><joint name="free_spin" type="hinge"/>
<geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>"""
d = pathlib.Path(tempfile.mkdtemp()); (d / "m.xml").write_text(xml)
convert(str(d / "m.xml"), str(d / "m.urdf"))
print((d / "m.urdf").read_text())   # <joint type="revolute" name="free_spin"> ... <limit lower="0.0" upper="0.0" .../>
```

URDF tools then treat the joint as frozen, or reject it (cuRobo: "lower limits must be less than
upper limits").

**Suggested fix (small PR):** in `convert`, when `model.jnt_limited[jntid] == 0`, write
`type="continuous"` and no `<limit>` (or `<limit effort velocity>` without lower/upper), and keep
`revolute` + limits otherwise. Add a test with an unlimited hinge.

**Workaround here:** `sim/urdf_export.py` rewrites the limits of unlimited joints (coupled range
for the PincOpen linkage, else ±π).

**Draft issue title:** "Unlimited hinge joints are exported as revolute with limits [0, 0]"

## 3–4. cuRobo, fixed on main

- **Saved config not loadable** (v0.8.0): `RobotBuilder.save` writes `load_collision_spheres` and
  `num_envs`, and `RobotCfg.create` passes them again → `TypeError`. Fixed on main. Possible ask:
  a patch release, since v0.8.0 is the documented install.
- **`lock_joints` with an unknown joint** (v0.8.0): `KeyError`. Main raises a clear `ValueError`.

## 5. cuRobo: IK without self-collision (unconfirmed)

With our generated SO-101 config, `self_collision_check=False` and no scene, `solve_pose` fails in
the MPPI seed stage (`torch.cat(): expected a non-empty list of Tensors`). With the bundled Franka
it works, so something in our config matters. Next step: bisect the config (spheres, cspace,
lock_joints) until it reproduces on a bundled robot, then file.
