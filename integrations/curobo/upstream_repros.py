"""Minimal reproductions of cuRobo bugs found while benchmarking (only cuRobo's bundled Franka).

    ~/tools/curobo/.venv/bin/python integrations/curobo/upstream_repros.py            # installed
    PYTHONPATH=~/tools/curobo-main ~/tools/curobo/.venv/bin/python \\
        integrations/curobo/upstream_repros.py                                       # upstream main

Each case runs in its own process (a CUDA device assert poisons the whole process) and prints
PASS / FAIL with the exception. Findings and draft issues: docs/UPSTREAM-BUGS.md.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap

CASES = {
    "builder_mimic": """
        # RobotBuilder on the Franka URDF with the standard finger mimic
        # (panda_finger_joint2 mimics panda_finger_joint1, as in franka_description)
        import re, tempfile, pathlib
        from curobo.content import get_assets_path
        from curobo.robot_builder import RobotBuilder
        d = pathlib.Path(get_assets_path()) / "robot/franka_description"
        txt = (d / "franka_panda.urdf").read_text()
        txt = re.sub(r'(<joint name="panda_finger_joint2"[^>]*>)',
                     r'\\1<mimic joint="panda_finger_joint1" multiplier="1" offset="0"/>', txt)
        urdf = d / "_repro_franka_mimic.urdf"
        urdf.write_text(txt)
        try:
            b = RobotBuilder(urdf_path=str(urdf), asset_path=str(d), tool_frames=["panda_hand"])
            b.fit_collision_spheres(sphere_density=0.3)
            b.compute_collision_matrix()
        finally:
            urdf.unlink()
    """,
    "builder_save_roundtrip": """
        # a config saved by RobotBuilder.save cannot be loaded by InverseKinematicsCfg.create
        import pathlib, tempfile
        from curobo.content import get_assets_path
        from curobo.robot_builder import RobotBuilder
        from curobo.inverse_kinematics import InverseKinematicsCfg
        d = pathlib.Path(get_assets_path()) / "robot/franka_description"
        b = RobotBuilder(urdf_path=str(d / "franka_panda.urdf"), asset_path=str(d),
                         tool_frames=["panda_hand"])
        b.fit_collision_spheres(sphere_density=0.3)
        b.compute_collision_matrix()
        out = pathlib.Path(tempfile.mkdtemp()) / "franka_built.yml"
        b.save(b.build(), str(out))
        InverseKinematicsCfg.create(robot=str(out), num_seeds=4)
    """,
    "ik_without_self_collision": """
        # IK with self_collision_check=False and no scene
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, Pose
        ik = InverseKinematics(InverseKinematicsCfg.create(
            robot="franka.yml", num_seeds=8, self_collision_check=False))
        goal = Pose(position=torch.tensor([[0.4, 0.0, 0.4]], device="cuda"),
                    quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device="cuda"))
        ik.solve_pose(GoalToolPose.from_poses({ik.tool_frames[0]: goal}, num_goalset=1))
    """,
}

# mimic joints whose *leader* joint is not in the tree (it drives a link without geometry, off
# the path to the tool frame), as in a four-bar gripper: the followers have meshes, the leader
# (motor/coupler) does not. Franka + a mesh-less "coupler" joint on the hand that both fingers
# mimic. The plain-Franka mimic case (builder_mimic) works.
CASES["builder_mimic_leader_without_geometry"] = """
    import re, pathlib
    from curobo.content import get_assets_path
    from curobo.robot_builder import RobotBuilder
    d = pathlib.Path(get_assets_path()) / "robot/franka_description"
    txt = (d / "franka_panda.urdf").read_text()
    coupler = (
        '<link name="coupler"/>'
        '<joint name="coupler_joint" type="revolute"><parent link="panda_hand"/>'
        '<child link="coupler"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 0 1"/>'
        '<limit lower="0" upper="0.04" effort="10" velocity="1"/></joint>'
    )
    txt = txt.replace("</robot>", coupler + "</robot>")
    for j in ("panda_finger_joint1", "panda_finger_joint2"):
        txt = re.sub(rf'(<joint name="{j}"[^>]*>)',
                     r'\\1<mimic joint="coupler_joint" multiplier="1" offset="0"/>', txt)
    urdf = d / "_repro_franka_coupler.urdf"
    urdf.write_text(txt)
    try:
        b = RobotBuilder(urdf_path=str(urdf), asset_path=str(d), tool_frames=["panda_hand"])
        b.fit_collision_spheres(sphere_density=0.3)
        b.compute_collision_matrix()
    finally:
        urdf.unlink()
"""

# lock_joints naming a joint that is not in the kinematic tree -> bare KeyError (here: a name
# that doesn't exist; in our robot, a real joint off the path to the tool frame)
CASES["lock_joint_not_in_tree"] = """
    from curobo.content import get_robot_configs_path
    from curobo.inverse_kinematics import InverseKinematicsCfg
    import yaml, pathlib
    cfg = yaml.safe_load((pathlib.Path(get_robot_configs_path()) / "franka.yml").read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    kin["lock_joints"] = dict(kin.get("lock_joints") or {}, not_a_joint=0.0)
    InverseKinematicsCfg.create(robot=cfg, num_seeds=4)
"""


def run_case(name: str, code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
    p = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=900)
    if p.returncode == 0:
        return True, ""
    lines = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
    return False, lines[-1] if lines else f"exit {p.returncode}"


def main() -> None:
    import curobo

    print(f"cuRobo {curobo.__version__} from {curobo.__file__}")
    for name, code in CASES.items():
        ok, err = run_case(name, code)
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {err}"))


if __name__ == "__main__":
    main()
