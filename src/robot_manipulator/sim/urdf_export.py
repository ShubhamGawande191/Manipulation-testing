"""URDF of the twin, generated from the MJCF (for cuRobo, ROS, Pinocchio, ...).

    uv run urdf-export                        # -> runs/urdf/so101_pinc/so101_pinc.urdf + meshes

One source of truth: the URDF is converted from ``sim/assets/so101/so101_pinc.xml`` (PincOpen,
flipped palm, camera mount) with mjcf-urdf-simple-converter (MIT, pypi), which loads the model in
MuJoCo and writes a revolute/fixed joint per body plus meshes (OBJ) taken from MuJoCo's own copies.
On top of the converter's output this adds:

- a ``fingertip`` link at the closed-fingertip point the IK and calibration use
  (``calib.extrinsics.TIP_OFFSET_M`` past the ``gripperframe`` site),
- ``<mimic>`` tags for the PincOpen four-bar (MJCF equality constraints), unless ``--no-mimic``,
- limits for the coupled finger joints (unlimited in the MJCF; the converter would write [0, 0]),
- ``<collision>`` elements reusing the visual meshes (tools that fit collision spheres read them).

Generated into ``runs/`` (gitignored): rebuild it whenever the MJCF changes.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import typer

from robot_manipulator.sim.model import ASSET_DIR

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs" / "urdf" / "so101_pinc"
app = typer.Typer(add_completion=False, help=__doc__)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def export(
    mjcf: str = "so101_pinc.xml", out: Path = OUT, name: str = "so101_pinc", mimic: bool = True
) -> Path:
    from mjcf_urdf_simple_converter import convert

    from robot_manipulator.calib.extrinsics import TipFK
    from robot_manipulator.sim.model import load_model

    out.mkdir(parents=True, exist_ok=True)
    urdf = out / f"{name}.urdf"
    cwd = os.getcwd()
    try:
        os.chdir(out)  # the converter writes its meshes relative to the working directory
        convert(str(ASSET_DIR / mjcf), str(urdf))
    finally:
        os.chdir(cwd)

    tree = ET.parse(urdf)
    root = tree.getroot()
    root.set("name", name)
    links = {link.get("name"): link for link in root.findall("link")}
    for link in links.values():  # collision = visual mesh
        for vis in link.findall("visual"):
            col = ET.SubElement(link, "collision", {"name": vis.get("name", "")})
            for child in vis:
                if child.tag in ("origin", "geometry"):
                    col.append(child)

    m = load_model(mjcf)
    joints = {j.get("name"): j for j in root.findall("joint")}
    coupled: dict[str, tuple[str, float, float]] = {}
    for i in range(m.neq):  # four-bar: finger joint = c0 + c1 * gripper
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        follower = m.joint(m.eq_obj1id[i]).name
        leader = m.joint(m.eq_obj2id[i]).name
        c0, c1 = float(m.eq_data[i][0]), float(m.eq_data[i][1])
        coupled[follower] = (leader, c0, c1)
        if mimic and follower in joints:
            ET.SubElement(
                joints[follower],
                "mimic",
                {"joint": leader, "multiplier": f"{c1:.9g}", "offset": f"{c0:.9g}"},
            )
    # unlimited MJCF joints come out of the converter as [0, 0]: give them the range their
    # coupling implies (finger = c0 + c1 * gripper over the gripper's range), else +-pi
    for j in range(m.njnt):
        name = m.joint(j).name
        if m.jnt_limited[j] or name not in joints:
            continue
        if name in coupled:
            leader, c0, c1 = coupled[name]
            ends = c0 + c1 * m.jnt_range[m.joint(leader).id]
            lo, hi = float(ends.min()), float(ends.max())
        else:
            lo, hi = -np.pi, np.pi
        lim = joints[name].find("limit")
        if lim is not None:
            lim.set("lower", f"{lo:.9g}")
            lim.set("upper", f"{hi:.9g}")

    fk = TipFK(m)
    body = m.body(fk.body).name
    ET.SubElement(root, "link", {"name": "fingertip"})
    j = ET.SubElement(root, "joint", {"name": "fingertip_fixed", "type": "fixed"})
    ET.SubElement(j, "parent", {"link": body})
    ET.SubElement(j, "child", {"link": "fingertip"})
    ET.SubElement(j, "origin", {"xyz": _fmt(fk.tip_local_nominal), "rpy": "0 0 0"})
    ET.indent(tree, "  ")
    tree.write(urdf, xml_declaration=True, encoding="utf-8")
    return urdf


def check_against_mujoco(urdf: Path, n: int = 50, seed: int = 0) -> float:
    """Max fingertip distance (m) between the URDF (yourdfpy) and MuJoCo FK over random poses."""
    import yourdfpy

    from robot_manipulator.calib.extrinsics import TipFK
    from robot_manipulator.sim.model import JOINTS, joint_ranges, load_model

    m = load_model("so101_pinc.xml")
    fk = TipFK(m)
    robot = yourdfpy.URDF.load(str(urdf), load_meshes=False)
    lo, hi = joint_ranges(m, JOINTS)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        q = np.r_[rng.uniform(lo[:5], hi[:5]), 0.0]
        cfg = dict(zip(JOINTS, q))
        robot.update_cfg({k: v for k, v in cfg.items() if k in robot.actuated_joint_names})
        tip = robot.get_transform("fingertip")[:3, 3]
        worst = max(worst, float(np.linalg.norm(tip - fk.tip(q))))
    return worst


@app.command()
def main(
    mjcf: str = "so101_pinc.xml",
    out: Path = OUT,
    check: bool = True,
    mimic: bool = typer.Option(
        True,
        help="write <mimic> for the finger coupling; --no-mimic for tools that choke on it "
        "(cuRobo 0.8.0's RobotBuilder): lock the finger joints at the gripper's pose instead",
    ),
) -> None:
    urdf = export(mjcf, out, name="so101_pinc" if mimic else "so101_pinc_nomimic", mimic=mimic)
    n_mesh = len([f for f in out.rglob("*") if f.suffix.lower() in (".obj", ".stl")])
    typer.echo(f"-> {urdf} ({n_mesh} meshes)")
    if check:
        try:
            err = check_against_mujoco(urdf)
            typer.echo(f"fingertip URDF vs MuJoCo FK, 50 random poses: max {1e6 * err:.2f} um")
        except ImportError:
            typer.echo("yourdfpy not installed here: FK check skipped (cuRobo's env has it)")


if __name__ == "__main__":
    app()
