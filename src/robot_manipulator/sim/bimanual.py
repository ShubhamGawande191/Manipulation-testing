"""Two SO-101 arms in one MuJoCo model, and the arm-to-arm transform from a shared board.

``config/bimanual.yaml`` places each arm (base position, yaw, gripper variant) in a world frame;
by convention the world is the **left arm's base**. Each arm is attached with a name prefix
(``left_shoulder_pan``, ``right_gripper``, ...; cameras ``left_wrist_cam`` ...), so per-arm code
uses ``prefixed(JOINTS, "left_")``.

    uv run bimanual check        # compile, names, contacts, clearance at HOME
    uv run bimanual transform runs/wristcal/joint_L.json runs/wristcal/joint_R.json [--write]

``transform`` takes the ``camcal-wrist solve-joint`` outputs of the two arms (run with
``SO101_ARM=left`` / ``right``) for sessions in which **both arms saw the same board pose**, and
gives the right base in the left base frame: R = R_L R_R^T, t = t_L - R t_R, with
``p_base = R_b p_board + t_b`` per arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import typer
import yaml

from robot_manipulator.sim.model import ASSET_DIR, JOINTS

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "config" / "bimanual.yaml"
ARM_MJCF = {"pinc": "so101_pinc.xml", "stock": "so101.xml"}
# HOME in CMD order (rad), gripper open-ish; matches sim/envs/reach.HOME_ARM for the arm joints
HOME = np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0])

app = typer.Typer(add_completion=False, help=__doc__)


def prefixed(names: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(prefix + n for n in names)


def load_config(path: Path = CONFIG) -> dict:
    doc = yaml.safe_load(path.read_text()) or {}
    arms = doc.get("arms") or {}
    if set(arms) != {"left", "right"}:
        raise ValueError(f"{path}: expected arms 'left' and 'right', got {sorted(arms)}")
    return doc


def build(cfg: dict) -> mujoco.MjModel:
    """Compile the two-arm model; options (timestep, solver) come from the arm MJCF."""
    world = mujoco.MjSpec()
    first = mujoco.MjSpec.from_file(
        str(ASSET_DIR / ARM_MJCF[cfg["arms"]["left"].get("gripper", "pinc")])
    )
    for field in ("timestep", "iterations", "ls_iterations", "impratio", "integrator", "cone"):
        setattr(world.option, field, getattr(first.option, field))
    world.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, -float(cfg.get("table_below_base_m", 0.0275))],
    )
    for name in ("left", "right"):
        a = cfg["arms"][name]
        child = mujoco.MjSpec.from_file(str(ASSET_DIR / ARM_MJCF[a.get("gripper", "pinc")]))
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            q, np.array([0.0, 0.0, 1.0]), np.radians(float(a.get("yaw_deg", 0)))
        )
        frame = world.worldbody.add_frame(pos=[float(v) for v in a["pos"]], quat=q.tolist())
        world.attach(child, frame=frame, prefix=f"{name}_")
    return world.compile()


def set_both(model: mujoco.MjModel, data: mujoco.MjData, q_left, q_right) -> None:
    for prefix, q in (("left_", q_left), ("right_", q_right)):
        for j, v in zip(prefixed(JOINTS, prefix), q, strict=True):
            data.qpos[model.joint(j).qposadr[0]] = v
    mujoco.mj_forward(model, data)


def inter_arm_clearance(model: mujoco.MjModel, data: mujoco.MjData, margin: float = 0.5) -> float:
    """Smallest distance (m) between left-arm and right-arm geoms (negative = overlap)."""
    body_arm = {}
    for b in range(model.nbody):
        n = model.body(b).name
        body_arm[b] = (
            "left" if n.startswith("left_") else "right" if n.startswith("right_") else None
        )
    best, fromto = np.inf, np.zeros(6)
    for g1 in range(model.ngeom):
        if body_arm[model.geom_bodyid[g1]] != "left":
            continue
        for g2 in range(model.ngeom):
            if body_arm[model.geom_bodyid[g2]] != "right":
                continue
            best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, margin, fromto))
    return float(best)


def arm_to_arm(left: dict, right: dict) -> tuple[np.ndarray, np.ndarray]:
    """Right base pose in the left base frame from the same board seen by both arms."""
    R_l, t_l = np.array(left["R"]), np.array(left["t_m"])
    R_r, t_r = np.array(right["R"]), np.array(right["t_m"])
    R = R_l @ R_r.T
    return R, t_l - R @ t_r


@app.command()
def check(config: Path = typer.Option(CONFIG)) -> None:
    """Compile the two-arm twin; report names, contacts and inter-arm clearance at HOME."""
    cfg = load_config(config)
    m = build(cfg)
    d = mujoco.MjData(m)
    typer.echo(
        f"nu {m.nu} (expect 12), nq {m.nq}, cameras {[m.camera(i).name for i in range(m.ncam)]}"
    )
    set_both(m, d, HOME, HOME)
    clear = inter_arm_clearance(m, d)
    typer.echo(f"HOME/HOME: {d.ncon} contacts, inter-arm clearance {1000 * clear:.0f} mm")
    for name in ("left", "right"):
        typer.echo(
            f"{name} gripperframe at HOME: {np.round(d.site(f'{name}_gripperframe').xpos, 3)}"
        )
    if clear < 0.02:
        typer.echo(
            "WARNING: arms within 20 mm at HOME - check placement before any shared-space task"
        )


@app.command()
def transform(
    left_json: Path = typer.Argument(..., help="left arm's camcal-wrist solve-joint json"),
    right_json: Path = typer.Argument(..., help="right arm's solve-joint json"),
    left_session: int = typer.Option(-1, help="index of the shared-board session in the left json"),
    right_session: int = typer.Option(
        -1, help="index of the shared-board session in the right json"
    ),
    write: bool = typer.Option(False, help="write the right arm pose into config/bimanual.yaml"),
    config: Path = typer.Option(CONFIG),
) -> None:
    """Right base in the left base frame from both arms' views of one board pose."""
    L = json.loads(left_json.read_text())["board_in_base"][left_session]
    R_ = json.loads(right_json.read_text())["board_in_base"][right_session]
    R, t = arm_to_arm(L, R_)
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    typer.echo(
        f"right base in left frame: t = {np.round(1000 * t, 1)} mm, yaw {yaw:.2f} deg, "
        f"tilt {tilt:.2f} deg (both bases on one table -> tilt ~ 0)"
    )
    if tilt > 2.0:
        typer.echo(
            "WARNING: tilt > 2 deg - the two sessions probably did not see the same board pose"
        )
    if write:
        doc = load_config(config)
        doc["arms"]["right"]["pos"] = [round(float(v), 4) for v in t]
        doc["arms"]["right"]["yaw_deg"] = round(float(yaw), 2)
        doc["arms"]["right"]["source"] = f"bimanual transform {left_json.name} {right_json.name}"
        config.write_text(yaml.safe_dump(doc, sort_keys=False))
        typer.echo(f"-> {config}")


if __name__ == "__main__":
    app()
