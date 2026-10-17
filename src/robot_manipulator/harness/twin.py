"""Live digital twin: mirror the real follower in a MuJoCo viewer through the sim2real joint map.

Read-only on the robot: torque is never enabled, so you move the arm by hand. (It cannot run
alongside ``uv run teleop``: only one process can own the serial port.)
Use it to validate ``config/sim2real.yaml`` (hardware ticket H2): move one joint at a time
and check the twin turns the same way. A joint that mirrors is a sign error; a constant
angle difference is an offset error.

    uv run twin                      # viewer window + per-joint readout
    uv run twin --flip wrist_flex    # try a sign hypothesis without editing the yaml
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import typer
from rich.console import Console

from robot_manipulator.harness.units import JointMap

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MAP = REPO / "config" / "sim2real.yaml"
LEROBOT_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ABBREV = ("pan", "lift", "elb", "wfl", "wrl", "grip")

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


def flipped(jm: JointMap, joints: list[str]) -> JointMap:
    """Copy of ``jm`` with the sign of the named body joints negated (for trying hypotheses)."""
    body = dict(jm.body)
    for j in joints:
        body[j] = dataclasses.replace(body[j], sign=-body[j].sign)
    return dataclasses.replace(jm, body=body)


@app.command()
def main(
    port: str = typer.Option("/dev/so101_follower"),
    robot_id: str = typer.Option("my_so101_follower"),
    joint_map: Path = typer.Option(DEFAULT_MAP),
    flip: list[str] = typer.Option([], help="negate the sign of this joint (repeatable)"),
    scene: str = typer.Option("scene_pinc.xml", help="MJCF scene in sim/assets/so101"),
    hz: float = typer.Option(30.0),
) -> None:
    import mujoco
    import mujoco.viewer
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config
    from robot_manipulator.sim.model import (
        JOINTS,
        apply_joint_couplings,
        joint_qpos_adr,
        load_model,
    )

    jm = JointMap.from_yaml(joint_map)
    if flip:
        jm = flipped(jm, flip)
    console.print("signs:", {j: c.sign for j, c in jm.body.items()})

    model = load_model(scene)
    data = mujoco.MjData(model)
    adr = joint_qpos_adr(model, JOINTS)

    robot = SOFollower(build_follower_config(port=port, robot_id=robot_id, cameras=False))
    robot.bus.connect()
    try:
        if any(robot.bus.sync_read("Torque_Enable", normalize=False).values()):
            raise RuntimeError("follower torque is on — run `uv run teleop --park-only` first")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            last_print = 0.0
            while viewer.is_running():
                pos = robot.bus.sync_read("Present_Position")
                vals = {f"{j}.pos": float(pos[j]) for j in LEROBOT_JOINTS}
                q = jm.lerobot_to_sim(vals)
                data.qpos[adr] = q
                apply_joint_couplings(model, data)  # PincOpen fingers follow the servo
                mujoco.mj_forward(model, data)
                viewer.sync()
                if time.monotonic() - last_print > 0.5:
                    last_print = time.monotonic()
                    # one overwritten line, short labels so the gripper fits an 80-column terminal
                    deg = " ".join(f"{a}{pos[j]:+6.1f}" for a, j in zip(ABBREV, LEROBOT_JOINTS))
                    print(f"\r{deg}  ", end="", flush=True)
                time.sleep(1.0 / hz)
    finally:
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    app()
