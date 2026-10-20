"""Leader-arm stow: move the leader out of the follower's workspace during autonomous runs.

The leader sits beside the follower, which hit it at a near-base pose on its left (survey
2026-10-18). ``LeaderStow`` is a context manager:

1. On enter it glides the leader (all joints, velocity-capped, low torque limit) to the stow pose,
   a full pose taught by hand or, by default, the rest pose with the pan turned to its left.
2. During the run it holds torque (``hold_torque_during_run``).
3. On exit it glides back to the **rest pose** and releases torque.

    with LeaderStow.from_config():            # harness reach / survey / gap live runs
        ...
    uv run leader-stow --dry-run | --teach | --teach-rest | --stow | --rest

It reuses ``hw.follower`` primitives (Goal = Present before torque, velocity-capped glide). If the
leader can't be stowed, the caller must not start the follower.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import typer
import yaml

from robot_manipulator.hw.follower import JOINTS, bus_pos, console, glide, send_goal

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "config" / "leader_stow.yaml"
HZ = 50.0


def connect_leader(port: str, leader_id: str):
    """Leader bus only, read-only until torque is explicitly enabled."""
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    leader = SOLeader(SOLeaderTeleopConfig(port=port, id=leader_id))
    leader.bus.connect()
    if not leader.bus.is_calibrated:
        leader.bus.disconnect(disable_torque=False)
        raise RuntimeError(f"leader calibration mismatch for id {leader_id!r}")
    return leader


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def stow_pose(cfg: dict) -> np.ndarray:
    if cfg.get("stow_pose_deg") is not None:
        return np.asarray(cfg["stow_pose_deg"], dtype=float)
    pose = np.asarray(cfg["rest_pose_deg"], dtype=float).copy()
    pose[0] = float(cfg["default_stow_pan_deg"])
    return pose


BUS_RETRY = 3  # one dropped status packet aborted a run (servo 4, 2026-10-20): retry writes


def torque_on(leader, torque_limit: int) -> np.ndarray:
    """Goal = Present, torque limit, torque on. Returns the present pose.

    Writes are retried. If torque still can't be enabled on every joint, the joints that did
    switch on are released again before re-raising, so the leader is never left half-energised.
    """
    here = bus_pos(leader)
    send_goal(leader, here)  # enabling torque must not jump
    for motor in JOINTS:
        leader.bus.write("Torque_Limit", motor, int(torque_limit), num_retry=BUS_RETRY)
    try:
        leader.bus.enable_torque(num_retry=BUS_RETRY)
    except Exception:
        try:
            leader.bus.disable_torque(num_retry=BUS_RETRY)
        finally:
            raise
    return here


def move_to(leader, target: np.ndarray, torque_limit: int, speed: float, release: bool) -> None:
    here = torque_on(leader, torque_limit)
    try:
        glide(leader, here, np.asarray(target, dtype=float), speed, HZ)
    finally:
        if release:
            leader.bus.disable_torque()


class LeaderStow:
    def __init__(self, cfg: dict, connect=connect_leader) -> None:
        self.cfg = cfg
        self._connect = connect
        self.leader = None

    @classmethod
    def from_config(cls, path: Path = CONFIG) -> LeaderStow:
        return cls(load_config(path))

    def __enter__(self) -> LeaderStow:
        c = self.cfg
        self.leader = self._connect(c["port"], c["leader_id"])
        try:
            target = stow_pose(c)
            console.print(f"[cyan]stowing leader → {np.round(target, 1).tolist()}[/]")
            move_to(
                self.leader,
                target,
                c["torque_limit"],
                c["speed_deg_s"],
                release=not c.get("hold_torque_during_run", True),
            )
        except BaseException:
            self.leader.bus.disable_torque()
            self.leader.bus.disconnect(disable_torque=False)
            self.leader = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self.leader is None:
            return
        c = self.cfg
        try:
            console.print("[cyan]returning leader to rest[/]")
            move_to(
                self.leader,
                np.asarray(c["rest_pose_deg"], dtype=float),
                c["torque_limit"],
                c["speed_deg_s"],
                release=True,
            )
        finally:
            self.leader.bus.disable_torque()
            self.leader.bus.disconnect(disable_torque=False)
            self.leader = None


def _save_pose(key: str, pose: np.ndarray, path: Path = CONFIG) -> None:
    """Replace one ``key: ...`` line in the yaml, keeping the comments."""
    lines = Path(path).read_text().splitlines()
    value = "[" + ", ".join(f"{v:.1f}" for v in pose) + "]"
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            comment = line[line.index("#") :] if "#" in line else ""
            lines[i] = f"{key}: {value}" + (f"  {comment}" if comment else "")
            break
    else:
        lines.append(f"{key}: {value}")
    Path(path).write_text("\n".join(lines) + "\n")


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="print the present, stow and rest poses"),
    teach: bool = typer.Option(False, "--teach", help="save the present pose as the stow pose"),
    teach_rest: bool = typer.Option(False, "--teach-rest", help="save the present pose as rest"),
    stow: bool = typer.Option(
        False, "--stow", help="move to the stow pose (torque released after)"
    ),
    rest: bool = typer.Option(False, "--rest", help="move to the rest pose and release"),
) -> None:
    """Inspect, teach or move the leader (the harness stows/restores automatically)."""
    cfg = load_config()
    leader = connect_leader(cfg["port"], cfg["leader_id"])
    try:
        here = bus_pos(leader)
        typer.echo(f"present {np.round(here, 1).tolist()}")
        typer.echo(f"stow    {np.round(stow_pose(cfg), 1).tolist()}")
        typer.echo(f"rest    {np.round(cfg['rest_pose_deg'], 1).tolist()}")
        if teach:
            _save_pose("stow_pose_deg", here)
            typer.echo("saved present pose as stow_pose_deg")
        elif teach_rest:
            _save_pose("rest_pose_deg", here)
            typer.echo("saved present pose as rest_pose_deg")
        elif stow or rest:
            target = stow_pose(cfg) if stow else np.asarray(cfg["rest_pose_deg"], dtype=float)
            move_to(leader, target, cfg["torque_limit"], cfg["speed_deg_s"], release=True)
            time.sleep(0.2)
            typer.echo(f"now {np.round(bus_pos(leader), 1).tolist()}, torque off")
    finally:
        leader.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    app()
