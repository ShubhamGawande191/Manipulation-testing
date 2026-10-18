"""Leader-arm stow: rotate the leader out of the follower's workspace during autonomous runs.

The leader sits beside the follower. The follower hit it at a near-base pose on its left
(survey 2026-10-18). ``LeaderStow`` is a context manager:

1. On enter, it glides the leader's pan to ``stow_pan_deg`` (everything else stays folded) under
   a low torque limit, then releases torque. Pan carries no gravity load, so the leader stays put
   and remains back-drivable.
2. On exit, it glides back to the pose it found and releases.

    with LeaderStow.from_config():            # in harness / survey / gap live runs
        ...
    uv run leader-stow [--dry-run|--stow|--restore-pan DEG]

It reuses the safe primitives from ``hw.follower`` (Goal = Present before torque, velocity-capped
glide). If the leader can't be stowed, the caller must not start the follower.
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


def move_pan(leader, pan_deg: float, torque_limit: int, speed: float, hz: float = HZ) -> None:
    """Glide the leader so pan = ``pan_deg`` (other joints held where they are), then release."""
    bus = leader.bus
    here = bus_pos(leader)
    send_goal(leader, here)  # Goal = Present: enabling torque must not jump
    for motor in JOINTS:
        bus.write("Torque_Limit", motor, int(torque_limit))
    bus.enable_torque()
    try:
        target = here.copy()
        target[0] = float(pan_deg)
        glide(leader, here, target, speed, hz)
    finally:
        bus.disable_torque()


class LeaderStow:
    def __init__(
        self,
        port: str,
        leader_id: str,
        stow_pan_deg: float,
        torque_limit: int,
        speed_deg_s: float,
        connect=connect_leader,
    ) -> None:
        self.port, self.leader_id = port, leader_id
        self.stow_pan_deg, self.torque_limit, self.speed = stow_pan_deg, torque_limit, speed_deg_s
        self._connect = connect
        self.leader = None
        self.original_pan: float | None = None

    @classmethod
    def from_config(cls, path: Path = CONFIG, **overrides) -> LeaderStow:
        cfg = {**yaml.safe_load(Path(path).read_text()), **overrides}
        return cls(
            cfg["port"],
            cfg["leader_id"],
            float(cfg["stow_pan_deg"]),
            int(cfg["torque_limit"]),
            float(cfg["speed_deg_s"]),
        )

    def __enter__(self) -> LeaderStow:
        self.leader = self._connect(self.port, self.leader_id)
        try:
            self.original_pan = float(bus_pos(self.leader)[0])
            console.print(
                f"[cyan]stowing leader: pan {self.original_pan:+.1f} → "
                f"{self.stow_pan_deg:+.1f} deg[/]"
            )
            move_pan(self.leader, self.stow_pan_deg, self.torque_limit, self.speed)
        except BaseException:
            self.leader.bus.disconnect(disable_torque=False)
            self.leader = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self.leader is None:
            return
        try:
            if self.original_pan is not None:
                console.print(f"[cyan]restoring leader pan → {self.original_pan:+.1f} deg[/]")
                move_pan(self.leader, self.original_pan, self.torque_limit, self.speed)
        finally:
            self.leader.bus.disconnect(disable_torque=False)
            self.leader = None


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="only print current and stow pan"),
    stow: bool = typer.Option(False, "--stow", help="rotate to the stow pan and leave it there"),
    restore_pan: float | None = typer.Option(None, help="rotate pan to this angle and release"),
) -> None:
    """Inspect or move the leader's pan (the harness does stow/restore automatically)."""
    s = LeaderStow.from_config()
    leader = connect_leader(s.port, s.leader_id)
    try:
        here = bus_pos(leader)
        typer.echo(f"leader pose deg {np.round(here, 1).tolist()}  stow pan {s.stow_pan_deg:+.1f}")
        if dry_run:
            return
        if stow:
            move_pan(leader, s.stow_pan_deg, s.torque_limit, s.speed)
        elif restore_pan is not None:
            move_pan(leader, restore_pan, s.torque_limit, s.speed)
        time.sleep(0.2)
        typer.echo(f"now pan {bus_pos(leader)[0]:+.1f} deg, torque off")
    finally:
        leader.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    app()
