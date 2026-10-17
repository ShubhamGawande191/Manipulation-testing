"""Staged leader→follower teleoperation with tracking-error and load logging.

Why not plain ``lerobot-teleoperate``: ``SOFollower.connect()`` enables torque on all six
servos at once without first setting Goal = Present, and the first leader action then drives
every joint at full speed. With the heavier PincOpen gripper this browns out the 12 V bus
("no status packet") and on disconnect the arm drops when torque turns off. This tool:

1. connects without LeRobot's ``configure()``, writes the same registers itself (plus optional
   per-joint P/I/D overrides), sets Goal = Present, then enables torque one joint at a time;
2. glides the follower to the leader pose one joint at a time (``--stage-speed`` deg/s);
3. mirrors the leader at ``--hz`` with a per-step velocity cap, logging commanded vs present
   position and Present_Load per joint;
4. on Ctrl+C / ``--duration``: glides back to the pose it started in (the rest pose), then
   disables torque joint by joint, and prints a tracking/load table (parquet in runs/teleop/).

    uv run teleop                                   # defaults: /dev/so101_*, my_so101_* ids
    uv run teleop --pid elbow_flex=24,0,32          # P,I,D override for one joint (repeatable)
    uv run teleop --duration 60 --max-speed 120

Body joints are degrees (LeRobot ``use_degrees``), gripper 0..100.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from robot_manipulator.core.safety import VelocityCap
from robot_manipulator.hw.follower import (  # noqa: F401  (re-exported for tests/callers)
    DEFAULT_PID,
    JOINT_PID,
    JOINTS,
    REST_ARM,
    SETTLE_S,
    STAGE_ORDER,
    STAGE_TOL,
    stage_targets,
)
from robot_manipulator.hw.follower import bus_pos as _bus_pos
from robot_manipulator.hw.follower import connect_follower as _connect_follower
from robot_manipulator.hw.follower import glide as _glide
from robot_manipulator.hw.follower import park_and_release as _park_and_release
from robot_manipulator.hw.follower import send_goal as _send

LOAD_EVERY = 5  # read Present_Load every N ticks (bus time)
# Leader rest pose (deg), read at the start of the 2026-10-17 teleop run with the leader folded.
LEADER_REST_ARM = np.array([-3.1, -107.0, 96.6, 77.3, -85.1])
LEADER_TORQUE_LIMIT = 300  # permille (RAM Torque_Limit): gentle, a hand can overpower it
MAX_LEADER_MISSES = 10  # consecutive failed leader reads (0.2 s at 50 Hz) before stopping
RUNS_DIR = Path("runs/teleop")

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


def parse_pid(specs: list[str]) -> dict[str, tuple[int, int, int]]:
    """``["elbow_flex=24,0,32"]`` → ``{"elbow_flex": (24, 0, 32)}``. Values are register units."""
    out: dict[str, tuple[int, int, int]] = {}
    for spec in specs:
        name, _, vals = spec.partition("=")
        name = name.strip()
        if name not in JOINTS:
            raise typer.BadParameter(f"unknown joint {name!r}; one of {JOINTS}")
        parts = [int(v) for v in vals.split(",")]
        if len(parts) != 3 or not all(0 <= v <= 254 for v in parts):
            raise typer.BadParameter(f"{spec!r}: need P,I,D integers in 0..254")
        out[name] = (parts[0], parts[1], parts[2])
    return out


@dataclass
class TrackLog:
    t: list[float] = field(default_factory=list)
    cmd: list[np.ndarray] = field(default_factory=list)
    pos: list[np.ndarray] = field(default_factory=list)
    load: list[np.ndarray | None] = field(default_factory=list)

    def add(self, t: float, cmd: np.ndarray, pos: np.ndarray, load: np.ndarray | None) -> None:
        self.t.append(t)
        self.cmd.append(cmd.copy())
        self.pos.append(pos.copy())
        self.load.append(None if load is None else load.copy())

    def summary(self) -> dict[str, dict[str, float]]:
        """Per joint: mean/p95/max |cmd - pos|, steady-state error (cmd nearly still), load."""
        if not self.t:
            return {}
        cmd, pos = np.array(self.cmd), np.array(self.pos)
        err = cmd - pos
        still = np.zeros(len(cmd), dtype=bool)
        if len(cmd) > 10:
            speed = np.abs(np.gradient(cmd, axis=0)).max(axis=1)
            still = speed < 0.05  # deg per tick
        loads = [x for x in self.load if x is not None]
        lo = np.abs(np.array(loads)) if loads else None
        out: dict[str, dict[str, float]] = {}
        for j, name in enumerate(JOINTS):
            e = np.abs(err[:, j])
            out[name] = {
                "mean_abs_err": float(e.mean()),
                "p95_abs_err": float(np.percentile(e, 95)),
                "max_abs_err": float(e.max()),
                "steady_err": float(err[still, j].mean()) if still.any() else float("nan"),
                "load_mean": float(lo[:, j].mean()) if lo is not None else float("nan"),
                "load_max": float(lo[:, j].max()) if lo is not None else float("nan"),
            }
        return out

    def save(self, path: Path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        cols: dict[str, list] = {"t": self.t}
        for j, name in enumerate(JOINTS):
            cols[f"cmd_{name}"] = [float(c[j]) for c in self.cmd]
            cols[f"pos_{name}"] = [float(p[j]) for p in self.pos]
            cols[f"load_{name}"] = [None if x is None else float(x[j]) for x in self.load]
        pq.write_table(pa.table(cols), path)
        return path


# ---- hardware ---------------------------------------------------------------------------


def _leader_pos(leader) -> np.ndarray:
    act = leader.get_action()
    return np.array([act[f"{j}.pos"] for j in JOINTS], dtype=np.float64)


def _park_leader(leader, speed: float, hz: float) -> None:
    """Drive the leader back to LEADER_REST_ARM under a low torque limit, then release it.

    The leader is normally unpowered so it can be back-driven; its 5 V servos can still position
    themselves gently. The gripper/trigger is left where it is.
    """
    bus = leader.bus
    here = _bus_pos(leader)
    _send(leader, here)  # Goal = Present: enabling torque must not jump
    for motor in JOINTS:
        bus.write("Torque_Limit", motor, LEADER_TORQUE_LIMIT)
    for motor in STAGE_ORDER:
        bus.enable_torque(motor)
    try:
        cmd = here
        target = np.r_[LEADER_REST_ARM, here[-1]]
        for _name, wp in stage_targets(cmd, target, tuple(reversed(STAGE_ORDER))):
            cmd = _glide(leader, cmd, wp, speed, hz)
    finally:
        bus.disable_torque()
    console.print("[green]leader parked, torque off[/]")


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    table = Table(title="Tracking (cmd − present) and load")
    for col in ("joint", "mean|e|", "p95|e|", "max|e|", "steady e", "load mean", "load max"):
        table.add_column(col, justify="right")
    for name, s in summary.items():
        table.add_row(
            name,
            *(f"{s[k]:.2f}" for k in ("mean_abs_err", "p95_abs_err", "max_abs_err", "steady_err")),
            f"{s['load_mean']:.0f}",
            f"{s['load_max']:.0f}",
        )
    console.print(table)
    console.print(
        "[dim]errors in deg (gripper %); load raw 0..1000 = % of max torque ×10. "
        "Large steady e on elbow/shoulder_lift with load near the cap → "
        "try a higher P or a little I.[/]"
    )


@app.command()
def main(
    follower_port: str = typer.Option("/dev/so101_follower"),
    follower_id: str = typer.Option("my_so101_follower"),
    leader_port: str = typer.Option("/dev/so101_leader"),
    leader_id: str = typer.Option("my_so101_leader"),
    hz: float = typer.Option(50.0, help="mirror loop rate"),
    max_speed: float = typer.Option(180.0, help="per-joint velocity cap while mirroring, deg/s"),
    stage_speed: float = typer.Option(30.0, help="velocity for staging/parking moves, deg/s"),
    duration: float = typer.Option(0.0, help="stop after N s (0 = until Ctrl+C)"),
    pid: list[str] = typer.Option([], help="joint=P,I,D register override, repeatable"),
    park: bool = typer.Option(
        True, "--park/--no-park", help="return to start pose before torque off"
    ),
    park_leader: bool = typer.Option(
        True, "--park-leader/--no-park-leader", help="drive the leader to its rest pose at the end"
    ),
    park_only: bool = typer.Option(
        False, "--park-only", help="no teleop: glide the follower to REST_ARM and drop torque"
    ),
) -> None:
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    overrides = parse_pid(pid)
    if park_only:
        robot, here = _connect_follower(follower_port, follower_id, overrides)
        try:
            console.print("present:", np.round(here, 1), "→ rest arm", REST_ARM)
            _park_and_release(robot, np.r_[REST_ARM, here[-1]], stage_speed, hz)
        finally:
            robot.bus.disconnect(disable_torque=False)
        return
    console.print(
        f"[cyan]PID[/] default {DEFAULT_PID}, per-joint {JOINT_PID}, "
        f"overrides {overrides or 'none'}"
    )

    leader = SOLeader(SOLeaderTeleopConfig(port=leader_port, id=leader_id))
    for attempt in range(3):  # leader ID 2 missed handshakes when overvolted (12 V on a 5 V arm)
        try:
            leader.connect(calibrate=False)  # leader torque is disabled by its configure()
            break
        except RuntimeError as exc:
            if attempt == 2:
                raise
            console.print(
                f"[yellow]leader handshake failed, retrying: {str(exc).splitlines()[0]}[/]"
            )
            leader.bus.port_handler.closePort()
            time.sleep(0.5)
    if not leader.is_calibrated:
        leader.disconnect()
        raise RuntimeError(
            f"leader calibration mismatch for id {leader_id!r}; recalibrate the leader"
        )
    robot = None
    log = TrackLog()
    try:
        robot, start = _connect_follower(follower_port, follower_id, overrides)
        goal = _leader_pos(leader)
        console.print("start (follower):", np.round(start, 1))
        console.print("leader pose     :", np.round(goal, 1))
        cmd = start.copy()
        for name, wp in stage_targets(start, goal):
            console.print(
                f"  staging {name}: {cmd[JOINTS.index(name)]:.1f} → {wp[JOINTS.index(name)]:.1f}"
            )
            cmd = _glide(robot, cmd, wp, stage_speed, hz)

        console.print("[green]mirroring — move the leader; Ctrl+C to stop[/]")
        cap = VelocityCap(max_speed)
        dt = 1.0 / hz
        t0 = time.monotonic()
        next_t = t0
        tick = 0
        misses = 0
        while duration <= 0 or time.monotonic() - t0 < duration:
            try:
                target = _leader_pos(leader)
                misses = 0
            except (ConnectionError, RuntimeError) as exc:
                # a dropped leader read must never move the follower: hold still
                misses += 1
                if misses > MAX_LEADER_MISSES:
                    console.print(f"[red]leader lost for {misses} ticks ({exc}); stopping[/]")
                    break
                target = cmd
            cmd = cap.clamp_step(cmd, target, dt)
            _send(robot, cmd)
            pos = _bus_pos(robot)
            load = None
            if tick % LOAD_EVERY == 0:
                raw = robot.bus.sync_read("Present_Load", normalize=False)
                load = np.array([raw[j] for j in JOINTS], dtype=np.float64)
            log.add(time.monotonic() - t0, cmd, pos, load)
            tick += 1
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        console.print("\n[yellow]stopping[/]")
    finally:
        if log.t:  # save before parking so the data survives anything that happens next
            path = log.save(RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.parquet")
            console.print(f"[dim]trace → {path}[/]")
        if robot is not None:
            try:
                if park:
                    console.print("parking at start pose…")
                    _park_and_release(robot, start, stage_speed, hz)
                else:
                    for motor in reversed(STAGE_ORDER):
                        robot.bus.disable_torque(motor)
            finally:
                robot.bus.disconnect(disable_torque=False)
        if park_leader:
            try:
                _park_leader(leader, stage_speed, hz)
            except (Exception, KeyboardInterrupt) as exc:  # leader parking must not hide results
                console.print(f"[yellow]leader park skipped: {exc}[/]")
        leader.disconnect()

    if log.t:
        _print_summary(log.summary())


if __name__ == "__main__":
    app()
