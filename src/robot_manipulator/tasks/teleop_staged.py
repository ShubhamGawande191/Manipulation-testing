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

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Heavy, gravity-loaded joints move alone and first; light ones after.
STAGE_ORDER = ("shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan", "wrist_roll", "gripper")
# LeRobot SOFollower.configure() defaults.
DEFAULT_PID = (16, 0, 32)
# Gravity-loaded joints sag at P=16 with the PincOpen: elbow settled 3.7° short, 0.14° at P=24
# (2026-10-17 teleop runs); shoulder_lift showed -1.1° steady error at 16.
JOINT_PID = {"shoulder_lift": (24, 0, 32), "elbow_flex": (24, 0, 32)}
STAGE_TOL = 2.0  # deg (gripper: % of range)
SETTLE_S = 1.5  # max wait for joints to follow after a staging move
LOAD_EVERY = 5  # read Present_Load every N ticks (bus time)
# Follower arm rest pose (deg): the arm folded on itself, safe to drop torque. Measured 2026-10-17
# and converted to the frame written by `calib-recal` that day (pan/elbow/wrist re-swept,
# wrist_roll aligned to the leader). Re-derive if the follower is recalibrated again. The gripper is
# left wherever it is when parking.
REST_ARM = np.array([-3.3, -102.8, 96.1, 78.2, -85.0])
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


def stage_targets(
    start: np.ndarray, goal: np.ndarray, order: tuple[str, ...] = STAGE_ORDER
) -> list[tuple[str, np.ndarray]]:
    """Waypoints that move one joint at a time from ``start`` to ``goal`` in ``order``."""
    cur = start.astype(np.float64).copy()
    steps: list[tuple[str, np.ndarray]] = []
    for name in order:
        i = JOINTS.index(name)
        cur = cur.copy()
        cur[i] = goal[i]
        steps.append((name, cur))
    return steps


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


def _bus_pos(robot) -> np.ndarray:
    pos = robot.bus.sync_read("Present_Position")
    return np.array([pos[j] for j in JOINTS], dtype=np.float64)


def _leader_pos(leader) -> np.ndarray:
    act = leader.get_action()
    return np.array([act[f"{j}.pos"] for j in JOINTS], dtype=np.float64)


def _send(robot, q: np.ndarray) -> None:
    robot.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, q)})


def _connect_follower(port: str, robot_id: str, pid: dict[str, tuple[int, int, int]]):
    """Connect the follower with LeRobot's registers but staged torque enable."""
    from lerobot.motors.feetech import OperatingMode
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config

    robot = SOFollower(build_follower_config(port=port, robot_id=robot_id, cameras=False))
    robot.bus.connect()
    if not robot.bus.is_calibrated:
        robot.bus.disconnect(disable_torque=False)
        raise RuntimeError(
            f"follower calibration mismatch for id {robot_id!r}; run `uv run calibrate` first"
        )
    bus = robot.bus
    torque = bus.sync_read("Torque_Enable", normalize=False)
    if any(torque.values()):
        # An interrupted earlier run left the arm holding a pose. Dropping torque here to write
        # registers would drop the arm, so keep holding: the registers are still configured
        # (servos keep them until power-off). Only Goal = Present is refreshed.
        if pid:
            bus.disconnect(disable_torque=False)
            raise RuntimeError(
                "follower torque is still on: run `uv run teleop --park-only` before --pid"
            )
        start = _bus_pos(robot)
        _send(robot, start)
        bus.enable_torque()  # any joints that were off; holds Goal = Present
        console.print("[yellow]follower torque was already on — kept holding[/]")
        return robot, start
    bus.disable_torque()
    bus.configure_motors()
    for motor in JOINTS:
        p, i, d = pid.get(motor, JOINT_PID.get(motor, DEFAULT_PID))
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.write("P_Coefficient", motor, p)
        bus.write("I_Coefficient", motor, i)
        bus.write("D_Coefficient", motor, d)
        if motor == "gripper":  # same protection as SOFollower.configure()
            bus.write("Max_Torque_Limit", motor, 500)
            bus.write("Protection_Current", motor, 250)
            bus.write("Overload_Torque", motor, 25)
    # Goal = Present so enabling torque holds the current pose instead of jumping.
    start = _bus_pos(robot)
    _send(robot, start)
    for motor in STAGE_ORDER:
        bus.enable_torque(motor)
        time.sleep(0.25)  # let the bus settle between joints (brown-out lesson)
    return robot, start


def _glide(robot, cmd: np.ndarray, target: np.ndarray, speed: float, hz: float) -> np.ndarray:
    """Velocity-capped move of the commanded pose to ``target``; returns the final command.

    Once the command arrives, waits at most ``SETTLE_S`` for the joints to follow. A joint that
    sags under gravity (low P, heavy gripper) never gets within ``STAGE_TOL``, so this must not
    block on it — the residual is printed instead (it is the steady-state error to tune away).
    """
    cap = VelocityCap(speed)
    dt = 1.0 / hz
    cmd = cmd.copy()
    while not np.allclose(cmd, target):
        cmd = cap.clamp_step(cmd, target, dt)
        _send(robot, cmd)
        time.sleep(dt)
    settle_end = time.monotonic() + SETTLE_S
    err = _bus_pos(robot) - target
    while time.monotonic() < settle_end and np.max(np.abs(err)) >= STAGE_TOL:
        time.sleep(dt)
        err = _bus_pos(robot) - target
    worst = int(np.argmax(np.abs(err)))
    if abs(err[worst]) >= STAGE_TOL:
        console.print(f"    [yellow]residual {JOINTS[worst]} {err[worst]:+.1f} (present − cmd)[/]")
    return cmd


def _park_and_release(robot, target: np.ndarray, speed: float, hz: float) -> None:
    """Glide to ``target`` (light joints first, shoulder_lift last), then drop torque joint by
    joint. Ctrl+C is ignored while this runs: an interrupted park leaves the arm mid-air."""
    import signal

    def _hold_on(signum, frame):  # noqa: ARG001
        console.print("[yellow]still parking — wait for torque off[/]")

    previous = signal.signal(signal.SIGINT, _hold_on)
    try:
        cmd = _bus_pos(robot)
        for _name, wp in stage_targets(cmd, target, tuple(reversed(STAGE_ORDER))):
            cmd = _glide(robot, cmd, wp, speed, hz)
        for motor in reversed(STAGE_ORDER):
            robot.bus.disable_torque(motor)
            time.sleep(0.1)
        console.print("[green]parked, torque off[/]")
    finally:
        signal.signal(signal.SIGINT, previous)


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
