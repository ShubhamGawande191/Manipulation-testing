"""Safe follower hardware primitives shared by teleop and the harness.

Lessons baked in (see docs/DIGITAL-TWIN-RUNBOOK.md §1-§3):
- ``SOFollower.connect()`` enables all six torques at once without Goal = Present, which can jump
  and brown out the 12 V bus. ``connect_follower`` writes LeRobot's registers itself (plus per-joint
  P gains: gravity joints sag at P=16 with the PincOpen) and enables torque joint by joint.
- ``disconnect()`` drops torque wherever the arm is. ``park_and_release`` glides to a rest pose
  first (Ctrl+C ignored), then releases joint by joint.

Units: LeRobot degrees for body joints, 0..100 for the gripper; order ``JOINTS``.
"""

from __future__ import annotations

import time

import numpy as np
from rich.console import Console

from robot_manipulator.core.safety import VelocityCap

console = Console()

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
# Follower arm rest pose (deg): the arm folded on itself, safe to drop torque. Measured 2026-10-17
# and converted to the frame written by `calib-recal` that day (pan/elbow/wrist re-swept,
# wrist_roll aligned to the leader). Re-derive if the follower is recalibrated again. The gripper is
# left wherever it is when parking.
REST_ARM = np.array([-3.3, -102.8, 96.1, 78.2, -85.0])


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


def bus_pos(robot) -> np.ndarray:
    pos = robot.bus.sync_read("Present_Position")
    return np.array([pos[j] for j in JOINTS], dtype=np.float64)


def send_goal(robot, q: np.ndarray) -> None:
    robot.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, q)})


def connect_follower(port: str, robot_id: str, pid: dict[str, tuple[int, int, int]]):
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
        start = bus_pos(robot)
        send_goal(robot, start)
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
    start = bus_pos(robot)
    send_goal(robot, start)
    for motor in STAGE_ORDER:
        bus.enable_torque(motor)
        time.sleep(0.25)  # let the bus settle between joints (brown-out lesson)
    return robot, start


def glide(robot, cmd: np.ndarray, target: np.ndarray, speed: float, hz: float) -> np.ndarray:
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
        send_goal(robot, cmd)
        time.sleep(dt)
    settle_end = time.monotonic() + SETTLE_S
    err = bus_pos(robot) - target
    while time.monotonic() < settle_end and np.max(np.abs(err)) >= STAGE_TOL:
        time.sleep(dt)
        err = bus_pos(robot) - target
    worst = int(np.argmax(np.abs(err)))
    if abs(err[worst]) >= STAGE_TOL:
        console.print(f"    [yellow]residual {JOINTS[worst]} {err[worst]:+.1f} (present − cmd)[/]")
    return cmd


def park_and_release(robot, target: np.ndarray, speed: float, hz: float) -> None:
    """Glide to ``target`` (light joints first, shoulder_lift last), then drop torque joint by
    joint. Ctrl+C is ignored while this runs: an interrupted park leaves the arm mid-air."""
    import signal

    def _hold_on(signum, frame):  # noqa: ARG001
        console.print("[yellow]still parking — wait for torque off[/]")

    previous = signal.signal(signal.SIGINT, _hold_on)
    try:
        cmd = bus_pos(robot)
        for _name, wp in stage_targets(cmd, target, tuple(reversed(STAGE_ORDER))):
            cmd = glide(robot, cmd, wp, speed, hz)
        for motor in reversed(STAGE_ORDER):
            robot.bus.disable_torque(motor)
            time.sleep(0.1)
        console.print("[green]parked, torque off[/]")
    finally:
        signal.signal(signal.SIGINT, previous)
