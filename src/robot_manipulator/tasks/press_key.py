"""Scripted key-press primitive — the Phase 0 exit task (roadmap M0).

Two steps:

    uv run python -m robot_manipulator.tasks.press_key teach --note 48
        Torque off; you physically place the arm (hover above the key, then
        key pressed down); waypoints saved to config/minilab_keys.json.

    uv run python -m robot_manipulator.tasks.press_key press --note 48
        Replays start -> hover -> press -> hover -> start with per-joint
        velocity caps and a heartbeat watchdog, captures live MIDI, scores
        with eval.scorer.PressKey, and writes a parquet EpisodeRecord to
        runs/<run_id>.parquet (+ the raw event log as JSONL).

Direct LeRobot path: joint values are DEGREES. Motion is only ever an
interpolation between positions a human physically demonstrated, which is
the real safety envelope here; velocity caps + watchdog bound how it moves,
not where.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from robot_manipulator.core.episode_log import EpisodeRecord, write_records
from robot_manipulator.core.safety import VelocityCap, Watchdog
from robot_manipulator.eval.scorer import PressKey, score

app = typer.Typer(add_completion=False)
console = Console()

KEYS_FILE = Path("config/minilab_keys.json")
RUNS_DIR = Path("runs")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HZ = 20.0
DT = 1.0 / HZ
# deg/s, conservative; gripper faster is fine
MAX_SPEED = np.array([30.0, 30.0, 30.0, 40.0, 40.0, 60.0])
TOL_DEG = 1.5  # per-joint arrival tolerance
MOVE_TIMEOUT_S = 8.0


def load_keys(path: Path = KEYS_FILE) -> dict[str, dict[str, list[float]]]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_key(note: int, hover: list[float], press: list[float], path: Path = KEYS_FILE) -> None:
    keys = load_keys(path)
    keys[str(note)] = {"hover": hover, "press": press}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(keys, indent=2) + "\n")


def _connect(port: str):
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config

    robot = SOFollower(build_follower_config(port=port, cameras=False))
    robot.connect(calibrate=False)
    return robot


def _obs_deg(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float64)


def _send_deg(robot, arr: np.ndarray) -> None:
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, arr)})


def glide(robot, target: np.ndarray, listener=None, sent_t: list[float] | None = None) -> bool:
    """Velocity-capped interpolation to target. Returns True if it arrived."""
    cap = VelocityCap(MAX_SPEED)
    wd = Watchdog(timeout_s=0.25)
    t0 = time.monotonic()
    current = _obs_deg(robot)
    setpoint = current.copy()
    while time.monotonic() - t0 < MOVE_TIMEOUT_S:
        if wd.stale:
            console.print("[red]watchdog stale — freezing[/]")
            return False
        setpoint = cap.clamp_step(setpoint, target, DT)
        _send_deg(robot, setpoint)
        wd.beat()
        if sent_t is not None:
            sent_t.append(time.monotonic())
        if listener is not None:
            listener.poll()
        if np.max(np.abs(_obs_deg(robot) - target)) < TOL_DEG:
            return True
        time.sleep(DT)
    return np.max(np.abs(_obs_deg(robot) - target)) < TOL_DEG * 2


@app.command()
def teach(note: int, port: str = "/dev/so101_follower") -> None:
    """Record hover + press waypoints for one key by hand-guiding the arm."""
    robot = _connect(port)
    robot.bus.disable_torque()
    console.print(
        f"[yellow]Torque OFF.[/] Move the arm so the fingertip HOVERS ~2cm "
        f"above note {note}, then press Enter."
    )
    input()
    hover = _obs_deg(robot).tolist()
    console.print("Now push the fingertip down so the key is PRESSED, then press Enter.")
    input()
    press_pos = _obs_deg(robot).tolist()
    robot.bus.enable_torque()
    robot.disconnect()
    save_key(note, hover, press_pos)
    console.print(f"[green]saved[/] note {note} to {KEYS_FILE}")
    console.print(
        {"hover": [round(v, 1) for v in hover], "press": [round(v, 1) for v in press_pos]}
    )


@app.command()
def press(
    note: int,
    dwell: float = 0.35,
    port: str = "/dev/so101_follower",
    run_id: str = "",
) -> None:
    """Execute the press and score it from live MIDI."""
    from robot_manipulator.io.midi import MidiListener

    keys = load_keys()
    if str(note) not in keys:
        console.print(f"[red]note {note} not taught — run `teach --note {note}` first[/]")
        raise typer.Exit(2)
    hover = np.array(keys[str(note)]["hover"])
    press_pos = np.array(keys[str(note)]["press"])
    run_id = run_id or f"press{note}-{uuid.uuid4().hex[:8]}"

    listener = MidiListener()
    midi_port = listener.open()
    console.print(f"MIDI: {midi_port}")

    robot = _connect(port)
    start = _obs_deg(robot)
    sent_t: list[float] = []
    t_run0 = time.monotonic()
    ok = True
    try:
        ok &= glide(robot, hover, listener, sent_t)
        ok &= glide(robot, press_pos, listener, sent_t)
        t_hold = time.monotonic()
        while time.monotonic() - t_hold < dwell:
            listener.poll()
            time.sleep(0.01)
        ok &= glide(robot, hover, listener, sent_t)
        ok &= glide(robot, start, listener, sent_t)
    finally:
        listener.poll()
        listener.close()
        robot.disconnect()
    e2e_ms = (time.monotonic() - t_run0) * 1000.0

    result = score(PressKey(note), listener.log.events, t_end=time.monotonic() - (listener.t0 or 0))
    intervals = np.diff(sent_t) if len(sent_t) > 2 else np.array([DT])
    record = EpisodeRecord(
        run_id=run_id,
        arm_id="follower",
        policy="scripted_press",
        executor="direct",
        chunk_len=1,
        precision="fp32",
        policy_latency_ms=0.0,
        e2e_latency_ms=e2e_ms,
        jitter_ms=float(np.std(intervals) * 1000.0),
        success=result.success,
        key_press_f1=result.key_press_f1,
        velocity_error=result.velocity_error,
        cc_error=result.cc_error,
        time_to_success_s=result.time_to_success_s,
        task=result.task,
    )
    RUNS_DIR.mkdir(exist_ok=True)
    write_records([record], RUNS_DIR / f"{run_id}.parquet")
    listener.log.save(RUNS_DIR / f"{run_id}.events.jsonl")

    color = "green" if result.success else "red"
    console.print(
        f"[{color}]success={result.success}[/] f1={result.key_press_f1} "
        f"t_success={result.time_to_success_s} motion_ok={ok} "
        f"events={len(listener.log.events)} e2e={e2e_ms:.0f}ms"
    )
    console.print(f"episode record: runs/{run_id}.parquet")
    raise typer.Exit(0 if result.success else 1)


@app.command()
def taught() -> None:
    """List taught keys."""
    console.print(load_keys() or "nothing taught yet")


if __name__ == "__main__":
    app()
