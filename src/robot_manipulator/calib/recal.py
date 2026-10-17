"""Partial follower recalibration: re-record selected joints and align wrist_roll to the leader.

LeRobot's stock ``calibrate()`` re-records every joint by hand, but the PincOpen gripper is not
back-drivable, so sweeping it would wreck its calibration (range 2005-3200, drive_mode 1). This
tool reuses LeRobot's own bus helpers on a subset of joints and keeps every other entry of the
calibration file untouched.

    uv run calib-recal                                  # pan, elbow, wrist_flex + wrist_roll
    uv run calib-recal --joints shoulder_pan --no-wrist-roll

Steps (torque stays OFF the whole time — you move the arm by hand):
  1. Range joints: hold them mid-range → Enter (``set_half_turn_homings``), then sweep each one
     stop-to-stop → Enter (``record_ranges_of_motion``). DEGREES zero = midpoint of that sweep.
  2. wrist_roll (no stops): turn the follower wrist to match the leader's visually → Enter. The
     follower homing is set so it reads the leader's current wrist_roll angle.
  3. Fleet check of the new ranges; accept → written to motors + JSON (backup kept), reject →
     the old calibration is written back.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from robot_manipulator.calib.check import DEFAULT_STATS, fleet_check, load_stats

FULL_TURN = 4096
MAX_RES = FULL_TURN - 1
HOMING_LIMIT = 2047  # Feetech Homing_Offset is 11-bit sign-magnitude
WRIST = "wrist_roll"
DEFAULT_JOINTS = "shoulder_pan,elbow_flex,wrist_flex"
ALIGN_TOL_DEG = 1.5

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


def homing_to_read(actual_raw: int, target_deg: float) -> int:
    """Homing offset that makes a full-turn joint at ``actual_raw`` read ``target_deg``.

    LeRobot DEGREES on a 0..4095 range: ``deg = (present - 4095/2) * 360/4095`` and Feetech
    ``present = (actual - homing) mod 4096``. The offset is wrapped by full turns into the
    register's ±2047 range (which leaves the reading unchanged modulo one turn).
    """
    target_present = target_deg * MAX_RES / 360.0 + MAX_RES / 2.0
    homing = round(actual_raw - target_present)
    homing = (homing + HOMING_LIMIT) % FULL_TURN - HOMING_LIMIT  # → [-2047, 2048]
    # ±2048 is one count past the register; 2047 reads ≤ 2 counts (0.18°) off, well inside tol
    return min(homing, HOMING_LIMIT)


def read_deg(actual_raw: int, homing: int) -> float:
    """LeRobot DEGREES reading of a full-turn joint (inverse of :func:`homing_to_read`)."""
    present = (actual_raw - homing) % FULL_TURN
    return (present - MAX_RES / 2.0) * 360.0 / MAX_RES


def merge_calibration(
    old: dict[str, dict[str, int]], new: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Old file with only the recalibrated joints replaced; ``id`` and ``drive_mode`` preserved."""
    out = {j: dict(v) for j, v in old.items()}
    for joint, entry in new.items():
        out[joint] = {
            **out[joint],
            "homing_offset": int(entry["homing_offset"]),
            "range_min": int(entry["range_min"]),
            "range_max": int(entry["range_max"]),
        }
    return out


def _to_motor_cal(cal: dict[str, dict[str, int]]):
    from lerobot.motors.motors_bus import MotorCalibration

    return {j: MotorCalibration(**v) for j, v in cal.items()}


@app.command()
def main(
    follower_port: str = typer.Option("/dev/so101_follower"),
    follower_id: str = typer.Option("my_so101_follower"),
    leader_port: str = typer.Option("/dev/so101_leader"),
    leader_id: str = typer.Option("my_so101_leader"),
    joints: str = typer.Option(DEFAULT_JOINTS, help="comma-separated range joints to re-record"),
    wrist_roll: bool = typer.Option(True, "--wrist-roll/--no-wrist-roll", help="align to leader"),
) -> None:
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config

    range_joints = [j.strip() for j in joints.split(",") if j.strip()]
    bad = [j for j in range_joints if j in ("gripper", WRIST)]
    if bad:
        raise typer.BadParameter(f"{bad}: gripper is never swept; wrist_roll uses --wrist-roll")

    robot = SOFollower(
        build_follower_config(port=follower_port, robot_id=follower_id, cameras=False)
    )
    fpath: Path = robot.calibration_fpath
    old = json.loads(fpath.read_text())
    bus = robot.bus
    bus.connect()
    leader = None
    try:
        if any(bus.sync_read("Torque_Enable", normalize=False).values()):
            raise RuntimeError("follower torque is on — run `uv run teleop --park-only` first")
        backup = fpath.with_name(f"{fpath.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(fpath, backup)
        console.print(f"[dim]backup → {backup}[/]")
        bus.disable_torque()  # also clears Lock so the EEPROM limit registers are writable
        new: dict[str, dict[str, int]] = {}

        if range_joints:
            input(f"\nHold {range_joints} near the MIDDLE of their range, then press ENTER…")
            homings = bus.set_half_turn_homings(range_joints)
            console.print(
                "Now sweep each of these joints slowly from one hard stop to the other "
                f"({', '.join(range_joints)}). Press ENTER when done."
            )
            mins, maxes = bus.record_ranges_of_motion(range_joints)
            for j in range_joints:
                new[j] = {"homing_offset": homings[j], "range_min": mins[j], "range_max": maxes[j]}

        if wrist_roll:
            from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

            leader = SOLeader(SOLeaderTeleopConfig(port=leader_port, id=leader_id))
            leader.bus.connect()
            input(
                "\nTurn the FOLLOWER wrist_roll so its gripper orientation matches the LEADER's "
                "exactly (look along the forearm), then press ENTER…"
            )
            target = float(leader.bus.sync_read("Present_Position", [WRIST])[WRIST])
            bus.reset_calibration([WRIST])  # homing 0 → present == actual
            actual = int(bus.sync_read("Present_Position", [WRIST], normalize=False)[WRIST])
            homing = homing_to_read(actual, target)
            new[WRIST] = {"homing_offset": homing, "range_min": 0, "range_max": MAX_RES}
            console.print(
                f"leader wrist_roll {target:+.1f}° → follower homing {homing} "
                f"(predicted reading {read_deg(actual, homing):+.1f}°)"
            )

        merged = merge_calibration(old, new)
        stats = load_stats(DEFAULT_STATS)
        issues = [i for i in fleet_check(merged, stats) if i.joint in new]
        for j in new:
            e = merged[j]
            console.print(
                f"  {j:13} homing {old[j]['homing_offset']:6} → {e['homing_offset']:6}   "
                f"range {old[j]['range_max'] - old[j]['range_min']:5} → "
                f"{e['range_max'] - e['range_min']:5}"
            )
        for i in issues:
            console.print(f"  [yellow]{i.joint}: {i.message}[/]")

        cal = _to_motor_cal(merged)
        bus.write_calibration(cal)
        misaligned = False
        if wrist_roll:
            got = float(bus.sync_read("Present_Position", [WRIST])[WRIST])
            want = float(leader.bus.sync_read("Present_Position", [WRIST])[WRIST])
            console.print(f"check: follower wrist_roll {got:+.1f}° vs leader {want:+.1f}°")
            if abs(got - want) > ALIGN_TOL_DEG:
                console.print("[red]wrist_roll did not land on the leader angle[/]")
                misaligned = True
        doubtful = bool(issues) or misaligned
        prompt = "Accept and save?" + (" (problems above)" if doubtful else "")
        if typer.confirm(prompt, default=not doubtful):
            robot.calibration = cal
            robot._save_calibration()
            console.print(f"[green]saved {fpath}[/] — run `uv run calib-check` to review")
        else:
            bus.write_calibration(_to_motor_cal(old))
            console.print("[yellow]rejected — old calibration written back to the motors[/]")
    except BaseException:
        # Ctrl+C / error mid-way: motors may hold half-written homings → restore the file's values
        if bus.is_connected:
            bus.write_calibration(_to_motor_cal(old))
            console.print("[yellow]aborted — old calibration written back to the motors[/]")
        raise
    finally:
        if leader is not None:
            leader.bus.disconnect(disable_torque=False)
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    app()
