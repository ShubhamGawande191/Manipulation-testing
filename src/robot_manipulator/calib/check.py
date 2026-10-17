"""Compare LeRobot SO101 calibration files with the NVIDIA fleet and with each other (read-only).

Conventions (LeRobot Feetech bus):
- `Present = Actual - Homing_Offset`, so an encoder count `Actual` is `Present + Homing_Offset`.
- In `DEGREES` mode, zero is the midpoint of the calibrated range, `mid = (min + max) / 2`.
- So the actual encoder count where a range-defined joint reads 0 deg is `zero_raw = mid + homing`.
  For full-turn joints (range 0..4095, no stops) the zero comes from the homing offset alone.

`frame_mismatch` returns the difference in zero location between two arms, in degrees. It is only a
hint. The calibration files cannot know the physical pose, so physical alignment is the ground truth
(the plan's `align` step, not built yet).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

COUNTS_PER_TURN = 4096
DEG_PER_COUNT = 360.0 / COUNTS_PER_TURN
HOMING_LIMIT = 2048  # |homing_offset| above this is outside the sign-magnitude range
FRAME_TOL_DEG = 5.0  # leader/follower zeros further apart than this are flagged (indicative)
GRIPPER = "gripper"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_STATS = REPO / "config" / "calibration_stats_so101.json"
CACHE = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
DEFAULT_FOLLOWER = CACHE / "robots" / "so_follower" / "my_so101_follower.json"
DEFAULT_LEADER = CACHE / "teleoperators" / "so_leader" / "my_so101_leader.json"


@dataclass(frozen=True)
class Issue:
    joint: str
    kind: str  # "fleet_range" | "homing" | "frame"
    value: float
    message: str


def load_calibration(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a LeRobot calibration JSON (plain JSON, no draccus)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of joints")
    cal: dict[str, dict[str, int]] = {}
    for joint, entry in data.items():
        missing = [k for k in FIELDS if k not in entry]
        if missing:
            raise ValueError(f"{path}: joint {joint!r} is missing {missing}")
        cal[joint] = {k: int(entry[k]) for k in FIELDS}
    return cal


def range_counts(cal: dict[str, dict[str, int]]) -> dict[str, int]:
    return {j: e["range_max"] - e["range_min"] for j, e in cal.items()}


def is_full_turn(entry: dict[str, int]) -> bool:
    return entry["range_min"] == 0 and entry["range_max"] == COUNTS_PER_TURN - 1


def zero_raw(entry: dict[str, int]) -> float:
    """Actual encoder count at which a range-defined joint reads 0 deg."""
    return 0.5 * (entry["range_min"] + entry["range_max"]) + entry["homing_offset"]


def _wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def frame_mismatch(
    cal_a: dict[str, dict[str, int]], cal_b: dict[str, dict[str, int]]
) -> dict[str, float]:
    """Estimated zero offset of arm A relative to arm B, in degrees, per joint (gripper excluded).

    Range-defined joints: `(zero_raw_a - zero_raw_b) * 360/4096`.
    Full-turn joints: `(homing_a - homing_b) * 360/4096`, which is the same formula when both
    ranges are 0..4095. Indicative only. See the module docstring.
    """
    out: dict[str, float] = {}
    for joint, a in cal_a.items():
        if joint == GRIPPER or joint not in cal_b:
            continue
        b = cal_b[joint]
        if is_full_turn(a) and is_full_turn(b):
            counts = a["homing_offset"] - b["homing_offset"]
        else:
            counts = zero_raw(a) - zero_raw(b)
        out[joint] = _wrap_deg(counts * DEG_PER_COUNT)
    return out


def fleet_check(
    cal: dict[str, dict[str, int]],
    stats: dict[str, Any],
    n_std: float = 2.0,
) -> list[Issue]:
    """Flag range outliers against the fleet and homing offsets outside the encoder's range.

    Joints whose stats entry has `"skip": true` are ignored entirely.
    """
    issues: list[Issue] = []
    for joint, entry in cal.items():
        s = stats.get(joint)
        if s is None or s.get("skip"):
            continue
        if s["std"] <= 0:
            raise ValueError(f"fleet std for {joint} must be positive")
        rng = entry["range_max"] - entry["range_min"]
        sigma = (rng - s["mean"]) / s["std"]
        if abs(sigma) > n_std:
            issues.append(
                Issue(
                    joint,
                    "fleet_range",
                    sigma,
                    f"range {rng} is {sigma:+.1f} sigma from fleet mean {s['mean']:.0f}",
                )
            )
        if abs(entry["homing_offset"]) > HOMING_LIMIT:
            issues.append(
                Issue(
                    joint,
                    "homing",
                    entry["homing_offset"],
                    f"homing offset {entry['homing_offset']} exceeds +-{HOMING_LIMIT}",
                )
            )
    return issues


def frame_issues(offsets: dict[str, float], tol_deg: float = FRAME_TOL_DEG) -> list[Issue]:
    return [
        Issue(
            joint,
            "frame",
            deg,
            f"zero is {deg:+.1f} deg from the other arm (indicative; confirm with physical align)",
        )
        for joint, deg in offsets.items()
        if abs(deg) > tol_deg
    ]


def load_stats(path: str | Path) -> dict[str, Any]:
    stats = json.loads(Path(path).read_text())
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def _arm_table(title: str, cal: dict[str, dict[str, int]], stats: dict[str, Any], issues):
    table = Table(title=title)
    for col in ("joint", "range", "fleet mean±σ", "σ-dev", "homing", "flags"):
        table.add_column(col)
    by_joint = {i.joint: [] for i in issues}
    for i in issues:
        by_joint.setdefault(i.joint, []).append(i.kind)
    for joint in JOINTS:
        if joint not in cal:
            continue
        entry = cal[joint]
        rng = entry["range_max"] - entry["range_min"]
        s = stats.get(joint)
        if s is None or s.get("skip"):
            fleet, sig = "—", "skip" if s and s.get("skip") else "—"
        else:
            fleet = f"{s['mean']:.0f}±{s['std']:.0f}"
            sig = f"{(rng - s['mean']) / s['std']:+.1f}"
        flags = ", ".join(by_joint.get(joint, [])) or "ok"
        table.add_row(joint, str(rng), fleet, sig, str(entry["homing_offset"]), flags)
    return table


def run_checks(
    follower: dict[str, dict[str, int]],
    leader: dict[str, dict[str, int]],
    stats: dict[str, Any],
    n_std: float,
    console: Console,
) -> int:
    f_issues = fleet_check(follower, stats, n_std)
    l_issues = fleet_check(leader, stats, n_std)
    offsets = frame_mismatch(follower, leader)
    flagged_frames = {i.joint for i in frame_issues(offsets)}
    console.print(_arm_table("follower vs fleet", follower, stats, f_issues))
    console.print(
        _arm_table(
            "leader vs fleet (fleet stats are follower arms; a hint)", leader, stats, l_issues
        )
    )

    # Frame offsets are printed, but they do not set the exit code. They are indicative only
    # (the files cannot see the physical pose), so physical alignment is the ground truth.
    title = (
        f"follower zero minus leader zero (indicative, not counted, tol {FRAME_TOL_DEG:.0f} deg)"
    )
    frame_table = Table(title=title)
    for col in ("joint", "offset (deg)", "differs"):
        frame_table.add_column(col)
    for joint, deg in offsets.items():
        frame_table.add_row(joint, f"{deg:+.1f}", "yes" if joint in flagged_frames else "no")
    console.print(frame_table)

    total = len(f_issues) + len(l_issues)
    if total:
        console.print(f"[bold red]{total} issue(s)[/bold red]")
    else:
        console.print("[bold green]no issues[/bold green]")
    return total


app = typer.Typer(add_completion=False, help="Check SO101 calibration files (read-only).")


@app.command()
def main(
    follower: Path = typer.Option(DEFAULT_FOLLOWER, help="follower calibration JSON"),
    leader: Path = typer.Option(DEFAULT_LEADER, help="leader calibration JSON"),
    stats: Path = typer.Option(DEFAULT_STATS, help="fleet stats JSON"),
    n_std: float = typer.Option(2.0, help="flag ranges further than this many sigma"),
) -> None:
    fol = load_calibration(follower)
    lead = load_calibration(leader)
    st = load_stats(stats)
    total = run_checks(fol, lead, st, n_std, Console())
    raise typer.Exit(code=1 if total else 0)


if __name__ == "__main__":
    app()
