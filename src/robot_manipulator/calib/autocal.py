"""Auto-calibration core: find mechanical stops by creeping, then home with LeRobot's convention.

Raw units only. Everything here works on raw encoder counts, with `read_raw` returning the position
*before* homing (0..4095). The real bus adapter is a stub on purpose; see `LeRobotJointBus`.

Conventions (checked against LeRobot, see tests/test_calib_autocal.py):
- Feetech: `Present = Actual - Homing_Offset` (feetech.py `read_calibration`,
  `_get_half_turn_homings`).
- LeRobot's half-turn homing: `Homing = raw - 2047`, so the current raw reads 2047.
- DEGREES: `deg = (present - mid) * 360 / 4095`, with `mid = (range_min + range_max) / 2`
  (motors_bus.py `_normalize`).

Consequence: in DEGREES mode the zero is the midpoint of the calibrated range, and homing cannot
move it. A homing-only zero shift (the plan's `zero_offsets_deg`) is therefore refused, not faked.
Shifting the zero needs an asymmetric range, and that needs a decision with the user.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

COUNTS = 4096
HALF_TURN = 2047  # the counts value that a centred joint reads
HOMING_LIMIT = 2047  # sign-magnitude, 11 bits
JOINT_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


class CalibError(RuntimeError):
    """A calibration could not be completed or failed validation."""


class JointBus(Protocol):
    """Raw-unit joint interface. Positions are counts 0..4095 before homing."""

    def read_raw(self, joint: str) -> int: ...

    def set_goal_raw(self, joint: str, raw: int) -> None: ...

    def read_load(self, joint: str) -> int: ...

    def set_torque_limit(self, joint: str, permille: int) -> None: ...

    def enable(self, joint: str) -> None: ...

    def disable(self, joint: str) -> None: ...


class LeRobotJointBus:
    """Placeholder for the real adapter over LeRobot's FeetechMotorsBus.

    Deliberately not implemented. Live autocalibration is reviewed with the user first.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("live autocal is reviewed with the user first")


@dataclass
class JointProfile:
    name: str
    method: Literal["hard_stop", "manual_zero"]
    expected_range: int | None = None
    range_tol: int = 40  # counts
    torque_permille: int = 250
    step_counts: int = 4  # per tick, about 17 deg/s at 50 Hz
    stall_err: int = 40
    stall_ticks: int = 10
    load_abort: int = 400
    backoff: int = 15

    def __post_init__(self) -> None:
        if self.method not in ("hard_stop", "manual_zero"):
            raise ValueError(f"{self.name}: unknown method {self.method!r}")
        if self.step_counts <= 0 or self.stall_ticks <= 0:
            raise ValueError(f"{self.name}: step_counts and stall_ticks must be positive")
        if not 0 < self.torque_permille <= 1000:
            raise ValueError(f"{self.name}: torque_permille must be in (0, 1000]")


@dataclass
class CalibProfile:
    robot: str
    joints: dict[str, JointProfile] = field(default_factory=dict)
    zero_offsets_deg: dict[str, float] = field(default_factory=dict)


def save_yaml(profile: CalibProfile, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(profile), sort_keys=False))
    return path


def load_yaml(path: str | Path) -> CalibProfile:
    data = yaml.safe_load(Path(path).read_text())
    joints = {name: JointProfile(**jp) for name, jp in data.get("joints", {}).items()}
    return CalibProfile(
        robot=data["robot"],
        joints=joints,
        zero_offsets_deg={k: float(v) for k, v in data.get("zero_offsets_deg", {}).items()},
    )


def _wrapdelta(x: int) -> int:
    """Shortest signed step between two raw positions, in -2048..2047."""
    return ((x + COUNTS // 2) % COUNTS) - COUNTS // 2


def _find_stop(
    bus: JointBus,
    joint: str,
    direction: int,
    prof: JointProfile,
    max_ticks: int,
    hint: int | None = None,
) -> tuple[int, str]:
    """Creep until a stop. Returns (unwrapped stop position, reason), reason "load" or "error".

    Positions are unwrapped across 4095 -> 0. Pass `hint` (an earlier unwrapped position) to
    continue in the same frame as an earlier search. Without it, the first raw reading starts a
    new frame.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    bus.set_torque_limit(joint, prof.torque_permille)
    bus.enable(joint)
    raw0 = bus.read_raw(joint)
    pos = raw0 if hint is None else hint + _wrapdelta(raw0 - hint % COUNTS)
    goal = pos
    bus.set_goal_raw(joint, goal % COUNTS)
    stalled = 0
    for _ in range(max_ticks):
        goal += direction * prof.step_counts
        bus.set_goal_raw(joint, goal % COUNTS)
        pos += _wrapdelta(bus.read_raw(joint) - pos % COUNTS)
        load = bus.read_load(joint)
        err = goal - pos
        stalled = stalled + 1 if abs(err) > prof.stall_err else 0
        reason = ""
        if abs(load) > prof.load_abort:
            reason = "load"
        elif stalled >= prof.stall_ticks:
            reason = "error"
        if reason:
            bus.set_goal_raw(joint, (pos - direction * prof.backoff) % COUNTS)
            return pos, reason
    raise CalibError(f"{joint}: no stop found in {max_ticks} ticks")


def find_stop(bus: JointBus, joint: str, direction: int, prof: JointProfile, max_ticks: int) -> int:
    """Unwrapped raw position of the stop in `direction`. Backs off `prof.backoff` counts after."""
    return _find_stop(bus, joint, direction, prof, max_ticks)[0]


def _homing_for(center_raw: float) -> int:
    """LeRobot's half-turn homing: the homing offset that makes `center_raw` read 2047."""
    raw_h = round(center_raw) - HALF_TURN
    homing = ((raw_h + COUNTS // 2) % COUNTS) - COUNTS // 2  # sign-magnitude window
    if abs(homing) > HOMING_LIMIT:
        raise CalibError(f"homing {homing} cannot be encoded (|homing| must be <= {HOMING_LIMIT})")
    return homing


def calibrate_joint(
    bus: JointBus,
    prof: JointProfile,
    *,
    zero_offset_deg: float = 0.0,
    zero_raw: int | None = None,
    max_ticks: int = 2000,
) -> dict[str, int]:
    """Return LeRobot's calibration entry (without `id`) for one joint.

    hard_stop: find both stops, home the stop midpoint to 2047, and set range_min/max to the stops
    in homed units. manual_zero: home `zero_raw` (the pose the user holds) to 2047 over the full
    range.
    """
    if zero_offset_deg != 0.0:
        raise CalibError(
            f"{prof.name}: zero_offset_deg={zero_offset_deg} is not supported. "
            "LeRobot's DEGREES zero"
            " is the calibrated-range midpoint, which homing cannot move. Decide the mechanism with"
            " the user first."
        )

    if prof.method == "manual_zero":
        if zero_raw is None or not 0 <= zero_raw < COUNTS:
            raise CalibError(f"{prof.name}: manual_zero needs zero_raw in 0..4095")
        homing = _homing_for(float(zero_raw))
        return {"drive_mode": 0, "homing_offset": homing, "range_min": 0, "range_max": COUNTS - 1}

    lo, _ = _find_stop(bus, prof.name, -1, prof, max_ticks)
    hi, _ = _find_stop(bus, prof.name, +1, prof, max_ticks, hint=lo)  # same unwrapped frame
    if lo > hi:
        lo, hi = hi, lo
    homing = _homing_for((lo + hi) / 2)
    range_min = (lo - homing) % COUNTS
    range_max = (hi - homing) % COUNTS
    width = range_max - range_min
    if not (0 <= range_min < range_max <= COUNTS - 1):
        raise CalibError(f"{prof.name}: homed range {range_min}..{range_max} is not in 0..4095")
    if prof.expected_range is not None and abs(width - prof.expected_range) > prof.range_tol:
        raise CalibError(
            f"{prof.name}: range {width} is outside {prof.expected_range} +- {prof.range_tol}"
        )
    return {
        "drive_mode": 0,
        "homing_offset": homing,
        "range_min": range_min,
        "range_max": range_max,
    }


def autocalibrate(
    bus: JointBus,
    profile: CalibProfile,
    order: list[str],
    manual_zero_raw: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """Calibrate the joints in `order`, one at a time, torque limited. Joints not in `order` are
    never touched. Each joint's torque stays on at its backoff pose. Releasing it is a live-step
    decision.
    """
    manual_zero_raw = manual_zero_raw or {}
    out: dict[str, dict[str, int]] = {}
    for name in order:
        jp = profile.joints.get(name)
        if jp is None:
            raise CalibError(f"{name}: no joint profile")
        offset = profile.zero_offsets_deg.get(name, 0.0)
        if jp.method == "manual_zero":
            if name not in manual_zero_raw:
                raise CalibError(f"{name}: manual_zero needs a zero_raw reading")
            out[name] = calibrate_joint(
                bus, jp, zero_offset_deg=offset, zero_raw=manual_zero_raw[name]
            )
        else:
            out[name] = calibrate_joint(bus, jp, zero_offset_deg=offset)
    return out


def fleet_profile(stats: str | Path | dict[str, Any]) -> CalibProfile:
    """Default profile from the fleet stats. Gripper is skipped. wrist_roll is manual_zero."""
    if not isinstance(stats, dict):
        stats = json.loads(Path(stats).read_text())
    joints: dict[str, JointProfile] = {}
    for name, s in stats.items():
        if name.startswith("_") or s.get("skip"):
            continue
        if name == "wrist_roll":  # full turn, no stops: take the zero from a held pose
            joints[name] = JointProfile(name=name, method="manual_zero")
            continue
        joints[name] = JointProfile(
            name=name,
            method="hard_stop",
            expected_range=round(s["mean"]),
            range_tol=max(round(3 * s["std"]), 40),
        )
    return CalibProfile(robot="so101_follower", joints=joints)
