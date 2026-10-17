"""Unit tests: auto-calibration core on the fake raw bus. No hardware.

LeRobot math is copied here, with citations, so the tests are independent of the installed version:
- `_get_half_turn_homings`: lerobot/motors/feetech/feetech.py:282-292 (`pos - int(max_res / 2)`)
- `Present = Actual - Homing`: lerobot/motors/feetech/feetech.py:282-286 (docstring)
- DEGREES `_normalize`: lerobot/motors/motors_bus.py:838-866 (`mid = (min + max) / 2`,
  `(val - mid) * 360 / max_res`, with `max_res = 4095`)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # fakes_calib lives next to this file

from fakes_calib import FakeJoint, FakeJointBus  # noqa: E402

from robot_manipulator.calib.autocal import (  # noqa: E402
    JOINT_IDS,
    CalibError,
    JointProfile,
    LeRobotJointBus,
    autocalibrate,
    calibrate_joint,
    fleet_profile,
    load_yaml,
    save_yaml,
)
from robot_manipulator.calib.autocal import _find_stop as find_stop_detail  # noqa: E402

STATS = Path(__file__).resolve().parents[1] / "config" / "calibration_stats_so101.json"
MAX_RES = 4095


def make_bus(seed: int = 0, noise: float = 1.0) -> FakeJointBus:
    return FakeJointBus(
        {
            # width 2700: a plain joint
            "shoulder_pan": FakeJoint(offset=2100, lo_link=-1350, hi_link=1350),
            # width 2222: another plain joint
            "elbow_flex": FakeJoint(offset=1800, lo_link=-1100, hi_link=1122),
            # width 2350: the stops straddle 4095 -> 0 in raw (3696 .. 1950)
            "shoulder_lift": FakeJoint(offset=200, lo_link=-600, hi_link=1750),
            # full turn, no stops
            "wrist_roll": FakeJoint(offset=3000, lo_link=-2000, hi_link=2000),
            "gripper": FakeJoint(offset=900, lo_link=-300, hi_link=1200),
        },
        seed=seed,
        noise=noise,
    )


def hard(name: str, width: int, **kw) -> JointProfile:
    return JointProfile(name=name, method="hard_stop", expected_range=width, range_tol=40, **kw)


# ---- LeRobot's math, copied with citations ------------------------------------------------


def lerobot_half_turn_homing(raw: int) -> int:
    """feetech.py:282-292 `_get_half_turn_homings`, for one motor: `pos - int(max_res / 2)`."""
    return raw - int(MAX_RES / 2)


def lerobot_present(actual: int, homing: int) -> int:
    """`Present_Position = Actual_Position - Homing_Offset`, modulo the 12-bit encoder."""
    return (actual - homing) % 4096


def lerobot_degrees(present: float, range_min: int, range_max: int) -> float:
    """motors_bus.py:838-866 `_normalize`, DEGREES branch."""
    mid = (range_min + range_max) / 2
    return (present - mid) * 360 / MAX_RES


# ---- tests --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "width"), [("shoulder_pan", 2700), ("elbow_flex", 2222), ("shoulder_lift", 2350)]
)
def test_recovers_range_within_backoff_plus_5(name, width):
    bus = make_bus()
    cal = calibrate_joint(bus, hard(name, width))
    recovered = cal["range_max"] - cal["range_min"]
    assert abs(recovered - width) <= 15 + 5  # backoff 15 counts, plus 5 of slack
    assert 0 <= cal["range_min"] < cal["range_max"] <= 4095


def test_wrapping_joint_is_homed_across_the_zero_crossing():
    bus = make_bus()
    lo_raw = bus.joints["shoulder_lift"].raw_stop("lo")  # 3696
    hi_raw = bus.joints["shoulder_lift"].raw_stop("hi")  # 1950
    assert lo_raw > hi_raw  # the wrap is really there
    cal = calibrate_joint(bus, hard("shoulder_lift", 2350))
    assert cal["range_max"] - cal["range_min"] == pytest.approx(2350, abs=20)


def test_homing_convention_matches_lerobot_and_stop_midpoint_reads_zero_degrees():
    # noise-free bus: with +-2 count noise the stop estimate can be off by one count (see next note)
    bus = make_bus(noise=0.0)
    for name, width in [("shoulder_pan", 2700), ("elbow_flex", 2222), ("shoulder_lift", 2350)]:
        joint = bus.joints[name]
        lo_raw, hi_raw = joint.raw_stop("lo"), joint.raw_stop("hi")
        # the midpoint in raw counts; midpoint of the wrapped interval when it straddles 0
        mid_link = (joint.lo_link + joint.hi_link) / 2
        mid_raw = round(mid_link + joint.offset) % 4096
        cal = calibrate_joint(bus, hard(name, width))
        h, rmin, rmax = cal["homing_offset"], cal["range_min"], cal["range_max"]

        # 1. same homing as LeRobot's half-turn rule, modulo the 12-bit encoder
        assert (lerobot_half_turn_homing(mid_raw) - h) % 4096 == 0, name
        # 2. the stop midpoint reads 2047 after homing
        assert lerobot_present(mid_raw, h) == 2047, name
        # 3. LeRobot DEGREES: the stop midpoint is 0.0 deg (half a count of rounding allowed)
        assert abs(lerobot_degrees(lerobot_present(mid_raw, h), rmin, rmax)) < 0.1, name
        # 4. the stops land at -/+ half the range
        lo_deg = lerobot_degrees(lerobot_present(lo_raw, h), rmin, rmax)
        hi_deg = lerobot_degrees(lerobot_present(hi_raw, h), rmin, rmax)
        half = (rmax - rmin) / 2 * 360 / MAX_RES
        assert lo_deg == pytest.approx(-half, abs=0.3), name
        assert hi_deg == pytest.approx(+half, abs=0.3), name


def test_stall_triggers_on_load_before_position_error_when_load_abort_is_low():
    bus = make_bus()
    prof = JointProfile(
        name="shoulder_pan", method="hard_stop", load_abort=60, stall_err=200, stall_ticks=10
    )
    _, reason = find_stop_detail(bus, "shoulder_pan", +1, prof, max_ticks=500)
    assert reason == "load"


def test_stall_triggers_on_position_error_when_load_abort_is_high():
    bus = make_bus()
    prof = JointProfile(
        name="shoulder_pan", method="hard_stop", load_abort=10_000, stall_err=40, stall_ticks=10
    )
    _, reason = find_stop_detail(bus, "shoulder_pan", +1, prof, max_ticks=500)
    assert reason == "error"


def test_range_out_of_tolerance_raises_calib_error():
    bus = make_bus()
    with pytest.raises(CalibError, match="outside"):
        calibrate_joint(bus, hard("shoulder_pan", 1000))


def test_manual_zero_puts_the_held_pose_at_2047():
    bus = make_bus()
    prof = JointProfile(name="wrist_roll", method="manual_zero")
    cal = calibrate_joint(bus, prof, zero_raw=1234)
    assert lerobot_present(1234, cal["homing_offset"]) == 2047
    assert (cal["range_min"], cal["range_max"]) == (0, 4095)
    # the held pose reads 0.0 deg within half a count (2047.5 is the true full-turn midpoint)
    assert abs(lerobot_degrees(lerobot_present(1234, cal["homing_offset"]), 0, 4095)) < 0.1


def test_manual_zero_needs_a_reading():
    with pytest.raises(CalibError, match="zero_raw"):
        calibrate_joint(make_bus(), JointProfile(name="wrist_roll", method="manual_zero"))


def test_max_ticks_raises():
    bus = make_bus()
    prof = JointProfile(name="shoulder_pan", method="hard_stop")
    with pytest.raises(CalibError, match="no stop found"):
        find_stop_detail(bus, "shoulder_pan", +1, prof, max_ticks=3)


def test_profile_yaml_round_trips(tmp_path):
    profile = fleet_profile(STATS)
    path = save_yaml(profile, tmp_path / "profile.yaml")
    assert load_yaml(path) == profile


def test_fleet_profile_matches_the_plan():
    profile = fleet_profile(STATS)
    assert set(profile.joints) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    }
    assert profile.joints["wrist_roll"].method == "manual_zero"
    pan = profile.joints["shoulder_pan"]
    assert pan.expected_range == 2725 and pan.range_tol == 96  # max(3 * 31.88, 40)
    assert profile.joints["elbow_flex"].range_tol == 40  # max(3 * 9.17, 40)


def test_gripper_is_never_touched_unless_in_order():
    bus = make_bus()
    profile = fleet_profile(STATS)
    autocalibrate(bus, profile, order=["shoulder_pan", "elbow_flex"], manual_zero_raw={})
    assert "gripper" not in bus.touched
    assert {"shoulder_pan", "elbow_flex"} <= bus.touched


def test_autocalibrate_returns_full_entries_for_each_joint():
    bus = make_bus()
    profile = fleet_profile(STATS)
    result = autocalibrate(
        bus,
        profile,
        order=["shoulder_pan", "elbow_flex", "shoulder_lift", "wrist_roll"],
        manual_zero_raw={"wrist_roll": 2000},
    )
    assert set(result) == {"shoulder_pan", "elbow_flex", "shoulder_lift", "wrist_roll"}
    for entry in result.values():
        assert set(entry) == {"drive_mode", "homing_offset", "range_min", "range_max"}
        assert abs(entry["homing_offset"]) <= 2047
    assert result["wrist_roll"]["range_max"] == 4095
    # torque limit was set before the stop search
    assert bus.torque_limits["shoulder_pan"] == 250


def test_nonzero_zero_offset_is_refused_not_faked():
    with pytest.raises(CalibError, match="not supported"):
        calibrate_joint(make_bus(), hard("shoulder_pan", 2700), zero_offset_deg=2.0)


def test_live_bus_stub_refuses_to_construct():
    with pytest.raises(NotImplementedError, match="reviewed with the user first"):
        LeRobotJointBus("/dev/so101_follower")


def test_joint_ids_match_the_cache_files():
    assert JOINT_IDS["shoulder_pan"] == 1 and JOINT_IDS["gripper"] == 6


@pytest.mark.parametrize("seed", range(4))
def test_midpoint_estimate_is_within_three_counts_under_read_noise(seed):
    """With +-2 count read noise, the stop midpoint still lands within 3 counts of 2047."""
    bus = make_bus(seed=seed)
    for name, width in [("shoulder_pan", 2700), ("elbow_flex", 2222), ("shoulder_lift", 2350)]:
        joint = bus.joints[name]
        mid_raw = round((joint.lo_link + joint.hi_link) / 2 + joint.offset) % 4096
        cal = calibrate_joint(bus, hard(name, width))
        assert abs(lerobot_present(mid_raw, cal["homing_offset"]) - 2047) <= 3, (name, seed)
