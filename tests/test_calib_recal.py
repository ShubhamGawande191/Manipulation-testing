import pytest

from robot_manipulator.calib.recal import (
    HOMING_LIMIT,
    homing_to_read,
    merge_calibration,
    read_deg,
)


@pytest.mark.parametrize("actual", [0, 37, 1024, 2047, 3000, 4095])
@pytest.mark.parametrize("target", [-179.0, -71.8, 0.0, 45.5, 170.0])
def test_homing_makes_joint_read_target(actual, target):
    h = homing_to_read(actual, target)
    assert abs(h) <= HOMING_LIMIT
    assert read_deg(actual, h) == pytest.approx(target, abs=2 * 360 / 4095)  # ≤ 2 counts


def test_read_deg_matches_lerobot_degrees_formula():
    # LeRobot DEGREES on range 0..4095: (present - mid) * 360 / max_res, present = actual - homing
    assert read_deg(2047, 0) == pytest.approx((2047 - 2047.5) * 360 / 4095)
    assert read_deg(1000, -500) == pytest.approx((1500 - 2047.5) * 360 / 4095)


def test_merge_keeps_untouched_joints_and_drive_mode():
    old = {
        "shoulder_pan": {
            "id": 1,
            "drive_mode": 0,
            "homing_offset": -1964,
            "range_min": 1236,
            "range_max": 3141,
        },
        "gripper": {
            "id": 6,
            "drive_mode": 1,
            "homing_offset": 1921,
            "range_min": 2005,
            "range_max": 3200,
        },
    }
    new = {"shoulder_pan": {"homing_offset": -300, "range_min": 690, "range_max": 3400}}
    m = merge_calibration(old, new)
    assert m["gripper"] == old["gripper"]
    assert m["shoulder_pan"] == {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -300,
        "range_min": 690,
        "range_max": 3400,
    }
    assert old["shoulder_pan"]["homing_offset"] == -1964  # input not mutated
