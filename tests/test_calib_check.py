"""Unit tests: calibration checker (fixtures copied from the cache files and the fleet stats)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_manipulator.calib.check import (
    app,
    fleet_check,
    frame_issues,
    frame_mismatch,
    load_calibration,
    load_stats,
    range_counts,
    zero_raw,
)

STATS_PATH = Path(__file__).resolve().parents[1] / "config" / "calibration_stats_so101.json"

# Copied from ~/.cache/huggingface/lerobot/calibration/ (read-only) on 2026-10-17.
FOLLOWER = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -1964,
        "range_min": 1236,
        "range_max": 3141,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -1512,
        "range_min": 918,
        "range_max": 3394,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -1636,
        "range_min": 865,
        "range_max": 2993,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -1707,
        "range_min": 913,
        "range_max": 3199,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": -2,
        "range_min": 0,
        "range_max": 4095,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 1,
        "homing_offset": 1921,
        "range_min": 2005,
        "range_max": 3200,
    },
}
LEADER = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -381,
        "range_min": 690,
        "range_max": 3402,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -1026,
        "range_min": 796,
        "range_max": 3411,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": 1947,
        "range_min": 827,
        "range_max": 3035,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -460,
        "range_min": 874,
        "range_max": 3231,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": -1253,
        "range_min": 0,
        "range_max": 4095,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 1037,
        "range_min": 2024,
        "range_max": 3307,
    },
}


@pytest.fixture(scope="module")
def stats():
    return load_stats(STATS_PATH)


def test_stats_file_is_attributed_and_skips_gripper():
    raw = json.loads(STATS_PATH.read_text())
    assert "Apache-2.0" in raw["_source"] and "n=15" in raw["_source"]
    assert raw["gripper"]["skip"] is True
    assert "_source" not in load_stats(STATS_PATH)


def test_range_counts_match_the_cache_numbers():
    assert range_counts(FOLLOWER)["shoulder_pan"] == 1905
    assert range_counts(LEADER)["shoulder_pan"] == 2712
    assert range_counts(FOLLOWER)["elbow_flex"] == 2128


def test_follower_pan_is_flagged(stats):
    issues = fleet_check(FOLLOWER, stats)
    pan = [i for i in issues if i.joint == "shoulder_pan" and i.kind == "fleet_range"]
    assert pan, "follower pan should be flagged"
    assert abs(pan[0].value) > 2.0  # about -25.7 sigma


def test_leader_pan_is_not_flagged(stats):
    issues = fleet_check(LEADER, stats)
    assert not [i for i in issues if i.joint == "shoulder_pan"]


def test_wrist_roll_homing_difference_is_about_110_deg():
    offsets = frame_mismatch(FOLLOWER, LEADER)
    assert offsets["wrist_roll"] == pytest.approx(110.0, abs=0.1)


def test_frame_mismatch_only_full_turn_joints():
    # raw counts of range-defined joints are not comparable across physical arms
    offsets = frame_mismatch(FOLLOWER, LEADER)
    assert set(offsets) == {"wrist_roll"}
    assert {i.joint for i in frame_issues(offsets)} == {"wrist_roll"}


def test_zero_raw_follows_lerobot_convention():
    # Present = Actual - Homing, and DEGREES zero is the midpoint of the calibrated range.
    assert zero_raw(FOLLOWER["shoulder_pan"]) == pytest.approx((1236 + 3141) / 2 - 1964)


def test_skipped_gripper_never_flags(stats):
    wild = {
        **LEADER,
        "gripper": {
            "id": 6,
            "drive_mode": 0,
            "homing_offset": 4000,
            "range_min": 0,
            "range_max": 9000,
        },
    }
    assert not [i for i in fleet_check(wild, stats) if i.joint == "gripper"]
    assert "gripper" not in frame_mismatch(wild, LEADER)


def test_load_calibration_rejects_missing_fields(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"shoulder_pan": {"id": 1}}))
    with pytest.raises(ValueError, match="missing"):
        load_calibration(bad)


def _write(path: Path, cal: dict) -> Path:
    path.write_text(json.dumps(cal))
    return path


def test_cli_exits_1_on_the_real_numbers(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "--follower",
            str(_write(tmp_path / "f.json", FOLLOWER)),
            "--leader",
            str(_write(tmp_path / "l.json", LEADER)),
            "--stats",
            str(STATS_PATH),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "shoulder_pan" in result.output


def test_cli_exits_0_on_a_clean_pair(tmp_path, stats):
    clean = {}
    for joint, s in stats.items():
        if s.get("skip"):
            clean[joint] = {
                "id": 0,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 0,
            }
        elif joint == "wrist_roll":
            clean[joint] = {
                "id": 5,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
        else:
            clean[joint] = {
                "id": 0,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 500,
                "range_max": 500 + round(s["mean"]),
            }
    result = CliRunner().invoke(
        app,
        [
            "--follower",
            str(_write(tmp_path / "f.json", clean)),
            "--leader",
            str(_write(tmp_path / "l.json", clean)),
            "--stats",
            str(STATS_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no issues" in result.output


def test_frame_only_difference_does_not_set_exit_code(tmp_path, stats):
    """A 110 deg wrist_roll zero difference is reported, but it is indicative and not an issue."""
    clean = {}
    for joint, s in stats.items():
        if s.get("skip") or joint == "wrist_roll":
            clean[joint] = {
                "id": 0,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
            if joint == "gripper":
                clean[joint] = {
                    "id": 0,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": 0,
                    "range_max": 0,
                }
        else:
            clean[joint] = {
                "id": 0,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 500,
                "range_max": 500 + round(s["mean"]),
            }
    off_zero = {**clean, "wrist_roll": {**clean["wrist_roll"], "homing_offset": -1253}}
    result = CliRunner().invoke(
        app,
        [
            "--follower",
            str(_write(tmp_path / "f.json", off_zero)),
            "--leader",
            str(_write(tmp_path / "l.json", clean)),
            "--stats",
            str(STATS_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrist_roll" in result.output and "-110.1" in result.output  # follower minus leader
