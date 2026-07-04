"""Unit tests: bench-check diagnosis logic (no hardware)."""

from robot_manipulator.scripts.bench_check import EMEET, FOLLOWER, MINILAB, build_report

GOOD_USB = [
    {"vid": FOLLOWER[0], "pid": FOLLOWER[1], "serial": FOLLOWER[2], "product": "USB Single Serial"},
    {"vid": MINILAB[0], "pid": MINILAB[1], "serial": None, "product": "Minilab3"},
    {"vid": EMEET[0], "pid": EMEET[1], "serial": None, "product": "eMeet C950"},
    {"vid": EMEET[0], "pid": EMEET[1], "serial": None, "product": "eMeet C950"},
]
GOOD_CAMS = [
    {"index": 0, "name": "eMeet", "captures": True, "resolution": "640x480"},
    {"index": 2, "name": "eMeet", "captures": True, "resolution": "640x480"},
]
GOOD_MIDI = ["Minilab3:Minilab3 MIDI 20:0"]


def test_all_good():
    r = build_report(GOOD_USB, [], GOOD_CAMS, GOOD_MIDI, follower_symlink_exists=True)
    assert r.ok


def test_arm_not_enumerating_is_diagnosed_as_physical():
    usb = [d for d in GOOD_USB if d["vid"] != FOLLOWER[0]]
    r = build_report(usb, [], GOOD_CAMS, GOOD_MIDI, follower_symlink_exists=False)
    assert not r.ok
    assert any("NOT enumerating" in p for p in r.problems)


def test_arm_enumerated_but_symlink_missing_is_diagnosed_as_udev():
    r = build_report(GOOD_USB, [], GOOD_CAMS, GOOD_MIDI, follower_symlink_exists=False)
    assert any("udev" in p for p in r.problems)


def test_dead_camera_flagged():
    cams = [GOOD_CAMS[0], {"index": 2, "name": "eMeet", "captures": False, "resolution": None}]
    r = build_report(GOOD_USB, [], cams, GOOD_MIDI, follower_symlink_exists=True)
    assert any("2 capturing cameras" in p for p in r.problems)


def test_minilab_present_but_no_midi_port():
    r = build_report(GOOD_USB, [], GOOD_CAMS, [], follower_symlink_exists=True)
    assert any("no MIDI input" in p for p in r.problems)
