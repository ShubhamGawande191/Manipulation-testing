"""Unit tests: MIDI scorer over task suite v0."""

import pytest

from robot_manipulator.eval.scorer import (
    EncoderDelta,
    FaderTarget,
    KeySequence,
    PressHold,
    PressKey,
    score,
)
from robot_manipulator.io.midi import CONTROL_CHANGE, NOTE_OFF, NOTE_ON, MidiEvent


def on(t, note, vel=100):
    return MidiEvent(t, NOTE_ON, 0, note, vel)


def off(t, note):
    return MidiEvent(t, NOTE_OFF, 0, note, 0)


def cc(t, num, val):
    return MidiEvent(t, CONTROL_CHANGE, 0, num, val)


def test_press_key_success_and_metrics():
    r = score(PressKey(60, target_velocity=90), [on(1.2, 60, 80)])
    assert r.success and r.time_to_success_s == 1.2
    assert r.key_press_f1 == 1.0
    assert r.velocity_error == 10.0


def test_press_key_wrong_key_fails_with_partial_f1():
    r = score(PressKey(60), [on(0.5, 61)])
    assert not r.success and r.key_press_f1 == 0.0
    r2 = score(PressKey(60), [on(0.5, 61), on(0.7, 60)])
    assert r2.success and 0 < r2.key_press_f1 < 1


def test_key_sequence_in_order():
    ok = score(KeySequence((60, 62, 64)), [on(0.1, 60), on(0.3, 62), on(0.5, 64)])
    assert ok.success and ok.time_to_success_s == 0.5 and ok.key_press_f1 == 1.0
    # right keys, wrong order -> not success, but F1 still 1.0 (all keys present)
    bad = score(KeySequence((60, 62, 64)), [on(0.1, 64), on(0.3, 62), on(0.5, 60)])
    assert not bad.success and bad.key_press_f1 == 1.0


def test_press_hold():
    held = score(PressHold(60, 0.5), [on(0.0, 60), off(0.8, 60)])
    assert held.success and held.time_to_success_s == pytest.approx(0.5)
    short = score(PressHold(60, 0.5), [on(0.0, 60), off(0.2, 60)])
    assert not short.success
    # never released: hold runs to t_end
    open_ended = score(PressHold(60, 0.5), [on(0.0, 60)], t_end=1.0)
    assert open_ended.success


def test_fader_target():
    trace = [cc(0.1, 14, 10), cc(0.2, 14, 60), cc(0.3, 14, 99), cc(0.4, 14, 101)]
    r = score(FaderTarget(14, 100, tol=3), trace)
    assert r.success and r.cc_error == 1.0
    assert r.time_to_success_s == 0.3  # entered tolerance at 99 and stayed
    miss = score(FaderTarget(14, 100, tol=3), trace[:2])
    assert not miss.success and miss.cc_error == 40.0


def test_fader_overshoot_resets_time_to_success():
    trace = [cc(0.1, 14, 99), cc(0.2, 14, 120), cc(0.3, 14, 100)]
    r = score(FaderTarget(14, 100, tol=3), trace)
    assert r.success and r.time_to_success_s == 0.3


def test_encoder_delta():
    trace = [cc(0.1, 15, 50), cc(0.5, 15, 58), cc(0.9, 15, 61)]
    r = score(EncoderDelta(15, 10, tol=3), trace)
    assert r.success and r.cc_error == 1.0
    r2 = score(EncoderDelta(15, -10, tol=3), trace)
    assert not r2.success


def test_no_events():
    assert not score(PressKey(60), []).success
    assert not score(FaderTarget(14, 100), []).success
