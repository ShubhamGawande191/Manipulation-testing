"""Unit tests: MIDI encode/decode round-trip, rolling state, event log."""

import mido
import numpy as np

from robot_manipulator.io.midi import (
    CONTROL_CHANGE,
    NOTE_OFF,
    NOTE_ON,
    EventLog,
    MidiEvent,
    MidiState,
    from_mido,
    to_mido,
)


def test_roundtrip_note_and_cc():
    for ev in [
        MidiEvent(0.5, NOTE_ON, 0, 60, 100),
        MidiEvent(1.0, NOTE_OFF, 0, 60, 0),
        MidiEvent(2.0, CONTROL_CHANGE, 0, 7, 42),
    ]:
        back = from_mido(to_mido(ev), ev.t)
        assert back == ev


def test_note_on_velocity_zero_normalizes_to_note_off():
    msg = mido.Message("note_on", note=64, velocity=0)
    ev = from_mido(msg, 0.0)
    assert ev is not None and ev.kind == NOTE_OFF


def test_untracked_kinds_return_none():
    assert from_mido(mido.Message("program_change", program=5), 0.0) is None


def test_state_vector():
    st = MidiState()
    st.update(MidiEvent(0.0, NOTE_ON, 0, 60, 127))
    st.update(MidiEvent(0.1, CONTROL_CHANGE, 0, 7, 64))
    v = st.vector()
    assert v.shape == (256,)
    assert v[60] == 1.0
    assert np.isclose(v[128 + 7], 64 / 127)
    assert st.active_notes() == {60}
    st.update(MidiEvent(0.2, NOTE_OFF, 0, 60, 0))
    assert st.vector()[60] == 0.0
    assert st.last_event_t == 0.2


def test_event_log_between_and_jsonl_roundtrip(tmp_path):
    log = EventLog()
    for t in (0.1, 0.5, 1.5):
        log.append(MidiEvent(t, NOTE_ON, 0, 60, 90))
    assert len(log.between(0.0, 1.0)) == 2
    path = log.save(tmp_path / "events.jsonl")
    assert EventLog.load(path).events == log.events
