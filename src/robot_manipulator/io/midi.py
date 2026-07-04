"""MIDI event log + rolling state vector (PLAN item 3).

The MiniLab 3 is both the task surface and the ground-truth sensor: every key
press, fader move, and encoder turn arrives here as a `MidiEvent`. Two
consumers:

- `EventLog` — append-only, timestamped on the episode clock; the scorer's
  input and part of the episode artifact.
- `MidiState` — rolling state vector (128 note velocities + 128 CC values,
  normalized to [0, 1]) used as a proprio-like observation feature.

Encode/decode to mido messages is unit-tested round-trip. Live capture
(`MidiListener`) needs the python-rtmidi backend (`uv sync --extra midi`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

NOTE_ON = "note_on"
NOTE_OFF = "note_off"
CONTROL_CHANGE = "control_change"
KINDS = (NOTE_ON, NOTE_OFF, CONTROL_CHANGE)


@dataclass(frozen=True)
class MidiEvent:
    t: float  # seconds on the episode clock
    kind: str  # one of KINDS
    channel: int
    number: int  # note number or CC number
    value: int  # velocity or CC value, 0-127


def from_mido(msg, t: float) -> MidiEvent | None:
    """Convert a mido message; returns None for kinds we don't track.

    Per MIDI convention, note_on with velocity 0 is normalized to note_off.
    """
    if msg.type == NOTE_ON:
        kind = NOTE_OFF if msg.velocity == 0 else NOTE_ON
        return MidiEvent(t, kind, msg.channel, msg.note, msg.velocity)
    if msg.type == NOTE_OFF:
        return MidiEvent(t, NOTE_OFF, msg.channel, msg.note, msg.velocity)
    if msg.type == CONTROL_CHANGE:
        return MidiEvent(t, CONTROL_CHANGE, msg.channel, msg.control, msg.value)
    return None


def to_mido(ev: MidiEvent):
    """Convert back to a mido message (t is not carried — mido time is delta)."""
    import mido

    if ev.kind in (NOTE_ON, NOTE_OFF):
        return mido.Message(ev.kind, channel=ev.channel, note=ev.number, velocity=ev.value)
    if ev.kind == CONTROL_CHANGE:
        return mido.Message(CONTROL_CHANGE, channel=ev.channel, control=ev.number, value=ev.value)
    raise ValueError(f"unsupported kind: {ev.kind}")


class MidiState:
    """Rolling state: current note velocities and CC values, as a flat vector."""

    def __init__(self) -> None:
        self.notes = np.zeros(128, dtype=np.float32)  # velocity/127 while held, else 0
        self.cc = np.zeros(128, dtype=np.float32)  # value/127, persists
        self.last_event_t: float | None = None

    def update(self, ev: MidiEvent) -> None:
        if ev.kind == NOTE_ON:
            self.notes[ev.number] = ev.value / 127.0
        elif ev.kind == NOTE_OFF:
            self.notes[ev.number] = 0.0
        elif ev.kind == CONTROL_CHANGE:
            self.cc[ev.number] = ev.value / 127.0
        self.last_event_t = ev.t

    def vector(self) -> np.ndarray:
        """Shape (256,): notes[0:128] then cc[128:256]."""
        return np.concatenate([self.notes, self.cc])

    def active_notes(self) -> set[int]:
        return set(np.flatnonzero(self.notes).tolist())


class EventLog:
    """Append-only event log, saved as JSONL next to the episode artifacts."""

    def __init__(self) -> None:
        self.events: list[MidiEvent] = []

    def append(self, ev: MidiEvent) -> None:
        self.events.append(ev)

    def between(self, t0: float, t1: float) -> list[MidiEvent]:
        return [e for e in self.events if t0 <= e.t < t1]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for ev in self.events:
                f.write(json.dumps(asdict(ev)) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> EventLog:
        log = cls()
        with Path(path).open() as f:
            for line in f:
                log.append(MidiEvent(**json.loads(line)))
        return log


class MidiListener:
    """Live capture from a MIDI input port onto the episode clock.

    Polling model: call `poll()` from the control loop (or a thread). Updates
    both the event log and the rolling state.
    """

    def __init__(
        self,
        port_substring: str = "Minilab3",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port_substring = port_substring
        self._clock = clock
        self.t0: float | None = None
        self.log = EventLog()
        self.state = MidiState()
        self._port = None

    @staticmethod
    def available_ports() -> list[str]:
        import mido

        return mido.get_input_names()

    def open(self) -> str:
        import mido

        names = [n for n in mido.get_input_names() if self.port_substring.lower() in n.lower()]
        if not names:
            raise RuntimeError(
                f"no MIDI input matching {self.port_substring!r}; "
                f"available: {mido.get_input_names()}"
            )
        self._port = mido.open_input(names[0])
        self.t0 = self._clock()
        return names[0]

    def poll(self) -> list[MidiEvent]:
        """Drain pending messages; returns the newly captured events."""
        if self._port is None or self.t0 is None:
            raise RuntimeError("call open() first")
        new: list[MidiEvent] = []
        for msg in self._port.iter_pending():
            ev = from_mido(msg, self._clock() - self.t0)
            if ev is not None:
                self.log.append(ev)
                self.state.update(ev)
                new.append(ev)
        return new

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None
