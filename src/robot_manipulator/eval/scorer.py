"""MIDI ground truth -> scores (PLAN item 6).

Task suite v0: press key K · 3-key sequence · press-and-hold · fader -> target
CC value · encoder -> delta-CC target. This module is the single scoring path
for every phase; the outputs flatten into `core.episode_log.EpisodeRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_manipulator.io.midi import CONTROL_CHANGE, NOTE_OFF, NOTE_ON, MidiEvent


@dataclass(frozen=True)
class PressKey:
    note: int
    target_velocity: int | None = None


@dataclass(frozen=True)
class KeySequence:
    notes: tuple[int, ...]  # must appear in order (other presses allowed between)
    target_velocity: int | None = None


@dataclass(frozen=True)
class PressHold:
    note: int
    min_hold_s: float


@dataclass(frozen=True)
class FaderTarget:
    cc: int
    target: int
    tol: int = 3


@dataclass(frozen=True)
class EncoderDelta:
    cc: int
    delta: int  # net change of the CC value over the episode
    tol: int = 3


Task = PressKey | KeySequence | PressHold | FaderTarget | EncoderDelta


@dataclass
class ScoreResult:
    task: str
    success: bool
    key_press_f1: float | None = None
    velocity_error: float | None = None  # MAE vs target velocity, matched notes only
    cc_error: float | None = None  # |final - target| (fader) or |net - delta| (encoder)
    time_to_success_s: float | None = None


def _note_ons(events: list[MidiEvent]) -> list[MidiEvent]:
    return [e for e in events if e.kind == NOTE_ON]


def _f1(expected: list[int], played: list[int]) -> float:
    """Multiset F1 between expected and played note numbers."""
    if not played and not expected:
        return 1.0
    if not played or not expected:
        return 0.0
    exp = list(expected)
    matched = 0
    for n in played:
        if n in exp:
            exp.remove(n)
            matched += 1
    precision = matched / len(played)
    recall = matched / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _velocity_error(target: int | None, matched: list[MidiEvent]) -> float | None:
    if target is None or not matched:
        return None
    return float(np.mean([abs(e.value - target) for e in matched]))


def _hold_spans(events: list[MidiEvent], note: int, t_end: float) -> list[tuple[float, float]]:
    """(on, off) spans for one note; an unreleased press runs to t_end."""
    spans: list[tuple[float, float]] = []
    t_on: float | None = None
    for e in events:
        if e.number != note:
            continue
        if e.kind == NOTE_ON and t_on is None:
            t_on = e.t
        elif e.kind == NOTE_OFF and t_on is not None:
            spans.append((t_on, e.t))
            t_on = None
    if t_on is not None:
        spans.append((t_on, t_end))
    return spans


def score(task: Task, events: list[MidiEvent], t_end: float | None = None) -> ScoreResult:
    """Score one episode's event log against one task."""
    t_end = t_end if t_end is not None else (events[-1].t if events else 0.0)
    ons = _note_ons(events)

    if isinstance(task, PressKey):
        hits = [e for e in ons if e.number == task.note]
        return ScoreResult(
            task=f"press_key:{task.note}",
            success=bool(hits),
            key_press_f1=_f1([task.note], [e.number for e in ons]),
            velocity_error=_velocity_error(task.target_velocity, hits),
            time_to_success_s=hits[0].t if hits else None,
        )

    if isinstance(task, KeySequence):
        want = list(task.notes)
        i = 0
        t_done: float | None = None
        matched: list[MidiEvent] = []
        for e in ons:
            if i < len(want) and e.number == want[i]:
                matched.append(e)
                i += 1
                if i == len(want):
                    t_done = e.t
        return ScoreResult(
            task=f"key_sequence:{','.join(map(str, task.notes))}",
            success=t_done is not None,
            key_press_f1=_f1(want, [e.number for e in ons]),
            velocity_error=_velocity_error(task.target_velocity, matched),
            time_to_success_s=t_done,
        )

    if isinstance(task, PressHold):
        spans = _hold_spans(events, task.note, t_end)
        good = [(a, b) for a, b in spans if b - a >= task.min_hold_s]
        return ScoreResult(
            task=f"press_hold:{task.note}x{task.min_hold_s}s",
            success=bool(good),
            key_press_f1=_f1([task.note], [e.number for e in ons]),
            time_to_success_s=(good[0][0] + task.min_hold_s) if good else None,
        )

    if isinstance(task, FaderTarget):
        trace = [e for e in events if e.kind == CONTROL_CHANGE and e.number == task.cc]
        if not trace:
            return ScoreResult(task=f"fader:{task.cc}->{task.target}", success=False)
        final = trace[-1].value
        err = abs(final - task.target)
        # first time the value enters tolerance and stays there to the end
        t_ok: float | None = None
        for e in trace:
            if abs(e.value - task.target) <= task.tol:
                if t_ok is None:
                    t_ok = e.t
            else:
                t_ok = None
        return ScoreResult(
            task=f"fader:{task.cc}->{task.target}",
            success=err <= task.tol,
            cc_error=float(err),
            time_to_success_s=t_ok,
        )

    if isinstance(task, EncoderDelta):
        trace = [e for e in events if e.kind == CONTROL_CHANGE and e.number == task.cc]
        if len(trace) < 1:
            return ScoreResult(task=f"encoder:{task.cc}+{task.delta}", success=False)
        # net change relative to the first observed value (absolute-mode encoder)
        net = trace[-1].value - trace[0].value
        err = abs(net - task.delta)
        return ScoreResult(
            task=f"encoder:{task.cc}+{task.delta}",
            success=err <= task.tol,
            cc_error=float(err),
            time_to_success_s=trace[-1].t if err <= task.tol else None,
        )

    raise TypeError(f"unknown task type: {type(task)!r}")
