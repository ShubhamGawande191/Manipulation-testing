"""Optional per-episode audio capture (PLAN item 4).

Raw wav first; spectrogram features come later. Capture needs the optional
`audio` extra (sounddevice); writing/reading wav and the timestamp alignment
math are stdlib + numpy so they stay unit-testable without hardware.
"""

from __future__ import annotations

import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioClip:
    samples: np.ndarray  # (n,) or (n, channels), int16
    samplerate: int
    t0: float  # episode-clock time of samples[0]

    def sample_time(self, i: int) -> float:
        """Episode-clock time of sample i."""
        return self.t0 + i / self.samplerate

    def duration_s(self) -> float:
        return len(self.samples) / self.samplerate


def write_wav(path: str | Path, clip: AudioClip) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(clip.samples)
    if samples.dtype != np.int16:
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(clip.samplerate)
        w.writeframes(samples.tobytes())
    return path


def read_wav(path: str | Path, t0: float = 0.0) -> AudioClip:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        if w.getnchannels() > 1:
            samples = samples.reshape(-1, w.getnchannels())
        return AudioClip(samples, w.getframerate(), t0)


class EpisodeAudioRecorder:
    """USB mic capture aligned to the episode clock.

    start() notes the episode-clock time of the first buffer so scorer/analysis
    code can line audio up against MIDI events.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._clock = clock
        self._blocks: list[np.ndarray] = []
        self._t0: float | None = None
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:  # pragma: no cover - needs optional extra
            raise RuntimeError("audio capture needs `uv sync --extra audio`") from e

        self._blocks = []
        self._t0 = None

        def callback(indata, frames, t, status):  # pragma: no cover - hw callback
            if self._t0 is None:
                self._t0 = self._clock()
            self._blocks.append(np.array(indata, dtype=np.int16, copy=True))

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> AudioClip:
        if self._stream is not None:  # pragma: no cover - hw teardown
            self._stream.stop()
            self._stream.close()
            self._stream = None
        samples = (
            np.concatenate(self._blocks)
            if self._blocks
            else np.zeros((0, self.channels), dtype=np.int16)
        )
        if self.channels == 1:
            samples = samples.reshape(-1)
        return AudioClip(samples, self.samplerate, self._t0 or 0.0)
