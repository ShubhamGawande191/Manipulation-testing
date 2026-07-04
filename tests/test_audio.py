"""Unit tests: wav write/read and episode-clock alignment (no hardware)."""

import numpy as np
import pytest

from robot_manipulator.io.audio import AudioClip, read_wav, write_wav


def test_wav_roundtrip_int16(tmp_path):
    sr = 16000
    t = np.arange(sr // 10) / sr
    samples = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    clip = AudioClip(samples, sr, t0=3.25)
    path = write_wav(tmp_path / "ep.wav", clip)
    back = read_wav(path, t0=3.25)
    assert back.samplerate == sr
    np.testing.assert_array_equal(back.samples, samples)
    assert back.duration_s() == pytest.approx(0.1)


def test_float_input_is_converted():
    clip = AudioClip(np.array([0.0, 0.5, -2.0]), 8000, t0=0.0)  # -2.0 must clip

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        back = read_wav(write_wav(Path(d) / "f.wav", clip))
    assert back.samples.dtype == np.int16
    assert back.samples[2] == -32767  # clipped, not wrapped


def test_sample_time_alignment():
    clip = AudioClip(np.zeros(16000, dtype=np.int16), 16000, t0=10.0)
    assert clip.sample_time(0) == 10.0
    assert clip.sample_time(8000) == pytest.approx(10.5)
