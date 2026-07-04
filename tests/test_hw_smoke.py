"""Hardware smoke tests — need the bench plugged in. Run manually:

uv run pytest -m hw
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.hw


def test_minilab3_midi_input_visible():
    mido = pytest.importorskip("mido")
    names = mido.get_input_names()
    assert any("minilab" in n.lower() for n in names), f"MiniLab 3 not found in {names}"


def test_follower_serial_symlink_present():
    assert Path("/dev/so101_follower").exists()


def test_workspace_and_wrist_cameras_present():
    assert Path("/dev/video0").exists()
