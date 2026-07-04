"""Unit tests: press-key task config round trip (no hardware)."""

from robot_manipulator.tasks.press_key import load_keys, save_key


def test_keys_file_roundtrip(tmp_path):
    path = tmp_path / "minilab_keys.json"
    hover = [0.0, -90.0, 90.0, 30.0, 0.0, 0.5]
    press = [0.0, -92.0, 91.0, 35.0, 0.0, 0.5]
    save_key(48, hover, press, path)
    save_key(50, hover, press, path)
    keys = load_keys(path)
    assert set(keys) == {"48", "50"}
    assert keys["48"]["hover"] == hover and keys["48"]["press"] == press


def test_load_missing_file_is_empty(tmp_path):
    assert load_keys(tmp_path / "nope.json") == {}
