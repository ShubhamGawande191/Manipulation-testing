import pytest

from robot_manipulator import profile

ARMS = """
arms:
  default: {}
  left:
    port: /dev/so101_left_follower
    robot_id: left_follower
    joint_map: config/sim2real_left.yaml
    wrist_camera: left_wrist
    wrist_mount: config/extrinsics/left_wrist_mount.yaml
"""


def test_without_file_single_arm_defaults(tmp_path):
    a = profile.arm(path=tmp_path / "missing.yaml")
    assert a.name == "default"
    assert a.port == "/dev/so101_follower" and a.robot_id == "my_so101_follower"
    assert a.joint_map == profile.REPO / "config" / "sim2real.yaml"
    assert a.calibration_json.name == "my_so101_follower.json"


def test_env_picks_arm_and_missing_fields_default(tmp_path, monkeypatch):
    f = tmp_path / "arms.yaml"
    f.write_text(ARMS)
    monkeypatch.setenv(profile.ENV, "left")
    a = profile.arm(path=f)
    assert a.port == "/dev/so101_left_follower" and a.robot_id == "left_follower"
    assert a.joint_map == profile.REPO / "config" / "sim2real_left.yaml"
    assert a.wrist_camera == "left_wrist" and a.workspace_camera == "workspace"
    assert a.wrist_mount.name == "left_wrist_mount.yaml"
    assert profile.arm("default", path=f).robot_id == "my_so101_follower"


def test_unknown_arm_names_the_choices(tmp_path):
    f = tmp_path / "arms.yaml"
    f.write_text(ARMS)
    with pytest.raises(KeyError, match="default, left"):
        profile.arm("right", path=f)


def test_repo_config_default_matches_single_arm_bench():
    a = profile.arm("default")
    assert a.robot_id == "my_so101_follower" and a.wrist_camera == "wrist"
