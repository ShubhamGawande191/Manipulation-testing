from robot_manipulator.cameras import camera_path, load_camera_config, v4l2_command


def test_config_has_stable_paths_and_auto_modes_first():
    cfg = load_camera_config()
    assert set(cfg) == {"workspace", "wrist"}
    for name, cam in cfg.items():
        assert camera_path(name, cfg).startswith("/dev/v4l/by-path/")
        keys = list(cam["controls"])
        # manual values only stick after the auto mode is switched off
        assert keys.index("auto_exposure") < keys.index("exposure_time_absolute")
        assert keys.index("white_balance_automatic") < keys.index("white_balance_temperature")


def test_v4l2_command_preserves_order():
    cmd = v4l2_command("/dev/x", {"auto_exposure": 1, "exposure_time_absolute": 300})
    assert cmd == [
        "v4l2-ctl",
        "-d",
        "/dev/x",
        "-c",
        "auto_exposure=1",
        "-c",
        "exposure_time_absolute=300",
    ]
