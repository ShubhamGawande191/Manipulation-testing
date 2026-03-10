"""Smoke-tests: verify imports load without a physical robot attached."""

import importlib


def test_camera_module_importable():
    mod = importlib.import_module("robot_manipulator.cameras")
    assert hasattr(mod, "OpenCVCamera")
    assert hasattr(mod, "OpenCVCameraConfig")
    assert hasattr(mod, "list_cameras")
    assert hasattr(mod, "make_opencv_config")


def test_scripts_importable():
    for script in ("calibrate", "teleoperate", "pick_and_place"):
        mod = importlib.import_module(f"robot_manipulator.scripts.{script}")
        assert hasattr(mod, "app")
