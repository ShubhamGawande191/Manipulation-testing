"""Camera helpers for workspace and wrist cameras.

Both cameras are INNO-MAKER U20CAM-1080P-S1 (standard UVC/V4L2).  LeRobot's
built-in ``OpenCVCamera`` handles all capture, buffering, and thread safety —
no custom OpenCV wrappers needed here.

Convenience functions
---------------------
``list_cameras()``
    Thin wrapper around ``lerobot-find-cameras opencv`` to discover available
    camera indices without requiring physical cameras to be present.

``make_opencv_config(index, ...)``
    Build an ``OpenCVCameraConfig`` with sensible defaults for the U20CAM.
"""

from __future__ import annotations

import subprocess
import sys

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

__all__ = [
    "OpenCVCamera",
    "OpenCVCameraConfig",
    "list_cameras",
    "make_opencv_config",
]


def list_cameras() -> None:
    """Print discovered OpenCV cameras to stdout (wraps ``lerobot-find-cameras``)."""
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.lerobot_find_cameras", "opencv"],
        check=False,
    )
    return result.returncode


def make_opencv_config(
    index: int,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> OpenCVCameraConfig:
    """Return an ``OpenCVCameraConfig`` for an INNO-MAKER U20CAM-1080P-S1."""
    return OpenCVCameraConfig(
        index_or_path=index,
        fps=fps,
        width=width,
        height=height,
    )
