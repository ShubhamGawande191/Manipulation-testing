"""Camera helpers for workspace and wrist cameras.

Both cameras are eMeet C950 webcams (standard UVC/V4L2). They share one USB serial, so they are
identified by USB port path (``config/cameras.yaml``), not by index.  LeRobot's
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
from pathlib import Path

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

__all__ = [
    "OpenCVCamera",
    "OpenCVCameraConfig",
    "apply_controls",
    "camera_path",
    "list_cameras",
    "load_camera_config",
    "make_opencv_config",
    "v4l2_command",
]

REPO = Path(__file__).resolve().parents[2]
CAMERA_CONFIG = REPO / "config" / "cameras.yaml"


def load_camera_config(path: str | Path = CAMERA_CONFIG) -> dict[str, dict]:
    """``{name: {"path": ..., "controls": {...}, ...}}`` from ``config/cameras.yaml``."""
    import yaml

    return yaml.safe_load(Path(path).read_text())["cameras"]


def camera_path(name: str, config: dict[str, dict] | None = None) -> str:
    """Stable device path (``/dev/v4l/by-path/...``) of camera ``name`` (workspace / wrist)."""
    return (config or load_camera_config())[name]["path"]


def v4l2_command(device: str, controls: dict[str, int]) -> list[str]:
    """``v4l2-ctl`` argv setting ``controls`` in order (auto modes first, as listed)."""
    cmd = ["v4l2-ctl", "-d", device]
    for key, value in controls.items():
        cmd += ["-c", f"{key}={int(value)}"]
    return cmd


def apply_controls(device: str, controls: dict[str, int]) -> None:
    """Apply V4L2 controls; raises CalledProcessError if the device or a control is missing."""
    subprocess.run(v4l2_command(device, controls), check=True, capture_output=True, text=True)


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


def setup_main(
    snapshot: bool = False,
    list_paths: bool = False,
) -> None:
    """Apply ``config/cameras.yaml`` controls to every camera (``uv run cam-setup``)."""
    import glob

    if list_paths:
        for p in sorted(glob.glob("/dev/v4l/by-path/*video-index0")):
            print(p)
        return
    cfg = load_camera_config()
    for name, cam in cfg.items():
        apply_controls(cam["path"], cam.get("controls", {}))
        print(f"{name:10} {cam['path']}  controls applied")
        if snapshot:
            import time

            import cv2

            cap = cv2.VideoCapture(cam["path"])
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.5:
                cap.read()
            ok, img = cap.read()
            cap.release()
            out = REPO / "runs" / "cam" / f"{name}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if ok:
                cv2.imwrite(str(out), img)
            where = f"-> {out}  mean {img.mean():.0f}" if ok else "FAILED"
            print(f"{'':10} snapshot {where}")


def cam_setup_app() -> None:
    import typer

    typer.run(setup_main)
