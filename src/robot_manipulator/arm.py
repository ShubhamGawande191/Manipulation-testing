"""SO101 arm helpers — thin wrappers around LeRobot 0.5+ config objects.

The SO101 12 V uses Feetech STS3215 servos.  LeRobot models the arm as two
separate halves:
  - follower  → ``SOFollowerRobotConfig`` / ``SOFollower`` robot
  - leader    → ``SOLeaderTeleopConfig``  / ``SOLeader``  teleoperator

Calibration and teleoperation are driven by the LeRobot CLI
(``lerobot-calibrate``, ``lerobot-teleoperate``); call them via
``scripts/calibrate.py`` and ``scripts/teleoperate.py`` in this repo.

Camera streams are injected directly into the follower config so that
LeRobot records and displays them automatically during teleoperation.
"""

from __future__ import annotations

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig


def build_follower_config(
    port: str,
    robot_id: str = "my_so101_follower",
    workspace_cam_index: int | str | None = None,
    wrist_cam_index: int | str | None = None,
    fps: int = 5,
    width: int = 1920,
    height: int = 1080,
    cameras: bool = True,
) -> SOFollowerRobotConfig:
    """Return a follower config with optional cameras attached.

    Cameras are registered inside the config so that ``lerobot-teleoperate``
    (and ``lerobot-record``) automatically capture frames without any extra
    code.  The *robot_id* must match the calibration file name under
    ``~/.cache/huggingface/lerobot/calibration/robots/so_follower/``.

    Set *cameras=False* to skip camera initialization (useful for joint-only
    operations where camera setup overhead is unwanted).

    Camera defaults come from ``config/cameras.yaml`` (stable ``/dev/v4l/by-path`` per USB port);
    ``/dev/videoN`` indices shuffle on replug. Run ``uv run cam-setup`` to lock exposure first.
    """
    cam_dict = {}
    if cameras:
        from robot_manipulator.cameras import camera_path

        if workspace_cam_index is None:
            workspace_cam_index = camera_path("workspace")
        if wrist_cam_index is None:
            wrist_cam_index = camera_path("wrist")
        cam_dict = {
            "workspace": OpenCVCameraConfig(
                index_or_path=workspace_cam_index,
                fps=fps,
                width=width,
                height=height,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_cam_index,
                fps=fps,
                width=width,
                height=height,
            ),
        }
    return SOFollowerRobotConfig(
        id=robot_id,
        port=port,
        cameras=cam_dict,
    )


def build_leader_config(port: str) -> SOLeaderTeleopConfig:
    """Return a leader (teleoperator) config."""
    return SOLeaderTeleopConfig(port=port)
