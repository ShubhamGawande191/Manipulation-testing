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
    workspace_cam_index: int = 0,
    wrist_cam_index: int = 1,
) -> SOFollowerRobotConfig:
    """Return a follower config with both cameras attached.

    Cameras are registered inside the config so that ``lerobot-teleoperate``
    (and ``lerobot-record``) automatically capture frames without any extra
    code.
    """
    return SOFollowerRobotConfig(
        port=port,
        cameras={
            "workspace": OpenCVCameraConfig(
                index_or_path=workspace_cam_index,
                fps=30,
                width=1920,
                height=1080,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_cam_index,
                fps=30,
                width=1920,
                height=1080,
            ),
        },
    )


def build_leader_config(port: str) -> SOLeaderTeleopConfig:
    """Return a leader (teleoperator) config."""
    return SOLeaderTeleopConfig(port=port)
