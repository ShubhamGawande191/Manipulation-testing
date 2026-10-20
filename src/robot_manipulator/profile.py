"""Per-arm defaults for benches with more than one SO-101 (e.g. a bimanual cell).

``config/arms.yaml`` lists the arms; the environment variable ``SO101_ARM`` picks one (default
``default``). Every real-arm tool takes its default port, LeRobot id, joint map, wrist camera and
wrist-camera mount file from the active arm, so one prefix switches a whole command:

    SO101_ARM=left uv run harness reach --backend real ...
    SO101_ARM=right uv run camcal-wrist run --backend real --live

Explicit CLI flags still win. Without ``config/arms.yaml`` the single-arm defaults below apply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ARMS_FILE = REPO / "config" / "arms.yaml"
ENV = "SO101_ARM"

_DEFAULT = {
    "port": "/dev/so101_follower",
    "robot_id": "my_so101_follower",
    "joint_map": "config/sim2real.yaml",
    "wrist_camera": "wrist",
    "workspace_camera": "workspace",
    "wrist_mount": "config/extrinsics/wrist_mount.yaml",
    "leader_port": "/dev/so101_leader",
    "leader_id": "my_so101_leader",
}


@dataclass(frozen=True)
class Arm:
    name: str
    port: str
    robot_id: str
    joint_map: Path
    wrist_camera: str  # camera name in config/cameras.yaml
    workspace_camera: str
    wrist_mount: Path
    leader_port: str = "/dev/so101_leader"  # the leader paired with this arm (teleop)
    leader_id: str = "my_so101_leader"

    @property
    def calibration_json(self) -> Path:
        """LeRobot follower calibration file for this arm's ``robot_id``."""
        root = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
        return root / "robots" / "so_follower" / f"{self.robot_id}.json"

    @property
    def leader_calibration_json(self) -> Path:
        root = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
        return root / "teleoperators" / "so_leader" / f"{self.leader_id}.json"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def load_arms(path: Path = ARMS_FILE) -> dict[str, dict]:
    """``{name: fields}`` from ``config/arms.yaml``; just ``default`` when the file is absent."""
    if not path.exists():
        return {"default": dict(_DEFAULT)}
    doc = yaml.safe_load(path.read_text()) or {}
    arms = doc.get("arms") or {}
    if not isinstance(arms, dict) or not arms:
        raise ValueError(f"{path}: expected a non-empty 'arms:' mapping")
    return arms


def arm(name: str | None = None, path: Path = ARMS_FILE) -> Arm:
    """The arm called ``name`` (else ``$SO101_ARM``, else ``default``), missing fields defaulted."""
    name = name or os.environ.get(ENV) or "default"
    arms = load_arms(path)
    if name not in arms:
        raise KeyError(f"arm {name!r} not in {path} (have: {', '.join(arms)}); check ${ENV}")
    f = {**_DEFAULT, **(arms[name] or {})}
    return Arm(
        name=name,
        port=str(f["port"]),
        robot_id=str(f["robot_id"]),
        joint_map=_resolve(f["joint_map"]),
        wrist_camera=str(f["wrist_camera"]),
        workspace_camera=str(f["workspace_camera"]),
        wrist_mount=_resolve(f["wrist_mount"]),
        leader_port=str(f["leader_port"]),
        leader_id=str(f["leader_id"]),
    )


ACTIVE = arm()
