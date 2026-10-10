"""Joint units at the LeRobot boundary. The only module that converts between them.

Inside the harness and the sim, joints are radians in CMD order (`CMD_JOINTS`).
LeRobot's SO101 follower reports body joints in degrees and the gripper in 0..100.
Per-joint `sign`/`offset_deg` come from `config/sim2real.yaml`. They stay placeholders
until hardware ticket H2, and `calibrated` stays false until then.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CMD_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
BODY_JOINTS: tuple[str, ...] = CMD_JOINTS[:5]


@dataclass(frozen=True)
class BodyCal:
    sign: int  # +1 or -1
    offset_deg: float

    def to_sim(self, deg: float) -> float:
        return self.sign * float(np.deg2rad(deg)) + float(np.deg2rad(self.offset_deg))

    def from_sim(self, rad: float) -> float:
        return float(np.rad2deg(self.sign * (rad - np.deg2rad(self.offset_deg))))


@dataclass(frozen=True)
class GripperCal:
    lerobot_open: float
    lerobot_closed: float
    sim_open: float
    sim_closed: float

    def __post_init__(self) -> None:
        if self.lerobot_open == self.lerobot_closed:
            raise ValueError("gripper lerobot_open and lerobot_closed must differ")
        if self.sim_open == self.sim_closed:
            raise ValueError("gripper sim_open and sim_closed must differ")

    def to_sim(self, v: float) -> float:
        t = np.clip((v - self.lerobot_open) / (self.lerobot_closed - self.lerobot_open), 0.0, 1.0)
        return float(self.sim_open + t * (self.sim_closed - self.sim_open))

    def from_sim(self, rad: float) -> float:
        t = np.clip((rad - self.sim_open) / (self.sim_closed - self.sim_open), 0.0, 1.0)
        return float(self.lerobot_open + t * (self.lerobot_closed - self.lerobot_open))


def _lerobot_key(joint: str) -> str:
    return f"{joint}.pos"


def _get(values: Mapping[str, float], joint: str) -> float:
    key = _lerobot_key(joint)
    if key not in values:
        raise KeyError(f"LeRobot observation/action is missing {key!r}")
    return float(values[key])


@dataclass(frozen=True)
class JointMap:
    body: Mapping[str, BodyCal]
    gripper: GripperCal
    calibrated: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> JointMap:
        raw = yaml.safe_load(Path(path).read_text())
        joints = raw["joints"]
        missing = [j for j in BODY_JOINTS if j not in joints]
        if missing:
            raise ValueError(f"{path}: missing joints {missing}")
        body: dict[str, BodyCal] = {}
        for j in BODY_JOINTS:
            sign = int(joints[j]["sign"])
            if sign not in (1, -1):
                raise ValueError(f"{path}: {j} sign must be +1 or -1, got {sign}")
            body[j] = BodyCal(sign=sign, offset_deg=float(joints[j]["offset_deg"]))
        g = raw["gripper"]
        gripper = GripperCal(
            lerobot_open=float(g["lerobot_open"]),
            lerobot_closed=float(g["lerobot_closed"]),
            sim_open=float(g["sim_open"]),
            sim_closed=float(g["sim_closed"]),
        )
        return cls(body=body, gripper=gripper, calibrated=bool(raw["calibrated"]))

    def lerobot_to_sim(self, values: Mapping[str, float]) -> np.ndarray:
        """LeRobot `{"<joint>.pos": value}` (degrees, gripper 0..100) -> radians, CMD order."""
        out = np.empty(len(CMD_JOINTS), dtype=np.float64)
        for i, j in enumerate(BODY_JOINTS):
            out[i] = self.body[j].to_sim(_get(values, j))
        out[-1] = self.gripper.to_sim(_get(values, "gripper"))
        return out

    def sim_to_lerobot(self, q: np.ndarray) -> dict[str, float]:
        """Radians, CMD order -> LeRobot `{"<joint>.pos": value}` for `send_action`."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (len(CMD_JOINTS),):
            raise ValueError(f"expected {len(CMD_JOINTS)} joints, got shape {q.shape}")
        out = {_lerobot_key(j): self.body[j].from_sim(q[i]) for i, j in enumerate(BODY_JOINTS)}
        out[_lerobot_key("gripper")] = self.gripper.from_sim(q[-1])
        return out
