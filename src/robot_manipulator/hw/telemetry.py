"""Hardware telemetry: one schema for every real run (teleop, harness, probes).

Every record stores commanded vs measured joints (LeRobot units AND sim radians via the joint
map in force at record time) and, every ``extra_every`` records, per-servo load / current /
temperature / voltage. Files: ``runs/telemetry/<stamp>_<source>.parquet`` + ``.json`` metadata
(joint map, calibration file, git commit) so the data stays interpretable after recalibration.
The workspace atlas (``sim/atlas.py``) aggregates these files.

Units: ``q_*_lr`` LeRobot (deg body, 0..100 gripper); ``q_*_rad`` sim radians (CMD order);
``load`` signed permille of max torque; ``current_ma`` (STS3215: 6.5 mA per raw unit);
``temp_c`` °C; ``voltage_v`` V.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "runs" / "telemetry"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CURRENT_MA_PER_UNIT = 6.5
EXTRA_REGISTERS = ("Present_Load", "Present_Current", "Present_Temperature", "Present_Voltage")


def _vec(values: Any) -> np.ndarray | None:
    """dict ``{joint: v}`` / ``{"joint.pos": v}`` / array -> float array[6] in JOINTS order."""
    if values is None:
        return None
    if isinstance(values, dict):
        return np.array(
            [float(values.get(j, values.get(f"{j}.pos", np.nan))) for j in JOINTS], dtype=float
        )
    return np.asarray(values, dtype=float).reshape(6)


def read_extras(bus) -> dict[str, np.ndarray]:
    """Per-servo load / current / temperature / voltage (raw registers, decoded sign)."""
    out = {}
    for reg in EXTRA_REGISTERS:
        raw = bus.sync_read(reg, normalize=False)
        out[reg] = np.array([float(raw[j]) for j in JOINTS])
    return {
        "load": out["Present_Load"],
        "current_ma": out["Present_Current"] * CURRENT_MA_PER_UNIT,
        "temp_c": out["Present_Temperature"],
        "voltage_v": out["Present_Voltage"] / 10.0,
    }


class TelemetryLogger:
    def __init__(
        self,
        source: str,
        robot_id: str,
        joint_map=None,
        calibration_path: Path | None = None,
        extra_every: int = 10,
        out_dir: Path = OUT_DIR,
    ) -> None:
        self.source, self.robot_id = source, robot_id
        self.joint_map, self.calibration_path = joint_map, calibration_path
        self.extra_every = max(1, int(extra_every))
        self.out_dir = Path(out_dir)
        self.t0 = time.monotonic()
        self.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.rows: list[dict[str, Any]] = []

    def _rad(self, lr: np.ndarray | None) -> np.ndarray | None:
        if lr is None or self.joint_map is None or np.isnan(lr).any():
            return None
        return self.joint_map.lerobot_to_sim({f"{j}.pos": float(v) for j, v in zip(JOINTS, lr)})

    def record(self, q_meas, q_cmd=None, bus=None, event: str | None = None) -> None:
        meas, cmd = _vec(q_meas), _vec(q_cmd)
        row: dict[str, Any] = {
            "t": time.monotonic() - self.t0,
            "q_meas_lr": meas.tolist(),
            "q_cmd_lr": None if cmd is None else cmd.tolist(),
            "event": event,
        }
        rad_m, rad_c = self._rad(meas), self._rad(cmd)
        row["q_meas_rad"] = None if rad_m is None else rad_m.tolist()
        row["q_cmd_rad"] = None if rad_c is None else rad_c.tolist()
        if bus is not None and len(self.rows) % self.extra_every == 0:
            try:
                for k, v in read_extras(bus).items():
                    row[k] = v.tolist()
            except Exception as exc:  # telemetry must never break control
                row["event"] = f"extras_failed: {exc}"
        self.rows.append(row)

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": self.source,
            "robot_id": self.robot_id,
            "stamp": self.stamp,
            "joints": list(JOINTS),
            "n_rows": len(self.rows),
            "extra_every": self.extra_every,
        }
        try:
            meta["git_head"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            meta["git_head"] = None
        if self.joint_map is not None:
            meta["joint_map"] = {
                "calibrated": bool(self.joint_map.calibrated),
                "body": {j: vars(c) for j, c in self.joint_map.body.items()},
                "gripper": vars(self.joint_map.gripper),
            }
        if self.calibration_path is not None and Path(self.calibration_path).is_file():
            data = Path(self.calibration_path).read_bytes()
            meta["calibration_file"] = str(self.calibration_path)
            meta["calibration_sha256"] = hashlib.sha256(data).hexdigest()
            meta["calibration"] = json.loads(data)
        return meta

    def save(self) -> Path | None:
        if not self.rows:
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"{self.stamp}_{self.source}"
        keys = sorted({k for r in self.rows for k in r})
        pq.write_table(
            pa.table({k: [r.get(k) for r in self.rows] for k in keys}), f"{base}.parquet"
        )
        Path(f"{base}.json").write_text(json.dumps(self.metadata(), indent=2, default=str) + "\n")
        return Path(f"{base}.parquet")
