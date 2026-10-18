"""Workspace atlas, layer 2: aggregate real telemetry (``runs/telemetry``) into the atlas grid.

For every logged sample with measured joints:

- EE position via the twin's FK (PincOpen model; the EE site does not depend on the gripper).
- Tracking error ``|q_cmd - q_meas|`` per joint (deg), when a command was logged.
- For quasi-static samples (|dq| < ``STATIC_RAD_S`` on every arm joint), the measured servo load
  (permille, signed) and the twin's predicted gravity load at that exact pose
  (``tau_g / tau_max``, permille). Their ratio calibrates the load reading against the model.

Per voxel we keep **medians** and sample counts, not maxima. The servo bus occasionally returns a
garbage register value (one 75 °C reading among steady 24 °C on 2026-10-18).

    uv run atlas real            # -> runs/atlas/real.parquet + per-joint calibration summary
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from robot_manipulator.sim.atlas import LO, VOXEL
from robot_manipulator.sim.model import (
    EE_SITE,
    JOINTS,
    apply_joint_couplings,
    joint_dof_adr,
    joint_qpos_adr,
    load_model,
)

TELEMETRY_DIR = Path(__file__).resolve().parents[3] / "runs" / "telemetry"
N_ARM = 5
STATIC_RAD_S = 0.15  # max joint speed for a sample to count as quasi-static
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class GravityModel:
    """EE position and predicted gravity load fraction (permille of tau_max) per arm joint."""

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        self.m = model if model is not None else load_model()
        self.d = mujoco.MjData(self.m)
        self.qadr = joint_qpos_adr(self.m, JOINTS)
        self.dadr = joint_dof_adr(self.m, JOINTS)[:N_ARM]
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        act = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM]
        self.tau_max = np.array([self.m.actuator_forcerange[a][1] for a in act])

    def __call__(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.qadr] = q
        apply_joint_couplings(self.m, self.d)
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        tau_g = self.d.qfrc_bias[self.dadr]
        return self.d.site_xpos[self.site].copy(), 1000.0 * tau_g / self.tau_max


def load_samples(paths: list[Path]) -> list[dict]:
    """Flatten telemetry files into per-sample dicts (with joint speeds from consecutive rows)."""
    import pyarrow.parquet as pq

    out: list[dict] = []
    for path in paths:
        meta = (
            json.loads(path.with_suffix(".json").read_text())
            if path.with_suffix(".json").exists()
            else {}
        )
        rows = pq.read_table(path).to_pylist()
        prev_t, prev_q = None, None
        for r in rows:
            q = r.get("q_meas_rad")
            if q is None:
                continue
            q = np.asarray(q, dtype=float)
            dq = None
            if prev_q is not None and r["t"] > prev_t:
                dq = (q - prev_q) / (r["t"] - prev_t)
            prev_t, prev_q = r["t"], q
            out.append(
                {
                    "file": path.name,
                    "source": meta.get("source"),
                    "q": q,
                    "dq": dq,
                    "q_cmd": None if r.get("q_cmd_rad") is None else np.asarray(r["q_cmd_rad"]),
                    "load": None if r.get("load") is None else np.asarray(r["load"], dtype=float),
                }
            )
    return out


def aggregate(samples: list[dict], grav: GravityModel | None = None) -> tuple[list[dict], dict]:
    """Per-voxel medians + a global measured/predicted load calibration per joint."""
    grav = grav or GravityModel()
    cells: dict[tuple, dict] = {}
    pairs: list[tuple[np.ndarray, np.ndarray]] = []  # (measured, predicted) static loads
    for s in samples:
        ee, pred = grav(s["q"])
        k = tuple(np.floor((ee - LO) / VOXEL).astype(int))
        c = cells.setdefault(k, {"n": 0, "track": [], "load": [], "pred": [], "sources": set()})
        c["n"] += 1
        c["sources"].add(s["source"])
        if s["q_cmd"] is not None:
            c["track"].append(np.degrees(np.abs(s["q_cmd"][:N_ARM] - s["q"][:N_ARM])))
        static = s["dq"] is not None and np.all(np.abs(s["dq"][:N_ARM]) < STATIC_RAD_S)
        if s["load"] is not None and static:
            c["load"].append(s["load"][:N_ARM])
            c["pred"].append(pred)
            pairs.append((s["load"][:N_ARM], pred))
    rows = []
    for k, c in cells.items():
        r = {
            "x": LO[0] + (k[0] + 0.5) * VOXEL,
            "y": LO[1] + (k[1] + 0.5) * VOXEL,
            "z": LO[2] + (k[2] + 0.5) * VOXEL,
            "n": c["n"],
            "n_static_load": len(c["load"]),
            "sources": sorted(x for x in c["sources"] if x),
        }
        if c["track"]:
            med = np.median(np.array(c["track"]), axis=0)
            r.update({f"track_med_deg_{j}": float(v) for j, v in zip(ARM, med)})
        if c["load"]:
            r.update({f"load_med_{j}": float(v) for j, v in zip(ARM, np.median(c["load"], axis=0))})
            r.update({f"pred_med_{j}": float(v) for j, v in zip(ARM, np.median(c["pred"], axis=0))})
        rows.append(r)
    calib: dict = {"n_static_samples": len(pairs)}
    if pairs:
        meas = np.array([p[0] for p in pairs])
        pred = np.array([p[1] for p in pairs])
        for i, j in enumerate(ARM):
            ok = np.abs(pred[:, i]) > 20  # only where the model predicts a meaningful gravity load
            if ok.sum() >= 5:
                # least-squares slope through the origin: measured ≈ gain * predicted
                gain = float(np.dot(meas[ok, i], pred[ok, i]) / np.dot(pred[ok, i], pred[ok, i]))
                corr = float(np.corrcoef(meas[ok, i], pred[ok, i])[0, 1])
                calib[j] = {"gain": round(gain, 3), "corr": round(corr, 3), "n": int(ok.sum())}
            else:
                calib[j] = {"gain": None, "corr": None, "n": int(ok.sum())}
    return rows, calib


def build_real(telemetry_dir: Path = TELEMETRY_DIR) -> tuple[list[dict], dict, list[Path]]:
    paths = sorted(Path(telemetry_dir).glob("*.parquet"))
    rows, calib = aggregate(load_samples(paths))
    return rows, calib, paths
