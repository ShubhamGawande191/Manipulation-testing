"""Static servo stiffness from held poses: sag vs the twin's gravity torque.

At rest a position servo balances gravity: ``k (q_cmd - q) = tau_g(q)`` (+ friction, which shows
up as scatter and an intercept). Regressing the tracking error on the twin's gravity torque over
quasi-static telemetry samples (``harness survey``) gives ``1/k`` per joint. That pins kp, which a
chirp fit cannot identify on its own (kp trades off with friction and armature; a policy trained
on such a fit hunted on the real arm, 2026-10-18).

Joints without a usable gravity signal (pan and wrist_roll carry ~no gravity torque, and a narrow
torque range gives a weak correlation) are filled from the best-measured joint, scaled by the
servo P coefficient (``hw.follower``): the STS3215 output stiffness is proportional to P.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_manipulator.hw.follower import DEFAULT_PID, JOINT_PID

R_MIN = 0.5  # minimum |corr(error, tau_g)| to trust a joint's own estimate
STATIC_RAD_S = 0.02  # all arm joints slower than this -> quasi-static sample
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass
class Stiffness:
    joint: str
    k_nm_per_rad: float
    kp_scale: float  # relative to the model's nominal position gain
    source: str  # "measured" or "P-scaled from <joint>"
    corr: float | None
    n: int
    intercept_deg: float | None


def p_gain(joint: str) -> int:
    return JOINT_PID.get(joint, DEFAULT_PID)[0]


def regress(err: np.ndarray, tau: np.ndarray) -> tuple[float, float, float]:
    """err ≈ slope * tau + c; returns (slope rad/Nm, c rad, corr)."""
    A = np.c_[tau, np.ones(len(tau))]
    (slope, c), *_ = np.linalg.lstsq(A, err, rcond=None)
    corr = float(np.corrcoef(tau, err)[0, 1]) if np.std(tau) > 1e-9 else 0.0
    return float(slope), float(c), corr


def static_samples(paths: list[Path], grav=None) -> tuple[np.ndarray, np.ndarray]:
    """(err, tau_g) arrays (n, 5) from quasi-static telemetry samples with a logged command."""
    from robot_manipulator.sim.atlas_real import GravityModel, load_samples

    grav = grav or GravityModel()
    errs, taus = [], []
    for s in load_samples(paths):
        if s["dq"] is None or s["q_cmd"] is None or np.any(np.abs(s["dq"][:5]) > STATIC_RAD_S):
            continue
        _, pred = grav(s["q"])
        errs.append(s["q_cmd"][:5] - s["q"][:5])
        taus.append(pred / 1000.0 * grav.tau_max)
    return np.array(errs).reshape(-1, 5), np.array(taus).reshape(-1, 5)


def estimate(err: np.ndarray, tau: np.ndarray, kp_nominal: np.ndarray) -> list[Stiffness]:
    """Per-joint stiffness; joints below R_MIN are P-scaled from the best-correlated joint."""
    fits = [regress(err[:, i], tau[:, i]) for i in range(5)]
    ok = [i for i, (slope, _, r) in enumerate(fits) if slope > 0 and r >= R_MIN]
    if not ok:
        raise ValueError(f"no joint with corr >= {R_MIN}: survey more loaded poses")
    ref = max(ok, key=lambda i: fits[i][2])
    k_ref = 1.0 / fits[ref][0]
    out = []
    for i, j in enumerate(ARM):
        slope, c, r = fits[i]
        if i in ok:
            k, src = 1.0 / slope, "measured"
        else:
            k, src = k_ref * p_gain(j) / p_gain(ARM[ref]), f"P-scaled from {ARM[ref]}"
        out.append(
            Stiffness(
                joint=j,
                k_nm_per_rad=float(k),
                kp_scale=float(k / kp_nominal[i]),
                source=src,
                corr=None if np.isnan(r) else float(r),
                n=len(err),
                intercept_deg=float(np.degrees(c)),
            )
        )
    return out
