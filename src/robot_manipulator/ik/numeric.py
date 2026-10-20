"""Damped-least-squares IK on the full MuJoCo FK (the generic numeric baseline).

Same task as ``AnalyticIK``: fingertip position + tool pitch, roll given. Iterates
``dq = J^T (J J^T + lambda^2 I)^-1 e`` on pan/lift/elbow/wrist_flex with a finite-difference
Jacobian, clamped to the joint limits after every step.
"""

from __future__ import annotations

import numpy as np

from robot_manipulator.ik.analytic import AnalyticIK

POS_TOL = 1e-4  # m
PITCH_TOL = np.radians(0.1)


def dls(
    ik: AnalyticIK,
    p,
    phi: float,
    roll: float,
    q0: np.ndarray,
    lam: float = 0.01,
    max_iter: int = 200,
    max_step: float = 0.3,
) -> tuple[np.ndarray, int]:
    """(joint vector, iterations). The result may not meet the tolerance: check it."""
    target = np.r_[np.asarray(p, dtype=float), phi]
    q = np.asarray(q0, dtype=float).copy()
    q[4], q[5] = roll, 0.0
    lo, hi = ik.lo, ik.hi
    for it in range(1, max_iter + 1):
        f = ik.task(q)
        e = target - f
        e[3] *= 0.1  # pitch error (rad) weighted like 10 cm of position
        if np.linalg.norm(e[:3]) < POS_TOL and abs(e[3]) < 0.1 * PITCH_TOL:
            return q, it
        J = np.empty((4, 4))
        for k in range(4):
            dq = np.zeros(6)
            dq[k] = 1e-6
            J[:, k] = (ik.task(q + dq) - f) / 1e-6
        J[3] *= 0.1
        step = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(4), e)
        step *= min(1.0, max_step / max(np.abs(step).max(), 1e-12))
        q[:4] = np.clip(q[:4] + step, lo[:4], hi[:4])
    return q, max_iter


def dls_restarts(ik: AnalyticIK, p, phi, roll, seeds: list[np.ndarray], **kw):
    """First DLS run (in seed order) that converges; (q, total iterations, converged)."""
    total = 0
    q = seeds[0]
    for s in seeds:
        q, it = dls(ik, p, phi, roll, s, **kw)
        total += it
        if solved(ik, q, p, phi):
            return q, total, True
    return q, total, False


def solved(ik: AnalyticIK, q, p, phi, pos_tol: float = 1e-3, pitch_tol=np.radians(0.5)) -> bool:
    if q is None or not ik.within_limits(q):
        return False
    f = ik.task(q)
    return bool(np.linalg.norm(f[:3] - np.asarray(p)) < pos_tol and abs(f[3] - phi) < pitch_tol)
