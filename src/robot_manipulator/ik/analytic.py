"""Closed-form IK for the SO-101 (+ PincOpen fingertip), polished on the full MuJoCo FK.

Task: fingertip position ``p`` (robot frame, m) + tool pitch ``phi`` (rad, 0 = horizontal, +pi/2 =
pointing straight down) + wrist roll ``roll`` (free: the fingertip sits on the roll axis, so roll
does not move it). Five joints, five task numbers.

Geometry (read from the MJCF, not hard-coded): the pan axis is vertical; shoulder_lift, elbow_flex
and wrist_flex are parallel (+y), so after pan the arm is a planar 3R chain. In the arm plane a
vector (r, z) is written as the complex number ``r - i z``; a joint rotation by ``q`` about +y is
then multiplication by ``exp(i q)``, and the tool pitch is exactly ``lift + elbow + wrist_flex``.

1. pan from the target's direction about the pan axis (or turned away, reaching back over it),
2. the wrist point = fingertip - L3 rotated by the pitch,
3. lift/elbow from the two-link triangle (two branches: elbow up / down),
4. wrist_flex = pitch - lift - elbow,
5. a few Newton steps on the real FK remove the sub-millimetre offsets the planar model ignores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_manipulator.calib.extrinsics import TipFK
from robot_manipulator.sim.model import JOINTS, joint_ranges


@dataclass
class ArmGeometry:
    pan_xy: np.ndarray  # pan axis position (x, y) in the robot frame
    pan_sign: float  # +1 if positive pan turns +x toward +y
    lift: complex  # lift axis in the arm plane (r - i z), at q = 0
    l1: complex  # lift -> elbow
    l2: complex  # elbow -> wrist_flex
    l3: complex  # wrist_flex -> fingertip

    @classmethod
    def from_fk(cls, fk: TipFK) -> ArmGeometry:
        m, d = fk.m, fk.d
        fk._pose(np.zeros(6))
        anchor = {j: d.xanchor[m.joint(j).id].copy() for j in JOINTS[:4]}
        tip = fk.tip(np.zeros(6))
        pan_xy = anchor["shoulder_pan"][:2]
        # positive pan about an axis along -z turns +x toward -y: sign = axis z component
        pan_sign = float(np.sign(d.xaxis[m.joint("shoulder_pan").id][2]))

        def plane(p: np.ndarray) -> complex:  # at q = 0 the arm plane is the x-z plane
            return complex(p[0] - pan_xy[0], -p[2])

        a_l, a_e, a_w = (plane(anchor[j]) for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))
        return cls(pan_xy, pan_sign, a_l, a_e - a_l, a_w - a_e, plane(tip) - a_w)


class AnalyticIK:
    """``solve(p, phi, roll)`` -> joint vector (rad, CMD order, gripper closed) or None."""

    def __init__(self, fk: TipFK | None = None, polish_iters: int = 3) -> None:
        self.fk = fk or TipFK()
        self.g = ArmGeometry.from_fk(self.fk)
        self.lo, self.hi = joint_ranges(self.fk.m, JOINTS)
        self.polish_iters = polish_iters

    def branches(self, p: np.ndarray, phi: float, roll: float = 0.0) -> list[np.ndarray]:
        """Up to four planar solutions (facing / reaching back x elbow branches), before limits
        and polish."""
        g = self.g
        dx, dy = p[0] - g.pan_xy[0], p[1] - g.pan_xy[1]
        pan0 = g.pan_sign * np.arctan2(dy, dx)
        r = np.hypot(dx, dy)
        a, b = abs(g.l1), abs(g.l2)
        al1, al2 = np.angle(g.l1), np.angle(g.l2)
        out = []
        # facing the target (r > 0), or turned away and reaching back over the pan axis (r < 0)
        for pan, rr in ((pan0, r), ((pan0 + 2 * np.pi) % (2 * np.pi) - np.pi, -r)):
            tip = complex(rr, -p[2])
            w = tip - g.l3 * np.exp(1j * phi)  # wrist_flex axis in the arm plane
            dvec = w - g.lift
            c = (abs(dvec) ** 2 - a**2 - b**2) / (2 * a * b)
            if abs(c) > 1.0:
                continue  # out of reach this way round
            for delta in (np.arccos(c), -np.arccos(c)):  # the two elbow branches
                q2 = delta - al2 + al1
                q1 = np.angle(dvec) - np.angle(a + b * np.exp(1j * delta)) - al1
                q3 = phi - q1 - q2
                q = np.array([pan, q1, q2, q3, roll, 0.0])
                q[1:4] = (q[1:4] + np.pi) % (2 * np.pi) - np.pi
                out.append(q)
        return out

    def task(self, q: np.ndarray) -> np.ndarray:
        """(fingertip xyz, pitch) of joint vector ``q``: pitch = lift + elbow + wrist_flex."""
        return np.r_[self.fk.tip(q), q[1] + q[2] + q[3]]

    def polish(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Newton steps on pan/lift/elbow/wflex against the full FK (finite-difference Jacobian)."""
        q = q.copy()
        for _ in range(self.polish_iters):
            e = target - self.task(q)
            if np.linalg.norm(e[:3]) < 1e-6 and abs(e[3]) < 1e-7:
                break
            J = np.empty((4, 4))
            for k in range(4):
                dq = np.zeros(6)
                dq[k] = 1e-6
                J[:, k] = (self.task(q + dq) - self.task(q)) / 1e-6
            q[:4] += np.linalg.lstsq(J, e, rcond=None)[0]
        return q

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        return bool(np.all(q[:5] >= self.lo[:5] + margin) and np.all(q[:5] <= self.hi[:5] - margin))

    def solve(
        self, p, phi: float, roll: float = 0.0, seed: np.ndarray | None = None, margin: float = 0.0
    ) -> np.ndarray | None:
        """The in-limits branch closest to ``seed`` (or the first), polished; None if none."""
        p = np.asarray(p, dtype=float)
        target = np.r_[p, phi]
        sols = [self.polish(q, target) for q in self.branches(p, phi, roll)]
        sols = [q for q in sols if self.within_limits(q, margin)]
        if not sols:
            return None
        if seed is not None:
            sols.sort(key=lambda q: float(np.abs(q[:5] - seed[:5]).max()))
        return sols[0]
