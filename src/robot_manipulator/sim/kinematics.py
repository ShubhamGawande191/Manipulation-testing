"""Forward kinematics, Jacobian, and reachability sampling on the SO101 MJCF.

Uses its own MjData, so it never touches a SimBackend's physics state. Joint vectors
are radians in CMD order, with all six entries (the gripper enters only through
`gripper` in sample_reachable).
"""

from __future__ import annotations

import mujoco
import numpy as np

from robot_manipulator.sim.model import (
    EE_SITE,
    JOINTS,
    apply_joint_couplings,
    joint_dof_adr,
    joint_qpos_adr,
    load_model,
)

N_ARM = 5  # the first five joints of JOINTS are the arm; the gripper is separate


class FK:
    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        self._qadr = joint_qpos_adr(self.model, JOINTS)
        self._dof = joint_dof_adr(self.model, JOINTS)
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if self._site < 0:
            raise KeyError(f"site {EE_SITE!r} not in model")

    def _set(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (len(JOINTS),):
            raise ValueError(f"expected {len(JOINTS)} joints, got shape {q.shape}")
        self.data.qpos[self._qadr] = q
        apply_joint_couplings(self.model, self.data)

    def ee_pos(self, q: np.ndarray) -> np.ndarray:
        """End-effector (gripperframe) position in the world frame, meters."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self._site].copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Positional Jacobian of the end effector, shape (3, 6), columns in JOINTS order."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._site)
        return jacp[:, self._dof]

    def sample_reachable(
        self,
        rng: np.random.Generator,
        n: int,
        lo: np.ndarray,
        hi: np.ndarray,
        z_min: float = 0.05,
        gripper: float = 0.0,
        max_tries_per_sample: int = 200,
        box: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample n arm configurations inside [lo, hi] whose end effector has z >= z_min
        and, if `box=(lo_xyz, hi_xyz)` is given, lies inside that base-frame box.

        `lo`/`hi` cover the five arm joints, or all six (the last entry is ignored). The
        gripper is fixed at `gripper`. Returns `(q[n, 6], ee[n, 3])`. Collisions are not checked.
        """
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)[:N_ARM]
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)[:N_ARM]
        if lo.shape != (N_ARM,) or hi.shape != (N_ARM,):
            raise ValueError("lo and hi need at least five arm entries")
        qs: list[np.ndarray] = []
        ees: list[np.ndarray] = []
        budget = max_tries_per_sample * max(n, 1)
        tries = 0
        while len(qs) < n:
            if tries >= budget:
                raise RuntimeError(f"only {len(qs)}/{n} reachable samples in {tries} tries")
            tries += 1
            q = np.r_[rng.uniform(lo, hi), gripper]
            ee = self.ee_pos(q)
            in_box = box is None or bool(np.all(ee >= box[0]) and np.all(ee <= box[1]))
            if ee[2] >= z_min and in_box:
                qs.append(q)
                ees.append(ee)
        return np.asarray(qs).reshape(-1, len(JOINTS)), np.asarray(ees).reshape(-1, 3)
