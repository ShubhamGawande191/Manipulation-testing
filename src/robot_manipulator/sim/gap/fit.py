"""Fit per-joint actuator parameters so the sim reproduces a recorded real trace.

For each joint, the fitter simulates the commanded sequence and compares the joint's measured
angle with the recording. It optimizes kp, joint damping, friction, and armature scales with
`scipy.optimize.least_squares`, once for each action delay in {0,1,2,3} ticks, and keeps the
delay with the lowest residual.

The simulation follows the runner's convention: the state is measured before each command, and
the delay applies to the command. Fit results are written to `config/actuator_fit.yaml`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

from robot_manipulator.sim.model import (
    JOINTS,
    actuator_ids,
    joint_dof_adr,
    joint_qpos_adr,
    load_model,
)
from robot_manipulator.sim.randomize import NominalParams

DELAYS = (0, 1, 2, 3, 4, 5, 6)  # real arm lags nominal sim 80-100 ms = 4-5 ticks (2026-10-18)
# kp, damping, frictionloss, armature scales. Widened 2026-10-18: the first real fit pinned kp at
# 0.2 and friction/armature at their bounds (LeRobot runs the servos at P=16-24, far softer).
LO = np.array([0.03, 0.05, 0.02, 0.02])
HI = np.array([5.0, 20.0, 40.0, 20.0])
# log-space starting points: nominal, softer/heavier, much softer with more friction
STARTS = (np.zeros(4), np.log([0.3, 2.0, 2.0, 2.0]), np.log([0.1, 3.0, 10.0, 4.0]))


@dataclass
class ActuatorParams:
    kp_scale: float = 1.0
    damping_scale: float = 1.0
    frictionloss_scale: float = 1.0
    armature_scale: float = 1.0
    delay_steps: int = 0


@dataclass
class FitResult:
    joint: str
    params: ActuatorParams
    rmse_rad: float
    nfev: int


def apply_params(
    model: mujoco.MjModel, nominal: NominalParams, joint: int, p: ActuatorParams
) -> None:
    """Set one joint's actuator and dof parameters from nominal values. Other joints are untouched.

    kp scales the position gain and its matching bias term (kv is unchanged).
    """
    a = actuator_ids(model, JOINTS)[joint]
    d = joint_dof_adr(model, JOINTS)[joint]
    model.actuator_gainprm[a, 0] = nominal.gainprm[a, 0] * p.kp_scale
    model.actuator_biasprm[a, 1] = nominal.biasprm[a, 1] * p.kp_scale
    model.dof_damping[d] = nominal.damping[d] * p.damping_scale
    model.dof_frictionloss[d] = nominal.frictionloss[d] * p.frictionloss_scale
    model.dof_armature[d] = nominal.armature[d] * p.armature_scale


def simulate_joint(
    model: mujoco.MjModel,
    joint: int,
    q_cmd: np.ndarray,
    delay: int,
    base: np.ndarray,
    n_substeps: int,
) -> np.ndarray:
    """Measured angle of `joint` over `q_cmd` (n, 6), using the runner's sample ordering."""
    data = mujoco.MjData(model)
    qadr = joint_qpos_adr(model, JOINTS)
    act = actuator_ids(model, JOINTS)
    data.qpos[qadr] = base
    data.ctrl[act] = base
    mujoco.mj_forward(model, data)
    pending: deque[np.ndarray] = deque(base.copy() for _ in range(delay))
    out = np.empty(len(q_cmd))
    for k, cmd in enumerate(q_cmd):
        out[k] = data.qpos[qadr[joint]]
        if delay > 0:
            pending.append(np.asarray(cmd, dtype=np.float64))
            cmd = pending.popleft()
        data.ctrl[act] = cmd
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
    return out


def fit_joint(
    joint: int,
    q_cmd: np.ndarray,
    q_meas: np.ndarray,
    *,
    model: mujoco.MjModel | None = None,
    hz: float = 50.0,
    delays: tuple[int, ...] = DELAYS,
    max_nfev: int = 60,
    kp_scale: float | None = None,
) -> FitResult:
    """Fit one joint. `q_cmd` (n, 6) is the command sequence, `q_meas` (n,) the recorded angle.

    ``kp_scale`` pins the stiffness (e.g. from static sag, `gap.stiffness`); only damping,
    friction, armature and delay are fitted then."""
    model = model if model is not None else load_model()
    nominal = NominalParams.snapshot(model)
    q_cmd = np.asarray(q_cmd, dtype=np.float64)
    q_meas = np.asarray(q_meas, dtype=np.float64)
    base = q_cmd[0].copy()
    n_substeps = max(1, round(1.0 / (hz * model.opt.timestep)))

    # Optimise log-scales: the bounds span ~3 decades (0.02x .. 40x), and on linear scales the
    # trust region converged to a low-stiffness local minimum that mimics delay (2026-10-18).
    free = slice(0, 4) if kp_scale is None else slice(1, 4)

    def full(z: np.ndarray) -> np.ndarray:
        return np.exp(z) if kp_scale is None else np.r_[kp_scale, np.exp(z)]

    def residual(z: np.ndarray, delay: int) -> np.ndarray:
        p = ActuatorParams(*full(z), delay_steps=delay)
        apply_params(model, nominal, joint, p)
        sim = simulate_joint(model, joint, q_cmd, delay, base, n_substeps)
        return sim - q_meas

    best: FitResult | None = None
    for delay in delays:
        # MuJoCo friction loss makes the objective non-smooth: a coarse finite-difference step and a
        # few starts avoid stalling in a low-stiffness minimum that mimics delay. kp, friction and
        # armature still trade off (weakly identifiable from a chirp); delay and damping are
        # identified. Judge fits by held-out prediction error (gap compare), not parameter values.
        sol = min(
            (
                least_squares(
                    residual,
                    x0=x0[free],
                    args=(delay,),
                    bounds=(np.log(LO[free]), np.log(HI[free])),
                    max_nfev=max_nfev,
                    diff_step=0.05,
                )
                for x0 in STARTS
            ),
            key=lambda r: r.cost,
        )
        rmse = float(np.sqrt(np.mean(sol.fun**2)))
        if best is None or rmse < best.rmse_rad:
            params = ActuatorParams(*full(sol.x), delay_steps=delay)
            best = FitResult(joint=JOINTS[joint], params=params, rmse_rad=rmse, nfev=int(sol.nfev))
    assert best is not None
    apply_params(model, nominal, joint, ActuatorParams())  # leave the model at nominal
    return best


def write_fit_yaml(
    results: list[FitResult],
    path: str | Path,
    hz: float = 50.0,
    source: str = "sim.gap.fit (least_squares per joint, delay grid 0..6)",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hz": hz,
        "source": source,
        "joints": {
            r.joint: {
                "kp_scale": float(r.params.kp_scale),
                "damping_scale": float(r.params.damping_scale),
                "frictionloss_scale": float(r.params.frictionloss_scale),
                "armature_scale": float(r.params.armature_scale),
                "delay_steps": int(r.params.delay_steps),
                "rmse_rad": float(r.rmse_rad),
            }
            for r in results
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def params_from_result(r: FitResult) -> dict:
    return asdict(r.params)
