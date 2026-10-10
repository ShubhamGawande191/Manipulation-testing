"""Motion scripts for actuation-gap capture. Generate once; commit the CSVs in config/gap_motions/.

- chirp_<joint>.csv: one joint swept 0.1 -> 2 Hz, +-0.3 rad around HOME, others held
- steps.csv:         each arm joint steps +-0.2 rad and holds 1 s (sequential over joints)
- sine_all.csv:      all arm joints sine at 0.1 Hz, +-0.2 rad, 30 s

Any sample outside the joint limits (with the 0.05 rad margin) is clipped, and the count is
reported. Clipping changes the motion, so check the report before using a motion for fitting.

The chirp peaks near 3.8 rad/s. It is too fast for `--max-vel 0.5` on hardware, so H3 needs
slower motions (lower `fmax` or `amp`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_manipulator.harness.units import CMD_JOINTS
from robot_manipulator.sim.envs.reach import GRIPPER_HOLD, HZ, JOINT_MARGIN
from robot_manipulator.sim.model import JOINTS, joint_ranges, load_model

N = len(CMD_JOINTS)
# Excitation is centred on the mid-range arm pose, not HOME: HOME sits within 0.3 rad of the
# shoulder_lift and elbow limits, and +-0.3 around it clipped 35-45 % of the chirp samples.
MID6 = np.r_[np.zeros(5), GRIPPER_HOLD]
ARM = CMD_JOINTS[:5]


def _base(base: np.ndarray | None) -> np.ndarray:
    return MID6.copy() if base is None else np.asarray(base, dtype=np.float64).reshape(N)


def chirp(
    joint: int,
    f0: float = 0.1,
    f1: float = 2.0,
    amp: float = 0.3,
    duration: float = 20.0,
    hz: float = HZ,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Linear chirp on one joint, instantaneous frequency f0 -> f1 over `duration` seconds."""
    t = np.arange(int(round(duration * hz))) / hz
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / duration)
    motion = np.tile(_base(base), (len(t), 1))
    motion[:, joint] += amp * np.sin(phase)
    return motion


def steps(
    amp: float = 0.2, hold: float = 1.0, hz: float = HZ, base: np.ndarray | None = None
) -> np.ndarray:
    """For each arm joint in turn: hold base, step +amp, hold, step -amp, hold, back to base."""
    b = _base(base)
    n_hold = int(round(hold * hz))
    segments = []
    for j in range(N - 1):
        for offset in (0.0, +amp, 0.0, -amp, 0.0):
            seg = np.tile(b, (n_hold, 1))
            seg[:, j] += offset
            segments.append(seg)
    return np.vstack(segments)


def sine_all(
    amp: float = 0.2,
    freq: float = 0.1,
    duration: float = 30.0,
    hz: float = HZ,
    base: np.ndarray | None = None,
) -> np.ndarray:
    t = np.arange(int(round(duration * hz))) / hz
    motion = np.tile(_base(base), (len(t), 1))
    motion[:, : N - 1] += amp * np.sin(2.0 * np.pi * freq * t)[:, None]
    return motion


def clip_to_limits(motion: np.ndarray, margin: float = JOINT_MARGIN) -> tuple[np.ndarray, int]:
    """Clip the arm joints to their sim limits shrunk by `margin`.

    The gripper is not moved by these motions and is held at GRIPPER_HOLD, which sits on its
    limit, so it is left alone. Returns (clipped, number of samples whose arm values changed).
    """
    lo, hi = joint_ranges(load_model(), JOINTS)
    lo = lo[:5] + margin
    hi = hi[:5] - margin
    clipped = motion.copy()
    clipped[:, :5] = np.clip(motion[:, :5], lo, hi)
    return clipped, int(np.sum(np.any(clipped[:, :5] != motion[:, :5], axis=1)))


def write_csv(path: str | Path, motion: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, motion, delimiter=",", header=",".join(CMD_JOINTS), comments="", fmt="%.6f")
    return path


def read_csv(path: str | Path) -> np.ndarray:
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    if tuple(header) != CMD_JOINTS:
        raise ValueError(f"{path}: header {header} does not match CMD order {CMD_JOINTS}")
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


@dataclass
class MotionReport:
    name: str
    samples: int
    clipped_samples: int


def generate_all(out_dir: str | Path, hz: float = HZ) -> list[MotionReport]:
    """Write the standard motion set into `out_dir`. Returns one report line per file."""
    out = Path(out_dir)
    named: dict[str, np.ndarray] = {}
    for j, joint in enumerate(ARM):
        named[f"chirp_{joint}"] = chirp(j, hz=hz)
    named["steps"] = steps(hz=hz)
    named["sine_all"] = sine_all(hz=hz)

    reports = []
    for name, motion in named.items():
        clipped, n_clip = clip_to_limits(motion)
        write_csv(out / f"{name}.csv", clipped)
        reports.append(MotionReport(name=name, samples=len(clipped), clipped_samples=n_clip))
    return reports
