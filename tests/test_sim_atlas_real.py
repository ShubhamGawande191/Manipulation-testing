import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulator.harness.survey import survey_poses  # noqa: E402
from robot_manipulator.sim.atlas_real import GravityModel, aggregate  # noqa: E402


def test_recovers_known_load_gain_and_ignores_moving_samples():
    grav = GravityModel()
    poses = survey_poses(12, seed=3, model=grav.m)
    samples = []
    for q in poses:
        _, pred = grav(q)
        for _ in range(3):  # held still: measured load = 0.6 x predicted
            samples.append(
                {
                    "source": "test",
                    "q": q,
                    "dq": np.zeros(6),
                    "q_cmd": q,
                    "load": np.r_[0.6 * pred, 0.0],
                }
            )
        samples.append(
            {
                "source": "test",
                "q": q,
                "dq": np.full(6, 1.0),
                "q_cmd": q,
                "load": np.r_[5.0 * pred, 0.0],
            }
        )  # moving: must be ignored
    rows, calib = aggregate(samples, grav)
    assert calib["n_static_samples"] == 36
    for j in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        assert calib[j]["gain"] == pytest.approx(0.6, abs=1e-6)
    assert sum(r["n"] for r in rows) == len(samples)
    assert all(r.get("track_med_deg_elbow_flex", 0.0) == 0.0 for r in rows)
