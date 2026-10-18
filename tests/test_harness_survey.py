import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulator.harness.survey import collision_free, run_survey, survey_poses  # noqa: E402
from robot_manipulator.sim.envs.reach import TARGET_BOX  # noqa: E402
from robot_manipulator.sim.kinematics import FK  # noqa: E402


def test_poses_spread_collision_free_in_workspace():
    fk = FK()
    poses = survey_poses(8, seed=1, model=fk.model)
    assert poses.shape == (8, 6)
    ee = np.array([fk.ee_pos(q) for q in poses])
    assert (ee >= TARGET_BOX[0] - 1e-9).all() and (ee <= TARGET_BOX[1] + 1e-9).all()
    assert all(collision_free(fk.model, q) for q in poses)
    d = np.linalg.norm(ee[:, None] - ee[None], axis=-1) + np.eye(8)
    assert d.min() > 0.03  # spread out, no duplicates


def test_sim_survey_reaches_and_holds(tmp_path):
    s = run_survey(backend="sim", n=4, hold_s=0.3, max_vel=1.0, seed=2, out=tmp_path)
    assert s["reached"] == 5  # HOME + 4 poses
    errs = [np.abs(r["err_deg"][:5]).max() for r in s["results"] if r.get("reached")]
    assert max(errs) < 3.0
    assert (tmp_path / "summary.json").exists()
