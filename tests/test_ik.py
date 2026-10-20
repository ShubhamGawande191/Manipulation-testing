import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulator.ik.analytic import AnalyticIK  # noqa: E402
from robot_manipulator.ik.numeric import dls, solved  # noqa: E402


@pytest.fixture(scope="module")
def ik():
    return AnalyticIK()


def test_analytic_round_trip_including_reach_back_poses(ik):
    rng = np.random.default_rng(3)
    for _ in range(60):
        q = np.r_[rng.uniform(ik.lo[:5] + 0.05, ik.hi[:5] - 0.05), 0.0]
        t = ik.task(q)
        sol = ik.solve(t[:3], t[3], roll=q[4], seed=q)
        assert sol is not None
        assert np.linalg.norm(ik.task(sol)[:3] - t[:3]) < 1e-5
        assert abs(ik.task(sol)[3] - t[3]) < 1e-6
        assert ik.within_limits(sol)


def test_out_of_reach_returns_none(ik):
    assert ik.solve(np.array([0.9, 0.0, 0.1]), 0.0) is None


def test_dls_converges_from_a_nearby_seed(ik):
    q = np.array([0.3, 0.2, 0.5, 0.9, -1.4, 0.0])
    t = ik.task(q)
    seed = q + np.r_[0.1, -0.1, 0.1, -0.1, 0, 0]
    sol, _ = dls(ik, t[:3], t[3], q[4], seed)
    assert solved(ik, sol, t[:3], t[3])
