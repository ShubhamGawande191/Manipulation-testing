import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_manipulator.sim.atlas import build_prior, max_scalar, query  # noqa: E402


def test_interval_solver():
    assert max_scalar(np.array([1.0]), np.array([1.0]), np.array([3.0])) == 2.0
    assert max_scalar(np.array([1.0]), np.array([-1.0]), np.array([3.0])) == 4.0
    assert max_scalar(np.array([4.0]), np.array([1.0]), np.array([3.0])) == 0.0


def test_small_prior_is_sane():
    rows, stats = build_prior(n=3000, seed=1)
    assert stats["valid"] > 1000 and len(rows) > 500
    assert all(r["payload_kg"] >= 0 and r["push_n"] >= 0 and r["margin"] <= 1 for r in rows)
    r = rows[0]
    assert query(rows, (r["x"], r["y"], r["z"])) is r
