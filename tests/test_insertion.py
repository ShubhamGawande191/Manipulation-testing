import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_manipulator.sim import insertion as ins  # noqa: E402


def test_wilson_interval_known_values():
    lo, hi = ins.wilson(19, 30)
    assert lo == pytest.approx(0.455, abs=0.005)
    assert hi == pytest.approx(0.781, abs=0.005)
    assert ins.wilson(30, 30)[1] == 1.0
    assert ins.wilson(0, 30)[0] == 0.0


def test_mcnemar_exact():
    assert ins.mcnemar_p(0, 0) == 1.0
    assert ins.mcnemar_p(5, 5) == 1.0
    # 10 discordant pairs all one way: 2 * 0.5**10
    assert ins.mcnemar_p(10, 0) == pytest.approx(2 / 1024)


def test_ideal_insertion_stays_out_of_the_walls():
    """The contact model must not let the peg sink into the fixture (that faked successes)."""
    model = ins.build(1.0)
    r = ins.run_episode(model, ins.Factors(), np.random.default_rng([0, 0]))
    assert r["ok"]
    assert r["pen_mm"] < ins.MAX_PEN_MM


def test_big_offset_cannot_insert():
    """3 mm off a hole with 0.5 mm radial room and no chamfer must jam, not push through."""
    model = ins.build(1.0)
    r = ins.run_episode(
        model, ins.Factors(), np.random.default_rng(0), hole_xy=ins.HOLE_XY + [0.003, 0.0]
    )
    assert not r["ok"]
    assert r["pen_mm"] < ins.MAX_PEN_MM


def test_chamfer_captures_an_offset_the_plain_hole_cannot():
    """1.2 mm off: no chamfer jams; a 1 mm 45 deg chamfer steers the peg in (ramps face in)."""
    off = ins.HOLE_XY + [0.0012, 0.0]
    plain = ins.run_episode(ins.build(1.0), ins.Factors(), np.random.default_rng(0), hole_xy=off)
    chamf = ins.run_episode(
        ins.build(1.0, 1.0), ins.Factors(), np.random.default_rng(0), hole_xy=off
    )
    assert not plain["ok"]
    assert chamf["ok"]
    assert chamf["pen_mm"] < ins.MAX_PEN_MM


def test_scripted_commands_descend_straight_and_stay_above_table():
    m = ins.scripted_commands(lift_m=0.03)
    tips = ins.peg_tips(ins.build(2.0), m)
    z = tips[:, 2]
    assert np.all(np.diff(z) <= 1e-6)  # hover, then monotone descent, then hold
    assert 1000 * (z[0] - z[-1]) == pytest.approx(22.0, abs=0.5)  # 5 mm above top -> 17 mm below
    lateral = 1000 * np.linalg.norm(tips[:, :2] - ins.HOLE_XY, axis=1)
    assert lateral.max() < 0.5  # straight down over the hole
    assert z.min() + ins.PEG_LEN - ins.TABLE_Z > 0.05  # fingertip >= 50 mm above the table
