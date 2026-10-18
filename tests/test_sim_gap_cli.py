import json

import pytest
from typer.testing import CliRunner

pytest.importorskip("mujoco")

from robot_manipulator.sim.gap import cli  # noqa: E402


def test_sim_run_then_compare_is_near_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    runner = CliRunner()
    res = runner.invoke(cli.app, ["run", "--backend", "sim", "--motion", "steps"])
    assert res.exit_code == 0, res.output
    run_dir = next(tmp_path.glob("*_sim"))
    assert (run_dir / "steps.npz").exists()
    assert json.loads((run_dir / "meta.json").read_text())["motions"] == ["steps"]
    res = runner.invoke(cli.app, ["compare", str(run_dir)])
    assert res.exit_code == 0, res.output
    report = json.loads((run_dir / "compare_nominal.json").read_text())["steps"]
    assert max(v["rmse_rad"] for v in report.values()) < 2e-3
    assert (run_dir / "steps_nominal.png").exists()


def test_unknown_motion_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    res = CliRunner().invoke(cli.app, ["run", "--backend", "sim", "--motion", "nope"])
    assert res.exit_code != 0
    assert not any(tmp_path.iterdir())  # rejected before any run folder is created
