"""CLI and run-level tests for `harness reach`. No hardware: the real backend is refused only."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
pytest.importorskip("onnxruntime")

from typer.testing import CliRunner  # noqa: E402

from robot_manipulator.harness.cli import app, run_reach  # noqa: E402

runner = CliRunner()


def test_fake_backend_writes_episodes_summary_and_trajectories(artifact, tmp_path):
    out = tmp_path / "fake"
    summary = run_reach(
        backend="fake",
        policy_path=artifact / "reach.onnx",
        n_episodes=2,
        seed=1,
        out=out,
    )
    assert summary["n_episodes"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert (out / "episodes.parquet").exists()
    assert (out / "summary.json").exists()
    assert (out / "traj" / "traj_home_1.parquet").exists()
    assert (out / "traj" / "traj_reach_2.parquet").exists()
    saved = json.loads((out / "summary.json").read_text())
    assert saved["policy"].endswith("reach.onnx")
    assert len(saved["episodes"]) == 2


def test_sim_backend_runs_with_domain_randomization(artifact, tmp_path):
    summary = run_reach(
        backend="sim",
        policy_path=artifact / "reach.onnx",
        n_episodes=1,
        seed=0,
        dr=True,
        out=tmp_path / "sim",
    )
    assert summary["dr"] is True
    assert summary["n_episodes"] == 1
    ep = summary["episodes"][0]
    assert ep["n_steps"] >= 0 and np.isfinite(ep["final_dist_m"])


def test_explicit_target_is_used(artifact, tmp_path):
    target = np.array([[0.2, 0.0, 0.2]])
    summary = run_reach(
        backend="fake",
        policy_path=artifact / "reach.onnx",
        targets=target,
        seed=0,
        out=tmp_path / "t",
    )
    np.testing.assert_allclose(summary["episodes"][0]["target_m"], [0.2, 0.0, 0.2])


def test_cli_reach_prints_summary(artifact, tmp_path):
    result = runner.invoke(
        app,
        [
            "reach",
            "--backend",
            "fake",
            "--policy",
            str(artifact / "reach.onnx"),
            "--random",
            "1",
            "--seed",
            "2",
            "--out",
            str(tmp_path / "cli"),
        ],
    )
    assert result.exit_code == 0, result.output
    brief = json.loads(result.output)
    assert brief["n_episodes"] == 1 and brief["backend"] == "fake"


def test_cli_rejects_malformed_target(artifact, tmp_path):
    result = runner.invoke(
        app,
        ["reach", "--backend", "fake", "--policy", str(artifact / "reach.onnx"), "--target", "1,2"],
    )
    assert result.exit_code != 0


def test_real_backend_refuses_live_while_uncalibrated(artifact, tmp_path):
    """The calibration guard runs before any serial connection, so this touches no hardware."""
    from robot_manipulator.harness.units import JointMap

    cfg = Path(__file__).resolve().parents[1] / "config" / "sim2real.yaml"
    assert JointMap.from_yaml(cfg).calibrated is False
    with pytest.raises(RuntimeError, match="calibrated"):
        run_reach(
            backend="real",
            policy_path=artifact / "reach.onnx",
            n_episodes=1,
            live=True,
            port="/dev/so101_follower",
            joint_map_path=cfg,
            out=tmp_path / "real",
        )
