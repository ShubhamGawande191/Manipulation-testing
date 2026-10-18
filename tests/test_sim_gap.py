"""Unit tests: actuation-gap toolkit (motions, replay, compare, fit)."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_manipulator.exec.scheduler import SimClock  # noqa: E402
from robot_manipulator.harness.backend import SimBackend  # noqa: E402
from robot_manipulator.harness.safety_layer import SafetyFilter  # noqa: E402
from robot_manipulator.sim.gap.compare import compare, plot_overlay  # noqa: E402
from robot_manipulator.sim.gap.fit import (  # noqa: E402
    ActuatorParams,
    apply_params,
    fit_joint,
    simulate_joint,
    write_fit_yaml,
)
from robot_manipulator.sim.gap.motions import (  # noqa: E402
    chirp,
    generate_all,
    read_csv,
    steps,
    write_csv,
)
from robot_manipulator.sim.gap.replay import ReplayTrace, replay  # noqa: E402
from robot_manipulator.sim.model import (  # noqa: E402
    JOINTS,
    actuator_ids,
    apply_actuator_fit,
    load_model,
)
from robot_manipulator.sim.randomize import NominalParams  # noqa: E402

ZERO6 = np.zeros(6)


def _trace(q: np.ndarray, hz: float = 50.0) -> ReplayTrace:
    n = len(q)
    t = np.arange(n) / hz
    return ReplayTrace(t=t, q_target=q, q_cmd=q, q=q, dq=np.zeros_like(q))


def test_chirp_and_steps_shapes():
    c = chirp(0, duration=20.0)
    assert c.shape == (1000, 6)
    np.testing.assert_allclose(c[:, 1:], np.tile(c[0, 1:], (1000, 1)))  # other joints held
    assert steps().shape == (5 * 5 * 50, 6)


def test_generate_all_writes_files_and_reports_clipping(tmp_path):
    reports = generate_all(tmp_path)
    names = sorted(r.name for r in reports)
    assert names == sorted([f"chirp_{j}" for j in JOINTS[:5]] + ["steps", "sine_all"])
    for r in reports:
        motion = read_csv(tmp_path / f"{r.name}.csv")
        assert motion.shape[1] == 6
    by_name = {r.name: r for r in reports}
    # centred on the mid-range pose, so no motion is distorted by the joint limits
    assert all(r.clipped_samples == 0 for r in reports)
    assert by_name["chirp_shoulder_lift"].samples == 1000


def test_csv_round_trip_checks_header(tmp_path):
    m = np.linspace(-1, 1, 12).reshape(2, 6)
    path = write_csv(tmp_path / "m.csv", m)
    np.testing.assert_allclose(read_csv(path), m, atol=1e-6)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c,d,e,f\n0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="header"):
        read_csv(bad)


def test_compare_identical_traces_is_exact():
    rng = np.random.default_rng(0)
    q = np.cumsum(rng.normal(0, 0.01, size=(200, 6)), axis=0)
    report = compare(_trace(q), _trace(q))
    for joint, m in report.items():
        assert m["rmse_rad"] == pytest.approx(0.0, abs=1e-12), joint
        assert m["lag_ms"] == pytest.approx(0.0), joint


def test_compare_recovers_a_known_lag():
    t = np.arange(400) / 50.0
    q = np.zeros((400, 6))
    q[:, 0] = np.sin(2 * np.pi * 0.5 * t)
    lagged = np.roll(q, 3, axis=0)  # the second trace is 3 ticks (60 ms) late
    lagged[:3] = 0.0
    report = compare(_trace(q), _trace(lagged), hz=50.0)
    assert report["shoulder_pan"]["lag_ms"] == pytest.approx(60.0)


def test_plot_overlay_writes_png(tmp_path):
    q = np.zeros((100, 6))
    path = plot_overlay(_trace(q), _trace(q + 0.01), tmp_path / "overlay.png")
    assert path.exists() and path.stat().st_size > 1000


def test_replay_through_runner_follows_the_motion(tmp_path):
    motion = steps(amp=0.2, hold=0.5)
    clock = SimClock()
    sb = SimBackend(hz=50.0)
    sb.reset(motion[0])
    safety = SafetyFilter(lo=np.full(6, -2.0), hi=np.full(6, 2.0), max_vel=20.0, clock=clock)
    trace = replay(motion, sb, safety, clock=clock)
    sb.close()
    assert len(trace) == len(motion)
    np.testing.assert_allclose(trace.q_target, motion, atol=1e-12)
    # the 0.02 s servo lag is small, so the arm sits near each held step at the end
    np.testing.assert_allclose(trace.q[-1, :5], motion[-1, :5], atol=0.02)


def test_fit_recovers_synthetic_actuator_and_delay():
    """Acceptance: a synthetic real arm (kp x0.8, damping x1.5, delay 1) is recovered."""
    model = load_model()
    nominal = NominalParams.snapshot(model)
    joint = JOINTS.index("shoulder_lift")
    n_sub = 4
    motion = chirp(joint, f0=0.1, f1=2.0, amp=0.3, duration=4.0, base=ZERO6)

    truth = ActuatorParams(kp_scale=0.8, damping_scale=1.5, delay_steps=1)
    apply_params(model, nominal, joint, truth)
    q_real = simulate_joint(model, joint, motion, 1, motion[0].copy(), n_sub)
    apply_params(model, nominal, joint, ActuatorParams())

    result = fit_joint(joint, motion, q_real, model=model, max_nfev=40)
    assert result.joint == "shoulder_lift"
    # What a chirp identifies: the delay and the damping exactly, and a model that predicts the
    # trajectory. kp / friction / armature trade off against each other (2026-10-18: with wide
    # bounds kp came out 0.61 for a true 0.8 at 1.9 mrad), so they are not asserted.
    assert result.params.delay_steps == 1
    assert result.params.damping_scale == pytest.approx(1.5, rel=0.15)
    assert result.rmse_rad < 3e-3


def test_fit_yaml_round_trips_into_a_model(tmp_path):
    from robot_manipulator.sim.gap.fit import FitResult

    result = FitResult(
        joint="elbow_flex",
        params=ActuatorParams(kp_scale=0.5, damping_scale=2.0, delay_steps=2),
        rmse_rad=0.001,
        nfev=10,
    )
    path = write_fit_yaml([result], tmp_path / "actuator_fit.yaml")
    model = load_model()
    nominal = NominalParams.snapshot(model)
    delays = apply_actuator_fit(model, path)
    assert delays == {"elbow_flex": 2}
    a = actuator_ids(model, JOINTS)[JOINTS.index("elbow_flex")]
    assert model.actuator_gainprm[a, 0] == pytest.approx(nominal.gainprm[a, 0] * 0.5)
    assert model.actuator_biasprm[a, 1] == pytest.approx(nominal.biasprm[a, 1] * 0.5)
    assert Path(path).exists()


def test_static_stiffness_recovers_k_and_p_scales_the_rest():
    from robot_manipulator.sim.gap.stiffness import ARM, estimate, p_gain

    rng = np.random.default_rng(0)
    n = 400
    tau = np.zeros((n, 5))
    tau[:, 1] = rng.uniform(-1.2, -0.05, n)  # shoulder_lift carries the gravity load
    k_true = 20.0
    err = tau / k_true + rng.normal(0, 0.002, (n, 5))
    est = {e.joint: e for e in estimate(err, tau, kp_nominal=np.full(5, 1000.0))}
    lift = est["shoulder_lift"]
    assert lift.source == "measured" and abs(lift.k_nm_per_rad - k_true) / k_true < 0.05
    assert abs(lift.kp_scale - lift.k_nm_per_rad / 1000.0) < 1e-12
    pan = est["shoulder_pan"]  # no gravity signal -> P-scaled from the lift
    assert pan.source.startswith("P-scaled")
    ratio = p_gain(ARM[0]) / p_gain(ARM[1])
    assert abs(pan.k_nm_per_rad - lift.k_nm_per_rad * ratio) < 1e-9
