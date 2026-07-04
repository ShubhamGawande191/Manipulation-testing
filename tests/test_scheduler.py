"""Unit tests: the four chunk executors against a SimClock + mock motor bus.

ScriptedPolicy predicts chunk[k] = obs["step"] + k (+ optional per-call bias),
and the mock bus obs_fn reports the number of actions sent so far — so a
correctly spliced executor sends exactly 0, 1, 2, ... and any stall, repeat,
or backtrack is visible in the sent values.
"""

import numpy as np
import pytest

from robot_manipulator.core.policy_iface import Health
from robot_manipulator.exec.scheduler import (
    RTC,
    AsyncClient,
    FakeInference,
    SimClock,
    SyncReplan,
    TemporalEnsemble,
    ThreadedInference,
)

# 8 Hz -> dt = 0.125, exactly representable in binary floating point, so the
# SimClock's accumulated tick times never drift an ulp away from the fake
# inference channel's ready-times and every trace below is exact.
HZ = 8.0
DT = 1.0 / HZ


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, chunk_len=8, clock=None, latency_s=0.0, call_bias=0.0):
        self.chunk_len = chunk_len
        self.clock = clock
        self.latency_s = latency_s
        self.call_bias = call_bias
        self.calls = 0

    def reset(self):
        self.calls = 0

    def act(self, obs):
        if self.latency_s and self.clock is not None:
            self.clock.sleep(self.latency_s)
        base = obs["step"] + self.call_bias * self.calls
        self.calls += 1
        return np.array([[base + k] for k in range(self.chunk_len)], dtype=np.float64)

    def health(self):
        return Health()


def make_bus():
    bus: list[float] = []

    def send(a):
        bus.append(float(np.asarray(a).ravel()[0]))

    def obs_fn():
        return {"step": len(bus)}

    return bus, send, obs_fn


def test_sync_replan_zero_latency_is_exact_and_regular():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    stats = SyncReplan().run(ScriptedPolicy(chunk_len=4), send, obs_fn, 12, HZ, clock)
    assert bus == list(range(12))
    np.testing.assert_allclose(stats.achieved_hz(), HZ)
    assert stats.jitter_ms() < 1e-9


def test_sync_replan_latency_stalls_the_loop():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    policy = ScriptedPolicy(chunk_len=4, clock=clock, latency_s=1.5 * DT)
    stats = SyncReplan().run(policy, send, obs_fn, 12, HZ, clock)
    assert bus == list(range(12))  # accurate, just slow
    assert stats.achieved_hz() < HZ
    assert stats.jitter_ms() > 0
    assert stats.mean_policy_latency_ms() == pytest.approx(1.5 * DT * 1000)


def test_temporal_ensemble_perfect_predictions_pass_through():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    TemporalEnsemble(m=0.0).run(ScriptedPolicy(chunk_len=3), send, obs_fn, 6, HZ, clock)
    assert bus == list(range(6))


def test_temporal_ensemble_averages_disagreeing_chunks():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    # call c predicts value t + 100*c for timestep t -> the executed action is
    # the mean over the (up to chunk_len) live predictions for each step.
    policy = ScriptedPolicy(chunk_len=3, call_bias=100.0)
    TemporalEnsemble(m=0.0).run(policy, send, obs_fn, 5, HZ, clock)
    np.testing.assert_allclose(bus, [0.0, 51.0, 102.0, 203.0, 304.0])


def test_async_client_stalls_then_replays_stale_chunks():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    policy = ScriptedPolicy(chunk_len=4)
    channel = FakeInference(policy, clock, latency_s=2 * DT)
    stats = AsyncClient(channel).run(policy, send, obs_fn, 10, HZ, clock)
    # 2 ticks with nothing to execute, then chunks restart from stale obs:
    # the repeated "1" is the obs-lag artifact RTC exists to fix.
    assert bus == [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert stats.stale_steps == 2
    assert stats.mean_policy_latency_ms() == pytest.approx(2 * DT * 1000)


def test_rtc_splices_with_no_stall_and_no_backtrack():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    policy = ScriptedPolicy(chunk_len=8)
    channel = FakeInference(policy, clock, latency_s=2 * DT)
    stats = RTC(channel, latency_budget_steps=3).run(policy, send, obs_fn, 12, HZ, clock)
    assert bus == list(range(12))  # perfect continuity across chunk boundaries
    assert stats.stale_steps == 0


def test_rtc_budget_too_small_holds_last_action():
    clock = SimClock()
    bus, send, obs_fn = make_bus()
    policy = ScriptedPolicy(chunk_len=4)
    channel = FakeInference(policy, clock, latency_s=3 * DT)
    stats = RTC(channel, latency_budget_steps=1).run(policy, send, obs_fn, 10, HZ, clock)
    assert stats.stale_steps > 0
    assert len(bus) == 10  # still sends every tick (held action), never crashes


def test_threaded_inference_round_trip():
    policy = ScriptedPolicy(chunk_len=2)
    ch = ThreadedInference(policy)
    try:
        ch.submit({"step": 5})
        deadline = 2.0
        import time

        t0 = time.monotonic()
        chunk = None
        while chunk is None and time.monotonic() - t0 < deadline:
            chunk = ch.poll()
            time.sleep(0.005)
        assert chunk is not None
        np.testing.assert_allclose(chunk.ravel(), [5.0, 6.0])
    finally:
        ch.close()
