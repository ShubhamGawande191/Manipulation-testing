"""Interchangeable chunk executors behind one interface (PLAN item 5).

Executors: `SyncReplan`, `TemporalEnsemble`, `AsyncClient` (policy server
pattern), `RTC` (real-time chunking). Stretch items (bidirectional decoding,
streaming/partial-denoise DP, consistency-distilled DP) slot in as further
`ChunkExecutor` subclasses.

Everything runs against a `Clock` so unit tests drive a `SimClock`
deterministically; hardware uses `WallClock`. The async executors talk to an
`InferenceChannel` — `ThreadedInference` in production, a fake with scripted
latency in tests.

Phase 1 measures exactly what these emit: `ExecStats` (policy latency,
achieved Hz, jitter, stale steps) against MIDI-scored task success.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from robot_manipulator.core.policy_iface import Obs, Policy

Send = Callable[[np.ndarray], None]
ObsFn = Callable[[], Obs]


class WallClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, t: float) -> None:
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)


class SimClock:
    """Deterministic clock for tests; `sleep` also models compute latency."""

    def __init__(self, t0: float = 0.0) -> None:
        self._t = t0

    def now(self) -> float:
        return self._t

    def sleep_until(self, t: float) -> None:
        self._t = max(self._t, t)

    def sleep(self, dt: float) -> None:
        self._t += dt


@dataclass
class ExecStats:
    executor: str
    sent_t: list[float] = field(default_factory=list)
    policy_latency_s: list[float] = field(default_factory=list)
    stale_steps: int = 0  # control ticks served by a held/repeated action (or none)

    def achieved_hz(self) -> float:
        if len(self.sent_t) < 2:
            return 0.0
        return 1.0 / float(np.mean(np.diff(self.sent_t)))

    def jitter_ms(self) -> float:
        if len(self.sent_t) < 3:
            return 0.0
        return float(np.std(np.diff(self.sent_t)) * 1000.0)

    def mean_policy_latency_ms(self) -> float:
        if not self.policy_latency_s:
            return 0.0
        return float(np.mean(self.policy_latency_s) * 1000.0)


class InferenceChannel(ABC):
    """Submit an observation snapshot, poll later for the resulting chunk."""

    latencies_s: list[float]

    @abstractmethod
    def submit(self, obs: Obs) -> None: ...

    @abstractmethod
    def poll(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class ThreadedInference(InferenceChannel):
    """Runs policy.act on a worker thread; at most one request in flight."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.latencies_s: list[float] = []
        self._obs: Obs | None = None
        self._result: np.ndarray | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, obs: Obs) -> None:
        with self._lock:
            self._obs = obs
        self._wake.set()

    def poll(self) -> np.ndarray | None:
        with self._lock:
            result, self._result = self._result, None
        return result

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            with self._lock:
                obs, self._obs = self._obs, None
            if obs is None:
                continue
            t0 = time.monotonic()
            chunk = self.policy.act(obs)
            with self._lock:
                self._result = chunk
            self.latencies_s.append(time.monotonic() - t0)


class FakeInference(InferenceChannel):
    """Test channel: the chunk (computed from the obs at submit time) becomes
    visible `latency_s` after submit, measured on the injected clock."""

    def __init__(self, policy: Policy, clock: SimClock, latency_s: float) -> None:
        self.policy = policy
        self.clock = clock
        self.latency_s = latency_s
        self.latencies_s: list[float] = []
        self._pending: tuple[float, np.ndarray] | None = None

    def submit(self, obs: Obs) -> None:
        self._pending = (self.clock.now() + self.latency_s, self.policy.act(obs))

    def poll(self) -> np.ndarray | None:
        if self._pending is not None and self.clock.now() >= self._pending[0]:
            chunk = self._pending[1]
            self._pending = None
            self.latencies_s.append(self.latency_s)
            return chunk
        return None


class ChunkExecutor(ABC):
    name = "base"

    @abstractmethod
    def run(
        self,
        policy: Policy,
        send: Send,
        obs_fn: ObsFn,
        n_steps: int,
        hz: float,
        clock=None,
    ) -> ExecStats: ...


class SyncReplan(ChunkExecutor):
    """Execute the whole chunk, then block on the policy for the next one.

    Inference latency stalls the control loop — that is the phenomenon this
    executor exists to measure against the others.
    """

    name = "sync_replan"

    def run(self, policy, send, obs_fn, n_steps, hz, clock=None) -> ExecStats:
        clock = clock or WallClock()
        dt = 1.0 / hz
        stats = ExecStats(self.name)
        next_t = clock.now()
        steps = 0
        while steps < n_steps:
            t0 = clock.now()
            chunk = policy.act(obs_fn())
            stats.policy_latency_s.append(clock.now() - t0)
            for a in np.atleast_2d(chunk):
                if steps >= n_steps:
                    break
                clock.sleep_until(next_t)
                send(a)
                stats.sent_t.append(clock.now())
                steps += 1
                next_t = max(next_t, clock.now()) + dt
        return stats


class TemporalEnsemble(ChunkExecutor):
    """ACT-style: query every step, execute the weighted average of every
    still-live prediction for the current step. Weights exp(-m * age), oldest
    prediction first (m=0 -> uniform)."""

    name = "temporal_ensemble"

    def __init__(self, m: float = 0.01) -> None:
        self.m = m

    def run(self, policy, send, obs_fn, n_steps, hz, clock=None) -> ExecStats:
        clock = clock or WallClock()
        dt = 1.0 / hz
        stats = ExecStats(self.name)
        preds: dict[int, list[np.ndarray]] = {}
        next_t = clock.now()
        for step in range(n_steps):
            t0 = clock.now()
            chunk = np.atleast_2d(policy.act(obs_fn()))
            stats.policy_latency_s.append(clock.now() - t0)
            for k, a in enumerate(chunk):
                preds.setdefault(step + k, []).append(np.asarray(a, dtype=np.float64))
            plist = preds.pop(step)  # chronological: oldest first
            w = np.exp(-self.m * np.arange(len(plist)))
            action = np.average(np.stack(plist), axis=0, weights=w)
            clock.sleep_until(next_t)
            send(action)
            stats.sent_t.append(clock.now())
            next_t = max(next_t, clock.now()) + dt
        return stats


class AsyncClient(ChunkExecutor):
    """Policy-server pattern: inference runs elsewhere; the control loop never
    blocks. Chunks are executed from the start on arrival, so obs-lag shows up
    as repeated/backtracked actions — RTC below is the fix; the gap between
    them is a Phase 1 result."""

    name = "async_client"

    def __init__(self, channel: InferenceChannel | None = None) -> None:
        self._channel = channel

    def run(self, policy, send, obs_fn, n_steps, hz, clock=None) -> ExecStats:
        clock = clock or WallClock()
        ch = self._channel or ThreadedInference(policy)
        owned = self._channel is None
        dt = 1.0 / hz
        stats = ExecStats(self.name)
        chunk: np.ndarray | None = None
        idx = 0
        last: np.ndarray | None = None
        ch.submit(obs_fn())
        next_t = clock.now()
        for _ in range(n_steps):
            clock.sleep_until(next_t)
            fresh = ch.poll()
            if fresh is not None:
                chunk, idx = np.atleast_2d(fresh), 0
            if chunk is not None and idx < len(chunk):
                a = chunk[idx]
                idx += 1
                last = a
            elif last is not None:
                a = last
                stats.stale_steps += 1
            else:
                stats.stale_steps += 1
                next_t = max(next_t, clock.now()) + dt
                continue
            send(a)
            stats.sent_t.append(clock.now())
            if fresh is not None:
                ch.submit(obs_fn())  # pipeline the next request
            next_t = max(next_t, clock.now()) + dt
        stats.policy_latency_s = list(ch.latencies_s)
        if owned:
            ch.close()
        return stats


class RTC(ChunkExecutor):
    """Real-time chunking: request the next chunk while the current one still
    has `latency_budget_steps` actions left; on arrival, splice in at the
    action matching the current step (actions for already-elapsed steps are
    dropped). No stall, no backtrack — if the budget covers the latency."""

    name = "rtc"

    def __init__(
        self, channel: InferenceChannel | None = None, latency_budget_steps: int = 3
    ) -> None:
        self._channel = channel
        self.latency_budget_steps = latency_budget_steps

    def run(self, policy, send, obs_fn, n_steps, hz, clock=None) -> ExecStats:
        clock = clock or WallClock()
        ch = self._channel or ThreadedInference(policy)
        owned = self._channel is None
        dt = 1.0 / hz
        stats = ExecStats(self.name)

        ch.submit(obs_fn())
        chunk = ch.poll()
        while chunk is None:  # startup: nothing to execute yet
            clock.sleep_until(clock.now() + dt)
            chunk = ch.poll()
        chunk = np.atleast_2d(chunk)
        idx = 0
        submitted_at: int | None = None
        last: np.ndarray | None = None
        next_t = clock.now()
        for step in range(n_steps):
            clock.sleep_until(next_t)
            if submitted_at is not None:
                fresh = ch.poll()
                if fresh is not None:
                    elapsed = step - submitted_at
                    chunk, idx = np.atleast_2d(fresh), elapsed
                    submitted_at = None
            if submitted_at is None and len(chunk) - idx <= self.latency_budget_steps:
                ch.submit(obs_fn())
                submitted_at = step
            if idx < len(chunk):
                a = chunk[idx]
                idx += 1
                last = a
            else:  # budget didn't cover the latency; hold last action
                a = last
                stats.stale_steps += 1
            send(a)
            stats.sent_t.append(clock.now())
            next_t = max(next_t, clock.now()) + dt
        stats.policy_latency_s = list(ch.latencies_s)
        if owned:
            ch.close()
        return stats


EXECUTORS: dict[str, type[ChunkExecutor]] = {
    SyncReplan.name: SyncReplan,
    TemporalEnsemble.name: TemporalEnsemble,
    AsyncClient.name: AsyncClient,
    RTC.name: RTC,
}
