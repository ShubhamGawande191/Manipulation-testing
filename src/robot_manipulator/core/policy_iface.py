"""Common plugin interface (PLAN item 1).

Everything that produces arm actions — teleop, classical scripts, learned
policies, later world-model MPC — implements the same three methods so the
arbiter and the chunk executors can treat them interchangeably.

An action chunk is a float array of shape (chunk_len, dof). chunk_len == 1 is
a plain single-step policy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

Obs = dict[str, Any]


@dataclass
class HealthLimits:
    """Thresholds above/below which a policy counts as degraded."""

    max_latency_ms: float = 100.0
    max_staleness_s: float = 0.2
    min_confidence: float = 0.3


@dataclass
class Health:
    latency_ms: float = 0.0
    staleness_s: float = 0.0
    confidence: float = 1.0

    def degraded(self, limits: HealthLimits | None = None) -> bool:
        lim = limits or HealthLimits()
        return (
            self.latency_ms > lim.max_latency_ms
            or self.staleness_s > lim.max_staleness_s
            or self.confidence < lim.min_confidence
        )


@runtime_checkable
class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: Obs) -> np.ndarray: ...

    def health(self) -> Health: ...


class BasePolicy(ABC):
    """Optional base that does the health bookkeeping for you.

    Subclasses implement `_act`; latency and observation staleness are measured
    here. Override `_confidence` if the policy has a self-estimate.
    """

    name = "base"

    def __init__(self) -> None:
        self._health = Health()

    def reset(self) -> None:
        self._health = Health()

    def act(self, obs: Obs) -> np.ndarray:
        t0 = time.monotonic()
        chunk = np.atleast_2d(np.asarray(self._act(obs), dtype=np.float64))
        latency_ms = (time.monotonic() - t0) * 1000.0
        obs_t = obs.get("t")
        staleness = max(0.0, time.monotonic() - obs_t) if isinstance(obs_t, (int, float)) else 0.0
        self._health = Health(latency_ms, staleness, self._confidence(obs, chunk))
        return chunk

    def health(self) -> Health:
        return self._health

    @abstractmethod
    def _act(self, obs: Obs) -> np.ndarray: ...

    def _confidence(self, obs: Obs, chunk: np.ndarray) -> float:
        return 1.0
