"""Unit tests: arbiter fallback state machine and transition log."""

import numpy as np

from robot_manipulator.core.arbiter import Arbiter, ArbiterState
from robot_manipulator.core.policy_iface import Health


class FakePolicy:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self._health = Health()
        self.reset_calls = 0

    def set_health(self, **kw):
        self._health = Health(**kw)

    def reset(self):
        self.reset_calls += 1

    def act(self, obs):
        return np.array([[self.value]])

    def health(self):
        return self._health


def make_arbiter(max_retries=1, teleop=True):
    primary = FakePolicy("primary", 1.0)
    recovery = FakePolicy("recovery", 2.0)
    tele = FakePolicy("teleop", 3.0) if teleop else None
    arb = Arbiter(primary, recovery, tele, max_retries=max_retries, recovery_steps=2)
    return arb, primary, recovery, tele


def test_healthy_primary_stays_active():
    arb, primary, *_ = make_arbiter()
    for _ in range(5):
        action = arb.step({})
        assert action is not None and action[0, 0] == 1.0
    assert arb.state is ArbiterState.ACTIVE
    assert arb.transitions == []


def test_degraded_freezes_then_recovers_then_retries():
    arb, primary, recovery, _ = make_arbiter()
    primary.set_health(confidence=0.0)
    assert arb.step({}) is None  # freeze tick: no command
    assert arb.state is ArbiterState.FROZEN

    a = arb.step({})  # frozen -> recovery, first recovery action
    assert a[0, 0] == 2.0 and arb.state is ArbiterState.RECOVERY
    assert recovery.reset_calls == 1

    arb.step({})  # recovery step 2 of 2
    primary.set_health(confidence=1.0)  # fault cleared during recovery
    assert arb.step({}) is None  # recovery done -> back to ACTIVE
    assert arb.state is ArbiterState.ACTIVE and arb.retries == 1
    assert primary.reset_calls == 1

    assert arb.step({})[0, 0] == 1.0  # primary healthy again


def test_persistent_fault_escalates_to_teleop():
    arb, primary, _, tele = make_arbiter(max_retries=1)
    primary.set_health(latency_ms=500.0)  # degraded via latency, stays broken
    guard = 0
    while arb.state is not ArbiterState.TELEOP and guard < 50:
        arb.step({})
        guard += 1
    assert arb.state is ArbiterState.TELEOP
    assert arb.step({})[0, 0] == 3.0  # teleop policy now drives
    # every hop was logged with a reason
    hops = [(t.src, t.dst) for t in arb.transitions]
    assert hops.count((ArbiterState.ACTIVE, ArbiterState.FROZEN)) == 2
    assert hops[-1][1] is ArbiterState.TELEOP
    assert all(t.reason for t in arb.transitions)


def test_teleop_without_policy_returns_none_and_manual_reset():
    arb, primary, *_ = make_arbiter(max_retries=0, teleop=False)
    primary.set_health(staleness_s=10.0)
    for _ in range(10):
        arb.step({})
    assert arb.state is ArbiterState.TELEOP
    assert arb.step({}) is None
    primary.set_health()  # healthy
    arb.reset()
    assert arb.state is ArbiterState.ACTIVE and arb.retries == 0
    assert arb.step({})[0, 0] == 1.0
