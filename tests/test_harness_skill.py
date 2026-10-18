import numpy as np


def test_oscillation_detects_hunting_but_not_a_smooth_approach():
    from robot_manipulator.harness.skill import Oscillation, RunContext

    hz = 50.0
    t = np.arange(0, 2.0, 1 / hz)
    # hunting: wrist_flex dithers +-0.02 rad at 5 Hz around a fixed point
    q_hunt = 0.02 * np.sin(2 * np.pi * 5 * t)
    dq_hunt = 0.02 * 2 * np.pi * 5 * np.cos(2 * np.pi * 5 * t)
    # smooth: exponential approach with a single small overshoot
    q_ok = 0.3 * (1 - np.exp(-3 * t))
    dq_ok = 0.9 * np.exp(-3 * t)

    def run(q, dq):
        o = Oscillation()
        for k in range(len(t)):
            qq = np.zeros(6)
            qq[3] = q[k]
            dd = np.zeros(6)
            dd[3] = dq[k]
            if o.check({"q": qq, "dq": dd}, RunContext({}, k, t[k], t[k])) == "failure":
                return t[k]
        return None

    assert run(q_hunt, dq_hunt) is not None and run(q_hunt, dq_hunt) <= 1.0
    assert run(q_ok, dq_ok) is None
