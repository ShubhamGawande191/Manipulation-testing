import numpy as np
import pytest
import typer

from robot_manipulator.tasks.teleop_staged import (
    JOINTS,
    STAGE_ORDER,
    TrackLog,
    parse_pid,
    stage_targets,
)


def test_parse_pid():
    assert parse_pid(["elbow_flex=24,0,32", "shoulder_lift=20,2,32"]) == {
        "elbow_flex": (24, 0, 32),
        "shoulder_lift": (20, 2, 32),
    }
    assert parse_pid([]) == {}


@pytest.mark.parametrize("bad", ["elbow=24,0,32", "elbow_flex=24,0", "elbow_flex=300,0,32"])
def test_parse_pid_rejects(bad):
    with pytest.raises(typer.BadParameter):
        parse_pid([bad])


def test_stage_targets_one_joint_at_a_time():
    start = np.zeros(6)
    goal = np.arange(1.0, 7.0)
    steps = stage_targets(start, goal)
    assert [name for name, _ in steps] == list(STAGE_ORDER)
    prev = start
    for name, wp in steps:
        changed = np.flatnonzero(wp != prev)
        assert list(changed) == [JOINTS.index(name)]
        prev = wp
    np.testing.assert_array_equal(steps[-1][1], goal)


def test_stage_targets_reverse_order_parks():
    here, start = np.arange(1.0, 7.0), np.zeros(6)
    steps = stage_targets(here, start, tuple(reversed(STAGE_ORDER)))
    assert steps[-1][0] == "shoulder_lift"
    np.testing.assert_array_equal(steps[-1][1], start)


def test_tracklog_summary_and_save(tmp_path):
    log = TrackLog()
    for k in range(40):
        cmd = np.full(6, 10.0)
        pos = cmd - np.array([0, 1, 2, 0, 0, 0])  # constant lag on lift/elbow
        log.add(k * 0.02, cmd, pos, np.full(6, 100.0) if k % 5 == 0 else None)
    s = log.summary()
    assert s["elbow_flex"]["steady_err"] == pytest.approx(2.0)
    assert s["shoulder_lift"]["mean_abs_err"] == pytest.approx(1.0)
    assert s["shoulder_pan"]["max_abs_err"] == 0.0
    assert s["gripper"]["load_mean"] == pytest.approx(100.0)
    path = log.save(tmp_path / "t.parquet")
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    assert t.num_rows == 40 and "load_elbow_flex" in t.column_names
