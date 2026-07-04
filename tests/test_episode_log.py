"""Unit tests: episode record -> parquet round trip."""

from robot_manipulator.core.episode_log import EpisodeRecord, load_records, write_records


def make_record(run_id="r1", **kw):
    defaults = dict(
        run_id=run_id,
        arm_id="follower_1",
        policy="scripted",
        executor="rtc",
        chunk_len=16,
        precision="fp32",
        policy_latency_ms=42.0,
        e2e_latency_ms=55.0,
        jitter_ms=1.5,
        success=True,
        key_press_f1=1.0,
        time_to_success_s=2.5,
        task="press_key:60",
    )
    defaults.update(kw)
    return EpisodeRecord(**defaults)


def test_parquet_roundtrip(tmp_path):
    path = write_records(
        [make_record(), make_record(success=False, cc_error=7.0)], tmp_path / "runs" / "r1.parquet"
    )
    rows = load_records(path)
    assert len(rows) == 2
    assert rows[0]["run_id"] == "r1" and rows[0]["success"] is True
    assert rows[1]["cc_error"] == 7.0
    assert rows[0]["velocity_error"] is None  # optional metric survives as null


def test_load_concatenates_a_directory(tmp_path):
    d = tmp_path / "runs"
    write_records([make_record("a")], d / "a.parquet")
    write_records([make_record("b")], d / "b.parquet")
    rows = load_records(d)
    assert sorted(r["run_id"] for r in rows) == ["a", "b"]
