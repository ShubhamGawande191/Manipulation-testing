import json
import os
import time

from robot_manipulator.viz.colony import ZONE_REAL, ZONE_SIM, ZONE_TWIN, build_feed

RUNBOOK = """# runbook
## Status of this setup (home bench)

| Stage | Status | Evidence |
|---|---|---|
| 1 Hardware sanity | ✅ | ok |
| 7 Extrinsics | 🟡 touch session pending | `runs/x` |
"""


def _run(root, name, summary=None, age_s=0.0):
    d = root / "runs" / "harness" / name
    d.mkdir(parents=True)
    (d / "traj.parquet").write_bytes(b"x" * 100)
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    t = time.time() - age_s
    for f in [d, *d.iterdir()]:
        os.utime(f, (t, t))
    return d


def test_feed_maps_runs_and_runbook_stages_to_bot_states(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "DIGITAL-TWIN-RUNBOOK.md").write_text(RUNBOOK)
    _run(
        tmp_path,
        "20261018-100000",
        {
            "backend": "real",
            "live": True,
            "policy": "p/a/reach.onnx",
            "n_episodes": 20,
            "success_rate": 1.0,
        },
        age_s=600,
    )
    _run(
        tmp_path,
        "20261018-110000",
        {"backend": "sim", "policy": "p/b/reach.onnx", "n_episodes": 10, "success_rate": 0.5},
        age_s=600,
    )
    _run(tmp_path, "20261018-120000", None, age_s=1)  # being written right now
    _run(tmp_path, "20261018-090000", None, age_s=3600)  # interrupted
    threads = {t["id"]: t for t in build_feed(tmp_path)["threads"]}

    real = threads["so101:harness:20261018-100000"]
    assert real["project"] == ZONE_REAL and real["prState"] == "merged" and not real["hasError"]
    assert "20/20" in real["title"] and real["sizeBytes"] >= 100
    sim = threads["so101:harness:20261018-110000"]
    assert sim["project"] == ZONE_SIM and sim["hasError"] and "prState" not in sim
    assert threads["so101:harness:20261018-120000"]["running"]
    assert threads["so101:harness:20261018-090000"]["archived"]  # interrupted: no result
    stage7 = threads["so101:stage:7"]
    assert stage7["project"] == ZONE_TWIN and stage7["unread"]
    assert not threads["so101:stage:1"]["unread"]
    assert all(t["id"].startswith("so101:") for t in threads.values())
