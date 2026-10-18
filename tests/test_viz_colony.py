import json
import os
import time

from robot_manipulator.viz.colony import DAY_MS, build_feed

TASKS = """
tasks:
  - {id: measure, title: Measure the fingertip, owner: you, status: todo, updated: 2026-10-18}
  - {id: solve, title: Solve, owner: agent, status: waiting, waits_for: measure}
  - {id: train, title: Train, owner: agent, status: doing}
  - {id: broke, title: Broke, owner: agent, status: failed}
  - {id: shipped, title: Shipped, owner: agent, status: done, updated: 2026-10-18}
  - {id: ancient, title: Ancient, owner: agent, status: done, updated: 2026-01-01}
  - {title: no id, owner: you}
  - not even a mapping
"""


def _touch(path, age_s):
    t = time.time() - age_s
    for f in [path, *path.rglob("*")]:
        os.utime(f, (t, t))


def _run(root, name, summary=None, age_s=0.0):
    d = root / "runs" / "harness" / name
    d.mkdir(parents=True)
    (d / "traj.parquet").write_bytes(b"x" * 100)
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    _touch(d, age_s)
    return d


def test_tasks_become_bots_in_the_repo_zone(tmp_path):
    repo = tmp_path / "myrepo"
    (repo / ".colony").mkdir(parents=True)
    (repo / ".colony" / "tasks.yaml").write_text(TASKS)
    now = int(1000 * time.mktime((2026, 10, 18, 18, 0, 0, 0, 0, -1)))
    t = {x["id"].split(":")[-1]: x for x in build_feed([repo], now_ms=now)["threads"]}
    assert set(t) == {
        "measure",
        "solve",
        "train",
        "broke",
        "shipped",
        "ancient",
    }  # bad ones skipped
    assert all(x["project"] == "myrepo" and x["projectPath"] == str(repo) for x in t.values())
    assert t["measure"]["unread"] and t["measure"]["title"].startswith("👤 you")
    assert not t["solve"]["unread"] and "waits for measure" in t["solve"]["title"]
    assert t["train"]["running"] and t["broke"]["hasError"]
    assert t["shipped"]["prState"] == "merged" and not t["shipped"]["archived"]
    assert t["ancient"]["archived"] and "prState" not in t["ancient"]


def test_only_latest_run_per_kind_and_running_ones(tmp_path):
    repo = tmp_path / "bench"
    real = {"backend": "real", "live": True, "policy": "p/a/reach.onnx"}
    _run(repo, "20261018-100000", {**real, "n_episodes": 5, "success_rate": 0.8}, age_s=7200)
    _run(repo, "20261018-110000", {**real, "n_episodes": 20, "success_rate": 1.0}, age_s=600)
    _run(repo, "20261018-090000", {"backend": "real", "live": False, "n_episodes": 3}, age_s=60)
    _run(repo, "20261018-120000", None, age_s=1)  # being written right now
    _run(repo, "20261018-080000", None, age_s=3600)  # interrupted: not shown
    _run(
        repo,
        "20261010-080000",
        {**real, "n_episodes": 1, "success_rate": 0.0},
        age_s=8 * DAY_MS / 1000,
    )  # stale: not shown
    threads = build_feed([repo])["threads"]
    names = sorted(x["id"].split(":")[-1] for x in threads)
    assert names == ["20261018-110000", "20261018-120000"]  # latest real reach + the running one
    latest = next(x for x in threads if x["id"].endswith("110000"))
    assert latest["prState"] == "merged" and "20/20" in latest["title"]
    assert next(x for x in threads if x["id"].endswith("120000"))["running"]
