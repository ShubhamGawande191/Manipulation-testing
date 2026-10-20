"""Lab board for Bot Crossing (github.com/Station-Sciences/bot-crossing): what is next, per repo.

Bot Crossing draws every coding-agent thread as a bot on a hex zone per repo. The
``integrations/bot-crossing/lab.mjs`` adapter reads the feed written here and adds, to each repo's
zone:

- **tasks** from ``<repo>/.colony/tasks.yaml``: the work that is next, whoever does it. Your
  to-dos (``owner: you``, ``status: todo``) hold a ``?``: data collection, measurements, live
  runs, anything an agent cannot do itself. Work in progress (``doing``) hammers, ``failed``
  slumps with ``!``, ``waiting`` (on another task) potters, and ``done`` celebrates for two days,
  then leaves the map. ``<repo>/.colony/tasks.private.yaml`` (same format, gitignored) adds tasks
  that must not be pushed with a public repo.
- **robot runs** (repos with ``runs/``, i.e. this one): only the latest run of each kind from the
  last two days, plus anything being written right now. Older runs stay on disk, off the map.

Repos: every ``<root>/*/.colony/tasks.yaml`` under ``~/Desktop``, ``~/projects``, ``~/code`` and
``~/src``, plus the paths in ``~/.config/lab-colony/projects.yaml`` (``projects: [path, ...]``).
Agent threads themselves come from Bot Crossing's own Claude Code / Codex adapters.

    uv run colony-feed               # write once
    uv run colony-feed --watch 3     # keep it fresh while Bot Crossing is open

Read-only over the repos; the only file written is the feed (atomic replace).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

REPO = Path(__file__).resolve().parents[3]
HOME = Path.home()
FEED = Path(
    os.environ.get("LAB_COLONY_FEED", HOME / ".local" / "state" / "lab-colony" / "threads.json")
)
CONFIG = HOME / ".config" / "lab-colony" / "projects.yaml"
SEARCH_ROOTS = [HOME / d for d in ("Desktop", "projects", "code", "src")]
TASKS_FILE = Path(".colony") / "tasks.yaml"
PRIVATE_TASKS_FILE = Path(".colony") / "tasks.private.yaml"  # gitignored overlay, same format
RUN_WINDOW_DAYS = 2.0
DONE_SHOWN_DAYS = 2.0
RUNNING_S = 30.0
TRAINING_RUNNING_S = 120.0
PASS_RATE = 0.9
DAY_MS = 86_400_000
OWNERS = ("you", "agent")
STATUSES = ("todo", "doing", "waiting", "done", "failed")

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------- helpers
def _mtime_ms(path: Path) -> int:
    times = [path.stat().st_mtime]
    if path.is_dir():
        times += [c.stat().st_mtime for c in path.iterdir()]
    return int(1000 * max(times))


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # missing, or being written right now


def _base(tid: str, title: str, repo: Path, **extra: Any) -> dict[str, Any]:
    t = {
        "id": f"lab:{tid}",
        "title": title,
        "preview": "",
        "project": repo.name,  # same name + path as the agent threads: one zone per repo
        "projectPath": str(repo),
        "worktree": "",
        "cwd": str(repo),
        "gitBranch": "",
        "model": "",
        "effort": "",
        "createdAt": 0,
        "lastActivityAt": 0,
        "lastFocusedAt": 0,
        "running": False,
        "unread": False,
        "hasError": False,
        "archived": False,
        "sizeBytes": 0,
        "source": "lab",
        "canOpen": True,
        "ref": {"path": str(repo)},
    }
    t.update({k: v for k, v in extra.items() if v is not None})
    return t


# ---------------------------------------------------------------- tasks
def _date_ms(v: Any, fallback: int) -> int:
    if isinstance(v, datetime):
        return int(1000 * v.timestamp())
    if isinstance(v, date):
        return int(1000 * datetime(v.year, v.month, v.day, 12).timestamp())
    if isinstance(v, str):
        try:
            return _date_ms(datetime.fromisoformat(v), fallback)
        except ValueError:
            pass
    return fallback


def task_threads(repo: Path, now_ms: int) -> list[dict]:
    """Bots for ``<repo>/.colony/tasks.yaml`` plus the private overlay; malformed entries or files
    are skipped, not fatal."""
    out = []
    for name in (TASKS_FILE, PRIVATE_TASKS_FILE):
        out += _file_tasks(repo, repo / name, now_ms)
    return out


def _file_tasks(repo: Path, f: Path, now_ms: int) -> list[dict]:
    try:
        doc = yaml.safe_load(f.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    file_ms = _mtime_ms(f)
    out = []
    for task in doc.get("tasks") or []:
        if not isinstance(task, dict) or not task.get("id") or not task.get("title"):
            continue
        owner = task.get("owner") if task.get("owner") in OWNERS else "agent"
        status = task.get("status") if task.get("status") in STATUSES else "todo"
        updated = _date_ms(task.get("updated"), file_ms)
        old_done = status == "done" and now_ms - updated > DONE_SHOWN_DAYS * DAY_MS
        who = "👤 you" if owner == "you" else "🤖 agent"
        title = f"{who} · {task['title']}"
        if status == "waiting" and task.get("waits_for"):
            title += f" (waits for {task['waits_for']})"
        link = task.get("link")
        target = (repo / link) if link else f
        out.append(
            _base(
                f"task:{repo.name}:{task['id']}",
                title,
                repo,
                preview=str(task.get("detail", ""))[:240],
                model=status,
                createdAt=updated,
                lastActivityAt=updated,
                running=status == "doing",
                unread=owner == "you" and status == "todo",
                hasError=status == "failed",
                prState="merged" if status == "done" and not old_done else None,
                archived=old_done,
                sizeBytes=2000 * (1 + len(str(task.get("detail", "")))),
                ref={"path": str(target if target.exists() else repo)},
            )
        )
    return out


# ---------------------------------------------------------------- robot runs
def _run(kind: str, d: Path, title: str, repo: Path, **extra: Any) -> dict:
    return _base(
        f"run:{repo.name}:{kind}:{d.name}",
        title,
        repo,
        createdAt=int(1000 * d.stat().st_ctime),
        lastActivityAt=_mtime_ms(d),
        sizeBytes=_size(d),
        cwd=str(d),
        ref={"path": str(d)},
        **extra,
    )


def _run_candidates(repo: Path, now_ms: int) -> list[tuple[str, dict]]:
    """(kind, thread) for every recent robot run; ``latest_runs`` keeps one per kind."""
    runs = repo / "runs"
    out: list[tuple[str, dict]] = []

    def recent(d: Path) -> bool:
        return now_ms - _mtime_ms(d) < RUN_WINDOW_DAYS * DAY_MS

    def fresh(d: Path, s: float = RUNNING_S) -> bool:
        return now_ms - _mtime_ms(d) < 1000 * s

    for d in sorted((runs / "harness").glob("2*")) if (runs / "harness").exists() else []:
        if not d.is_dir() or not recent(d):
            continue
        s = _json(d / "summary.json")
        if s is None:
            if fresh(d):
                out.append(("reach", _run("reach", d, "⚙ reach · running", repo, running=True)))
            continue
        live = s.get("backend") == "real" and bool(s.get("live"))
        if s.get("backend") == "real" and not live:
            continue  # dry runs move nothing: not worth a bot
        rate, n = s.get("success_rate"), s.get("n_episodes", 0)
        med = s.get("median_final_dist_m")
        policy = Path(s.get("policy", "")).parent.name
        title = f"⚙ reach · {'real arm' if live else 'sim'} · {policy}"
        if rate is not None:
            title += f" · {round(rate * n)}/{n}"
        if med is not None:
            title += f" · med {100 * med:.2f} cm"
        kind = "reach-real" if live else "reach-sim"
        out.append(
            (
                kind,
                _run(
                    kind,
                    d,
                    title,
                    repo,
                    model=policy,
                    hasError=rate is None or rate < PASS_RATE,
                    prState="merged" if live and rate == 1.0 else None,
                ),
            )
        )
    for name, kind, meta, label in (
        ("survey", "survey", "summary.json", "⚙ survey"),
        ("gap", "gap", "meta.json", "⚙ gap replay"),
        ("wristcal", "wristcal", "session.json", "⚙ wrist-camera calibration"),
    ):
        for d in sorted((runs / name).glob("2*")) if (runs / name).exists() else []:
            if not d.is_dir() or not recent(d):
                continue
            m = _json(d / meta)
            if m is None:
                if fresh(d):
                    out.append((kind, _run(kind, d, f"{label} · running", repo, running=True)))
                continue
            if m.get("backend") == "sim":
                continue  # sim rehearsals: not news
            out.append((kind, _run(kind, d, f"{label} · {d.name}", repo)))
    ppo = runs / "ppo_reach"
    for d in sorted(ppo.iterdir()) if ppo.exists() else []:
        if not d.is_dir() or not recent(d):
            continue
        log = d.with_suffix(".log")
        done = (d / "reach.onnx").exists()
        running = not done and fresh(log if log.exists() else d, TRAINING_RUNNING_S)
        if not (done or running):
            continue
        ev = _json(d / "eval_final.json") or {}
        rate = ev.get("success_rate")
        title = f"⚙ train · {d.name}" + (f" · eval {100 * rate:.0f}%" if rate is not None else "")
        out.append(("train", _run("train", d, title, repo, model="PPO", running=running)))
    return out


def latest_runs(repo: Path, now_ms: int) -> list[dict]:
    """Anything running, plus the latest finished run of each kind."""
    latest: dict[str, dict] = {}
    running = []
    for kind, t in _run_candidates(repo, now_ms):
        if t["running"]:
            running.append(t)
        elif kind not in latest or t["lastActivityAt"] > latest[kind]["lastActivityAt"]:
            latest[kind] = t
    return running + list(latest.values())


# ---------------------------------------------------------------- repos + feed
def discover_repos() -> list[Path]:
    repos: set[Path] = set()
    for root in SEARCH_ROOTS:
        if root.is_dir():
            repos.update(p.parent.parent for p in root.glob(f"*/{TASKS_FILE}"))
    if CONFIG.exists():
        try:
            listed = (yaml.safe_load(CONFIG.read_text()) or {}).get("projects") or []
        except yaml.YAMLError:
            listed = []
        repos.update(Path(p).expanduser() for p in listed if Path(p).expanduser().is_dir())
    return sorted(repos)


def build_feed(repos: list[Path] | None = None, now_ms: int | None = None) -> dict:
    now_ms = int(1000 * time.time()) if now_ms is None else now_ms
    repos = discover_repos() if repos is None else repos
    threads: list[dict] = []
    for repo in repos:
        threads += task_threads(repo, now_ms)
        if (repo / "runs").is_dir():
            threads += latest_runs(repo, now_ms)
    return {
        "version": 2,
        "generatedAt": now_ms,
        "repos": [str(r) for r in repos],
        "threads": threads,
    }


def write_feed(path: Path = FEED, repos: list[Path] | None = None) -> dict:
    feed = build_feed(repos)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(feed, ensure_ascii=False))
    tmp.replace(path)  # atomic: the adapter never sees half a file
    return feed


@app.command()
def main(
    watch: float = typer.Option(0.0, help="rewrite every N seconds (0 = once)"),
    out: Path = typer.Option(FEED, help="feed path (the adapter's default location)"),
) -> None:
    feed = write_feed(out)
    typer.echo(f"{len(feed['threads'])} bots from {len(feed['repos'])} repos -> {out}")
    for r in feed["repos"]:
        typer.echo(f"  {r}")
    while watch > 0:
        time.sleep(watch)
        try:
            write_feed(out)
        except OSError as e:  # a folder vanishing mid-scan is fine; try again next tick
            typer.echo(f"skipped one refresh: {e}")


if __name__ == "__main__":
    app()
