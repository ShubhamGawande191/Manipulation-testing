"""Feed for Bot Crossing (github.com/Station-Sciences/bot-crossing): the bench as a colony.

Bot Crossing draws every agent thread on the machine as a little bot on a hex zone per repo. Its
``integrations/bot-crossing/so101.mjs`` adapter reads the feed written here, so the robot work
shows up next to the Claude Code threads:

- zone **SO-101 real arm**: live real-arm runs (harness reach, survey, gap replay)
- zone **SO-101 sim**: sim harness runs, sim gap replays and PPO training runs
- zone **SO-101 twin**: one bot per stage of the digital-twin runbook status table

Bot state follows Bot Crossing's precedence (first match wins): a run being written right now
hammers (``running``); a failed run slumps (``hasError``); a clean live real-arm run celebrates
(``prState: merged``); a runbook stage in progress (🟡) waits on you (``unread``, the ``?``).
Building size is the run folder's size.

    uv run colony-feed               # write once
    uv run colony-feed --watch 3     # keep it fresh while Bot Crossing is open

Read-only over ``runs/`` and ``docs/``; the only file written is the feed (atomic replace).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import typer

REPO = Path(__file__).resolve().parents[3]
FEED = Path(
    os.environ.get(
        "BOT_CROSSING_SO101_FEED",
        Path.home() / ".local" / "state" / "so101-colony" / "threads.json",
    )
)
RUNBOOK = REPO / "docs" / "DIGITAL-TWIN-RUNBOOK.md"
RUNNING_S = 30.0  # a run folder written to this recently, without a summary, is still running
TRAINING_RUNNING_S = 120.0
MAX_AGE_DAYS = 30.0
PASS_RATE = 0.9
ZONE_REAL, ZONE_SIM, ZONE_TWIN = "SO-101 real arm", "SO-101 sim", "SO-101 twin"

app = typer.Typer(add_completion=False, help=__doc__)


def _mtime_ms(path: Path) -> int:
    """Latest mtime under ``path`` (the folder itself and its direct children), epoch ms."""
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


def _thread(
    tid: str, title: str, zone: str, path: Path, now_ms: int, **extra: Any
) -> dict[str, Any]:
    last = _mtime_ms(path)
    t = {
        "id": f"so101:{tid}",
        "title": title,
        "preview": extra.pop("preview", ""),
        "project": zone,
        "projectPath": str(REPO),
        "worktree": "",
        "cwd": str(path if path.is_dir() else path.parent),
        "gitBranch": "",
        "model": extra.pop("model", ""),
        "effort": "",
        "createdAt": int(1000 * path.stat().st_ctime),
        "lastActivityAt": last,
        "lastFocusedAt": 0,
        "running": False,
        "unread": False,
        "hasError": False,
        "archived": False,
        "sizeBytes": _size(path),
        "source": "so101",
        "canOpen": True,
        "ref": {"path": str(path)},
    }
    t.update(extra)
    return t


def _recent(path: Path, now_ms: int) -> bool:
    return now_ms - _mtime_ms(path) < MAX_AGE_DAYS * 86_400_000


def harness_threads(runs: Path, now_ms: int) -> list[dict]:
    out = []
    for d in sorted((runs / "harness").glob("2*")):
        if not d.is_dir() or not _recent(d, now_ms):
            continue
        s = _json(d / "summary.json")
        if s is None:  # no summary yet: running if fresh, else an interrupted run
            fresh = now_ms - _mtime_ms(d) < 1000 * RUNNING_S
            out.append(
                _thread(
                    f"harness:{d.name}",
                    f"reach · {d.name} · {'running' if fresh else 'interrupted'}",
                    ZONE_SIM,
                    d,
                    now_ms,
                    running=fresh,
                    archived=not fresh,  # Ctrl+C'd experiment, no result: back to the ship
                )
            )
            continue
        real_live = s.get("backend") == "real" and bool(s.get("live"))
        policy = Path(s.get("policy", "")).parent.name
        rate = s.get("success_rate")
        n = s.get("n_episodes", 0)
        med = s.get("median_final_dist_m")
        backend = s.get("backend")
        dry = " dry-run" if backend == "real" and not real_live else ""
        title = f"reach · {backend}{dry} · {policy}"
        if rate is not None:
            title += f" · {round(rate * n)}/{n}"
        if med is not None:
            title += f" · med {100 * med:.2f} cm"
        flags = [k for k in ("settle", "dr") if s.get(k)]
        if s.get("actuator_fit"):
            flags.append(f"fit {Path(s['actuator_fit']).stem}")
        ok = rate is not None and rate >= PASS_RATE
        out.append(
            _thread(
                f"harness:{d.name}",
                title,
                ZONE_REAL if real_live else ZONE_SIM,
                d,
                now_ms,
                preview=" · ".join(
                    [d.name, *([f"max_vel {s['max_vel']}"] if "max_vel" in s else []), *flags]
                ),
                model=policy,
                hasError=not ok and not dry,  # a dry run never moves: 0 % is expected
                prState="merged" if real_live and rate == 1.0 else None,
            )
        )
    return out


def survey_threads(runs: Path, now_ms: int) -> list[dict]:
    out = []
    for d in sorted((runs / "survey").glob("2*")):
        s = _json(d / "summary.json")
        if not d.is_dir() or s is None or not _recent(d, now_ms):
            continue
        live = s.get("backend") == "real" and bool(s.get("live"))
        want = s.get("n_poses", 0) + 1  # + HOME
        out.append(
            _thread(
                f"survey:{d.name}",
                f"survey · {s.get('backend')} · {s.get('reached')}/{want} poses",
                ZONE_REAL if live else ZONE_SIM,
                d,
                now_ms,
                hasError=s.get("reached", 0) < want,
            )
        )
    return out


def gap_threads(runs: Path, now_ms: int) -> list[dict]:
    out = []
    for d in sorted((runs / "gap").glob("2*_*")):
        if not d.is_dir() or not _recent(d, now_ms):
            continue
        m = _json(d / "meta.json")
        fresh = now_ms - _mtime_ms(d) < 1000 * RUNNING_S
        if m is None:
            out.append(
                _thread(
                    f"gap:{d.name}",
                    f"gap replay · {d.name}",
                    ZONE_SIM,
                    d,
                    now_ms,
                    running=fresh,
                    archived=not fresh,
                )
            )
            continue
        live = m.get("backend") == "real" and bool(m.get("live"))
        out.append(
            _thread(
                f"gap:{d.name}",
                f"gap replay · {m.get('backend')} · {len(m.get('motions', []))} motions",
                ZONE_REAL if live else ZONE_SIM,
                d,
                now_ms,
                preview=", ".join(m.get("motions", [])),
                hasError=not m.get("motions"),
            )
        )
    return out


def training_threads(runs: Path, now_ms: int) -> list[dict]:
    out = []
    for d in sorted((runs / "ppo_reach").iterdir()) if (runs / "ppo_reach").exists() else []:
        if not d.is_dir() or not _recent(d, now_ms):
            continue
        log = d.with_suffix(".log")
        ev = _json(d / "eval_final.json") or {}
        done = (d / "reach.onnx").exists()
        active = log if log.exists() else d
        running = not done and now_ms - _mtime_ms(active) < 1000 * TRAINING_RUNNING_S
        rate = ev.get("success_rate")
        title = f"train · {d.name}" + (f" · eval {100 * rate:.0f}%" if rate is not None else "")
        out.append(
            _thread(
                f"train:{d.name}",
                title,
                ZONE_SIM,
                d,
                now_ms,
                model="PPO",
                running=running,
                hasError=not done and not running,
            )
        )
    return out


STAGE_ROW = re.compile(r"^\|\s*(\d+)\s+([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|")


def stage_threads(runbook: Path, now_ms: int) -> list[dict]:
    """One bot per row of the runbook's status table ("| 7 Extrinsics | 🟡 ... | evidence |")."""
    if not runbook.exists():
        return []
    text = runbook.read_text()
    section = text.split("## Status of this setup", 1)[-1]
    out = []
    for line in section.splitlines():
        m = STAGE_ROW.match(line)
        if not m:
            continue
        num, name, status, evidence = m.groups()
        status = status.replace("**", "")
        out.append(
            _thread(
                f"stage:{num}",
                f"{num} {name.strip()} · {status[:70]}",
                ZONE_TWIN,
                runbook,
                now_ms,
                preview=(evidence or status)[:200],
                unread="🟡" in status,  # in progress: wants you
                hasError="❌" in status,
            )
        )
    return out


def build_feed(repo: Path = REPO, now_ms: int | None = None) -> dict:
    now_ms = int(1000 * time.time()) if now_ms is None else now_ms
    runs = repo / "runs"
    threads = (
        harness_threads(runs, now_ms)
        + survey_threads(runs, now_ms)
        + gap_threads(runs, now_ms)
        + training_threads(runs, now_ms)
        + stage_threads(repo / "docs" / RUNBOOK.name, now_ms)
    )
    for t in threads:
        if t.get("prState") is None:
            t.pop("prState", None)
    return {"version": 1, "generatedAt": now_ms, "repo": str(repo), "threads": threads}


def write_feed(path: Path = FEED, repo: Path = REPO) -> int:
    feed = build_feed(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(feed, ensure_ascii=False))
    tmp.replace(path)  # atomic: the adapter never sees half a file
    return len(feed["threads"])


@app.command()
def main(
    watch: float = typer.Option(0.0, help="rewrite every N seconds (0 = once)"),
    out: Path = typer.Option(FEED, help="feed path (the adapter's default location)"),
) -> None:
    n = write_feed(out)
    typer.echo(f"{n} threads -> {out}")
    while watch > 0:
        time.sleep(watch)
        try:
            write_feed(out)
        except OSError as e:  # a run folder vanishing mid-scan is fine; try again next tick
            typer.echo(f"skipped one refresh: {e}")


if __name__ == "__main__":
    app()
