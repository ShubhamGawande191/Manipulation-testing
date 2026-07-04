# Execution roadmap — planned with Fable 5, executed by Opus over the next month

Context: heavy planning happened while Fable 5 quota lasted (2026-07-04/05). From then
on, a weaker-but-capable model executes. Rules for the executing model:

- Work top-to-bottom through the milestones; don't reorder without a written reason here.
- Every milestone ends with: green `uv run pytest`, an evidence page under `docs/`,
  a commit, a push. Claims need a pytest + a run artifact (see `minilab/PLAN.md`).
- If something is ambiguous, prefer the smallest change that keeps tests green,
  and leave a `TODO(fable-review):` comment rather than redesigning.
- After any reboot/replug: `uv run bench-check` first (see `.claude/skills/bench-check`).

## M0 — Phase 0 exit (hardware, ~1 session)

Blocked only on the follower arm enumerating (plug into a **direct** USB port; bench-check
must show `/dev/so101_follower`).

1. `tasks/press_key.py`: scripted classical primitive — from HOME, move above a chosen
   key (hand-tuned joint targets are fine; store them in `config/minilab_keys.json`),
   press, release, HOME. Direct LeRobot path (`arm.py`, degrees) is fine for v1;
   respect `core.safety` (watchdog beat every cycle, VelocityCap on every step).
2. During the run: `io.midi.MidiListener` captures events; `eval.scorer.score(PressKey(k))`
   decides success; write one `EpisodeRecord` parquet to `runs/`.
3. Add `@hw` test asserting the primitive's joint targets stay inside `JOINT_LIMITS`.
4. Evidence: update `docs/phase0-mvp-harness.md` with the run_id + numbers. Commit, push.

## M1 — Task suite v0 coverage (~2-3 sessions)

Primitives for: 3-key sequence, press-and-hold, fader→CC target (closed-loop: read CC
from MidiState, servo the wrist), encoder→ΔCC. Each scored, each logged to parquet.
Fader/encoder need the closed-loop pattern — that's the template for everything later.

## M2 — Data collection + first learned policy (Phase 1 opens)

1. LeRobot record integration: episodes = camera frames + joint states + **MIDI state
   vector** (`io.midi.MidiState.vector()` as a proprio feature) + wav (optional extra).
2. Record ~50 demos of `press key K` via teleop (leader arm if fixed, else scripted-
   with-noise as bootstrap).
3. Train ACT (chunk 16) on the 3060: small batch + grad accum; freeze vision tower if
   VRAM forces it. Evaluate with the SAME scorer + episode log as M0/M1.

## M3 — Executor shootout (Blog 1)

Grid: {ACT, DP} × {sync_replan, temporal_ensemble, async_client, rtc} × chunk {8,16,32}
× {fp32, bf16}. All four executors already exist in `exec/scheduler.py` with the stats
fields the blog tables need (`ExecStats` → `EpisodeRecord`). ThreadedInference is the
production channel; add a policy-server client only if latency demands it.
Report: latency / achieved Hz / jitter / stale_steps vs MIDI success + velocity error.
Tables generate from `runs/*.parquet` — never hand-copy numbers.

## M4 — Ongoing hygiene (every session)

- Push after every green milestone. Update README Progress Tracker.
- `uv lock --upgrade` monthly + when Dependabot complains; `uv run pytest` gates it.
- Once `gh auth login` is done: `scripts/github_backlog.sh` creates labeled issues;
  keep the GitHub Project board in sync (columns: Backlog / This week / Done).

## Deferred (do NOT start early)

- MuJoCo MiniLab MJCF + sim2real (PLAN Phase 5) — after Phase 1 ships.
- pixi migration — only alongside a RoboStack ROS2 decision (see TODO.md).
- Multi-arm (Phase 4), world model (Phase 5), ASPIRE-lite (Phase 6).

## Model-switch note (for the human)

When Fable quota ends: `/model opusplan` (Opus plans, Sonnet executes) or plain Opus.
If using opusplan with a 200k cap, optionally set `CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000`
in `~/.claude/settings.json` env — do NOT set that while on `claude-fable-5[1m]`,
it would fight the 1M window.
