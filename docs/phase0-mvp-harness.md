# Phase 0 — MVP harness (evidence page)

Per `minilab/PLAN.md`: a claim counts as true only with (a) a pytest, (b) a
logged run artifact, (c) a wiki page linking both. This page is the local
draft of the wiki page.

## Claim

The Phase 0 software harness exists and its components behave as specified,
without hardware: policy interface, arbiter fallback, MIDI codec + state
vector, audio alignment, all four chunk executors, MIDI scorer over task
suite v0, episode-log parquet schema, safety watchdog/caps/boxes.

## Setup

- `src/robot_manipulator/core/` — `policy_iface.py`, `arbiter.py`,
  `safety.py`, `episode_log.py`
- `src/robot_manipulator/io/` — `midi.py`, `audio.py`
- `src/robot_manipulator/exec/scheduler.py` — `sync_replan`,
  `temporal_ensemble`, `async_client`, `rtc` behind one `ChunkExecutor`
  interface, driven by a `SimClock` + mock motor bus in tests
- `src/robot_manipulator/eval/scorer.py` — task suite v0: press key ·
  3-key sequence · press-and-hold · fader→CC target · encoder→ΔCC

## Evidence

- **pytest:** `uv run pytest` → 35 passed (2026-07-04). Executor tests
  assert exact deterministic traces: RTC splices chunks with zero stall and
  zero backtrack; async_client shows the obs-lag replay artifact RTC fixes.
- **hardware smoke:** `uv run pytest -m hw` → MiniLab 3 visible as MIDI
  input (`Minilab3:Minilab3 MIDI 20:0`), opened live via
  `io.midi.MidiListener`; cameras present. **Follower arm absent** at test
  time (no `/dev/ttyACM*` — unplugged/unpowered), so the arm smoke test is
  red until the bench is reconnected.
- **run artifact:** none yet — first parquet episode record lands with the
  exit-criterion run below.

## Conclusion / next

Software harness done. Phase 0 exits when a scripted classical primitive
presses a key on the real MiniLab 3, the scorer marks it successful from the
live MIDI log, and the episode record is written to parquet.
