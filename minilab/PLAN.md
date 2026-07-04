# SO101 Multi-Arm Orchestration — Project Plan v2 (locked; amended 2026-07-04: platform is Ubuntu now, not Windows)

Instructions for Claude Code: apply the *Repo adjustments* section as minimal, incremental changes to the existing
Manipulation-testing repo. Do not restructure beyond what is listed. Every feature lands with a test + a wiki
evidence page. MVP first; stretch items only when their phase begins.

## Locked constraints & decisions
- Compute: RTX 3060 (~6 GB VRAM assumed). **Ubuntu 24.04 from day one** (amended — originally planned on Windows).
  - JAX track (Octo-small 27M, MJX) is therefore ungated and may run whenever Phase 1 bandwidth allows.
  - TensorRT + torch.compile both available; schedule TensorRT as round 2 of Phase 1.
- Stack: **LeRobot v3** (dataset format v3, current plugin/record APIs).
- Sim: **MuJoCo** (Menagerie `trs_so_arm100` + custom MJCF for the MiniLab 3). Genesis = optional appendix if it
  runs acceptably on 6 GB. No Isaac.
- Arms: 2× 12 V followers + 2× leaders converted to 7.4 V followers (per-arm PSU, calibration, config — already handled).
- Teleop: user's existing Quest 3 VR implementation. No phospho.
- Policies: **ACT, Diffusion Policy, Multitask DiT, SmolVLA**. On 6 GB: small batch + grad accumulation,
  freeze vision towers / LoRA where needed. Precision decided empirically.
- Action space: absolute + relative **joint positions** now; reserve an `ee_delta` config flag for the later
  human-video phase (not MVP).
- Action chunk length: small (8–16 default) for precision; treated as a swept variable, not a constant.
- Extra observations: **MIDI state vector** (current notes + CC values) as a proprio-like feature;
  **USB mic** audio logged per episode (raw wav first; spectrogram feature later).
- Evidence culture (non-negotiable): a claim counts as true only with (a) a pytest, (b) a logged run artifact,
  (c) a GitHub wiki page linking both.

## Repo adjustments (minimal, in order)
1. `core/policy_iface.py` — common plugin interface: `reset()`, `act(obs) -> action_chunk`,
   `health() -> {latency_ms, staleness, confidence}` . All methods (teleop, classical scripts, learned policies,
   later world-model MPC) implement it.
2. `core/arbiter.py` — small state machine over plugins. Fallback rule: health degraded → freeze →
   classical recovery primitive → retry or escalate to teleop. Log every transition.
3. `io/midi.py` — mido/python-rtmidi. Event log + rolling state vector (notes, velocities, CC values).
   Unit-tested encode/decode.
4. `io/audio.py` — optional per-episode wav capture, timestamp-aligned to episode clock.
5. `exec/scheduler.py` — interchangeable chunk executors behind one interface:
   `sync_replan`, `temporal_ensemble`, `async_client` (policy server), `rtc` (real-time chunking).
   Stretch: bidirectional decoding, streaming/partial-denoise for DP, consistency-distilled DP.
6. `eval/scorer.py` — MIDI ground truth → success flag, key-press F1, velocity error, fader/CC target error,
   time-to-success. This is the single scoring path for every phase.
7. Episode log schema — `run_id, arm_id, policy, executor, chunk_len, precision, policy_latency, e2e_latency,
   jitter, scorer outputs` → parquet (wandb optional). All blog tables generate from this.
8. `tests/` — unit (MIDI codec, scorer, executors against a simulated clock + mock motor bus),
   integration (record/replay vs mock bus), hardware smoke tests marked `@hw` (manual).
9. `docs/` + GitHub wiki — one page per claim: hypothesis → setup → run_ids → result table → conclusion.
10. Safety — heartbeat watchdog (freeze if command stream stale >100 ms), per-arm velocity/current caps,
    per-arm workspace boxes; keyboard-zone **region mutexes** (reserved for Phase 4 multi-arm).

## Task suite v0 (MIDI-scored, reused by every phase)
press key K · 3-key sequence · press-and-hold · fader → target CC value · encoder → ΔCC target.

## Phases (each ends with a blog + wiki evidence pages)
- **Phase 0 — MVP harness.** Items 1–10 above. Exit criterion: a scripted classical primitive passes task suite
  v0 with logged evidence and green tests.
- **Phase 1 — Blog 1: inference-method shootout (single arm, RTX 3060).**
  Policies: ACT, DP, DiT, SmolVLA. Executors: sync vs temporal ensembling vs async vs RTC (+ stretch methods).
  Systems: fp32/bf16/int8 (TorchAO), torch.compile; TensorRT in round 2. Chunk_len ∈ {8, 16, 32}.
  Report: latency / achieved Hz / jitter vs MIDI success + velocity error.
- **Phase 2 — Blog 2: self-scoring improvement loop.**
  Curriculum over task suite; DAgger-style corrections via VR teleop; optional HIL-SERL (RL with MIDI reward).
  Add MIDI-state + audio features to the observation and measure the delta they contribute.
- **Phase 3 — Blog 3: classical vs learned vs orchestrated → hybrid-best.**
  Classical: SAM 3 concept-prompt masks → 3D (AprilTag board or fixed-plane prior) → IK primitives.
  Functional scene graph with **interactive MIDI verification** (FunFact-lite: touch → event → collapse edge belief).
  After the 3-way ablation, compose the best pieces (e.g. classical reach + learned press; learned locate +
  classical CC-feedback servo) and benchmark the hybrid against all three.
- **Phase 4 — Blog 4: multi-arm allocator ("mothership").**
  Per-arm async policy clients, one GPU policy server (batched). Allocator: task graph → subtask leases per arm +
  keyboard-zone mutexes. New metric: **idle-arm utilization**. Operator attention routed to the arm whose policy
  reports lowest confidence. Ablation: monolithic bimanual policy vs allocator + per-arm agents.
- **Phase 5 — Blog 5: sim + tiny world model.**
  MJCF MiniLab (keys = hinges w/ springs, faders = sliders, encoders = free hinges); sim2real2sim: fit key
  stiffness/damping so sim MIDI velocities match real logs. Tiny world model over (joints ⊕ MIDI vector) for MPC;
  DINO-WM / TD-MPC2 optional. Genesis speed test as appendix.
- **Phase 6 — Blog 6: ASPIRE-lite (Fable in the loop).**
  Trace-recording per primitive call (inputs, outputs, frames). Skill library entries = {code, test, evidence
  run_ids}, retrievable as context. Claude Code as the debugging agent via MCP tools: `capture_views`,
  `get_scene_graph`, `move_ee`, `run_skill`, `read_midi`, `record`, `train`, `eval` — inside the safety envelope.
  MIDI = automatic success detection; arm 4 = auto scene-reset. Small-scale evolutionary search only if token
  budget allows.

## Priorities
P0 = Phases 0–1. P1 = Phases 2–3. P2 = Phase 4. P3 = Phases 5–6.
Stretch (any phase): ee_delta + human videos, BID/streaming executors, Genesis, evolutionary search, JAX/Octo track.

## Risks / notes
- 6 GB VRAM: DiT variant size TBD — verify it fits before committing it to the Phase 1 roster.
- Converted 7.4 V arms: lower torque → light payloads, per-arm speed scaling in config.
- USB bandwidth: 4 motor buses + 3–4 cameras + mic; cameras on direct ports, not shared hubs.
  (Confirmed the hard way 2026-07-04: the follower board fails to enumerate behind the ThinkPad
  dock/hub chain — `uv run bench-check` diagnoses this; keep motor buses on direct ports too.)
