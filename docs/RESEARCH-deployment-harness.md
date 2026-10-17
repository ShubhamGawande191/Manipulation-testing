# Research → harness design notes (2026-10-17)

These notes cover seven sources the user flagged for the deployment harness, which needs to deploy, debug, find
issues and fix them. Each source gets a short summary of what it actually claims (numbers taken from the paper or page)
and then **what we take for this repo**. Where I'm guessing at something a page didn't state, I say so.

## 1. Post-training: "Towards Universal Post-Training for Robotics" (Perry Dong, Chelsea Finn)
https://pd-perry.github.io/posts/post-training.html

- **Thesis.** Pretrained VLA and WAM policies aren't reliable enough to deploy. Robotics lacks the LLM-style
  post-training recipe: a stable RL algorithm, rewards, resets and HIL protocols. Today, RL on real robots is a craft.
- **Why the usual methods fail.**
  - On-policy PG needs cheap samples, which real robots don't give.
  - SAC/TD3 assume Gaussian actions, which doesn't fit diffusion or flow policies.
  - Steering in noise space or sample-and-select can only recombine what the base model already does.
- **EXPO / EXPO-FT** has five parts:
  - a frozen imitation **base policy**;
  - a small **edit policy** making bounded corrections toward higher Q;
  - at inference, sample N candidates → edit → score with Q → execute the best;
  - successes are **absorbed** back into the base through continued BC;
  - human interventions are used as training signal.
- **Result.** Fine-tuning π0.5 on 6 real tasks reached an average of 30/30, using about 19 minutes of robot
  interaction (wall clock is dominated by compute).
- **Open protocol questions it names.**
  - Reward: no RLVR equivalent exists for robots.
  - Resets: usually done by humans.
  - How much HIL feedback to use, and when.
  - Defaults for hyperparameters such as UTD.

**For us:** our rig already answers the two hardest open questions:
- **Reward:** a verifiable success signal from an object-through-hole sensor.
- **Resets:** the gravity chute does them automatically.

That makes "BC skill + RL to improve that skill" a good fit for EXPO-style post-training. The bounded edit means RL
volatility can't exceed a set bound (`|Δq| ≤ ε` per step), which composes with `harness/safety_layer.py`.
Implementation order: best-of-N with a critic first (see Mulligan / HiL-IDQL), then a trained edit policy.

## 2. Offline validation metric: CI-MSE (arXiv 2606.29898, Huang et al.)
https://arxiv.org/abs/2606.29898 · https://ci-mse.github.io/

- **Problem.** Raw validation MSE is a poor predictor of rollout success.
- **The metric** has four steps:
  1. **Filter** to *critical intervals*, which:
     - start about 5 cm before object contact, or within about 10 cm of a precision target, or in a risky phase;
     - exclude transit, idle, homing and retreat;
     - keep only the final attempt when a subgoal was retried.
  2. **Align** actions the way they are executed:
     - **temporal ensembling** with H = 8: `f^TE(o_t)_i = (1/H) Σ_h f(o_{t-h})_{i+h}`;
     - or RTC inpainting with delay Δ;
     - plus **DTW** with window w = 1.
  3. Use squared L2 error summed over action dims, with no per-dim normalization.
  4. Take the **median** over critical frames.
- **Numbers (27 sim checkpoints).**
  - Spearman ρ: CI-MSE −0.87 vs raw MSE −0.61, with CI [−0.90, −0.80].
  - Regret@1: 0 vs 42.5.
  - Raw MSE has an *inverted* correlation across data scales.
- **Robustness.** Interval length off by < 1 s or shifted by < 40 % hardly matters.
- **Limitations.**
  - Real-world results are weaker when collectors or operator styles differ.
  - It doesn't see dynamics.
  - It suits short-horizon tasks.
- **Real-world scoring.** Ordinal outcome levels → Elo (start 1500, K = 32).
- **Annotation.** Intervals are annotated zero-shot by an agent skill: a subagent per episode, coarse-to-fine frame
  inspection, boundaries at 0.1 s.

**For us:**
- **Cheap critical intervals.** Our episodes are instrumented, so we don't need a VLM annotator. The interval is
  [first tick ≤ 5 cm from the object or ≤ 10 cm from the hole (FK on measured q)] → [success or contact end], with
  contact taken from `PincGripper` load / `Present_Load`.
- **Alignment.** We already have `exec/scheduler.TemporalEnsemble`, which matches the paper's H = 8 TE alignment.
- **Use.** Rank BC checkpoints offline before spending real rollouts, and validate the ranking against a few real
  evaluations, as the paper does.
- **Selection optimism.** T8 already hit this: a checkpoint picked at 96 % scored 82 % on fresh episodes. Checkpoint
  selection and the final eval must use disjoint episode sets.

## 3. Success-guided sampling (AlphaNIST + SGS)
https://alpha-nist.github.io/ · https://sgs-rl.github.io/

- **AlphaNIST.** One PPO policy, trained in sim, learns all 20 NIST Task Board #1 assemblies (parts as small as 4 mm).
  - Starts come from loose parts, in-hand grasps and cached partial assemblies. No demos.
  - **Binary reward beat dense**: shared policy 98.2 % vs 17.8 %.
- **SGS.** PPO plus an outer loop over N fixed task configurations (s0, goal, env):
  - Each configuration keeps a success history, and resets are sampled weighted toward configurations solved about
    half the time.
  - Example weights: 0.14 at 0/64, 0.99 at 27/64, 0.73 at 59/64.
  - Configurations are split across Reaching / Stable-Grasp / Near-Goal resets.
  - At 1 M envs: nut-bolt 0.70 vs uniform 0.06; ANYmal 0.73 vs 0.00. At 32 K envs the advantage is small.
  - The exact Beta formula isn't published.

**For us:**
- **Reach (T7/T8).** Replace uniform target sampling with a fixed pool of targets plus SGS weights. Our own formula,
  since theirs isn't public: the Beta(s+1, f+1) posterior mean p̂, weight ∝ `p̂(1−p̂) + floor`.
- **Insertion task.** Use AlphaNIST's start mix (reach / in-hand / near-hole cached states) with a **binary** success
  reward. Re-test our dense reach shaping against binary + SGS there; don't assume dense helps.

## 4. Data collection: Mulligan (arXiv 2610.05882, Ankile et al.)
https://mulligan.page/ · code github.com/ankile/mulligan · data huggingface.co/mulligan

- **Supervised deployment.** The operator places objects and intervenes on failures. The *initial-state distribution
  is the decision*.
- **Initial states** are parameterized object poses, scaled to [0, 1] with periodic yaw.
  - Round 0: Sobol points.
  - Later rounds: **promote up to m_f failures** from the last eval (earliest failure stage first), then fill with the
    **farthest point** from all collected states.
  - Operators reproduce poses with a **printed reset card**.
- **Round structure.**
  - train → eval (K = 50 autonomous episodes, fresh Sobol block, shared by all methods) → choose states →
    collect M = 50 supervised episodes.
  - Round 0 has 100 demos.
  - **Counterfactual demos**: reset to the same state and demo fully, instead of teaching recovery from a big mistake.
- **HiL-IDQL.**
  - Actor: Diffusion Policy, BC on human-controlled frames of successful episodes.
  - Critic: IQL Q/V, expectile, chunked λ-return, distributional, trained on **all data, failures included**.
  - Deployment is **best-of-N by Q**.
- **Eval protocol.**
  - Separate from collection, with no interventions and fresh states.
  - Methods interleaved and operator-blind, with paired per-state tests.
  - Wilson ±1 SE intervals; 2,550 blinded episodes.
- **Results.** +10 to +34 pp over HG-DAgger. 98 % of 1,009 collection episodes were completed. Adding autonomous
  successes to actor data **hurt** (−3.4 pp).

**For us:**
- **Initial state with the chute.** We can't *place* the initial state: the chute drops the object with noise. So we
  *measure* it (workspace camera → object pose in the tray), log it, and use Mulligan's selection to decide **which
  episodes get operator attention** (promote observed failure states). A printed grid on the tray gives a reset card
  for manual rounds.
- **Interventions.** `uv run teleop` becomes the takeover device. It needs a takeover trigger and per-step
  `intervention` flags in the log.
- **Dataset rules.** Keep failures labelled for the critic, and keep the actor's BC set to human-controlled frames.

## 5. Agentic skill discovery: ASPIRE (NVlabs, arXiv 2607.00272)
https://github.com/NVlabs/ASPIRE (Apache-2.0)

- **What it is.** Coding agents discover robot skills by writing, testing and debugging policy code.
  - "Training" = refining a repository of skills.
  - Sim workflows: **iterative-debugging** and **evosearch**.
  - LIBERO-Pro, 10 tasks: 15 % → 99 % depending on the model (99 % with Claude Opus 5.5).
  - Real robot: YAM, driven via registered MCP tools (`mcp__cap__*`) under `AGENTS.md` rules.
- **Real-robot rules worth copying.**
  - Act only through registered tools: no ad-hoc motion code, no bypassing collision constraints.
  - **Pose and gripper results don't prove a grasp**: a stage stays active until its *physical predicate* is
    verified from fresh observations.
  - An async receipt means accepted, not done.
  - An occluded view means unknown, not failed.
  - Recovery is a temporary path back to the parent stage.
  - Report blocked only when no capability remains.
  - Keep a compact object-centric task state.
- **Safety.** It runs LLM-generated code with full imports, so it needs an isolated host and **no sim-agent access to
  hardware**.

**For us:**
- This is the "debugging / finding issues" half of the harness: expose `harness` as MCP tools (`run_skill`,
  `read_state`, `camera`, `park`, `eval`) with an `AGENTS.md` like theirs. `minilab/PLAN.md` Phase 6 already lists
  `run_skill` / `move_ee`.
- Skills' end conditions (`harness/skill.py`) are the "physical predicates". The rule that the gripper closing isn't a
  grasp maps directly to our PincOpen contact/load checks.
- Agent experiments go in sim first; hardware access only through the harness tools.

## 6. Edge policy: SmolWAM (arXiv 2610.13740)
https://arxiv.org/abs/2610.13740 · github.com/iitimii/SmolWAM

- **What it is.** Distilled from Cosmos Policy (2B params):
  - single-frame latents, cached text features;
  - 12 of 28 blocks, 1,008 tokens;
  - actions and future latents produced in one rectified-flow step.
- **Numbers.** LIBERO 95.5 % over 400 episodes; **57 ms and 2.4 GiB per 16-action chunk**; 3.6 GPU-hours of training.
- **Gaps.** No real-robot or deployment results; no stated limitations.

**For us:** it's the first world-action model that fits our 6 GB GPU for inference. It's a candidate policy class
behind `core/policy_iface.Policy` once we have demos, and its 16-action chunks run through our chunk executors. It
needs our own real-world validation.

## 7. Real2Sim: RoboSimStudio (arXiv 2610.18835, Chen, …, Coumans, Macklin, Goldberg)
https://robosimstudio.github.io/

- **Pipeline.** Phone video → a coding agent builds the scene, camera and physics in about 1 h (one agent verified in
  56 min).
  - The solver, contact model and physical parameters are explicit and editable.
  - Humans tune with sliders in Viser, or with **force feedback through a Gello leader arm** (deadband against sim
    lag).
  - Agents get primitives plus a text/visual digest: target error, contacts, slip, progress.
- **Stack.** Built on **Newton**: articulated robots use **MuJoCo-Warp**, cloth and cables use VBD, soft bodies use
  co-simulated FEM (SuperDex).
  - 18 tasks; robots: Franka, UR5e, YAM.
  - Sim-real agreement is **qualitative only**; sim2real is shown on one task (wipe).

**For us:**
- **Rig geometry.** For the tray, chute and hole, follow RSS's approach: an editable scene plus parameter tuning
  against video of real drops.
- **Arm.** Keep the quantitative SAGE-lite gap fit (`sim/gap/`) for the arm, which RSS lacks.
- **Newton later.** MuJoCo-Warp underneath means our MJCF assets carry over if we try Newton later.
- **Leader arm.** Our SO101 leader is a Gello-class device, so leader-in-the-loop sim teleop works (NVIDIA Strategy 1
  too).

---

## Backlog for the harness (proposed order)

| # | Item | Draws on | Size |
|---|---|---|---|
| 1 | **Eval protocol module** (`harness/eval.py`): fixed held-out initial-state blocks (Sobol), selection vs final eval on disjoint sets, Wilson intervals, ordinal stage outcomes (approach/grasp/lift/insert), paired comparisons between checkpoints | Mulligan, CI-MSE, T8 lesson | S |
| 2 | **Episode schema v2**: initial-state params, stage reached, failure stage, per-step `intervention` flag, success sensor value | Mulligan, EXPO, CI-MSE | S |
| 3 | **SGS in sim training** (reach now, insertion later) + binary-vs-dense reward ablation | AlphaNIST/SGS | S |
| 4 | **Teleop takeover** in `uv run teleop` (hold-to-intervene, flags logged) | Mulligan, EXPO HIL | M |
| 5 | **CI-MSE offline ranking** with instrumented critical intervals + TE H = 8 + DTW w = 1, median | CI-MSE | M |
| 6 | **Supervised deployment loop** on the rig: failure-promoted states, counterfactual demos, chute sensor reward | Mulligan, post-training | M |
| 7 | **Best-of-N + critic** (HiL-IDQL-style), then an **EXPO edit policy** with bounded Δ | Mulligan, EXPO | L |
| 8 | **Agent interface**: harness as MCP tools + `AGENTS.md` rules (verified predicates, recovery returns to parent) | ASPIRE | M |
| 9 | **Rig real2sim**: tray, chute and hole in MJCF, tuned against drop videos | RoboSimStudio | M |
| 10 | Evaluate **SmolWAM** as a BC policy class on our demos | SmolWAM | L |
