# Lab notebook

Dated entries: what we tried, what came out (with n and intervals), what was wrong, and why we
decided what we did. This is the raw material for the blog posts; the runbook holds the procedures,
`.colony/tasks.yaml` holds what's next. Negative results and our own mistakes go in here too.

## 2026-10-20

**Fingertip accuracy, run 3** (`runs/accuracy/20261020-170433_real`, 5 corners × 4 approaches).
In-plane error per corner: 10.5 mm median raw → 4.9 mm after the settle trim; play between
approach directions 3.1 → 1.3 mm. All 20 arrivals improved. Leftover: a 2–7 mm bias along base x,
same sign at every corner, larger at the far ones. Likely the fingertip model or the board
estimate, not the servo. Not yet shown: needs the ruler check of the closed fingertip.

**Insertion factor study (MuJoCo), three mistakes before a valid run.**
1. Protocol: the descent started while the soft servo was still moving, and the trim was frozen
   during the descent. "Soft servo + trim" read 0 %; fixed, it reads 100 %.
2. Friction: MuJoCo's default μ = 1.0 has a 45° friction angle, the same as the chamfer's slope,
   so a peg couldn't slide down it (μ ≥ tan 45°). Now μ = 0.35 (PLA on PLA). That was reasoning,
   not an observation: at the time the ramps were also reversed (bug 4), so the chamfer couldn't
   have worked either way.
3. Contacts: with the default 5 ms step and soft contacts, the position-controlled arm pushed the
   peg 2–4 mm into the walls, so episodes "inserted" through them. Caught because 1 mm perception
   error at 1 mm clearance gave 69 % where geometry allows ~15 %. Now 1 ms + stiff contacts,
   penetration < 0.1 mm, same result at 0.5 ms; every sweep checks penetration and flags itself
   invalid above 0.2 mm. Earlier sweep files are listed as invalid in `runs/insertion/README.md`.

4. Chamfer: the four ramp boxes were rotated the wrong way (surface normal pointing away from the
   hole). With 100 paired episodes, chamfer 0 vs 1 changed 1-6 outcomes (McNemar p 0.5-1.0):
   "the chamfer does nothing" would have been a confident, wrong finding. Caught by asking why a
   1 mm chamfer didn't widen the capture window, then printing the ramp normals. Now a test checks
   that a 1.2 mm offset jams without the chamfer and inserts with it.

Also found with the trace: from the base side, the stiff arm can land on the wall top up to
~2.6 mm short and **drag the peg across the top face into the hole** (the blocked arm's joint
errors push the tip sideways). Physically plausible for a position-controlled arm, direction-
dependent; a hypothesis for the printed-peg test, not a finding yet.

Lesson: a sim result that beats a back-of-the-envelope bound is a bug until shown otherwise; so is
a design change (a chamfer) that changes nothing.
Sweeps now print 95 % Wilson intervals, use paired episodes (episode i draws the same errors in
every row), and `insertion-study compare` runs an exact McNemar test.

**Insertion factor study, first valid result** (stiff contacts, correct ramps, 100 paired episodes per
cell; `runs/insertion/sweep_20261020-182557.json` chamfer 0, `-183359.json` chamfer 1). 95 % Wilson
intervals:

| factors | 1 mm, no chamfer | 1 mm, 1 mm chamfer | 2 mm, no chamfer | 2 mm, 1 mm chamfer |
|---|---|---|---|---|
| ideal / lag 68 ms | 100 [96-100] | 100 [96-100] | 100 [96-100] | 100 [96-100] |
| 2 deg deadband or soft servo, no trim | 0 [0-4] | 0 [0-4] | 0 [0-4] | 0 [0-4] |
| same + settle trim | 100 [96-100] | 100 [96-100] | 100 [96-100] | 100 [96-100] |
| perception error 1 mm | 34 [25-44] | 81 [72-87] | 73 [64-81] | 95 [89-98] |
| perception error 2 mm | 12 [7-20] | 40 [31-50] | 35 [26-45] | 63 [53-72] |
| all measured factors + trim | 24 [17-33] | 75 [66-82] | 62 [52-71] | 92 [85-96] |

- Once the trim fixes the controller, perception error decides 1 mm vs 2 mm.
- The chamfer is the biggest lever: paired McNemar, outcomes changed 47-0, 22-0, 28-0, 29-1, 51-0,
  30-0 in its favour. Geometry predicts ~75 % for 1 mm error with a +-1.5 mm window; measured 81 %.
- 2 of 400 chamfer successes went 0.21/0.25 mm into a wall; counting them as failures moves a rate
  by at most 1 point.
- Still a MuJoCo claim: perception error is modelled as centred Gaussian noise, the arm's lateral
  give comes from the twin. The printed-peg test decides it.

**Model-based command correction vs closed loop** (insertion study, 100 paired episodes, no chamfer;
`runs/insertion/sweep_20261020-223912.json`, `-224207.json`). Commands planned for an ideal actuator,
run on the servo-like plant (2 deg play, 68 ms lag, soft P loop):

| correction | 1 mm | 2 mm |
|---|---|---|
| none | 0 % [0-4] | 0 % [0-4] |
| settle trim (closed loop) | 100 % [96-100] | 100 % [96-100] |
| identified model: gravity feedforward + play lead, exact | 100 % [96-100] | 100 % [96-100] |
| model -10 % / -20 % (k and play underestimated) | 100 % / 100 % | 100 % / 100 % |
| model -30 % | 0 % [0-4] | 100 % [96-100] |
| model +10 % / +20 % / +30 % | 0 % | 0 % |
| model + trim | 100 % | 100 % |
| with 1 mm perception error: model vs trim | 9 % vs 24 % | 45 % vs 62 % |

- An exact servo model fixes the commands as well as the trim does, but it is brittle and lopsided:
  overestimating stiffness/play by 10 % already fails everywhere (the play lead overshoots), while
  underestimating by up to 20 % still works. The closed loop doesn't care.
- With perception error the trim beats the model, paired: 22 vs 7 and 27 vs 10 discordant episodes
  (McNemar p = 0.008 both). Practical rule: correct with a deliberately conservative (low) model and keep
  a closed loop on top.
- Still sim: the play model is a symmetric play operator; the real stop error may not be.

**Sim commands replayed on the real arm** (`insertion-study export-motion`, `gap run --live` x3 at
0.5 rad/s, `replay-check`; runs/gap/20261020-2311**_real). The scripted straight-down insertion from the
twin, raised 30 mm into free air, replayed open loop on the home SO-101. Peg-tip position from the
encoders (FK) in the final hold:

| run | real vs command, lateral | z | per-joint error lift / elbow / wflex |
|---|---|---|---|
| 1 | 3.22 mm (dx -3.22) | -1.24 mm | +0.27 / +0.24 / +0.46 deg |
| 2 | 2.87 mm (dx -2.87) | -1.04 mm | +0.27 / +0.15 / +0.46 deg |
| 3 | 3.39 mm (dx -3.39) | -1.46 mm | +0.35 / +0.24 / +0.46 deg |

- The twin predicts 0.2 mm; the arm lands ~3.2 mm short toward the base and ~1.2 mm low, repeatable
  to +-0.25 mm. Each joint is only 0.15-0.46 deg off, but all in the same direction (sag + stop error),
  so they add up at the tip.
- Systematic, not random: correctable (trim, or a conservative feedforward), but uncorrected it already
  exceeds a 1 mm-chamfer capture range at 1 and 2 mm clearance.
- Encoders only: link flex and the tool model are extra (tip-accuracy found a further 2-7 mm bias in
  the same -x direction), so the world-frame error of open-loop sim commands is larger still.

**Closed-loop twin: first one that reproduces "box calm, fit hunts"** (`gap closed-loop`; same 20 targets,
`--settle --max-vel 1.0`; runs/harness/twinplay/). Real arm: box policy 0/20 oscillation-guard trips, 24
command reversals/s; fit policy 20/20 trips, 51/s.

| twin | box: trips, reversals/s | fit: trips, reversals/s |
|---|---|---|
| nominal | 20/20, 217 | 20/20, 217 |
| nominal + play (0.3/1/1/0.7/0 deg) +- 0.088 deg quantisation | 20/20, ~174 | 19-20/20, ~165 |
| open-loop fit + play | 0/20, 23 | 0/20, 20 |
| fit^0.25 + play | 3/20, 135 | 20/20, 132 |
| fit^0.5 + play | 1/20, 42 | 7/20, 88 |
| **fit^0.75 + play + quant, 3 seeds** | **0/60, 18-38** | **24/60, 47-55** |
| fit^0.75 without play | 2/20, 1.8 | 11/20, 38 |

- The open-loop fit is over-damped closed-loop (both policies calm); nominal is under-damped (both hunt).
  Pulling the fit 25 % back toward nominal in log space, with the measured play, is the first twin where
  the ranking matches the arm: box calm (0/60) and fit hunting (24/60).
- The play operator is what gives the calm policy its real reversal rate (1.8/s without, 18-38/s with,
  real 24/s).
- Still short: the twin trips on 40 % of fit episodes vs 100 % real, and doesn't reproduce the fit
  policy's real misses (100 % vs 85 % success). Saved as `config/actuator_fit_a075.yaml` (candidate).
- Consistent with Tune to Learn: the policy trained on the over-damped fit overshoots on a livelier
  plant; the twin that reproduces it is livelier than the open-loop fit.

**Intrinsic insertion challenge + `intrinsic-core` agent skills (read 2026-10-20).** What the winning
teams did matches what the insertion study says:
- Perception and action split: a CAD-based foundation pose model for sub-mm part poses (the open repo wires
  FoundationPose, which needs RGB **and depth**), then classical insertion (spiral search, alignment
  correction) on real force-torque feedback. Our study: once the controller is fixed, perception error
  decides 1 vs 2 mm, and a chamfer buys the most; a search-on-contact strategy is the next baseline to add.
- Their force-control rule of thumb: the target contact force must exceed 2x the sensor's deadband, and the
  arm must settle before taring. Our analog: the servo's ~2 deg approach-dependent stop and friction set the
  smallest contact the following-error "sensor" can see; measure it before relying on it.
- The sim-to-real gap they hit was contact (force, snap-fit clicks, cables) - the same place our sim broke.
- Adopted the structure of their agent skills (hub + progressive-disclosure references, input-aware decision
  trees, paired guardrails, pre-action checkpoint, retry caps, contrastive failing/succeeding tables,
  completion criteria) for `.claude/skills/so101-*`, filled with this repo's own lessons.

**Decisions.**
- No servo teardown: the blog gets an animated schematic from public specs + our register reads.
- No weighing of links: keep model masses, randomise them in training, say so.
