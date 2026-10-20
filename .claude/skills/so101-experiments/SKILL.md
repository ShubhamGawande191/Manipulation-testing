---
name: so101-experiments
description: >-
  Designing, running, validating and reporting experiments and sweeps in this lab (sim factor studies, real-arm
  accuracy runs, policy evaluations, paper claims) so that no number is reported without its sample size,
  interval, validity checks and failure modes. The anti-hype protocol.
  Triggers: run a sweep or study, compare two variants, report a success rate or error, write a notebook/blog
  result, a sim result looks too good or a change does nothing, reading a paper's results, proposal claims.
  Subsystems: sim/insertion.py (sweep, compare, wilson, mcnemar_p), harness eval, tip-accuracy analyze,
  gap closed-loop, docs/LAB-NOTEBOOK.md.
  Dependencies: numpy, MuJoCo; runs/<tool>/<ts>/ outputs.
  Anti-keywords: building the twin (digital-twin), hardware bring-up (bench-check).
---

# Experiments that hold up

The user's standing rule: no hyped results. A number without n, an interval and a validity check is not a result.
Negative results and our own bugs get written up; they are what makes the rest credible.

## Before running: design checkpoint

1. **Question and decision.** What decision changes depending on the answer? Write the expected result first.
2. **Kill criterion.** What outcome means "this doesn't work"? (e.g. "if A doesn't beat B by more than the
   interval on >= 2 of 3 tasks, that is the result").
3. **Simplest strong baseline** in the comparison (in the insertion study a 10-line integral loop beat everything).
4. **Pairing.** Same seeds / initial states / perturbation draws for every variant (episode i draws from
   `rng([seed, i])`), so variants are compared episode by episode.
5. **Power.** Pick n from the effect you need to see (two-sided, 80 % power, alpha 0.05, unpaired):

   | from -> to | n per arm |
   |---|---|
   | 30 % -> 50 % | ~91 |
   | 60 % -> 75 % | ~150 |
   | 60 % -> 70 % | ~354 |

   Real-arm success rates rarely reach these n: prefer continuous outcomes (mm, deg, s) on the real arm.
6. **Units and outputs:** where results land (`runs/<tool>/<ts>/`), what the json records (seed, n, versions).

## While running: validity checks (a result that beats a back-of-envelope bound is a bug until shown otherwise)

| Check | How | Caught (this repo) |
|---|---|---|
| Analytic bound | compare with geometry/physics you can compute by hand | 69 % insertion where geometry allows ~15 % -> peg went through walls |
| Conservation / penetration | log the deepest contact per episode; flag successes above a limit | soft contacts 2-4 mm deep (`MAX_PEN_MM`) |
| Convergence | halve the timestep (or double n_sub); the answer must not move | 1 ms vs 0.5 ms: 35 % vs 33 % |
| A change that changes nothing | if a design change (chamfer, gain) moves nothing, inspect it directly | chamfer ramps faced away from the hole |
| Known-input test | known offsets / known perturbations in a unit test | `tests/test_insertion.py` offset and chamfer tests |
| Protocol timing | does the system reach steady state before you measure? | soft servo still moving at descent -> 0 % |
| Defaults you never chose | friction, contact softness, timestep, solver iterations | MuJoCo friction 1.0 = 45 deg friction angle |

## After running: statistics

- Every rate with **n and a 95 % Wilson interval** (`insertion.wilson(k, n)`).
- Compare variants **paired**: exact McNemar on discordant pairs (`uv run insertion-study compare A rowA B rowB`).
- Report effects in **physical units** and **absolute points**, never relative-% headlines on small bases.
- With small real n (20-40 trials): say what the interval allows; don't rank variants whose intervals overlap.
- Continuous outcomes: median + spread per condition; say how many trials/poses/corners.
- Details and formulas: [references/statistics.md](references/statistics.md).

## Reporting

1. `docs/LAB-NOTEBOOK.md`: dated entry - question, setup, table with n + intervals, what was wrong, decision.
2. Invalid or superseded runs listed in `runs/<tool>/README.md` so nobody quotes them.
3. Runbook lessons / status table if a procedure changed; board task status updated.
4. Sim findings are hypotheses until one real-arm point confirms them - say so in the text.

## Contrastive bifurcation table (failing vs succeeding protocol)

| Domain | Failing (tau-) | Succeeding (tau+) | Invariant |
|---|---|---|---|
| Settle before measuring | descend after a fixed 1 s; soft servo still moving -> 0 % | settle until still (and on target with trim) | measure at steady state |
| Integral action | trim frozen during the descent | trim on during the descent (bounded) | keep the corrective loop on while it matters |
| Contact model | MuJoCo default 5 ms, soft contacts | 1 ms + stiff solref/solimp, penetration logged | contacts must not give way more than the tolerance studied |
| Geometry under test | build, run, trust | print/inspect the geometry (normals, poses) + a known-input test | verify the thing you're varying |
| Twin choice | best open-loop fit -> policy hunts on the arm | broad DR around nominal; accept twins only via `gap closed-loop` | judge a twin closed-loop |
| Reading a result | "63 % vs 53 %, chamfer does nothing" | intervals 46-78 / 36-70 overlap -> data can't tell | no ranking inside the interval |

## Reading papers (same rules)

Read in full, extract per claim: n per cell, trials, seeds, whether intervals exist, absolute vs relative
numbers, what is measured vs asserted, ablations with numbers vs qualitative. Recompute significance when
counts are given (e.g. 62/90 vs 55/90 -> Fisher p ~ 0.35). See [references/reading-papers.md](references/reading-papers.md).

## Circuit breaker

- If a result contradicts a bound or a change does nothing: stop sweeping; inspect one episode end to end
  (trace positions, contacts, geometry) before running more cells.
- If two "fixes" in a row don't move the outcome, the model of the problem is wrong; go back to the trace.

## Completion criteria

- [ ] Kill criterion and baseline stated before the run.
- [ ] Validity checks passed (bound, penetration/conservation, convergence, known-input test).
- [ ] Every number has n and an interval; comparisons are paired where possible.
- [ ] Notebook entry written, invalid runs marked, board updated; sim claims labelled as sim.
