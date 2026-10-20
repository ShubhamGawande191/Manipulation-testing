# Reading papers critically

Use when evaluating related work, a research proposal, or a method we might adopt. Private reading notes
live under `third_party/` (gitignored, never committed).

## Protocol

1. **Read in full**, appendix included (setups, prompts, thresholds and compute often sit there). If a
   section was only skimmed, say so.
2. For each headline claim extract: tasks, trials per cell, seeds, n of demos/data, absolute numbers for every
   baseline, whether intervals or tests exist.
3. Recompute: relative -> absolute points; counts -> Wilson intervals; two arms -> Fisher/McNemar.
4. Separate **measured** from **asserted**: ablations with numbers vs qualitative figures; "we observe" vs a table.
5. Note the knob they set **by eye** (thresholds, loss weights, control weights) - often the open problem.
6. Note their **stated limitations and open problems** verbatim (short quotes, < 15 words).
7. Note compute and hardware (GPU memory, hours, robot) for feasibility on our machines.
8. Write one line: what is taken, what is open, what is weakly evidenced and could be done properly.

## Red flags

- Relative gains on small bases (e.g. "+179.9 %" from 13.9 % -> 38.9 % on 24 trials/task).
- 20-40 real trials per cell with 5-10 point differences and no intervals.
- Ablations shown only as pictures.
- A filter that removes 5 % of data and gains 2-3 points.
- Sim actions or generated actions reused on a different (cheaper) actuator without a check.

## A finding can still be worth doing again

"Done before" is not "done well". If the prior work's evidence is weak (no intervals, global metrics,
knob set by eye), a careful, well-powered measurement of the same idea is a contribution - say exactly
what is measured better and why it matters.
