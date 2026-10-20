# Statistics for small robot experiments

## Success rates (binary outcomes)

- **Wilson 95 % interval** for k of n (use instead of +-sqrt(p(1-p)/n), which fails near 0 % / 100 %):
  `c = (p + z^2/2n) / (1 + z^2/n)`, `h = z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)`, interval `c +- h`, z = 1.96.
  Code: `robot_manipulator.sim.insertion.wilson(k, n)`.
- Reference widths (n = 30): 19/30 -> 46-78 %, 16/30 -> 36-70 %, 30/30 -> 89-100 %, 0/30 -> 0-11 %.
  n = 100: 34/100 -> 25-44 %, 81/100 -> 72-87 %. n = 20 (5 % steps) is too coarse to rank close variants.

## Comparing two variants

- **Paired (same episodes/seeds/initial states):** exact McNemar on discordant pairs. b = only A succeeds,
  c = only B succeeds; p = 2 * sum_{i<=min(b,c)} C(b+c, i) / 2^(b+c). Code: `insertion.mcnemar_p(b, c)`;
  CLI: `uv run insertion-study compare <A.json> "<row>" <B.json> "<row>"`. Report b and c, not just p.
- **Unpaired:** Fisher's exact test on the 2x2 table (`scipy.stats.fisher_exact`). Example from GRA (arXiv
  2606.24448): 62/90 vs 55/90 -> p ~ 0.35, i.e. their headline +7.8 pts is not significant.
- **Many factor cells:** a logistic regression on the factor (e.g. log2 gain) with a corrected test on the
  hypothesis region is stronger than cell-by-cell comparisons (Tune to Learn, arXiv 2604.02523).

## Sample size (unpaired, two-sided alpha 0.05, power 0.8)

`n = (1.96 + 0.84)^2 (p1(1-p1) + p2(1-p2)) / (p2 - p1)^2` per arm.
30->50 %: 91; 60->75 %: 150; 60->70 %: 354. Pairing reduces this a lot when outcomes are correlated.

## Continuous outcomes (mm, deg, s)

- Report median and spread (IQR or min-max) with the number of units (poses, corners, episodes).
- Paired differences (same target, before/after trim) -> sign test or Wilcoxon signed-rank; "all 20 of 20
  improved" is itself strong evidence (sign test p ~ 2e-6).
- Separate bias (same sign everywhere) from scatter (repeatability); they have different fixes.

## What to write

`81 % [72-87], n = 100, paired with chamfer 0: changed 47-0, McNemar p < 0.001` - not `+138 %`.
