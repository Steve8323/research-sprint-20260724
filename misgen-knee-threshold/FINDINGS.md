# Data-driven knee for the barrier census

## Question

`stats/paper_stats.py` promises a "data-driven knee" connectivity threshold for the
60-seed permutation-aligned barrier census but hardcodes `thr_abs = 0.05` (the knee is
comment-only, lines 313-316). The README flags this: "the threshold is ad hoc (a planned
data-driven knee was not implemented)". Does the paper's connectivity verdict -- all 60
models merge into ONE component at thr = 0.05 -- survive when the threshold is chosen by
a principled knee on the barrier distribution itself?

## Data

Analysis-only, no retraining. Saved 60x60 permutation-aligned barrier matrix
(`stats/results.json` -> `basins/barrier_matrix`; verified symmetric, zero diagonal,
min/median/max reproduce the saved summary: 0.00045 / 0.02437 / 0.35612, M = 1770 pairs)
and per-seed deployment-alignment scores (`multimodality/scores[:60]`; verified against
`basins/cluster_align`: mean 0.4493, 12 aligned (>0.8), 17 misaligned (<0.2), 31 hedgers).

## Pre-stated method (this section written BEFORE computing the knee or any census at it)

**Primary threshold (Kneedle-type elbow on the sorted barrier curve).** Let
b_(1) <= ... <= b_(M) be the M = 1770 sorted pairwise barriers. Normalize
x_i = (i-1)/(M-1), y_i = (b_(i) - b_(1)) / (b_(M) - b_(1)). The curve is increasing and
(for low-mode + heavy-tail data) convex, so the elbow is the point of maximum vertical
distance BELOW the diagonal: i* = argmax_i (x_i - y_i), thr_knee = b_(i*). This is the
standard max-distance-to-chord / Kneedle elbow with no tuning parameters. Rationale: the
census premise is that same-basin pairs have barrier ~ 0 under permutation alignment;
the elbow separates the near-zero mode from the tail.

**Primary outcome, decided in advance.** Build adj = (B <= thr_knee), diagonal = 1
(identical to `census()` in paper_stats.py), take connected components, and report:
(a) number of components and their sizes;
(b) whether any component contains BOTH an aligned (score > 0.8) and a misaligned
    (score < 0.2) seed;
(c) fraction of the 12 x 17 = 204 aligned-misaligned pairs directly connected
    (barrier <= thr_knee), with exact Clopper-Pearson 95% CI.

**Decision rule, stated in advance.**
- All 60 seeds one component at thr_knee -> published "one basin at the chosen
  threshold" conclusion UNCHANGED (and now data-driven).
- Aligned and misaligned seeds in disjoint components -> conclusion CHANGES to full
  worldview separation at the data-driven threshold.
- Multiple components but at least one mixes aligned and misaligned -> intermediate:
  partial fragmentation, worldviews still connected.

**Pre-stated robustness / secondary analyses (labeled secondary):**
- S1. Alternative thresholds: Otsu (max between-class variance, 512-bin) on barriers,
  and 2-component GMM on log10(barriers) with threshold at the posterior crossing;
  census at each.
- S2. Leave-one-seed-out jackknife: drop each seed, recompute thr_knee + census on the
  59x59 submatrix; report thr_knee range and how often verdicts (a)/(b) change.
- S3. Census curve n_components(thr) on a fine grid; percolation threshold = smallest
  thr at which the graph is fully connected (bottleneck of the minimum spanning tree);
  which seeds merge last.
- S4. Hedger-chain test at thr_knee: delete the 31 hedger seeds (0.2 <= score <= 0.8),
  recompute components among the 29 committed seeds; do aligned and misaligned still
  connect without hedger intermediaries?
- Descriptive cross-check: worldview-grouped barrier means (cross vs within) reproduced
  from the matrix against the README's 0.064 vs 0.029 and Mann-Whitney p.

Everything below the rule was written AFTER running the pre-stated analysis.

---
## Result

**Primary (pre-registered): the knee lands at thr_knee = 0.0858 — ABOVE the ad hoc 0.05
— and the census verdict is unchanged: one component containing all 60 seeds.**
Per the pre-stated decision rule: the published "one basin at the chosen threshold"
conclusion is UNCHANGED, and is now data-driven rather than ad hoc.

| quantity | value |
|---|---|
| pre-registered knee (Kneedle on 1770 sorted pairwise barriers) | thr_knee = 0.0858, at pair-quantile 0.90 |
| census at thr_knee | 1 component, size 60 (12 aligned + 17 misaligned + 31 hedgers) |
| aligned & misaligned share a component | yes |
| direct aligned-misaligned edges at thr_knee | 149/204 = 0.730, Clopper-Pearson 95% CI [0.664, 0.790] |
| fraction of all pairs <= thr_knee | 0.902 |

**Secondaries (pre-stated):**
- S1 alternatives: Otsu = 0.0869 -> identical verdict (1 comp, 151/204 direct cross).
  2-comp GMM on log10(barriers) crosses much lower, 0.0205 -> 2 comps [59, 1]; the
  singleton is seed 17 (a hedger, score 0.79), and aligned + misaligned STILL share the
  59-seed component. No principled threshold separates the worldviews.
- S2 leave-one-seed-out jackknife (n = 60 drops): thr_knee in [0.0845, 0.0858];
  1 component and shared-worldview verdict in 60/60 drops. The knee is stable.
- S3 percolation: the graph is already fully connected at thr = 0.0206 (bottleneck of
  the barrier MST), 4.2x below the knee and 2.4x below the paper's 0.05. The last seed
  to merge is a hedger (seed 17, score 0.79), not a committed model.
- S4 hedger-chain test: deleting all 31 hedgers, the 29 committed seeds still form ONE
  component at thr_knee. Aligned and misaligned connect directly, not only via hedgers.

**Post-hoc (labeled): the README's hedger-chain claim fails even at 0.05.** At the
paper's own threshold, 105/204 = 51.5% (CI95 [0.444, 0.585]) of aligned-misaligned
pairs are directly connected and the 29 committed seeds form one component with all
hedgers deleted. The sentence "All 60 models do merge into one component at that
threshold — but only via chains through hedger seeds" is contradicted by the saved
matrix on its second half. (Saved in `results.json` under `posthoc_at_paper_thr`.)

**Descriptive cross-check:** cross-worldview barrier mean 0.064 (n = 204) vs
within-group 0.029 (n = 202), Mann-Whitney p = 2.3e-15 — reproduces the README exactly.
The "partial separation in barrier HEIGHTS" half of the conclusion stands.

**Interpretation.** The barrier distribution has no near-zero "same-basin mode +
separated tail" structure that a knee could exploit to split worldviews: the Kneedle
elbow sits at the 90th percentile of pairs (it separates the heavy tail of the worst
0.36-high barriers, not a zero mode), and full connectivity is reached already at 0.021
— barely above the median barrier 0.024. Under every principled threshold tried
(knee, Otsu, GMM crossing, percolation), aligned and misaligned models end up in the
same connected component. The honest summary is unchanged from the README but now
threshold-free: worldview shows up in barrier heights (p ~ 2e-15) but never in the
component structure, at any data-driven threshold.

## Caveats

- Analysis-only on the single saved 60x60 matrix (one training config: width-8 MLP,
  154 params, seeds 700-759). No new seeds were trained; n = 60 seeds / 1770 pairs is
  what the paper shipped.
- Pairwise barriers are not independent (each seed appears in 59 pairs); the
  Clopper-Pearson CI on 149/204 treats pairs as independent and is therefore
  anti-conservative. The jackknife (S2) is the dependence-aware stability check.
- The knee is on the pooled barrier distribution, as the paper's comment promised; a
  knee on the n_components(thr) curve itself would be an alternative formulation (the
  full curve is saved in results.json; it is a cliff at 0.021 and flat after, so any
  elbow on it gives the same one-component verdict).
- Barriers are linear-path max-loss after activation-matching permutation alignment —
  an upper bound on true (nonlinear-path) barriers; lower true barriers would only
  strengthen the one-component verdict.
- The GMM log-crossing threshold (0.0205) sits in the steep part of the census curve;
  verdicts just below it differ in component count (2 vs 1) but not in the
  worldview-sharing outcome, which is the pre-stated question.

## Files

- `knee_census.py` — full analysis (analysis-only, ~5 s CPU)
- `results.json` — all numbers above, incl. the 800-point census curve and the
  labeled post-hoc block at thr = 0.05
- `fig_knee_census.png` — sorted-barrier curve with knee; n_components(thr) with all
  four thresholds marked

Verification note (2026-07-24): the full pipeline was re-run and every headline number
(knee 0.08576 at quantile 0.902, 1 component, 149/204, jackknife range, percolation
0.02060 with seed 17 last, 105/204 and 1-committed-component at 0.05, MW p = 2.3e-15)
was additionally reproduced by an independent re-implementation of the knee and census
against `stats/results.json`. The published `census_by_thr` values (2/1/1/1 components
at 0.02/0.05/0.10/0.20) also reproduce from the saved matrix.
