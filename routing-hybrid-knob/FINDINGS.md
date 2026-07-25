# Random <-> structural routing interpolation: Pareto of (unlearning quality, relearning-resistance)

## Question

In ~/gradient_routing_experiment, random per-weight waste masks beat structural
(whole-neuron) masks on heavy-deletion capacity/absorption/composability but lose on
relearning-resistance (published README: "recovery AUC 0.65 vs 0.72 (6/6 seeds)",
structural more resistant). Is there an interpolation knob alpha between the two
partitions whose intermediate settings trace a useful Pareto frontier of
(unlearning quality, relearning-resistance)?

## Pre-registered design (written BEFORE running the full grid)

**Knob.** alpha in {0, 0.25, 0.5, 0.75, 1.0}. Mask construction per routed layer
(fc1.weight, fc2.weight), total expected waste fraction fixed at p_waste:
- a fraction `alpha * p_waste` of output rows become whole waste rows (structural part);
- remaining entries are scattered waste i.i.d. with probability
  `(1-alpha)*p_waste / (1 - alpha*p_waste)` so total expected waste mass is exactly
  p_waste at every alpha.
alpha=0 reproduces the repo's "random" partition distribution; alpha=1 reproduces
"structural". Same RNG discipline as `harness.make_masks` (seeded generator).

**Training / attack.** Exactly the published tamper_experiment pipeline, reusing its
verified code: ablation-aware clean-scalpel training (fix_experiment.step_aa, aa=True),
MLP h=128, 8 epochs, bs=128, Adam 1e-3, 6000 train / 2000 test MNIST subsample,
forget={0..4}. Then permanent prune + freeze of waste entries, then relearning attack:
Adam 1e-3 on 200 forget examples, probe forget TEST accuracy at steps
{0,10,20,30,50,75,100}.

**Regimes.** Two p_waste values, both pre-planned (not post-hoc):
- p_waste=0.5: regime of the published relearning result (quality axis may be
  near-degenerate here since both endpoints reached retain ~0.95).
- p_waste=0.9: heavy-deletion regime where the published capacity gap lives
  (retain 0.90 random vs 0.80 structural), so the quality axis has real spread.

**Seeds.** 6 seeds (0-5), matching the published tamper experiment. n=6 per cell.

**Metrics.**
- Unlearning quality U = retain_pruned - forget_pruned (both components also reported).
  Validity gate: a cell only counts as successful unlearning if mean forget_pruned <= 0.05.
- Relearning-resistance = LOW recovery AUC: trapezoid mean of forget test accuracy over
  attack steps 0..100 (identical to the published AUC). Lower AUC = more resistant.

**Primary pre-stated tests (alpha level 0.05):**
- T1 (resistance endpoint replication, p_waste=0.5): per-seed paired AUC(alpha=1) <
  AUC(alpha=0). Exact one-sided Wilcoxon signed-rank, n=6.
- T2 (quality endpoint replication, p_waste=0.9): per-seed paired
  retain_pruned(alpha=0) > retain_pruned(alpha=1). Exact one-sided Wilcoxon, n=6.
- T3 (frontier, descriptive with pre-stated criterion): compute the non-dominated set
  of the 5 alpha grid points in (mean U, mean -AUC) space at each p_waste. The knob is
  "Pareto-useful" iff at least one interior alpha (0.25/0.5/0.75) is non-dominated AND
  strictly better than one endpoint on each axis (i.e. sits strictly inside the span,
  not tied to an endpoint). Otherwise the frontier is just the two endpoints and the
  honest conclusion is "the knob buys nothing".
- Secondary (labeled as such): Spearman trend of alpha vs per-seed AUC; per-seed Pareto
  membership counts; 95% t-CIs (n=6) on all cell means.

**Budget.** 5 alphas x 6 seeds x 2 p_waste = 60 train+attack runs, single process,
2 torch threads. A timing probe of 1 run gates the grid; if projected > ~9 min the
p_waste=0.9 arm drops alpha in {0.25, 0.75} first (this downscoping rule is pre-stated).

## Results

Full pre-registered grid ran as planned (60 runs, 51 s, no downscoping needed).
All 10 cells pass the validity gate: mean forget_pruned = 0.000 everywhere.

### Cell means (n=6 seeds, 95% t-CI)

p_waste = 0.5:

| alpha | retain_pruned | U | recovery AUC (lower = more resistant) |
|---|---|---|---|
| 0.00 (random) | 0.949 [0.942,0.956] | 0.949 | 0.736 [0.714,0.757] |
| 0.25 | 0.944 [0.938,0.951] | 0.944 | 0.721 [0.701,0.740] |
| 0.50 | 0.945 [0.938,0.953] | 0.945 | 0.711 [0.690,0.732] |
| 0.75 | 0.945 [0.940,0.949] | 0.945 | 0.690 [0.659,0.720] |
| 1.00 (structural) | 0.943 [0.935,0.951] | 0.943 | 0.650 [0.612,0.689] |

p_waste = 0.9:

| alpha | retain_pruned | U | recovery AUC |
|---|---|---|---|
| 0.00 (random) | 0.928 [0.920,0.937] | 0.928 | 0.365 [0.284,0.445] |
| 0.25 | 0.931 [0.922,0.941] | 0.931 | 0.353 [0.301,0.406] |
| 0.50 | 0.928 [0.918,0.937] | 0.928 | 0.352 [0.344,0.361] |
| 0.75 | 0.928 [0.919,0.937] | 0.928 | 0.334 [0.272,0.397] |
| 1.00 (structural) | 0.883 [0.844,0.922] | 0.883 | 0.180 [0.072,0.288] |

### Pre-registered tests

- **T1 PASSED**: structural more relearning-resistant than random at p_waste=0.5,
  6/6 seeds, exact one-sided Wilcoxon W=0, p=0.0156. (Replicates the published
  AUC 0.65-vs-0.72 result almost exactly: 0.650 vs 0.736 here.)
- **T2 PASSED**: random better retain than structural at p_waste=0.9, 6/6 seeds,
  W=21, p=0.0156 (0.928 vs 0.883).
- **T3 (frontier)**: on cell means, interior alphas are non-dominated at both regimes
  (p=0.5: {0.5, 0.75}; p=0.9: {0.25, 0.75}) and meet the pre-stated "Pareto-useful"
  criterion. BUT the honest magnitude read differs by regime (see below).

### What the frontier actually looks like

- **p_waste=0.5: a shallow, near-monotone trade.** Resistance improves steadily with
  alpha (AUC 0.736 -> 0.650; secondary pooled Spearman rho=-0.74, p=2.5e-6,
  descriptive) while quality pays only ~0.006 retain, and even that tiny cost is
  paired-consistent (post-hoc two-sided Wilcoxon retain a=0 vs a=1: p=0.031).
  Interior alphas sit between the endpoints on both axes (a=0.5 strictly inside on
  both axes in 5/6 seeds, secondary). The knob works as a knob here, but the whole
  quality axis spans only 0.6 points of retain -- in practice you'd take alpha=1.
- **p_waste=0.9: the trade is threshold-like, not gradual.** Interior alphas
  (0.25-0.75) all match random's retain (post-hoc a=0.75 vs a=0 two-sided p=0.84)
  and gain little resistance (a=0.75 vs a=0: 5/6 seeds but exact p=0.078, n.s.).
  Nearly all of structural's resistance advantage arrives only at the pure endpoint
  (a=1 vs a=0.75: AUC 0.180 vs 0.334, 5/6 seeds, post-hoc exact p=0.031), and that is
  exactly where the 4.5-point retain cost appears (T2). So in the heavy-deletion
  regime the frontier is effectively TWO points -- "random-like" (retain 0.93,
  AUC 0.35) and "fully structural" (retain 0.88, AUC 0.18) -- with no useful
  interior compromise: partial structure buys ~nothing.

Figure: `pareto.png` (per-seed points + means, both regimes).
Raw + analysis: `results.json` (keys: runs, analysis.cell_summary, analysis.T1/T2/
T3_pareto, analysis.posthoc -- all post-hoc items are labeled).

## Verification (independent re-run, 2026-07-24 afternoon)

A second agent instance audited and re-ran the full pipeline from scratch:
- `hybrid_knob_experiment.py` re-run end-to-end (60 runs, 42 s): all 60 run records
  **bit-identical** to the first run (deterministic seeding confirmed;
  first run preserved as `results_prior_run.json`).
- `analyze.py` re-run: T1 (W=0, p=0.0156), T2 (W=21, p=0.0156), T3 sets, and all
  cell means/CIs match the tables above exactly.
- Reproducibility gap fixed: the post-hoc tests and `pareto.png` originally had no
  generating code on disk. Added `posthoc_figure.py`; its recomputed post-hoc W/p
  values match the originals exactly and it regenerates the figure.
- Faithfulness of the knob's endpoints checked against `harness.make_masks`:
  structural there is whole-row Bernoulli(p_waste) masks on fc1.weight/fc2.weight
  (both HIDDEN layers; fc3 is the untouched output head), same seed+999 generator.
  alpha=1 matches it exactly for fc1 and distributionally for fc2 (RNG consumption
  order differs); alpha=0 matches the scattered partition distributionally.

### Interpretation (mechanism guess, untested)

Relearning-resistance seems to come from removing WHOLE hidden units: only at alpha=1
do entire neurons vanish, shrinking the effective width available to the attacker.
Any scattered component (alpha<1) leaves every unit partially alive, and the attack
re-uses them at similar speed regardless of how much of the mask is row-structured.
That would explain both the alpha=1 jump at p=0.9 and the milder monotone trend at
p=0.5 (where masks remove at most half of each row's mass). Untested; labeled a guess.

## Caveats

- Toy scale: MLP h=128 on 6000-example MNIST subsample, class-split "capability".
- n=6 seeds; exact tests used throughout; smallest achievable one-sided p is 0.0156.
- Recovery AUC over 100 attack steps saturates near the ceiling at p=0.5, so AUC
  mostly measures recovery SPEED (same caveat as the published tamper experiment).
- The alpha grid has 5 points; "threshold-like at p=0.9" means "between 0.75 and 1.0";
  a finer grid near alpha=1 was not run (budget).
- Post-hoc items (endpoint-jump test, retain flatness tests) are labeled as such and
  their p-values are not corrected for the 4 comparisons made.
- The interpolation is one specific knob (row-fraction mixing at fixed total mass);
  other interpolations (e.g. block sizes, per-row partial widths) might behave
  differently.

