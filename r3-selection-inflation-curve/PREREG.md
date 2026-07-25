# PREREGISTRATION — How much does a sweep inflate the winner? A correction rule.

Stamped with `date` at time of writing, before any code in this directory was run:

```
Sat Jul 25 10:12:59 EDT 2026
```

Round-3 methods project. All writes go only to
`/Users/steven/research_sprint_20260724/r3-selection-inflation-curve/`. Prior-round directories
and `~/` repos are read-only inputs.

## HONESTY DISCLOSURE (read first)

This is a **pre-specification of the analysis, not an outcome-blind preregistration.** The
validation targets in Part B are numbers already published in
`r2-fragility-prereg/FINDINGS.md` and `noiseprobe-prereg-replication/FINDINGS.md`, and I read
those files before writing this document. There is no way to be blind to them and still do the
project. The mitigations, all fixed here before any computation:

1. The correction rule's **functional form and every parameter** are fixed below and are
   estimated **only** from (i) the original sweep's cell means and (ii) *variance-only*
   quantities (an item x cell interaction SD) from the replication data. No fresh-prompt *mean*
   is allowed to touch the rule's parameters.
2. The **decision criteria** (pass/fail thresholds, tests, n) are written below before any
   prediction is computed.
3. Part A (the derivation + simulation) is genuinely confirmatory: that simulation has not been
   run and its pass criteria are stated below in advance.
4. Anything not named below is labeled POST-HOC in FINDINGS.md.

## Background

`r2-fragility-prereg/` found that a published layer-specific claim from
`~/misalignment_noise_probe` was a real but layer-GENERAL effect that a ~144-cell sweep had
over-fit: the post-hoc selected contrast c = mean(d(L4), d(L8), d(L11)) - d(L6) = **-2.72 nats**
in the original sweep, but **+0.22 [-0.72, +1.16]** on 24 fresh prompts (one-sided exact
Wilcoxon p = 0.76, failed replication), while never-selected layers showed a clean layer-general
effect (pooled d = **-0.64**, p = 2.5e-5, n = 32 fresh prompts). Goal of this project: turn that
one anecdote into a reusable, validated deflation rule.

## Part A — Derivation and simulation of max-of-k inflation (CONFIRMATORY)

**Estimator family (fixed).** A sweep of k cells. Cell j has an unknown true value
theta_j ~ N(mu, tau^2) and an observed cell mean D_j = theta_j + e_j, where e_j is the mean of
n exchangeable item-level scores, so e_j ~ N(0, s^2) with s = sigma_w / sqrt(n) and sigma_w the
item x cell interaction SD. Selection picks j* = argmax_j (D_j) in the hypothesised direction
(or a top-m-minus-bottom-1 contrast). Reliability rho = tau^2 / (tau^2 + s^2).

**Claimed closed forms (to be tested, not assumed).**
- Expected inflation of the winner as an estimate of its own true value:
  `E[D_j* - theta_j*] ~= a_k * s * sqrt(1 - rho)`, with `a_k = E[max of k iid N(0,1)]`,
  approximated by Blom `a_k ~= Phi^-1((k - 0.375)/(k + 0.25))`.
- Operational deflation (empirical-Bayes conditional mean):
  `theta_hat_j = Dbar + rho_hat * (D_j - Dbar)`, with method-of-moments
  `rho_hat = max(0, 1 - s^2 / S^2)`, `S^2 = (1/(k-1)) sum_j (D_j - Dbar)^2`.
  For a zero-sum contrast of cells, this reduces to a pure multiplicative shrink:
  `c_hat = rho_hat * c_obs`.
- Rule of thumb with no item-level data: a completely null sweep (tau = 0) produces a winner
  whose excess over the sweep mean is on average `a_k * sigma_w / sqrt(n)`. Define the
  **selection-inflation ratio** `R = |D_j* - Dbar| / (a_k * s)`; R <= 1 means the winner is
  fully explicable as selection noise.

**Simulation grid (fixed now).** k in {5, 10, 24, 50, 144}; n in {6, 12, 32, 100};
rho in {0, 0.25, 0.5, 0.75, 0.9}; item-noise distribution in {normal, t_5 (scaled to unit
variance)}; exchangeable cell-error correlation r in {0, 0.5} (a common set-level component
that cancels within a sweep). sigma_w = 1 throughout (all results in sigma_w units).
2000 replicates per grid cell, numpy, seed 20260725. 5*4*5*2*2 = 400 grid cells.

**PRE-STATED PASS CRITERIA (Part A).**
- **P1 (closed form).** On the normal, r = 0 sub-grid (100 cells), the closed form
  `a_k * s * sqrt(1-rho)` matches the simulated mean inflation `E[D_j* - theta_j*]` with
  relative error <= 15% in >= 90/100 cells. FAIL otherwise.
- **P2 (rule works).** Over all 400 grid cells, the EB rule with `rho_hat` estimated from the
  same sweep satisfies BOTH: (a) |bias| <= 0.25 * s in >= 90% of cells, and (b)
  RMSE_EB <= RMSE_naive in >= 90% of cells. FAIL otherwise. (This can genuinely fail: at high
  rho and small k, `rho_hat` is noisy and over-shrinks.)

Estimand throughout: theta_j* (the selected cell's true value), i.e. what a fresh, infinitely
large replication of that same cell would return.

## Part B — Validation against the real noise-probe sweep (PRE-SPECIFIED, NOT BLIND)

**Sweep data (rule inputs).** `~/misalignment_noise_probe/layer_importance_results.json`:
per-layer mean margin change for base and misaligned at sigma in {0.1, 0.2, 0.4} over 24 decoder
layers (144 cells total; 6 prompts x 3 seeds per cell, only cell means saved). The published
selection happened within the sigma = 0.2 row, so **k = 24** and **n = 6**. Sweep quantity:
`d_orig(L) = mis(L) - base(L)`; `Dbar` = mean over all 24 layers; `S^2` = variance over the 24
layers.

**sigma_w (variance-only, from replication data).** Two-way (prompt x layer) ANOVA residual SD
of the per-prompt differential `d_i(L)` computed from `r2-fragility-prereg/results.json`
(32 prompts x 8 layers = 256 cells, layers {2,4,6,8,11,12,16,20}). This uses only the
interaction variance, never a layer mean. Then `s = sigma_w / sqrt(6)` for the original sweep.

**Fresh replication values (validation targets).** `fresh(L)` = mean over the same 32 fresh
prompts of `d_i(L)`, for the 8 layers above, from `r2-fragility-prereg/results.json`. The r1
24-prompt values are used only for the V1 CI.

**V1 (headline anecdote).** Rule prediction for the published selected contrast
`c_sel = mean(d_orig(4), d_orig(8), d_orig(11)) - d_orig(6) = -2.72` is `rho_hat * c_sel`.
PASS iff (a) the prediction lies inside the r1 fresh 95% CI of c, [-0.72, +1.16], AND (b) the
naive value -2.72 lies outside it. Graded companion (reported either way): fraction of the
observed shrinkage explained = `(naive - pred) / (naive - fresh_observed)`.

**V2 (calibration, PRIMARY real-data test).** For the 8 layers with fresh data, compare
`|EB(L) - fresh(L)|` vs `|naive(L) - fresh(L)|`, where `naive(L) = d_orig(L)` and
`EB(L) = Dbar + rho_hat * (d_orig(L) - Dbar)`. Test: one-sided exact Wilcoxon signed-rank,
n = 8, H1: EB errors smaller, alpha = 0.05 (minimum attainable exact p = 0.0039). Also report
RMSE ratio. PASS iff p < 0.05.

**V2b (pre-stated secondary).** Same comparison after removing the mean level difference
between `fresh` and each predictor over the 8 layers (prompt-set register shift is a nuisance
that both predictors share).

**V3 (over-shrinkage / negative control, pre-stated).** The rule must not deflate what was never
selected. For the 4 never-selected layers {2, 12, 16, 20}: report the signed errors of EB and
naive and a two-sided t-test of EB signed error = 0. Interpretation stated now: if the rule is
over-shrinking, EB signed errors on unselected layers will be systematically opposite in sign to
their `d_orig` deviations.

**V4 (positive control, pre-stated).** The unselected pooled quantity: sweep grand mean `Dbar`
(no selection, hence no deflation) vs the fresh pooled untested-layer mean -0.642 [-0.918,
-0.367]. Report whether `Dbar` falls in that CI. This is the "unselected numbers replicate
without correction" control.

## Part C — Apply the rule to other selected-from-sweep numbers in the portfolio

Procedure fixed now for each target: name the selection structure (k, n), estimate s from saved
item-level data, compute `Dbar`, `S`, `rho_hat`, the selection-inflation ratio
`R = |D_sel - Dbar| / (a_k s)`, and the deflated estimate. **Pre-stated verdict scale:**
- `IMMUNE` — the number was pre-specified or the statistic is selection-free (no max over cells).
- `SUSPECT` — R < 1.5 (the winner's excess is within 1.5x what a null sweep produces).
- `BORDERLINE` — 1.5 <= R < 3.
- `ROBUST` — R >= 3.

Targets fixed now:
1. `routing-hybrid-knob/` endpoint localization: the claim that essentially all
   relearning-resistance arrives at alpha = 1 (post-hoc alpha=1 vs alpha=0.75 at p_waste=0.9,
   exact p = 0.031), and the T3 "interior alpha is non-dominated" claim. k = 5 alphas per
   regime (10 cells over both regimes), n = 6 seeds; per-seed AUC / retain in `results.json`.
2. `misgen-bimodality-honest/` regime selection: k = 25 regimes, n = 100 seeds; per-regime
   scores saved. Apply to the dip statistic across the 5 widths at equal cost c = 1.0 (the
   parent repo selected its published regime by searching widths for bimodality), and state
   whether the Holm-corrected count is immune (Holm corrects p-values, not effect sizes).
3. `misgen-connectivity-graph/` theta selection: the primary AUC is threshold-free (expected
   IMMUNE), but the reported maxima over a 1770-point theta grid — peak modularity Q = 0.22,
   max ARI = 0.108, and the "Q exceeds its 97.5th pointwise percentile for theta in
   [0.0015, 0.090]" window — are max-over-k statistics with k = 1770. Because a pointwise
   permutation band is the wrong null for a maximum over the grid, I will recompute a
   **max-over-theta** node-label permutation null for Q from the saved 60x60 barrier matrix
   (`~/moral_framework_misgen/stats/results.json`, read-only), 1000 permutations, seed 0,
   using an incremental edge-addition computation of Q. Decision: the peak Q is SUSPECT iff
   Q_max_obs <= 95th percentile of the permutation distribution of max_theta Q. ARI is
   secondary (subgrid stride 10, 200 permutations) and is reported descriptively if it fits
   the budget.

## Compute budget

Pure numpy simulation + analysis of saved artifacts. No model loads, no downloads, single
process. Target < 5 min wall clock, hard cap 15 min. Pre-stated downscoping order if the cap is
threatened: (1) drop the ARI permutation null, (2) reduce Part-A replicates from 2000 to 500,
(3) drop the t_5 arm of the Part-A grid.

## What counts as success

Negative results are fine and will be reported plainly. Specifically: if P2 fails, the rule is
not usable as stated; if V2 fails, the rule does not predict the real shrinkage and the r2
anecdote stays an anecdote; if V4 fails, the whole "unselected numbers replicate" premise is
wrong and the framing must change.
