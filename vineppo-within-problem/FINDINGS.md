# Within-problem credit assignment: bootstrap CIs for all three estimators

**Question.** The vineppo_replication README's headline numbers are mostly pooled correlations,
which are dominated by between-problem difficulty (87.5% of V* variance). The honest
credit-assignment signal is the WITHIN-problem correlation (simulator was quoted as r ~ 0.62
within-problem, but with no CI). Recompute all three estimators (simulator, hidden-state probe,
verbalized judge) within-problem, with problem-level (cluster) bootstrap CIs.

**Data.** `/Users/steven/vineppo_replication/results/data.jsonl` (read-only): 23 GSM8K problems,
3 prefixes each = 69 rows. Analysis only; no new generation.

## Pre-registered analysis (written before running)

- **Estimators.**
  - simulator: `v_sim` (8 disjoint rollouts).
  - verbalized: `v_verb` (one-pass P("Yes") from logits).
  - probe: out-of-fold Ridge(alpha=10) on the stored 1536-dim hidden states, 3-fold
    problem-level CV with permutation seed 0 (exactly matching `analyze2.py`), predictions
    clipped to [0,1]. Probe predictions are computed once and held FIXED across bootstrap
    resamples; probe training variability is assessed separately by a CV-seed sweep
    (secondary, below).
- **Primary statistic.** For each estimator: pooled within-problem Pearson r, i.e.
  Pearson correlation between (v_est - problem mean of v_est) and (v_gold - problem mean
  of v_gold) over all 69 prefixes. Problems whose 3 gold values are identical contribute
  zero-deviation gold points (they stay in, as in standard within-cluster correlation).
- **Primary uncertainty.** Nonparametric cluster bootstrap over the 23 problems
  (resample problems with replacement; a problem carries its 3 already-centered prefixes),
  B = 10,000, numpy seed 0, percentile 95% CI. Degenerate resamples (zero variance in
  either centered vector) are dropped and counted.
- **Decision rule.** An estimator carries detectable within-problem signal iff its 95% CI
  excludes 0. Report the point estimate and CI for all three regardless of outcome.
- **Pre-stated secondary analyses** (same bootstrap resamples, same seed):
  1. Pairwise differences in within-problem r (sim-probe, sim-verb, probe-verb) with 95%
     percentile CIs; a difference is claimed iff its CI excludes 0.
  2. Mean per-problem Spearman (only problems with variance in both vectors; 3 points per
     problem, so very coarse) with problem-bootstrap CI.
  3. Pooled (uncentered) r for each estimator with the same cluster bootstrap, for the
     pooled-vs-within contrast.
- **Labeled post-hoc / supplementary** (interpretation aids, not tests):
  1. Simulator noise-only ceiling: parametric simulation with p_hat = (8*v_sim + 16*v_gold)/24
     per prefix, drawing independent Binomial(8, p_hat)/8 and Binomial(16, p_hat)/16, recomputing
     within-problem r, 2,000 draws -> the within-problem r that pure binomial sampling noise
     predicts for an estimator identical to the "gold" up to rollout count.
  2. Probe CV-seed sweep: 20 permutation seeds for the 3-fold problem split; spread of the
     within-problem r point estimate.
  3. Count of problems with zero within-problem gold variance; median P(Yes) of the judge.

## Results

All numbers: 23 problems, 69 prefixes, cluster bootstrap B=10,000, seed 0, percentile 95% CIs.
Within-problem share of V* variance: 12.5% (4 of 23 problems have zero within-problem gold variance).

### Primary (pre-registered): within-problem Pearson r

| estimator  | r      | 95% CI            | CI excludes 0? |
|------------|--------|-------------------|----------------|
| simulator  | +0.621 | [+0.236, +0.850]  | yes            |
| probe      | +0.219 | [+0.026, +0.396]  | yes (barely)   |
| verbalized | -0.203 | [-0.468, +0.078]  | no             |

- The README's quoted within-problem simulator r ~ 0.62 replicates exactly and now has a CI:
  the 8-rollout simulator carries within-problem (step-level) signal clearly distinguishable from 0.
- NEW vs the README: the hidden-state probe, which looks like nothing pooled (r = +0.02,
  CI [-0.31, +0.33]), shows a small within-problem signal whose pre-registered CI excludes 0.
  The pooled near-zero number was hiding a weak step-level signal under the difficulty confound
  (the probe's pooled failure is a between-problem failure). But see fragility notes below.
- The verbalized judge is null within-problem, consistent with the README's "degenerate judge"
  reading (median P(Yes) = 0.072).

### Secondary (pre-registered)

Pairwise differences in within-problem r (same resamples):

| contrast              | diff   | 95% CI            | excludes 0? |
|-----------------------|--------|-------------------|-------------|
| simulator - probe     | +0.402 | [-0.053, +0.733]  | no          |
| simulator - verbalized| +0.824 | [+0.388, +1.161]  | yes         |
| probe - verbalized    | +0.422 | [+0.027, +0.786]  | yes         |

At this n the data CANNOT establish that the simulator beats the probe within-problem;
it does establish both beat the verbalized judge.

Mean per-chain Spearman (3 points/chain, very coarse): simulator +0.450 [+0.120, +0.737]
(18 chains); probe +0.244 [-0.110, +0.570] (19); verbalized +0.012 [-0.369, +0.402] (19).
Same ordering; only the simulator's CI excludes 0 on this coarser statistic.

Pooled r for contrast (cluster bootstrap): simulator +0.878 [+0.777, +0.936];
probe +0.024 [-0.306, +0.330]; verbalized -0.225 [-0.508, +0.182].

### Post-hoc / supplementary (labeled)

1. **Simulator noise-only ceiling.** If the 8-rollout estimator differed from the 16-rollout
   gold only by binomial sampling noise, the predicted within-problem r is 0.690, 95% PI
   [0.523, 0.826] (2,000 parametric draws). Observed 0.621 sits inside this interval:
   the simulator's within-problem r is fully consistent with it being the SAME estimator as
   gold up to rollout count. It remains a reliability coefficient, not accuracy vs external truth.
2. **Probe CV-seed sweep** (20 seeds for the 3-fold problem split): mean +0.245,
   range [-0.039, +0.383]. One of 20 splits gives ~0, so the seed-0 point estimate is not an
   outlier but the signal is split-sensitive.
3. **Step-index control** (post-hoc, not pre-registered): within-problem r(k, gold) = -0.155;
   partial within-problem r controlling k: probe +0.199, CI [-0.036, +0.397] (now marginally
   includes 0); simulator +0.610, CI [+0.227, +0.841] (robust). The probe's signal is not
   explained by step position, but after this control it is no longer individually significant.

### Caveats

- The probe result is the fragile one: the pre-registered CI excludes 0 by a hair
  ([+0.026, ...]), the post-hoc k-controlled CI includes 0, and the bootstrap holds the probe's
  out-of-fold predictions fixed (training variability is NOT inside the CI; the seed sweep shows
  it matters). Honest read: "suggestive within-problem signal, needs more problems," not
  "probe reads value."
- 23 clusters is few for a percentile bootstrap; CIs are approximate. Only 19 problems carry
  any within-problem gold variance.
- Single run, unseeded rollouts upstream; all of this inherits the original run's data.
- Same-estimator circularity for the simulator is unchanged (post-hoc check 1 makes it
  quantitative within-problem).

### Files

- `within_problem.py` - main pre-registered analysis (writes `results.json`)
- `results.json` - all numbers, machine-readable (incl. post-hoc `step_index_control`)

