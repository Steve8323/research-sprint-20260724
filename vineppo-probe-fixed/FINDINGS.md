# Properly-trained one-pass value probe (fix of the audited strawman)

**Question.** In ~/vineppo_replication, the one-pass hidden-state probe scored r ~ 0.06 vs gold
step-value V*, but it was Ridge(alpha=10) on a single 42/27 split in 1536 dims — an audited
strawman. Trained properly (cross-validated, cluster-respecting, alpha-tuned, plus a small MLP,
plus a learning curve), is step-value readable in one pass from these saved states at all?

**Data.** results/data.jsonl (read-only): 23 GSM8K problems x 3 step-prefixes = 69 points.
Features: 1536-dim last-token hidden state of Qwen2.5-1.5B-Instruct (4-bit) at the prefix end.
Target: v_gold = fraction of 16 rollouts from that prefix reaching the correct answer.
Points are clustered by problem (README: ICC ~ 0.88), so all CV is at the problem level.
Analysis-only: no generation, no model loading.

## Pre-registered analysis (written before running)

**Primary test.** Leave-one-problem-out (LOPO, 23 folds) ridge regression on standardized
hidden states, alpha selected per outer fold by inner GroupKFold(5) over
alpha in logspace(1e0..1e6, 13 points), minimizing MSE. Statistic: pooled Pearson r between
out-of-fold predictions and v_gold over all 69 prefixes. Inference: problem-level bootstrap
(resample the 23 problems with replacement, 10,000 draws, seed 0), 95% percentile CI.
"Readable" requires the CI to exclude 0.

**Secondary tests (all pre-stated).**
1. Within-problem r: center predictions and v_gold per problem, Pearson r on residuals;
   permutation test shuffling v_gold within each problem block (10,000 draws, seed 1),
   two-sided p. This is the step-level credit-assignment component.
2. Between-problem r: problem-mean prediction vs problem-mean v_gold (n = 23), Fisher 95% CI.
3. Small MLP: 1536->64->1, ReLU, Adam(lr 1e-3, weight decay 1e-2), 300 full-batch epochs,
   same LOPO CV, 5 seeds (0-4). Metric: pooled r of seed-averaged out-of-fold predictions;
   per-seed r mean +/- sd also reported. Same bootstrap CI as primary.
4. Learning curve (ridge): training-set sizes m in {4, 8, 12, 16, 20, 22} problems,
   50 random draws each (seed 2), inner GroupKFold(min(5, m)) alpha selection, evaluate
   pooled r on all held-out problems' prefixes; report mean and [2.5, 97.5] percentiles
   across draws. Question: is r still climbing at m = 22, or flat?
5. Pipeline positive control: identical LOPO ridge pipeline predicting log(prefix character
   length). If this fails (r < 0.5), the pipeline itself is broken and a null on value is
   uninformative. If it succeeds and value is null, the null is meaningful.
6. Reference lines from the same saved data: simulator (8 rollouts) vs v_gold pooled and
   within-problem r; verbalized judge pooled r.

Anything beyond this list will be explicitly labeled post-hoc.

**Power note (stated up front).** With 23 problems, the problem-level bootstrap on a pooled r
has roughly n_eff = 23; the CI will be wide (~ +/- 0.35 around 0). This experiment can detect
"clearly readable" (r >~ 0.5) and can show "no evidence of readability," but cannot
distinguish r = 0 from r ~ 0.2. That is a data limitation (no generation allowed), not fixable here.

## Results

Runtime: main run 95 s + diagnostic ~5 s, single process, single-threaded BLAS
(a first attempt with default OpenBLAS threading thrashed on the shared machine and was
killed at ~11 min wall; the code was unchanged except for thread pinning).
Positive control PASSED: the identical pipeline decodes log prefix length at
pooled r = +0.750 [boot95 +0.652, +0.828], so nulls on value below are meaningful.

### Primary (pre-registered): LOPO ridge, pooled r

r = -0.591, problem-bootstrap 95% CI [-0.720, -0.466]; n = 69 prefixes, 23 problems.
The CI excludes 0 -- but in the NEGATIVE direction, and the post-hoc diagnostic shows this is a
CV artifact, not inverted value-reading:

- D1 (post-hoc): a zero-feature predictor that outputs the training-fold mean of v_gold gets
  pooled r = -0.936 under the same LOPO. Leave-one-group-out with clustered targets makes the
  training mean mechanically anticorrelate with the held-out problem's mean; near-mean
  predictions (chosen alphas were 1e4-1e6, i.e. heavy shrinkage) inherit that.
- D2 (post-hoc): ridge does NOT beat the mean-only predictor: per-problem MSE 0.1384 vs 0.1338
  (worse by +0.0045, problem-bootstrap 95% CI [+0.0006, +0.0096]); ridge better on 12/23
  problems (exact sign test p = 1.0).
- D3 (post-hoc): artifact-corrected between-problem statistic (problem-mean prediction minus
  fold training mean, vs problem-mean v_gold): r = -0.328, Fisher 95% CI [-0.652, +0.097] -- no
  evidence of between-problem readability.

Honest reading of the primary: NULL. The features add nothing over a mean-only baseline for
problem-level value. (Methodological note, stated plainly: the pre-registered primary statistic
-- pooled r under LOPO with clustered targets -- was itself a poor choice; it can go strongly
negative with zero signal. The pre-registration is what made this visible.)

### Secondary results (pre-registered)

| Test | Result | Inference |
|---|---|---|
| Within-problem r, ridge (step-level signal, artifact-immune) | +0.128 | perm p = 0.237 (10k draws) |
| Within-problem r, MLP seed-avg | +0.247 | perm p = 0.026 uncorrected; NOT significant after multiplicity (~6 correlation tests, Bonferroni threshold 0.008) |
| MLP pooled r (pre-stated MLP metric), seed-avg | -0.077 | boot95 [-0.317, +0.226]; per-seed mean -0.075 +/- 0.036 (5 seeds) |
| Learning curve, pooled r (m = 4/8/12/16/20/22 problems, 50 reps each) | -0.001 / -0.009 / +0.022 / -0.025 / +0.207 / +0.032 | flat around 0; no rising trend; percentile bands span 0 everywhere |
| Learning curve, within-problem r | +0.086 / +0.194 / +0.222 / +0.213 / +0.270 / +0.032 | weak upward drift m=4 to 20, all bands include 0 (m=22 has 1 test problem, uninformative) |
| Reference: simulator (8 rollouts) | pooled +0.878 [boot95 +0.782, +0.936]; within +0.621, perm p = 0.0023 | rollouts DO read step value on this data |
| Reference: verbalized judge | pooled -0.225; within -0.203 | degenerate, as in original repo |

### Answer to the question

Is step-value readable in one pass from these frozen hidden states? With proper
cluster-respecting CV and tuned models, still NO evidence at this data scale:

1. Problem-level (difficulty) component: the probe cannot beat predicting the mean (D2), and the
   artifact-corrected between-problem r is ~0-or-negative (D3). This is a genuine null, not
   starvation-by-bad-split -- the same pipeline reads prefix length at r = 0.75.
2. Step-level (within-problem) component: point estimates are positive (+0.13 ridge, +0.25 MLP)
   and the within-problem learning curve drifts upward, but nothing survives multiplicity;
   the rollout simulator reads the same component at 0.62.
3. Learning curve is flat near 0 for the pooled statistic through m = 22 problems -- no sign
   that a few more problems would change the answer; the within-problem drift suggests the
   step-level component is the one worth testing with a properly powered dataset (hundreds of
   problems), which requires new generation and is out of scope here.

So the original repo's "r = 0.06, underpowered" row is superseded by: "with the strongest
analysis this dataset supports, one-pass linear/shallow readability of value is null; a small
positive step-level trend exists but is not significant; power remains the binding constraint
for effects r <~ 0.3."

### Caveats

- 23 problems is the hard ceiling; n_eff ~ 23 for anything pooled. CIs are wide exactly as
  pre-stated in the power note. This experiment can rule out "clearly readable" (r >~ 0.5 would
  have been detected: the control at r = 0.75 was), not "modestly readable."
- D1-D3 are post-hoc, run to interpret the pre-registered primary; labeled throughout.
- The MLP within-problem p = 0.026 is the sole nominally significant probe result and is best
  treated as a hypothesis for a bigger dataset, not a finding (uncorrected, one of ~6 tests,
  and the pre-stated MLP metric was pooled r, which is null).
- Hidden states are from a 4-bit 1.5B model at the last prefix token only; "not readable here"
  does not mean value is absent elsewhere in the network (other layers/tokens), nor does it
  bound a fine-tuned value head (VinePPO's actual claim target).
- Learning-curve draws share the same 23 problems, so points along the curve are not independent
  of each other or of the primary.

### Files

- probe_fixed.py -- pre-registered experiment (this file wrote results.json)
- posthoc_diagnostic.py -- labeled post-hoc artifact diagnostic (D1-D3)
- results.json -- all numbers incl. posthoc_diagnostic block
- run.log -- console log of the main run
