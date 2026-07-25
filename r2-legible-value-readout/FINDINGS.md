# Legible-word value readouts, done right (round 2) — FINDINGS

**Headline.** When the calibration miss is fixed so both arms carry real value variance, the
difficulty-removed step-value signal is significantly larger from the ERT's legible words than
from a matched continuous model's hidden states (within-input ridge r 0.055 vs 0.004, 5/5 seeds,
MW p=0.0079) — but the effect is tiny, the continuous states read essentially zero, and the same
"legible beats states" ordering does NOT reproduce on real GSM8K tokens (vineppo), where at 23
problems neither cheap text features nor frozen hidden states reliably read difficulty-removed
step-value and hidden states are if anything numerically ahead.

All tests, n, seeds, CIs are pre-registered in PREREG.md (stamped Fri Jul 24 23:10:10 EDT 2026,
before any run). Amendment 1 to Part-1 calibration was made after the pilot sweep, before the 5-seed
main run, and is documented in PREREG.md; its selection is confirmed by the logged pilot grid.

## PART 1 — ERT toy, variance-matched and calibration-fixed (the study's primary)

### What round 1 got wrong, and the fix
Round-1's primary (pooled MLP r) came out reversed (words 0.14 < hidden 0.56) but was confounded two
ways: the baseline's pooled advantage was pure input-difficulty memorization, and ERT was trained to
acc~0.99 so 6/15 of its V* targets were degenerate (std<0.05 — no variance to read). Round 2 fixes
both: (1) calibrate BOTH models by matching pooled V* std to 0.28 within a working-accuracy band
[0.80,0.98] (Amendment 1 — the naive V*std rule would have compared a working ERT to a near-chance
baseline); (2) make the difficulty-removed (within-input) contrast the pre-registered primary.

### Calibration outcome (variance-match: degeneracy fixed, residual mismatch flagged)
Chosen configs (from the logged pilot grid): ERT steps=60 (pilot acc 0.88, V*std 0.26); baseline
sigma=1.0 steps=25 (pilot acc 0.96, V*std 0.20). Realized over 5 fresh seeds:
- Degeneracy eliminated: 0/15 ERT and 0/15 baseline seed-boundaries are degenerate (round-1 had
  6/15 ERT). The calibration fix worked.
- ERT acc across seeds [0.795, 0.667, 0.848, 0.835, 0.619] (mean 0.75) — ERT training at this
  operating point is unstable. Baseline acc [0.965, 0.988, 0.937, 0.968, 0.967] (stable).
- Consequence: mean V*std ERT 0.353 vs baseline 0.178 (ratio ~2.0, per-seed 1.2-4.0). The residual
  variance mismatch is now REVERSED vs round 1: ERT carries MORE value variance than the baseline.
  The pre-registered <=1.5x success flag is therefore False. This matters for reading the primary.

### PRIMARY (pre-registered): difficulty-removed r — ERT-words > base-hidden, but tiny
Within-input partial Pearson r of the ridge probe at n=256, mean over 3 boundaries, per seed:
- ERT-words (legible):     [0.042, 0.061, 0.056, 0.050, 0.066]  mean 0.055, 95% t-CI [0.043, 0.067]
- base-hidden (continuous):[0.006, 0.011, 0.002, 0.011, -0.012] mean 0.004, 95% t-CI [-0.008, 0.015]
- Exact two-sided Mann-Whitney U (5v5): U=25, p=0.0079 — complete separation, ERT-words > base-hidden.
- Robustness: the MLP secondary probe gives the same direction and significance (ERT 0.059 vs base
  0.028, MW p=0.0159). The pre-registered pooled-secondary contrast is no longer reversed (ERT 0.736
  vs base 0.582, MW p=0.69, n.s.) — round-1's headline reversal was an artifact of the calibration miss.

Two things are true at once:
1. The continuous hidden states read essentially zero difficulty-removed step-value (r=0.004, CI
   covers 0) despite having ample non-degenerate variance to read (V*std 0.10-0.24). The null is not
   starvation — this is the robust half.
2. The legible words read a small but reliably-nonzero difficulty-removed step-value (r=0.055, CI
   excludes 0), and the gap grows with probe data — learning curve (within-input, mean over seeds):
   ERT-words 0.015->0.031->0.055->0.091->0.124 at n=64->1024, vs base-hidden 0.005->0.001->0.004->0.018->0.039.

Caveat that bounds claim (2): the residual variance mismatch is in the direction that would inflate
ERT's r (more V* spread to correlate with), so the SIZE of the ERT-words number is an upper-ish bound,
not a clean estimate. The direction and the base-hidden~=0 finding are solid; the exact 0.055 is fragile.

### Secondary: dictionary distillation no longer dominates (contrast with round 1)
Round-1's one clean "yes" was that class-featurized (public-dictionary-distilled) words read step-value
far better than raw words (within-input 0.36 vs 0.06). At the variance-matched operating point that
advantage largely evaporates: classfeat within-input r [0.039, 0.071, 0.181, 0.065, 0.040] (mean 0.079)
is only +0.024 over raw words on average (one-seed-driven). Round-1's big distillation win was partly
an artifact of the ERT arm being near-degenerate — raw word one-hots already carry the (small) signal
once the value target actually varies. Pooled, classfeat still tracks difficulty strongly (r 0.83).

### Part-1 verdict
With the calibration fixed and difficulty removed: legible words are more value-readable than matched
continuous states (p=0.0079, hypothesized direction), reversing round-1's confounded result — but the
effect is small (r~0.055 vs ~0.004), the continuous states read ~zero, and the size is upper-bounded
by a residual variance mismatch favoring the ERT. Legibility buys a little difficulty-removed
step-value that continuous states don't carry at all; "a little" is the operative word.

## PART 2 — real-token analog on SAVED vineppo artifacts (no new generation)

Probe step-value V* (v_gold, 12-rollout MC) from cheap TEXT features of the saved step strings vs the
frozen 1536-d Qwen hidden states, identical leave-one-PROBLEM-out CV (23 folds, nested-CV ridge alpha),
difficulty-removed. Data: ~/vineppo_replication/results/data.jsonl, 23 problems x 3 prefixes = 69 rows.

### A pipeline artifact found and corrected
My pre-registered metric ("within-problem Pearson r of the OOF probe predictions") is artifact-prone:
an OOF probe fits a global slope dominated by BETWEEN-problem structure, and its within-centered
predictions can flip sign vs the raw within-feature relationship. Concretely: raw within-problem
corr(prefix length, v_gold) = -0.27, yet the length-probe's OOF-within-r = +0.43. The prereg'd metric
is retained in results_part2.json (flagged); the trustworthy metric is a within-model probe trained on
within-centered features -> within-centered target (slope learned only from within-problem variation).

### PRIMARY (honest within-model probe): text does NOT beat frozen states; nothing clears 0
Within-problem r [95% cluster-bootstrap CI over 23 problems, B=5000]:
  hidden (1536-d frozen)   +0.328  [-0.013, +0.631]  clears 0? no (barely)
  bow (bag-of-words)       +0.229  [-0.022, +0.440]  no
  char (3-5 char n-grams)  +0.085  [-0.146, +0.272]  no
  len (2 length scalars)   +0.158  [-0.102, +0.459]  no
- No feature set reads difficulty-removed step-value with a CI excluding 0 at 23 problems.
- Directional hypothesis (text >= hidden): NOT supported. Paired within-model diffs (text - hidden):
  bow-hidden = -0.104 [-0.442, +0.263] (n.s.); char-hidden = -0.254 [-0.518, -0.002] (WRONG way —
  hidden beats char). Frozen states are numerically ahead of cheap text.
- Reproduces round-1's hidden-probe within-problem signal (~0.22-0.33) under a stricter LORO CV.

### Secondary: the length/step confound is the real driver
Raw single-feature within-problem correlations are all negative: len -0.27, num_tokens -0.29,
step-index k -0.15. After k-control, the prereg'd-metric within-r for text collapses (bow 0.13->0.04).
On real GSM8K chains, what little difficulty-removed step-value exists is entangled with position/length,
and neither cheap legible text nor frozen states pull a clean signal out at this sample size.

### Part-2 verdict
The ERT toy's "legible words beat continuous states" does NOT reproduce on real tokens. At 23 problems,
difficulty-removed step-value is not reliably readable from EITHER cheap text features or frozen hidden
states (all CIs cover 0), and frozen states are numerically ahead of bag-of-words / char-ngrams — the
opposite of the toy. An honest null on the brief's central transfer claim.

## Combined answer to the brief
- Toy (Part 1): after fixing calibration and removing difficulty, legible words carry a small,
  reliably-nonzero step-value signal that matched continuous states do not (p=0.0079) — but small, and
  its magnitude upper-bounded by a residual variance mismatch. Round-1's confounded reversal is gone;
  round-1's dramatic distillation advantage was a degeneracy artifact.
- Real tokens (Part 2): the effect does not transfer. Cheap text features do not beat (and slightly
  underperform) frozen hidden states, and neither reliably reads difficulty-removed step-value at n=23.
- Net: "step-value is more readable from legible words" is a toy-scale, small-effect phenomenon that
  survives the round-2 controls in the toy but fails to appear in the one real-token dataset testable
  on saved artifacts.

## Caveats
- Part 1: 5 seeds/arm (min exact p 0.0079); ERT unstable at the variance-matched operating point (acc
  0.62-0.85), per-seed V*std ratio 1.2-4.0, <=1.5x match flag False; the mismatch direction inflates
  ERT's r, so treat 0.055 as an upper-ish bound. One task, d=64, 24 classes, K=16 MC. The base-hidden~=0
  finding is robust (ample variance yet reads nothing).
- Part 2: 23 problems (effective n~23, ICC~0.88), 3 prefixes each, single unseeded upstream run, one
  1.5B-4bit model, one dataset; 23-cluster percentile-bootstrap CIs are approximate. The prereg'd
  OOF-within metric had a sign-flip artifact (corrected via within-model probe; both reported). Text
  features are cheap by design; a learned text encoder was out of scope.
- The toy's continuous baseline uses additive Gaussian boundary noise as the "continuous" analog; a
  different noise model or operating point could carry more state-dependent value.

## Files (under /Users/steven/research_sprint_20260724/r2-legible-value-readout/)
- PREREG.md — pre-registration + Amendment 1 (dated).
- part1_ert.py — Part 1 code (variance-matched ERT rerun; imports round-1 machinery read-only).
- part2_vineppo.py — Part 2 code (text vs hidden probes on saved vineppo data; within-model fix).
- results_part1.json, results_part2.json — full per-seed / per-feature numbers.
- results.json — merged summary. part1.log — Part 1 run log incl. calibration grid.
- merge_results.py — combines the two into results.json.
