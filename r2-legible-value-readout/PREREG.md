# PREREG — Legible-word value readouts, done right (round 2)

Stamped at write time (shell `date`):
```
Fri Jul 24 23:10:10 EDT 2026
```
Everything below this prereg was fixed BEFORE any experiment was run. Anything reported
in FINDINGS that is not specified here is labeled post-hoc.

## Background / what round 1 left open
Round-1 (`ert-value-probe/`) tested "is step-value more readable from legible words than from
matched continuous states?" The PREREGISTERED PRIMARY (pooled Pearson r of an MLP probe, ERT
words vs baseline hidden) came out significantly REVERSED (words 0.14 vs hidden 0.56, MW p=0.0079)
— but two controls dissolved it: (a) the baseline's pooled advantage is ~entirely
input-difficulty memorization (input-only probe matches full hidden probe), and (b) the ERT arm
was left near-degenerate by a calibration miss (ERT trained to acc≈0.99 → 6/15 boundaries had
V* std < 0.05, i.e. no value variance to read). The one representation that read genuine
difficulty-removed step-value was DICTIONARY-DISTILLED words (within-input r≈0.36 vs ≈0.03 for
continuous states). Two flaws to fix: value distributions were NOT variance-matched, and the
primary was the wrong (pooled, confounded) contrast. Separately, the vineppo real-token side
tested only frozen hidden states → V* (a real null), never CHEAP TEXT FEATURES of the step
strings vs those hidden states under the same pipeline.

## Two-part design.

---
## PART 1 — ERT toy, variance-matched and calibration-fixed (PRIMARY of this study)

### Hypothesis (directional, from the brief)
After removing problem difficulty and at MATCHED value-target variance and matched probe capacity,
step-value V* is MORE readable from the ERT's legible words than from a matched continuous model's
hidden states. Null or reversed will be reported plainly.

### Task / models (unchanged from round-1 substrate, `~/english_bottleneck_lm`)
- Task M: input word of class c0; 4 layers, each applies one hop of a fixed class permutation g;
  correct final = g^4(c0). SEQ_LEN=4, readout position 0. d=64, 4 layers, eps=0.5, rho=1.0.
- Model A (legible): ERT with commit(Gumbel tau=0.3)+paraphrase between layers. Boundary state =
  committed+paraphrased word at pos 0.
- Model B (continuous): NoisyBaseline = ContinuousBaseline + additive Gaussian sigma·N(0,I) at each
  boundary, trained+evaluated with noise. Boundary state = post-noise hidden at pos 0.
- V*(episode, boundary l) = mean over K=16 MC resumptions of 1[final class == g^4(c0)], l in {0,1,2}.
  Split-half (SB-corrected) reliability reported per boundary.

### THE FIX — variance-matched calibration (the core change vs round 1)
Round 1 selected each model by |acc − 0.80| and trained to a fixed step count; ERT overshot to
acc≈0.99 (degenerate V*). Instead, both models are calibrated DIRECTLY ON THE TARGET VARIANCE:
- Grid of ERT training steps CKPTS_E = {20, 30, 40, 60, 90, 140, 220} and baseline
  (sigma, steps) over sigma∈{1,2,4} × steps∈{15,25,40,70}. (These bracket the region where each
  crosses chance→ceiling, per the round-1 amendment-1 finding that both cross before step ~100.)
- ONE pilot run per model type (pilot seeds ERT=99, base=98). For each candidate config compute the
  pooled V* std over the 3 boundaries on a pilot dataset of 1024 episodes (K=16).
- TARGET_VSTD = 0.28 (mid-range of the round-1 baseline's realized V* std 0.21–0.33). Chosen config
  per model = argmin |pooledV*std − TARGET_VSTD|. That config is FIXED for all 5 seeds of that model.
- Post-hoc realized V* std per seed/boundary is reported; a seed-boundary with std < 0.05 is flagged
  degenerate and the count is reported for both arms (variance-match is judged a success iff neither
  arm has more than ~2/15 degenerate boundaries AND arm-mean V*std within a factor ~1.5 of each other).

### AMENDMENT 1 to Part-1 calibration (made after the pilot sweep, BEFORE the 5-seed main run)
The pilot V*std sweep (logged in results_part1.json.calibration) revealed an asymmetry the naive
"argmin |V*std−0.28|" rule mishandles: the noisy BASELINE only reaches V*std≈0.25 by being at
near-CHANCE accuracy (sigma=2,steps=15: V*std 0.248 but acc 0.13; sigma=4,steps=25: V*std 0.20 but
acc 0.08), whereas the ERT reaches V*std≈0.26 at a WORKING operating point (steps=60, acc 0.88). The
unconstrained rule would compare a working ERT to a broken baseline — a new confound. Amended rule
(endpoint/primary test UNCHANGED): restrict the calibration grid to configs with pilot eval-accuracy
in [0.80, 0.98] (both models must be working), then pick argmin |V*std − 0.28| within that band.
Under this band the selections are ERT steps=60 (acc≈0.88, V*std≈0.26) and baseline sigma=1,steps=25
(acc≈0.96, V*std≈0.20). The two arms' V*std then differ by ≈1.3× (< the 1.5× success threshold) at
comparable accuracy. If no baseline config lands in the band with V*std>0.05, calibration is declared
FAILED and the primary is reported as un-runnable at matched variance (honest null on the method).

### Probes (identical recipe both arms; matched capacity)
Features standardized by train stats. PRIMARY probe = ridge (closed form), alpha selected on a
held-out val split from {0.1,1,10,100,1000}. (Round-1 used MLP as primary; ridge is chosen here as
the matched-capacity primary because an MLP's capacity is not obviously matched across a 106-d words
arm vs a 170-d hidden arm, and ridge with tuned alpha adapts capacity to each feature set. MLP is
run as a SECONDARY probe.) Trained to regress V*. Probe-train n=256 (primary), curve over
{64,128,256,512,1024}. Episodes/seed = 1024 pool + 256 val + 2048 test (= 3328).

### Main arms (features always prepend the input token one-hot, 106-d, because V* is goal-relative)
- A-words: input one-hot ++ boundary word (pos 0) one-hot   [legible, raw symbols]
- B-hidden: input one-hot ++ boundary hidden state (pos 0, 64-d)   [continuous]
Secondary arms (n=256, ridge + mlp): input-only (both models); A-classfeat (public-dictionary class
one-hots of input & word, the distilled legible reading); B-hidden-allpos.

### PRE-STATED PRIMARY ENDPOINT AND TEST
The primary is the DIFFICULTY-REMOVED contrast (this is the change from round 1, which used the
pooled contrast as primary). For each of 5 model seeds: compute the within-input partial Pearson r
of the RIDGE probe at n=256 (test-set predictions and V* both centered by input-token group mean),
averaged over the 3 boundaries → r_within. Compare the 5 A-words r_within against the 5 B-hidden
r_within with a two-sided exact Mann–Whitney U (5v5, min possible p=0.0079). Report per-seed values,
means, and 95% t-CIs regardless of outcome. Secondary confirmatory: same test on the pooled r_bar.

### Pre-stated secondary analyses
1. Learning curves r_within(n) and r_pooled(n), both probes, both main arms.
2. A-classfeat vs A-words and vs B-hidden, within-input r, per-seed paired deltas (descriptive;
   n=5 Wilcoxon min p=0.0625). Directional pre-stated claim: classfeat ≥ words ≥ hidden within-input.
3. Input-only control per model (difficulty-only baseline).
4. On-track oracle r per boundary (descriptive).

### Decision rule (Part 1)
"Legible words are more value-readable" is supported ONLY IF the variance-match succeeds (above)
AND A-words r_within > B-hidden r_within with MW p<0.05. If variance-match succeeds but the raw-word
contrast is null while classfeat wins, the honest conclusion is "legibility helps only through cheap
semantic featurization, not raw symbols" (the round-1 descriptive finding, now on matched variance).

---
## PART 2 — Real-token analog on SAVED vineppo artifacts (no new generation)

### Hypothesis
On the saved vineppo prefixes, after removing problem difficulty and under an IDENTICAL
problem-level-CV pipeline, step-value V* (v_gold) is at least as readable from CHEAP TEXT FEATURES of
the saved step strings (bag-of-words / char n-grams) as from the frozen 1536-d hidden states. The
round-1 vineppo follow-up showed hidden→V* is a real difficulty-removed null; the open question is
whether the cheap legible-text reading does better, matching the ERT toy's "distilled words win".

### Data (READ-ONLY, no new generation)
`/Users/steven/vineppo_replication/results/data.jsonl`: 23 GSM8K problems × 3 prefixes = 69 rows.
Each row: `prefix` (the step-prefix TEXT), `hidden` (1536-d last-token frozen Qwen state), `v_gold`
(MC step-value, 12 rollouts, in {0,1/12,...,1}), `k` (prefix index 1..3), `base_sol` (for step count).
Effective independent n ≈ 23 (ICC≈0.88); analysis is clustered by problem throughout.

### Feature sets (all built ONLY from saved data)
- TEXT-bow: bag-of-words counts of the prefix text (lowercased, simple \w+ tokenizer),
  min_df=2, no idf; a legible surface reading an overseer could compute.
- TEXT-char: char n-gram (3–5) counts, min_df=2 (captures numbers/operators/length cheaply).
- HIDDEN: the stored 1536-d hidden state (the continuous reference, exactly round-1's probe input).
- LEN control: two scalar features [num chars, num \w tokens] of the prefix (pure length/difficulty
  proxy; positive control that difficulty is trivially readable, and a thing text features could be
  "cheating" with — so it must be difficulty-removed).

### Pipeline (IDENTICAL across all feature sets — this is the whole point)
- Leave-one-PROBLEM-out CV (23 folds): train on the other problems' prefixes, predict the held-out
  problem's 3 prefixes. Train/test NEVER share a problem. Out-of-fold predictions collected for all 69.
- Probe = Ridge with alpha selected by NESTED leave-one-problem-out CV on the training fold over
  alpha∈{1,10,100,1000,10000} (same grid for every feature set; larger alphas because bow/char and
  1536-d hidden are high-dim at n≈66 train). Predictions clipped to [0,1].
- Metrics on the 69 out-of-fold predictions:
  (a) POOLED Pearson r(pred, v_gold), and pooled ΔMSE vs the mean-only predictor of v_gold
      (positive ΔMSE = worse than predicting the grand mean, as in round-1 vineppo).
  (b) PRIMARY: WITHIN-PROBLEM (difficulty-removed) Pearson r: center pred and v_gold by
      problem mean, correlate over 69. This is the round-1 vineppo primary metric.
- Uncertainty: cluster bootstrap over the 23 problems, B=5000, numpy seed 0, percentile 95% CI, for
  both pooled and within-problem r of EVERY feature set, AND for the paired differences
  (TEXT-bow − HIDDEN), (TEXT-char − HIDDEN) in within-problem r on the same resamples. A difference /
  a signal is "detected" iff its 95% CI excludes 0.

### PRE-STATED PRIMARY ENDPOINT AND TEST (Part 2)
Within-problem Pearson r of TEXT-bow vs of HIDDEN, and the paired difference (TEXT-bow − HIDDEN) with
its cluster-bootstrap 95% CI. Directional claim from the brief/ERT analogy: TEXT-bow within-problem r
> HIDDEN within-problem r (CI of the difference excludes 0 and is positive). Null (CI covers 0) is a
fine, expected outcome given round-1's 23-problem power and will be stated plainly.

### Pre-stated secondary (Part 2)
1. Same for TEXT-char.
2. LEN positive control: pooled r should be clearly > 0 (difficulty IS readable from length);
   within-problem r should collapse toward 0 (length carries little within-problem step signal) —
   a sanity check that the difficulty-removal is doing its job.
3. Step-index (k) control: within-problem partial r of each feature set controlling for k.
4. Report the round-1 hidden-probe number reproduced under THIS pipeline as a cross-check.

### Decision rule (Part 2)
"Cheap legible text beats/matches frozen states for step-value" is supported iff TEXT-bow (or
TEXT-char) within-problem r has a CI excluding 0 AND the paired difference vs HIDDEN has a CI
excluding 0 in the positive direction. If neither text nor hidden clears 0 within-problem (the
likely 23-problem outcome), the honest headline is "at 23 problems, difficulty-removed step-value is
not reliably readable from EITHER cheap text or frozen states — the ERT toy's legibility advantage
does not reproduce on real tokens at this scale", with CIs given.

## Compute budget
Single process, CPU, torch threads ≤ 4. Part 1 target < 10 min (round-1 was 210s; variance-match adds
a pilot grid but seeds unchanged). Part 2 is pure sklearn on 69×1536, < 30 s. Hard stop: if Part 1
pilot+seed-0 exceeds 6 min wall, drop the MLP secondary probe and the learning-curve tails
(keep ridge primary + n=256) and note it. Total ≤ ~12 min.

## Seeds
Part 1: 5 model seeds {0..4} (fresh init, data, episodes); probe seeds derived deterministically.
Part 2: rollouts already fixed in saved data; bootstrap seed 0; CV permutation is deterministic
(problems in sorted id order).
