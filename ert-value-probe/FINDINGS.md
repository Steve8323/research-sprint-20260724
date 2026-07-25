# Is step-value more readable from legible intermediates? (ERT value probe)

**Question.** In the English-Residual Transformer toy (`~/english_bottleneck_lm`), the state
crossing every layer boundary is a literal word. Is *step-value* (P(final answer correct |
current boundary state)) more readable by a small probe from those legible word intermediates
than from a matched continuous model's hidden-state intermediates? This is the
`~/vineppo_replication` credit-assignment question at toy scale.

## Pre-registration (written BEFORE the experiment was run; nothing below this line existed yet)

### Setup
- Task M from the ERT repo: input word of class c0; each of 4 layers must apply one hop of a
  fixed class permutation g; correct final answer is g^4(c0). SEQ_LEN=4, readout at position 0.
- Model A ("legible"): ERT, d=64, 4 layers, rho=1.0 paraphrase, repo deep supervision
  (train recipe copied from `train.py`, tau schedule anchored to 700 steps, B=256, lr=3e-3).
  Boundary state = the committed + paraphrased words (what the next layer actually reads).
- Model B ("continuous"): the repo's ContinuousBaseline with an added stochastic continuous
  boundary channel: x <- x + sigma * N(0, I) between layers (trained and evaluated with the
  noise). Rationale (decided a priori): without boundary stochasticity the baseline is
  deterministic, so "value" degenerates to memorizing which of 96 inputs the model fails on;
  the noise is the continuous analog of the ERT's commit (Gumbel tau=0.3) + paraphrase
  stochasticity. Boundary state = the post-noise hidden state (what the next layer reads).
- Undertraining calibration (so intermediate states actually vary in value), fixed rule:
  one pilot run per model type (pilot seed), eval accuracy (tau=0.3, as in `eval_M`) measured
  at checkpoints {100, 200, 400, 700} steps; for the baseline additionally sigma in
  {0.5, 1.0, 2.0}. Chosen config = argmin |acc - 0.80| over checkpoints (and sigma for B).
  That config is then fixed for all 5 seeds of that model type. All seeds are included in the
  analysis regardless of their realized accuracy (flagged if outside [0.55, 0.92]).
- Value target: V*(episode, boundary l) = mean over K=16 independent Monte-Carlo resumptions
  from the recorded boundary state (fresh commit/paraphrase noise for A, fresh Gaussian noise
  for B) of 1[final class == g^4(c0)]. Boundaries l in {0,1,2} (after layers 1..3).
  Target reliability reported via split-half r (8 vs 8 rollouts), Spearman-Brown corrected.
  Degeneracy flag (pre-stated): an arm/boundary is labeled degenerate if std(V*_test) < 0.05.
- Episodes per model per seed: 3328 fresh Task-M episodes; fixed split: probe-train pool 1024,
  probe-val 256, probe-test 2048.
- Probes (identical recipe in every arm): features standardized by train stats;
  primary probe = MLP (D -> 64 ReLU -> 1), Adam lr 1e-2, wd 1e-4, 300 epochs full batch,
  best-val-MSE checkpoint (val checked every 10 epochs); secondary probe = ridge closed form,
  alpha in {0.1, 1, 10, 100, 1000} chosen on the val split. Trained to regress V*.
  Probe-train sizes n in {32, 64, 128, 256, 512, 1024} drawn as the first n of the pool.
- Main arms (features always include the episode input token one-hot, 106-d, because value is
  defined relative to the episode's goal, exactly as VinePPO prefixes include the problem):
  - A-words: input one-hot ++ boundary word (position 0) one-hot  [legible]
  - B-hidden: input one-hot ++ boundary hidden state (position 0, 64-d)  [continuous]
- Controls / secondary arms (at n=256 only):
  - input-only probe for each model (how much of the "readable value" is just problem
    difficulty -- the vineppo-audit confound);
  - A-hidden: ERT's own pre-commit continuous activation lnf(x)[pos 0] (within-model
    comparison; note the asymmetry: V* conditions on the committed word, which is downstream
    of the commit noise, so this arm has extra irreducible noise -- stated, not hidden);
  - A-embfeat: public-dictionary featurization E[input] ++ E[word] (an overseer reading words
    may use the public embedding);
  - A-classfeat: class one-hots of input and word (25-d each incl. specials) -- the maximally
    distilled legible reading;
  - all-positions variants of both main arms (words at all 4 positions / hidden at all 4
    positions);
  - descriptive oracle: r(V*, on-track indicator 1[class(word_l) == g^(l+1)(c0)]) for A, and
    the analogous model-head-decoded on-track proxy for B.
- Seeds: 5 model seeds per model type (fresh init, fresh training data, fresh episodes).
  Deterministic probe seeds derived from (seed, arm, boundary, n).

### Pre-stated primary endpoint and test
For each model seed: r_bar = Pearson r(probe prediction, V*) on the 2048 held-out episodes,
MLP probe, n=256, averaged over the 3 boundaries. Compare the 5 A-words r_bar values against
the 5 B-hidden r_bar values with a two-sided exact Mann-Whitney U test. The brief's hypothesis
is A-words > B-hidden ("value is MORE readable from legible intermediates"); a null or reversed
result will be reported as such.

### Pre-stated secondary analyses
1. Learning curves r(n) for both main arms (MLP and ridge), n in {32..1024}.
2. Delta over the input-only control at n=256 (per-seed paired differences; descriptive --
   with n=5 a Wilcoxon two-sided cannot go below p=0.0625).
3. Within-input partial r at n=256 (test-set predictions and V* centered by input-token group
   mean) -- the "difficulty-removed" number, analog of vineppo's within-problem r.
4. The A-hidden / A-embfeat / A-classfeat / all-positions arms, descriptive.
Anything not listed here that appears in Results will be explicitly labeled post-hoc.

### Compute
Single process, CPU, torch threads capped at 4, target < 10 min wall. If the time guard
(460 s at the start of secondary arms) trips, secondary arms are dropped and noted.


### Amendment 1 (made after a smoke pilot, BEFORE the main run; endpoint/test unchanged)
The smoke pilot showed both models cross from chance to ~1.0 eval accuracy before step 100
(baseline reached 1.0 at 30-60 steps even at sigma=2), so the pre-registered calibration grids
could not land near the 0.80 target. Amended grids: checkpoints {30, 50, 75, 100, 150, 250,
400, 700} (both models), sigma {1, 2, 4, 6}. Selection rule unchanged (argmin |acc - 0.80|).

## Results (5 model seeds per arm; 3328 episodes/seed/model, test n=2048; V* from K=16 MC resumptions; wall time 210 s CPU)

### Calibration outcome (and its partial failure — read this first)
Chosen configs: ERT steps=700 (pilot acc 0.923); baseline sigma=2.0, steps=30 (pilot acc 0.883).
Realized eval accuracy across the 5 fresh seeds: ERT [0.995, 0.993, 0.946, 0.997, 0.970] — far above
the pilot's 0.923 (training is unstable near the end of the tau anneal; the pilot seed happened to land
low). Baseline: [0.867, 0.933, 0.895, 0.893, 0.898]. Consequence: 6/15 ERT seed-boundary V* targets are
degenerate by the pre-stated flag (std < 0.05); 0/15 for the baseline. V* std: ERT 0.03-0.22 vs baseline
0.21-0.33. **The two arms' value targets are therefore not variance-matched, which attenuates the ERT
probe's r; the primary comparison is confounded and should be read together with the controls below.**
V* split-half (SB-corrected) reliability: ERT 0.81-1.0, baseline 0.95-1.0.

### Primary endpoint (pre-registered): REVERSED, but confounded
r_bar = held-out Pearson r(MLP probe, V*), n=256, mean over 3 boundaries, per seed:
- ERT-words  (legible):    [0.049, 0.037, 0.450, -0.000, 0.161]   mean 0.139, 95% t-CI [-0.089, 0.367]
- Base-hidden (continuous): [0.622, 0.608, 0.519, 0.559, 0.484]   mean 0.558, 95% t-CI [0.486, 0.631]
- Exact two-sided Mann-Whitney U (5 vs 5): U=0, p=0.0079 (complete separation, direction OPPOSITE to
  the brief's hypothesis).
Taken alone this says value was LESS readable from the legible words. It should not be taken alone:

### Pre-registered controls: the baseline's advantage is difficulty memorization, not credit assignment
- Input-only probe (no intermediate state at all): baseline [0.588, 0.575, 0.519, 0.571, 0.444] —
  within noise of the full base-hidden probe (paired deltas +0.034, +0.033, -0.000, -0.012, +0.040;
  mean +0.019). The continuous probe's r is almost entirely "which of the 96 inputs does this model
  fail on", i.e. the exact pooled-r confound the vineppo audit flagged.
- Within-input partial r (difficulty removed, the analog of vineppo's within-problem r):
  base-hidden [0.002, 0.049, 0.030, 0.041, 0.017] (mean 0.028); base-hidden-allpos ~0 as well
  [-0.019 .. 0.018]. **The continuous hidden states yield essentially zero probe-readable step-level
  value beyond input identity at n=256.**
- ERT raw-word one-hots are also ~zero after difficulty removal: [0.047, 0.030, 0.056, 0.006, 0.165]
  (mean 0.061). ERT input-only: [0.003, 0.004, 0.484, -0.017, 0.041] (within-input r undefined for
  input-only by construction — sanity check passed).

### The one representation that did read genuine step-value: dictionary-distilled words
ERT class-featurized words (class one-hots of input and boundary word via the public dictionary,
50-d — a legitimate legible reading since the dictionary is public/frozen by construction in this toy):
- pooled r_bar [0.389, 0.239, 0.742, 0.265, 0.655] (mean 0.458), beats raw word one-hots in 5/5 seeds
  (mean delta +0.319);
- within-input partial r [0.419, 0.215, 0.339, 0.216, 0.631] (mean 0.364) — the highest
  difficulty-removed value readout of ANY arm in EVERY seed (vs 0.028 for continuous hidden states).
  (Arm and metric were pre-registered; the emphasis on this specific cross-arm ranking is post-hoc.)
Other secondary arms: ERT pre-commit continuous activations: within-input r ~0.16 on the finite
boundaries (boundary 0 degenerate/nan in all seeds); public-embedding features E[input]++E[word] do NOT
rescue the raw words (within-input mean 0.065 — the eps=0.5 within-class embedding noise apparently
costs more than the one-hot sparsity at this n). Learning curves: the pooled gap persists to n=1024
(ridge: words 0.414 vs hidden 0.711), consistent with pooled r being difficulty-driven.

### Answer to the question
On this toy, at this probe budget: **No clean "yes".** (1) The pre-registered primary comparison came
out significantly REVERSED (p=0.0079), but that pooled result is confounded — the baseline's advantage
is ~entirely input-difficulty memorization, and the ERT's value targets were left near-degenerate by a
calibration miss. (2) After the pre-registered difficulty correction, NEITHER raw representation
supports step-level credit readout (within-input r ~0.03-0.06 both arms). (3) The only genuine
difficulty-removed step-value readout anywhere in the experiment came from the legible intermediates
after distillation through the public dictionary (within-input r ~0.36 vs ~0.03 for matched continuous
states, 5/5 seeds). So legibility helped, but only because the legible symbols admit a cheap semantic
featurization — not because raw symbols are probe-friendlier than activations.

### Caveats
- The primary comparison is on unmatched value distributions (calibration miss; 6/15 ERT boundaries
  degenerate). A rerun with per-seed accuracy matching would be needed for a fair primary.
- The baseline's calibrated operating point (30 steps, sigma=2) produced failures that are mostly
  input-determined rather than noise-driven; a noise-limited operating point might carry more
  state-dependent value in its hidden states. The "continuous reads ~zero step-value" finding is
  specific to this operating point.
- 5 seeds/arm (min possible exact p = 0.0079); one task, one probe recipe, toy scale (d=64, 4 layers,
  24 classes); V* estimated with K=16 rollouts (SB reliability 0.81-1.0).
- The within-ERT words-vs-pre-commit-hidden comparison has the pre-stated conditioning asymmetry
  (V* conditions on the committed word).
- Class-featurization uses the toy's public word->class dictionary; in real models the analogous
  distillation (text -> semantic state) is exactly what is contested, so this is an existence result
  about the toy, not evidence the distillation is cheap in general.

Files: experiment.py (all code), results.json (all numbers incl. per-seed/per-boundary), run.log.
