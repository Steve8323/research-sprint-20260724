# MoE legibility inside the ERT (moe-ert-bridge)

**Question.** If we put a small mixture-of-experts layer inside the English-Residual
Transformer (ERT, ~/english_bottleneck_lm) — where the inter-layer stream is literal
English words with *known* meaning classes — is expert specialization legible in the
emitted English? Concretely: (a) do experts align with meaning classes, and (b) do
short plain-language expert profiles predict routing on held-out data, including for
surface forms (synonyms) never used to build the profile?

This bridges the ERT toy (ground-truth meaning classes, legible stream) and the
moe_legibility question (are plain-language expert profiles predictive?), at toy
scale where, unlike OLMoE, the ground truth for "what a token means" is exact.

## Method (fixed before the full run)

- Model: ERT (d=64, 4 layers, seq_len=4, eps=0.5, rho=1.0 paraphrase channel),
  identical to the published toy, except each block's dense MLP is replaced by a
  Switch-style top-1 MoE: 8 experts (Linear 64->128->GELU->128->64), softmax gate,
  output scaled by the chosen gate prob, load-balance aux loss (weight 0.01).
- Training: Task M (meaning chain) with deep supervision, 1200 steps, B=256,
  Adam lr=3e-3, tau annealed 1.0->0.3, exactly as in the original train.py.
- Seeds: 4 (seeds 0-3; model init, data stream, and held-out-synonym choice all
  re-seeded per run).
- Routing data: at each layer we record the *incoming word* at the readout position
  (RPOS=0) — i.e. the literal English word crossing the layer boundary, which is the
  legible stream — and the router's top-1 expert for that token. Two independent
  held-out splits: PROFILE split (3000 sequences) and EVAL split (3000 sequences).
  Analysis uses content words only (specials excluded; the excluded fraction is
  reported).
- Held-out synonyms: per seed, one of the 4 synonyms in each of the 24 classes is
  held out. Profiles are built ONLY from tokens whose word is not held out
  (~75% of profile-split instances). The profile-prediction test is evaluated ONLY
  on eval-split instances whose word IS a held-out synonym (~25%), so word-identity
  memorization cannot help — only the class-level (plain-language) abstraction can.
- Plain-language profiles: per (layer, expert), the set of meaning classes assigned
  to it (class c -> argmax-count expert at that layer, from the profile split),
  rendered as English ("expert 3 handles words meaning big, cold, rich"). The
  profile predictor routes a word to the expert whose profile contains its class.

## Pre-registered primary tests (written BEFORE the full run)

- **P1 (expert-class alignment).** Adjusted mutual information (sklearn AMI)
  between top-1 expert and the incoming word's true meaning class at RPOS, per
  (seed, layer), on the EVAL split (n ~ 3000 instances per layer per seed).
  Significance: label-permutation test, 1000 shuffles,
  p = (1+#{AMI_perm >= AMI_obs})/1001.
  Claim "experts align with meaning classes" only if ALL 16 (seed,layer) cells have
  p < 0.01 AND the per-seed mean AMI (averaged over layers) has mean >= 0.30 across
  seeds. Report mean +/- 95% t-CI over the 4 seeds.
- **P2 (profiles predict routing).** Accuracy of the profile predictor on
  held-out-synonym instances of the EVAL split (pooled over layers, per seed),
  compared to the majority-expert-per-layer baseline on the SAME instances
  (paired). Test: exact McNemar (two-sided binomial on discordant pairs), per seed.
  Claim "plain-language profiles predict routing" only if accuracy > majority
  baseline in ALL 4 seeds with McNemar p < 0.01. Report pooled accuracy with
  Wilson 95% CI, plus chance = 1/8.

Anything not listed above (per-layer breakdowns, spec scores in the moe_legibility
convention, word-level lookup skylines, load-balance stats, example profiles) is
secondary/descriptive and labeled as such.

Negative results are acceptable and will be reported plainly (e.g., router collapse
-> high majority accuracy, AMI ~ 0, McNemar null).

## Results

(to be filled in after the run — nothing below this line was written before results existed)

**Pre-run deviation (recorded before the full run):** training steps reduced
1200 -> 800 for the shared-machine runtime budget (~10 min total). Decided after
the smoke run (100 steps, seed 0 only) but before any full-run results existed;
smoke task-M acc was already 0.889 at 100 steps, so 800 is ample. No other
change to the pre-registered design or tests.

Run: 4 seeds x 800 steps, wall time 504 s, CPU (2 threads). All numbers from
`results.json` (n per cell stated inline).

**Task capability retained.** MoE-ERT task-M accuracy per seed: 1.000, 0.999,
0.925, 0.994 (dense ERT published value: 0.996; chance 0.042). The MoE swap does
not break the ERT.

**P1 — experts align with meaning classes: SUPPORTED.**
Mean AMI(expert, meaning class) across seeds = **0.495**, 95% t-CI over 4 seeds
[0.448, 0.543]; per-(seed,layer) AMI range 0.393-0.551 (n = 3000 instances per
layer per seed; content fraction 0.955-1.0). All 16 (seed,layer) permutation
tests give p = 0.001 (the minimum possible with 1000 shuffles). Pre-registered
criteria (all p < 0.01, mean AMI >= 0.30) are met.
*Post-hoc context:* with 8 experts and 24 classes a perfectly class-deterministic
router caps AMI at ~0.79 (coarsening bound), so 0.50 is substantial but partial.

**P2 — plain-language profiles predict routing on held-out synonyms: SUPPORTED.**
Profile-predictor accuracy on synonyms never used for profiling, pooled over
layers: per seed 0.607 [0.590,0.624], 0.580 [0.562,0.598], 0.550 [0.532,0.569],
0.555 [0.536,0.574] (Wilson 95% CIs; n = 3031, 2969, 2792, 2640) — cross-seed
mean **0.573**, t-CI [0.531, 0.615]. Majority-expert baseline on the same
instances: 0.078-0.212 (mean 0.153); chance = 0.125. Exact McNemar vs majority:
p <= 4.5e-172 in every seed (discordant pairs e.g. 1701 vs 98). Pre-registered
criteria (accuracy > majority in all seeds, McNemar p < 0.01) are met.

**Secondary / descriptive (not pre-registered).**
- Routing is ~83% word-deterministic: a word-level lookup table from the profile
  split predicts routing of NON-held-out eval words at 0.828-0.838. The
  class-level English profile recovers 0.55-0.61 on words it has never seen —
  i.e. most, but not all, of the predictable routing generalizes through the
  meaning class; the remainder is surface/context-driven and NOT captured by the
  plain-language profile.
- Layer gradient: profile predictability rises with depth (layer 0: 0.32-0.46;
  layer 3: 0.66-0.71 across seeds) — echoing the moe_legibility finding that
  specialization concentrates in later layers, now visible at toy scale.
- All 8 experts are used at (almost) every layer; mean spec score in the
  moe_legibility convention 0.44-0.46 (OLMoE reference: 0.19), max up to 1.0.
- Example profiles are short readable English, e.g. seed 0 layer 2:
  E3 "expert handles words meaning: cold, old, rich, calm, clean";
  E5 "small, fast, hot, broke, angry"; E7 "happy".

**Conclusion.** Inside the ERT, where the inter-layer stream is literal English
and ground truth is exact, a small top-1 MoE develops partial but statistically
unambiguous meaning-class specialization (AMI ~ 0.5 of a ~0.79 ceiling), and a
class-level plain-English profile of each expert predicts held-out routing far
above baseline — including for surface forms the profiler never saw, which rules
out word-memorization as the mechanism. Legibility of routing is real here, but
incomplete: about a third of word-predictable routing is not explained by the
meaning-class profile.

## Caveats

- Toy scale (d=64, 4 layers, 8 experts, 24 synthetic classes); meaning classes
  are constructed, not learned; results are mechanism demonstrations.
- Profiles are rendered into English mechanically from class assignments (no LLM
  labeller, unlike moe_legibility's 3B describer) — legible by construction
  because the ERT stream is words; the substantive finding is that routing is a
  function of the class abstraction that generalizes to unseen synonyms.
- Pre-run deviation: 1200 -> 800 training steps (runtime budget; see above).
- Cross-seed CIs use only 4 seeds (t-intervals are wide); within-seed inference
  is exact/permutation-based with n ~ 2.6k-3k per test.
- The AMI ceiling argument, the layer gradient, spec scores, and the word-lookup
  skyline are post-hoc/descriptive.
- Profile and eval splits share the model and task distribution; generalization
  is over surface forms and fresh samples, not over tasks.
