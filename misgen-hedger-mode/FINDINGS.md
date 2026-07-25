# What are the hedgers?

**Question.** `stats/paper_stats.py` (120 seeds, width-8 MLP, equal-cost XOR cues) finds a
3-mode GMM with 55/120 "hedgers" (deploy alignment in [0.2, 0.8]). Are hedgers (a) a
patchwork mixture of the two worldviews (deterministically following the true cue on some
cue-disagreement input patterns and the proxy on others), (b) genuinely uncommitted
(near-zero logit margins on disagreement inputs, per-example noise decides), or (c) a
distinct third strategy? And does hedger-ness predict flip susceptibility under
disagreeing-data fine-tuning?

**Setup.** Exact replication of the `stats/paper_stats.py` config: 2-hidden-layer width-8
MLP (154 params), 7-dim inputs (true XOR cue dims 0-1, proxy XOR cue dims 2-3, 3 noise
dims), margin 2.0, noise 0.6, N=800, Adam lr 5e-3, wd 2e-3, 2000 full-batch steps, seeds
700..819 (n=120), deploy eval on 4000 fresh examples (proxy anti-correlated). Reproduction
is checked against the archived per-seed scores in `stats/results.json`.

On deployment the 4 cue dims take 8 sign patterns where the cues disagree, indexed by
(v, c_t, c_p): true block = (c_t, c_t XOR v), proxy block = (c_p, c_p XOR (1-v)), label v.
For each model and pattern we estimate p_k = P(prediction follows the true cue) over 500
fresh noise draws (noise tensor shared across all models). Determinism index
D = mean_k |2 p_k - 1| (1 = per-pattern deterministic, 0 = coin-flip per pattern).

## Pre-stated analysis plan (written BEFORE running any experiment)

- **Primary test 1 (characterization).** Compute D for every model. Decision rule, fixed
  in advance: hedgers are "input-space patchworks of the two worldviews" if median
  D(hedgers) >= 0.8; "uncommitted / noise-driven" if median D(hedgers) <= 0.5; otherwise
  "intermediate". Also report Mann-Whitney U of D, hedgers vs committed (aligned +
  misaligned pooled), noting the committed group's D is partly mechanically high.
- **Primary test 2 (flip susceptibility).** Fine-tune every non-aligned model (misaligned
  + hedger) from its trained weights for 1000 Adam steps (same lr/wd) on a shared N=800
  train-style dataset containing a fraction f = 0.005 of anti-proxy rows (proxy disagrees,
  label = truth) - the repo's "removal" protocol (geom_persist). Flip = final deploy
  alignment > 0.8. Test: Fisher exact on flip rate, hedgers vs misaligned. Known
  confound, stated in advance: hedgers start closer to the 0.8 line, so a positive result
  here is partly mechanical; that is why test 3 exists.
- **Pre-stated secondary test 3 (controlled).** Among the same non-aligned models,
  Spearman partial correlation between D and final alignment at f = 0.005, controlling
  for initial deploy alignment (rank-regress-residual method, t approximation, df = n-3).
  Directional hypothesis: lower D (less committed) -> higher final alignment.
- **Secondary dose** f = 0.02, same protocol (robustness of tests 2/3). Aligned models
  run at f = 0.005 as a stay-aligned control only.
- Everything else (hedger subtype census: constant-output vs patchwork vs noise-driven,
  with rule: D >= 0.8 & all 8 patterns predict one class = constant-fallback; D >= 0.8
  otherwise = patchwork; D <= 0.5 = noise-driven; else intermediate. Effect decomposition
  of the 8-pattern profile onto v / c_t / c_p main effects + interactions. Linear-probe
  decodability of each cue from penultimate activations on independent-cue data.
  Input-gradient reliance. First-layer weight mass) is EXPLORATORY / post-hoc and will be
  labeled as such.
- CIs: 2000-resample bootstrap for means, Wilson for proportions. All n reported.
- Nothing above this line will be edited after the run; results appended below.

## Results (appended after the run; plan above unedited)

Total runtime 140 s, single process, CPU. Raw numbers: `results.json`; console: `run.log`.

### Reproduction

Retrained all 120 seeds: max |deploy alignment - archived stats/results.json score| =
0.0000; groups 31 aligned / 34 misaligned / 55 hedgers, identical to the paper. Sanity:
alignment reconstructed from the 8-pattern profile correlates r = 0.9999 with measured
deploy alignment, so the pattern analysis captures the behavior.

### Primary test 1: hedgers are UNCOMMITTED, not patchworks

- Median D(hedgers) = 0.315 (IQR 0.251-0.457, n=55) => pre-stated rule fires the
  **"uncommitted / noise-driven"** verdict (<= 0.5). 44/55 hedgers have D < 0.5.
  Committed models: median D = 0.868 (n=65); Mann-Whitney p = 8.0e-21.
- Mean |logit margin| on disagreement patterns: hedgers 1.67 [boot CI 1.61, 1.75] vs
  committed 3.57 [3.24, 3.96] - about half, not near zero: hedgers lean, noise decides.
- So a hedger is NOT a deterministic per-input-pattern mixture of the two worldviews
  (that hypothesis predicted D >= 0.8 and is cleanly rejected). On any given
  cue-disagreement input its two cue contributions largely cancel, leaving a small
  seed-specific lean plus input noise - a superposed weighted vote.

### Primary test 2: hedgers flip far more easily at the marginal dose (confounded, as pre-stated)

- f = 0.005 (4/800 anti-proxy rows), 1000 steps: flip (>0.8) 54/55 hedgers (98%, Wilson
  [0.90, 1.00]) vs 18/34 misaligned (53%, [0.37, 0.69]); Fisher exact p = 1.6e-07.
  Mean final alignment 0.905 (from 0.481) vs 0.758 (from 0.083). Median steps to cross
  0.8 (censored at 1050): 50 vs 275.
- f = 0.02 (17/800 rows): ceiling - 55/55 vs 34/34, Fisher p = 1.0 (final means 0.980
  vs 0.931). Aligned controls at f = 0.005: 31/31 stay aligned.
- As pre-stated, this comparison is partly mechanical (hedgers start nearer 0.8).

### Secondary test 3 (controlled): MIXED - null at the primary dose

- f = 0.005: partial Spearman(D, final | initial alignment) = -0.043, p = 0.69 (n=89).
  **Null.** The pre-stated controlled test does NOT support commitment predicting flip
  resistance beyond starting alignment at the marginal dose.
- f = 0.02 (pre-stated robustness dose): rho_partial = -0.317, p = 2.6e-3, in the
  pre-stated direction. Simple correlations for reference: rho(init, final) = +0.41 /
  +0.36; rho(D, final) = -0.34 / -0.47 at the two doses.
- Honest summary: after controlling for where a model starts, extra "commitment" (D)
  predicts how far the disagreeing-data FT moves it at the larger dose but not at the
  marginal one. Post-hoc reading (labeled as such): at f = 0.005 hedger finals sit at
  ceiling and misaligned finals are near-bimodal (flip vs stuck), leaving little graded
  variance for D to explain.

### Exploratory (all post-hoc)

- Subtype census (pre-stated rules): 44/55 noise-driven, 11/55 intermediate,
  **0 patchwork, 0 constant-fallback**. The two "mixture-of-worldviews" subtypes are
  literally empty.
- Profile variance decomposition: committed models put 99% of pattern-profile variance
  on the v (worldview) axis; hedgers 64% v + 16% raw cue bits + 20% interactions - but
  on small-amplitude profiles, so this is a weak lean, not a third deterministic rule.
- Representation: linear probes (penultimate layer, independent-cue data, 3000/2000
  train/test) - aligned models decode true cue 0.978 but proxy only 0.722; misaligned
  decode proxy 0.969 but true 0.730; hedgers decode BOTH at 0.859/0.862. Weaker-cue
  decodability: hedgers 0.822 vs committed 0.726, Mann-Whitney p = 2.8e-5. Committing
  partially destroys the losing cue's representation; hedgers keep both - consistent
  with the arch/ over-capacity story, but visible already at width 8.
- Input-gradient true-cue reliance: aligned 0.66 +/- 0.10, hedgers 0.49 +/- 0.04,
  misaligned 0.37 +/- 0.07; r(reliance, alignment) = 0.85 (p = 2.6e-35, n=120).
  First-layer weight mass on the true block: 0.73 / 0.49 / 0.29.
- Steps-to-flip vs D among uncensored f=0.005 models: Spearman +0.53 (p = 1.8e-6,
  n=72) - post-hoc and start-point-confounded.

## Caveats

- The pre-stated controlled test (test 3) FAILED at its primary dose and passed only at
  the secondary dose; the "commitment predicts flip resistance beyond starting point"
  claim is therefore weak/mixed. The unambiguous flip result (test 2) carries the
  start-point confound stated in advance.
- One shared FT dataset per dose (paired across models; exactly 4 and 17 anti-proxy
  rows); dataset-draw variance unsampled. Flip threshold 0.8 and the 1000-step budget
  inherited from the repo, not tuned.
- D from 500 noise draws/pattern (SE(p_k) ~ 0.022); committed-group D is partly
  mechanically high (extreme mean alignment forces one-sided p_k), which is why the
  verdict rests on the absolute hedger median, per the pre-stated rule.
- geom_persist's removal curve used a 1-hidden-layer mlp8; this population is the
  2-hidden-layer stats/ model, so flip rates are not directly comparable (their 37%
  removed at f=0.005 vs 53% of misaligned here).
- Single architecture (width-8, 2-hidden-layer MLP), the repo's searched-for equal-cost
  construction, seeds 700-819 only; nothing here says hedging in real models looks like
  this.

## Files

- `run.py` - full experiment (single process, 140 s)
- `results.json` - config, per-model rows, all tests
- `run.log` - console output
