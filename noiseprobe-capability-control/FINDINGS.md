# Capability-matched control for the noise probe's L4/L8/L11 "misalignment-specific" differential

## Question

`layer_importance.py` (Qwen2.5-0.5B-Instruct base vs the ModelOrganismsForEM bad-medical-advice
LoRA) reported that under RMS-scaled activation noise the misaligned-base safety-margin
differential dS_diff(L) is most negative at layers 4, 8, 11 (-2.8/-2.6/-2.3 nats at sigma=0.2),
and the README sells these as candidate "misalignment-exposing" layers. The repo's own
Limitations section admits there is **no capability-matched control**: nothing separates "noise
at L makes the LoRA model incoherent faster" from "noise at L reveals misalignment". This
experiment adds that control.

## Pre-registered design (written BEFORE running the measurement)

**Data reuse.** dS_diff(L) at sigma=0.2 is taken from the saved `layer_importance_results.json`
(read-only). Saved generations cannot provide the covariate (they exist only for the 3 post-hoc
hot layers, misaligned model only), so the covariate comes from a minimal rerun of the same
noise harness on the same cached 0.5B model pair - teacher-forced scoring only, no generation,
no downloads.

**New covariate (neutral-competence degradation), per (model, layer, sigma, seed, item) cell.**
Six fixed neutral factual QA items, chat-templated exactly like the safety task, each with a
correct and a length-matched incorrect continuation:

- Primary covariate **dNLL**: change (noised - clean) in mean per-token NLL of the *correct*
  neutral continuation. This is the "perplexity-style" degradation measure. Differential
  covariate: dD_diff(L) = dNLL_mis(L) - dNLL_base(L), averaged over 3 seeds x 6 items
  (seed scheme identical to the original: hook seed = seed*1000 + L).
- Secondary covariate **dCM**: change in the neutral *competence margin*
  logP(correct) - logP(incorrect); structurally identical to the safety margin but with zero
  safety valence. Robustness check only.

**Primary test (sigma=0.2, n=24 layers).**
1. Spearman correlation between dS_diff(L) and dD_diff(L) across the 24 layers, two-sided
   permutation p (20,000 permutations). Under the degradation-confound hypothesis the expected
   sign is negative (misaligned degrades more -> misaligned margin erodes more).
2. Survival test: OLS of dS_diff on dD_diff; z-score the residuals against the 24-layer
   residual distribution; compare L4/L8/L11 residual z to their uncontrolled z (~ -2.3/-2.1/-1.7).

**Decision rule (stated in advance).**
- "Explained by degradation": Spearman p < 0.05 with negative sign AND all of L4/L8/L11 fall to
  |z_resid| < 1.5 after control.
- "Survives the control": at least two of L4/L8/L11 keep z_resid <= -1.5 after control.
- Otherwise: partial/mixed - report as such, no headline of survival.

**Secondary / labelled post-hoc analyses.**
- Re-run the sigma=0.2 safety margins with per-item logging (the original saved only cell
  means): reproduction check (Pearson r, rerun vs saved dS per model across 24 layers) and
  bootstrap CIs on the hot-layer differentials (resampling the 6 prompt items, 10,000 draws).
- Same covariate at sigma=0.1 and sigma=0.4 if runtime budget (<= ~10 min) allows.
- Absolute dNLL at the hot cells, to say how degraded the text actually is there.

**Sample sizes.** 24 layers x 3 sigma (primary sigma=0.2) x 3 noise seeds x 6 neutral items x 2
continuations x 2 models for the covariate; 6 safety prompts x 3 seeds x 2 continuations x 2
models x 24 layers for the safety rerun. Model: Qwen2.5-0.5B-Instruct (fp32, MPS), adapter
ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice, both already on disk.

## Result

**Measurement integrity.** Clean safety margins reproduce the saved run exactly (base +6.897 vs
saved 6.8972; misaligned +7.393 vs 7.3935). The batched teacher-forced scorer matches the
original sequential scorer to max |dlogp| = 6e-5. Full sweep: 432 cells (24 layers x 3 sigma x
3 seeds x 2 models), per-item logging, 418 s on MPS.

**Primary (pre-registered), sigma=0.2, n=24 layers.** The differential neutral degradation
dD_diff is essentially ZERO at the hot layers: +0.005 / -0.033 / -0.049 nats/token at L4/L8/L11
(SEM ~0.02-0.05); across layers it does not significantly predict the saved differential
(Spearman rho = -0.29, permutation p = 0.167, 20,000 perms; Pearson r = -0.18, R^2 = 0.03).
After regressing out the covariate the hot-layer residual z barely moves:
L4 -2.29 -> -2.19, L8 -2.10 -> -2.25, L11 -1.71 -> -1.97; the bottom-3 residual layers are
still exactly {8, 4, 11}. Covariate-bootstrap 95% CIs on the controlled z: L4 [-2.52, -1.65],
L8 [-2.55, -1.37], L11 [-2.23, -1.03]. **Pre-registered decision: survives_control.**
Same picture at sigma=0.1 (controlled z -1.68/-2.46/-1.87) and sigma=0.4 (-0.99/-1.64/-1.17).
Absolute degradation at the hot cells is tiny (dNLL/token <= 0.10 base, <= 0.07 misaligned), so
sigma=0.2 single-layer noise is NOT making either model incoherent at these layers.

**Secondary covariate (pre-registered robustness).** The structure-matched competence-margin
covariate DOES correlate with the differential (Spearman rho = +0.51, p = 0.013; ~21% of layer
variance is degradation-shaped), but the hot layers again survive controlling for it
(z = -2.27/-1.62/-2.39; joint two-covariate control: -2.35/-1.41/-2.30).

**Reproduction check (pre-registered secondary) - this is the real finding.** The per-item
rerun of the identical safety measurement (same prompts, continuations, hook, seed scheme;
fresh noise realizations) gives a sigma=0.2 differential at L4/L8/L11 of
**+0.67 / +0.19 / +0.64** nats (item-bootstrap 95% CIs [-1.90, 3.32] / [-1.07, 1.68] /
[-1.08, 2.37]) - wrong sign, CIs spanning zero - and the rerun 24-layer differential profile
correlates with the saved one at **r = -0.10**. In the rerun distribution the hot trio sits on
the POSITIVE side (z = +1.57/+0.95/+1.54). Post-hoc band tests on the rerun: mean dS_diff over
all 24 layers -0.55 [-1.21, +0.06], over L3-L11 -0.75 [-1.94, +0.57], over {4,8,11}
+0.50 [-0.28, +1.47] - nothing excludes zero. Post-hoc seed-noise accounting: the seed-level SD
of a 3-seed cell-mean differential is 0.2-2.5 nats (median SE 0.45); 22/24 layers'
saved-vs-rerun discrepancies fall within 2 x sqrt(2) x SE3 (L8 is the exception at 4.2 sigma,
though the 2-dof variance estimate is itself loose).

**Conclusion.** The pre-registered control comes out the way the original paper would have
hoped - the L4/L8/L11 differential is not explained by capability degradation - but the
victory is empty: the differential itself does not replicate under fresh noise draws. The
saved hot layers at z ~ -2 against a 24-cell distribution, estimated from 3 seeds x 6 prompts
per cell and selected post hoc, are consistent with noise-realization luck. The correct reading
is "no stable misalignment-specific layer signal at this n", not "signal confirmed clean" and
not "signal explained by degradation".

## Caveats

- The rerun uses batched scoring (validated against sequential to 6e-5 on clean logprobs), so
  its noise realizations are fresh iid draws, not bit-identical to the original run. The
  non-replication statement is about stability under new draws at the original n (3 seeds x 6
  prompts), which is exactly the n at which the original claim was made. Both runs are noisy
  estimates of the same quantity.
- The degradation covariate is per-token NLL (and a correct-vs-wrong competence margin) on six
  short factual QA continuations; it can miss degradation modes specific to free generation
  (repetition loops). Generation-based stats per cell were not computable from saved data
  (generations exist only for the 3 post-hoc hot layers, misaligned model only) and a
  generation grid was outside the runtime budget - this is the downscope from the brief.
- n = 24 layers for the correlation tests: power only for |rho| >~ 0.4.
- Labelled post-hoc: band tests, seed-noise accounting, the covariate-bootstrap CIs, and the
  interpretation that L8's 4.2-sigma discrepancy reflects an extreme original draw.
- Single model pair (Qwen2.5-0.5B-Instruct + bad-medical-advice LoRA), single prompt battery
  (the original 6 harmful prompts); no multiple-testing correction beyond the pre-registered
  three layers, as in the original.
- A first sequential run was killed after ~4 min when its projected time (~19 min) exceeded the
  budget; the batched rerun (418 s) replaced it. Total compute including the false start
  ~11 min.

## Files

- run_control.py - batched measurement sweep (noise hook copied verbatim from
  layer_importance.py; self-check vs sequential scorer built in)
- raw_control_measurements.json - per-item raw margins/NLLs, 432 cells + clean baselines
- analyze.py - pre-registered analysis; writes results.json
- results.json - all numbers quoted above (primary, secondary, post-hoc addenda)
- run_control.log - run log
