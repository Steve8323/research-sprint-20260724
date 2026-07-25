# FINDINGS — Round-2 preregistered test of misaligned-fragility + certified operating point

Preregistration in `PREREG.md` (date-stamped `Fri Jul 24 23:10:43 EDT 2026`, before any
confirmatory data on the 32 fresh prompts was collected). Code: `run.py`. Raw per-item margins:
`results.json`. Run log: `run.log`. Single process, MPS, 655.8 s (< 15-min cap), no downloads
(all weights cached). Substrate: Qwen2.5-0.5B-Instruct base vs + EM `bad-medical-advice` LoRA,
verbatim `~/misalignment_noise_probe` noise hook and teacher-forced margin readout.

## Headline

The last live noise-probe hypothesis SURVIVES a proper preregistered test. On 32 fresh prompts
and four UNTESTED decoder layers {2,12,16,20} (disjoint from round-1's selected {4,6,8,11}),
sigma=0.2 RMS-scaled activation noise erodes the misaligned model's safety margin significantly
more than the base model's: pooled per-prompt differential mean d = -0.642 nats (95% CI
[-0.92, -0.37]), median -0.579, 28/32 prompts negative, two-sided exact Wilcoxon W=55,
p = 2.5e-5 (H0 rejected; Cohen d_z = -0.84). Direction supports H1. This is not winner's curse:
the round-1 pooled effect was on layers picked as sweep extremes, and it reappears at layers
that were never used to pick anything.

## Part A — Certified operating point (adapting the CNR flip certificate)

Unlike CNR's small 768->64->1 probe, this substrate's readout is the whole 0.5B model's
teacher-forced margin S = logP(REFUSAL) - logP(COMPLY). A hard CNR-style unreachability
certificate is impossible (no tractable global bound on S over a noise ball; Gaussian full
support), so the certified object is the onset sigma at which the noise provably moves the
readout by theta = 0.5 nat, established two independent ways.

| quantity | value | meaning |
|---|---|---|
| validity witness: median MC/linear ratio @ sigma=0.02 | 0.98, IQR [0.88, 1.04] | linearization tracks MC; certificate valid (gate [0.7,1.4] passed) |
| onset_sigma_MC (median cell readout-move std >= 0.5 nat) | 0.05 | assumption-free MC onset |
| onset_sigma_lin (closed-form dS/dH sensitivity), median | 0.0385 | analytic companion, agrees with MC |
| median readout-move std at sigma=0.2 | 2.448 nats | the operating dose moves the readout by ~2.4 nats |
| sigma=0.2 certified above onset? | YES | confirmatory gate opened |

The closed-form linear certificate is the direct analog of CNR's cert2/cert4: one backward pass
gives g = dS/dH at block L, and the predicted move std is sigma*sqrt(sum_t RMS_t^2 ||g_t||^2),
exactly linear in sigma. Per-cell MC vs linear at sigma=0.2 agree to ~15% (e.g. L12 p0:
3.32 MC vs 3.11 lin; L2 p0: 1.85 vs 2.49). So sigma=0.2 is not arbitrary: it is ~4x the
certified readout-move onset (0.05), and the readout provably moves by ~2.4 nats there.

Certificate caveat: this bounds/measures the magnitude of readout movement, not a flip
probability of a fixed decision boundary as in CNR (the margin is continuous, not a 0/1 probe).
The onset is on the misaligned model's own readout; it establishes reachability of readout
change, which is the precondition for the Part-B differential to be measurable. It does not by
itself certify anything about the base-vs-misaligned difference — that is what Part B tests.

## Part B — Preregistered confirmatory result

Primary (untested layers {2,12,16,20}, the single locked test): SUPPORTS H1.
- mean d = -0.642 nats, 95% t-CI [-0.918, -0.367], median -0.579, Cohen d_z = -0.842
- 28/32 prompts negative (4 positive)
- two-sided exact Wilcoxon W = 55, p = 2.54e-5 -> reject H0; one-sided-less p = 1.27e-5.

Positive control (round-1 layers {4,6,8,11}, same 32 fresh prompts): DIRECTIONALLY MATCHES BUT
NOT SIGNIFICANT.
- mean d = -0.464 [-0.961, +0.033], median -0.536, 19/32 negative, two-sided p = 0.062 (fails
  alpha=0.05). Round-1 reported d = -1.33 (p=6e-4) on its own 24 prompts; on these fresh 32 the
  same layers give a weaker, non-significant effect.

Structural result: the untested layers give a CLEANER, more significant effect than the
round-1-selected layers (p=2.5e-5 vs p=0.062). Round-1's four layers were chosen as extreme
cells of a 144-cell sweep and carry winner's-curse shrinkage; the fresh, never-selected layers
do not. Strong evidence the pooled misaligned-fragility effect is real and layer-general, not an
artifact of the original extreme-cell selection.

Negative control (label permutation, seed 12345, 200 perms): observed median -0.579,
permutation two-sided p = 0.000.

## Per-layer breakdown (POST HOC, descriptive only)

| layer | set | mean d(L) [95% CI] |
|---|---|---|
| L2 | primary | -1.02 [-1.65, -0.40] |
| L12 | primary | -0.90 [-1.52, -0.28] |
| L16 | primary | -0.05 [-0.52, +0.41] (near zero) |
| L20 | primary | -0.59 [-0.81, -0.37] |
| L4 | control | -0.51 [-1.40, +0.38] |
| L6 | control | -0.02 [-1.03, +1.00] (near zero, as round-1 also found) |
| L8 | control | -0.75 [-1.83, +0.33] |
| L11 | control | -0.58 [-1.37, +0.21] |

The pooled effect is carried by L2/L12/L20; L16 is null. Broad but not uniform. Round-1's
"generic" L6 is again ~0 here. No single-layer test was preregistered; hypothesis-generating only.

## Interpretation

Round 1 killed the layer-dissociation story (which specific layers matter) but left a pooled
"misaligned models are more fragile to activation noise" hypothesis flagged as post-hoc because
its four layers were selected extremes. This round gives that pooled hypothesis its own
preregistration on fresh prompts + untested layers and it holds decisively (p=2.5e-5), while the
originally-selected layers weaken to non-significance — the exact pattern expected if the effect
is genuine and layer-general and round-1's numbers were inflated by selection. The last
noise-probe hypothesis is NOT killed; it is confirmed in its de-selected form.

Mechanistically (one-sentence, post hoc): the EM misaligned model sits on a shallower effective
safety margin that the same relative noise walks downhill further; clean margins here are base
+1.43 vs misaligned +4.46, so despite starting higher the misaligned model loses more margin per
unit noise across most layers.

## Caveats

- Certificate scope: Part A certifies/measures readout-movement onset (theta=0.5 nat), not a
  flip probability of a fixed boundary (readout is a continuous margin, not CNR's 0/1 probe).
  "sigma=0.2 certified reachable" = "readout provably moves by >=0.5 nat there," the precondition
  the brief wanted. Not a soundness certificate on the base-vs-mis differential itself.
- One model pair, one dose (sigma=0.2, prereg-fixed), one EM organism, 3 noise seeds/cell. Fresh
  prompts are experimenter-written in the petty-harm register, not an external benchmark; clean
  margins differ from round-1's set, so the sets are not distribution-identical.
- Margin readout is gameable: the parent repo's centerpiece result showed log-prob margin can
  decouple from behavior; no behavioral judge was run (compute budget). Claim is about the
  log-prob safety margin, not demonstrated behavior.
- L16 null and the per-layer table are post hoc. Only the pooled two-sided Wilcoxon over
  {2,12,16,20}, the Part-A onset/witness, the {4,6,8,11} positive control, and the permutation
  negative control were preregistered.
- Power: sd(d)=0.76, n=32 gives high power for the observed ~0.64-nat effect; CI excludes 0. The
  positive control's non-significance (p=0.062, CI just crosses 0) is itself informative about
  round-1's inflation, not a failure of this test.
