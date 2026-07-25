# PREREG — Confirmatory test of the CONFIDENTLY-INCONSISTENT hazard

STAMP (`date`, written before any confirmatory run): see the STAMP line at the bottom.
Nothing above the "RESULTS" divider is edited after that stamp.

## Why this experiment exists

Round 3 produced its most paper-relevant safety statement **post hoc**: an *unselected* routed
model is not a hedger but a **per-input patchwork** — it is confidently inconsistent. The
evidence (`r3-switch-mechanism/posthoc.py`, self-labeled POST-HOC): determinism index
D = 0.808 [boot 0.595, 0.925] against that run's own preregistered prediction of D <= 0.5;
mean |logit margin| 3.72 vs a standard model's 1.74 (2.1x); 16/22 switchable routed seeds
contain **both** a deterministic-true-cue pattern and a deterministic-proxy-cue pattern; the
sister experiment `misgen-hedger-mode` found 0/55 patchworks among standard-trained hedgers,
and r3's own 30 standard controls gave 0/30. In the transformer organism the qualitative
agreement exists (`r3-lm-organism`: routed 7/12 committed vs control 1/30 in round 1) but the
comparative claim is at n = 12.

This run gives that claim **confirmatory status**: the criterion, the thresholds, the n, and
the control comparison are fixed here, in advance, and the test is run on **fresh seeds** in
**both substrates**. It also preregisters the safety corollary the claim implies but that
round 3 never tested: **is the patchwork invisible to a confidence monitor?**

## Sources (READ-ONLY; code copied, never imported from, never written to)

- MLP substrate: config/data/model/routing/scoring copied verbatim from
  `r3-switch-mechanism/experiment.py` (itself a verbatim copy of round-1
  `routing-worldview-select/experiment.py`). Validated by a REPLICATION ANCHOR (below).
- Sequence substrate: `r3-lm-organism/core.py` copied to `tf_core.py` byte-for-byte;
  training loop / hyperparameters copied from `r3-lm-organism/experiment.py`.

## Declared pilot (peeking disclosure)

Before writing this file I ran `timing_pilot.py`: one control and one routed transformer on a
throwaway seed (9999), printing **only elapsed seconds** (8.6 s and 16.6 s per seed) and the
parameter count. No alignment, deployment, determinism, patchwork or confidence quantity was
computed or displayed. Its only use is to fix n for the transformer stage under today's
shared-machine contention. Every outcome quantity below is unseen at stamp time.

## Substrates and fresh seeds

Fresh seeds are outside every range used by earlier rounds (700-819 misgen family,
21000-21250 removal-dose):

- **MLP** (154-param 2-hidden-layer width-8 MLP, 7-dim inputs, MARGIN 2.0, NOISE 0.6, N = 800,
  Adam lr 5e-3, wd 2e-3, 2000 full-batch steps): seeds **3000-3059**, n = 60 per arm
  (`routed` = oracle column routing, `control` = standard training).
- **Transformer** (68,546-param 2-layer 4-head d=64 encoder over the word-order sequence task;
  AdamW, OneCycle max_lr 8e-3, wd 0.01, batch 512 from a 4096 pool, 700 steps, MPS):
  seeds **3100-3123**, n = 24 per arm (`routed`, `control`). Arms are trained **interleaved by
  seed** so that any early stop leaves the arms balanced.

Evaluation sets are fresh and fixed: deploy (cues anti-correlated) gen seed **50002**, n = 4000;
held-out agreeing gen seed **60002**, n = 4000. Both substrates.

## REPLICATION ANCHOR (must pass, else the run is void and is reported as void)

- MLP: retrain routed seed 700 and control seed 700 with the copied code and compare
  `full_align / selA_align / selA_train / selB_align / selB_train` against round-1's archived
  `routing-worldview-select/results.json`. Required: **max |diff| < 1e-6**.
- Transformer: retrain control seed 700 on r3's eval seeds and compare `deploy_full` to r3's
  logged 0.2215 and `held_full` to 1.000. Required: **|deploy diff| <= 0.02 and held >= 0.99**
  (looser than the MLP because MPS reductions are not bit-reproducible).

## Definitions fixed here

**Determinism index D** (protocol of `misgen-hedger-mode/run.py`, unchanged in form): a fixed set
of K cue-disagreement input PATTERNS; for each, M fresh draws of the label-irrelevant variation;
`p_k` = P(prediction follows the TRUE cue, i.e. equals the label v); `D = mean_k |2 p_k - 1|`.
D = 1 -> per-pattern deterministic; D = 0 -> per-pattern coin flip.

- MLP: **K = 8** patterns indexed by (v, c_t, c_p) exactly as in rounds 1-3; **M = 500** Gaussian
  noise draws, one shared noise tensor across all models (rng seed **1999**, fresh).
- Transformer: **K = 60** patterns = v in {0,1} x (ordered pair of distinct chunk slots for the
  cue fields F0, F1: 6 x 5 = 30). Deploy semantics: bit(F0) = v, bit(F1) = 1 - v. **M = 500**
  draws of the label-irrelevant variation = (the 4 distractor bits, the assignment of the 4
  distractor fields to the 4 remaining chunk slots), one shared draw set across all models
  (rng seed **1999**).

**Deterministic-true pattern**: p_k > 0.9. **Deterministic-proxy pattern**: p_k < 0.1.
**PATCHWORK(model) := (#det-true >= 1) AND (#det-proxy >= 1)** — the identical criterion r3 used.
**STRICT PATCHWORK** (secondary, multiplicity-robust): >= max(1, ceil(0.125 K)) patterns of each
type (MLP: >= 1 each, i.e. identical; transformer: >= 8 of 60 each).

**Confidence** = max softmax probability of the 2-way head (a monotone function of |logit margin|
for 2 classes). Mean |logit margin| is reported alongside.

**Routed target set** (the population the claim is about):
- MLP: the **switchable set S** = routed seeds meeting rounds 1-3's designated-success rule
  (selA_align > 0.8 AND selA_train > 0.9 AND selB_align < 0.2 AND selB_train > 0.9), because that
  is the population r3's 16/22 was measured on. **Contingency:** if |S| < 40 after 60 seeds,
  extend seeds 3060, 3061, ... in order up to a hard cap of 80 seeds; if |S| < 40 at the cap,
  report the achieved n and its power and do not claim the preregistered n.
- Transformer: routed seeds passing the competence gate `held_full > 0.9` (r3's seed 707 failed
  it at 0.587). Count reported; failures are never silently dropped.
**Control set**: ALL control seeds, no gating (60 MLP, 24 transformer) — matching r3's "0/30".

## PRIMARY 1 — H1 in the MLP substrate (n >= 40 routed, 60 control)

H1: *an unselected routed model is confidently inconsistent — a per-input patchwork, not a hedger,
and standard training does not produce this mode.* Three components, ALL must hold:

- **H1a NOT-A-HEDGER.** 2000-resample bootstrap 95% CI for median D over the routed target set
  has **lower bound > 0.5** (0.5 is the hedger-regime ceiling fixed by `misgen-hedger-mode`).
- **H1b PATCHWORK PREVALENT.** patchwork rate >= 0.5 with **Wilson 95% lower bound > 0.25**.
- **H1c ROUTING-SPECIFIC.** Fisher exact, one-sided (routed > control), on the patchwork
  indicator: **p < 0.001**.

Verdict: **CONFIRMED** iff H1a & H1b & H1c; otherwise the failing component is named and the
outcome is reported as PARTIAL (which components held) or DISCONFIRMED (H1b fails). No post-hoc
rescue of a failed component inside the confirmatory analysis.

Additional **pre-stated point predictions**, reported PASS/FAIL but **not gating** the verdict
(they are r3's point estimates, tested for replication):
(i) median D >= 0.8; (ii) median mean|logit margin| routed / control >= 1.5 (r3: 2.1x);
(iii) patchwork rate in [0.5, 0.9] (r3: 16/22 = 0.727).
Also reported: the full routed arm (all 60 seeds, not just S) as a robustness readout, so that
the S-conditioning cannot be the whole effect.

## PRIMARY 2 — H1 in the transformer substrate (n = 24 per arm)

Identical H1a / H1b, and **H1c with threshold p < 0.01** (smaller n).
Same verdict logic.

**Pre-stated risk, written before running (this is a prediction of a possible failure, not an
excuse written afterwards):** the sequence organism has **no continuous per-input noise channel** —
the only label-irrelevant variation is discrete distractor content, which a competent model may
ignore entirely. If control models are therefore also near-deterministic per pattern, **H1c can
fail while H1a and H1b pass**. That outcome will be reported as *"the patchwork mode is not
routing-specific in the sequence substrate"* — a partial disconfirmation of the round-3 framing —
and will NOT be rescued by redefining the criterion afterwards.

## PRIMARY 3 — H2: is the patchwork invisible to a confidence monitor? (MLP; transformer as
## replication if the compute budget allows)

**M2 (input-level, within model — the primary safety readout).** For each routed PATCHWORK model,
on the 4000-row deploy set, split inputs into *proxy-following* (prediction != v) and
*true-following* (prediction == v), and compute
`AUROC = P(confidence on a proxy-following input < confidence on a true-following input)`,
i.e. the monitor rule "low confidence flags the inputs where the model has switched worldview".
0.5 = no signal. Report the **median over models with a 2000-resample bootstrap 95% CI** and the
fraction of models with AUROC > 0.5.
**Equivalence (TOST-style), margin +-0.05:** declare *"per-input confidence carries no usable
signal"* only if the bootstrap **90% CI for the median lies entirely inside [0.45, 0.55]**.
Otherwise report the CI and the achieved bound and do NOT claim a null.
Exclusion rule fixed now: a model with fewer than 100 deploy inputs in either class is excluded
from M2 (AUROC unstable); the count of exclusions is reported.

- **POSITIVE CONTROL (information is present, code detects it).** Same AUROC with the confidence
  replaced by a routing-derived mechanistic score: MLP `|Delta_B| - |Delta_A|` (the r3 margin
  race; higher -> the proxy column wins); transformer `1[pred == selB pred] - 1[pred == selA
  pred]`. Pre-stated expectation **AUROC >= 0.9**. If this fails, M2's null is uninterpretable
  and will be reported as such.
- **NEGATIVE CONTROL (a population where the monitor should work).** The same input-level
  confidence AUROC on CONTROL models with deploy alignment in [0.2, 0.8] (ordinary underdetermined
  hedgers, which `misgen-hedger-mode` showed are noise-driven with small margins). Pre-stated
  prediction: median AUROC > 0.5, and **greater than the routed patchworks'**; one-sided
  Mann-Whitney (control-hedger > routed-patchwork), **p < 0.05**.

**M1 (model-level, secondary).** score s(model) = mean confidence over the deploy set; positive
class = routed patchwork models, negative class = control non-patchwork models; monitor rule
"lower mean confidence => patchwork", AUROC(-s), 2000-resample bootstrap 95% CI.
Decision: **USEFUL** if CI lower bound > 0.7; **UNINFORMATIVE** if the CI contains 0.5;
**ANTI-INFORMATIVE** (the monitor ranks patchwork models as the *safest*) if the CI is entirely
below 0.5. Pre-stated interpretation guard: M1 compares two differently-trained populations, so a
difference could be caused by routing per se rather than by patchwork-ness; that is exactly why
M2 (within-model) is the primary safety readout.

Headline descriptive number, pre-declared: mean confidence of routed patchwork models **on the
inputs where they follow the proxy**.

## Statistics and power discipline

Proportions: Wilson 95%. Medians: 2000-resample bootstrap 95% (90% for the TOST). Group
comparisons: Fisher exact (one-sided where the direction is pre-stated) or Mann-Whitney U.
Every reported null carries either its power against a pre-stated effect size or an equivalence
bound with its margin. Pre-computed power notes:
- H1c MLP at n ~ 44 vs 60: >0.99 power against 0.5 vs 0.0; ~0.97 against a 0.30 difference.
- H1c transformer at 24 vs 24: ~1.0 against 0.7 vs 0.0; ~0.80 against a 0.35 difference. A null
  here therefore bounds the routed-minus-control patchwork gap at roughly <= 0.35, not at 0.
- M2 equivalence at n ~ 40 models with per-model AUROC sd ~ 0.10: bootstrap CI half-width for the
  median ~ 0.04, so the +-0.05 margin is reachable; if the observed sd is larger, the achieved
  bound is reported instead of an equivalence claim.

## Compute budget and downscope rule

Single process per stage, no cloud, no downloads. MLP stage: 120 trainings x ~1.1-2.2 s ~= 4 min
(CPU, `torch.set_num_threads(2)`). Transformer stage: 24 x (8.6 + 16.6) s ~= 10.1 min + ~1 min
evaluation (MPS). Total ~15 min.
**Downscope rule (fixed now):** in the transformer stage, before starting seed index i, if the
elapsed stage time exceeds **800 s**, stop and report the achieved n (arms stay balanced because
seeds are interleaved). If the transformer stage stops with n < 16 per arm, PRIMARY 2 is reported
as under-powered with its bound and PRIMARY 3's transformer replication is skipped and stated.

## Anything not specified above is POST-HOC / EXPLORATORY and will be labeled so in FINDINGS.md.

---
STAMP:
Sat Jul 25 11:13:29 EDT 2026
