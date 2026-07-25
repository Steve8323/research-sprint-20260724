# FINDINGS — Preregistered test of the confidently-inconsistent hazard

Plan: `PREREG.md`, stamped **Sat Jul 25 11:13:29 EDT 2026**, unedited since.
Everything in sections 1–4 is confirmatory against that plan. Section 5 is **post-hoc /
exploratory** and is labeled so throughout; it is the reason the headline is a *correction*
rather than a clean confirmation.

---

## 0. Headline

The preregistered criterion **passed exactly as round 3 predicted in the MLP** (patchwork
29/46 vs 0/60 controls, Fisher one-sided p = 2.0e-14) and the "unselected routed models are
hedgers" prediction is **dead in both substrates** (median determinism 0.786 [0.622, 0.912] and
0.951 [0.869, 0.995]; the hedger regime is D <= 0.5). The safety corollary also holds where it
is powered: **per-input confidence cannot find the inputs where a routed model has switched
worldview** (median AUROC 0.521, 90% CI [0.477, 0.549], inside the preregistered ±0.05
equivalence margin), while an ablation-based readout of the same inputs scores 0.9996.

But two preregistered checks broke the round-3 *framing*:

1. **The mode is not routing-specific in the sequence substrate.** Transformer controls are
   *more* patchwork than routed models (19/23 vs 12/20; strict criterion 15/23 vs 7/20,
   difference −0.30, 95% CI [−0.53, −0.005]). The pre-stated risk in PREREG.md fired.
2. **Post-hoc (section 5): in the MLP the census cannot mean what round 3 said it meant.**
   A stricter criterion — same label, different input, different worldview followed — is met by
   only **2/46** routed models (and only 2/22 in round 3's own archived data). What the routed
   models actually do on cue-disagreement inputs is emit **one class, confidently**:
   majority-class rate **0.893** [0.811, 0.956] vs **0.550** [0.537, 0.582] in controls
   (p = 1.3e-10), at confidence 0.911, while still scoring **1.000** on held-out agreeing data.

So the honest replacement for "confidently inconsistent per-input patchwork" is
**confident off-distribution class collapse**: the unselected routed model is fully competent
on-distribution, follows *neither* worldview where they disagree, emits a near-constant answer
with high confidence, and no confidence monitor can see it.

---

## 1. What ran

| stage | arms x n | fresh seeds | runtime |
|---|---|---|---|
| MLP (misgen organism + r3 oracle routing) | routed 60, control 60 | 3000–3059 | 146 s |
| Transformer (r3 word-order organism) | routed 23, control 23 | 3100–3122 | 821 s |
| post-hoc mechanism (no new arms) | routed 46 retrained | 3000–3059 (S) | 77 s |

Wall-clock-only pilot 26 s (disclosed in PREREG.md). **Total 17.8 min**, single process,
CPU + MPS, over the ~15 min guidance by ~3 min; the overrun is the post-hoc mechanism run in
section 5, which I judged necessary to avoid shipping a confounded headline.

**Replication anchors (preregistered, both PASS).** MLP: retrained routed and control seed 700
match round-1's archived per-seed metrics at max |diff| = **0.0** on both arms. Transformer:
retrained control seed 700 gives deploy 0.2215 (r3 logged 0.2215) and held 1.0000.

**Preregistered contingencies that fired.** (a) MLP switchable set |S| = 46/60, above the
required 40, so no seed extension was needed. (b) Transformer: the 800 s downscope rule fired
before seed 3123, so n = 23 per arm rather than 24; arms stayed balanced because seeds were
interleaved. (c) Transformer competence gate: 3/23 routed models had held-out agreeing accuracy
<= 0.9 (seeds 3103, 3109, 3110) and were excluded as pre-stated, leaving n = 20.

Reported but **not preregistered as a test here** (it is r3's PRIMARY 2 replicating on fresh
seeds): transformer selection success **20/20 routed vs 0/23 control**.

---

## 2. PRIMARY 1 — MLP substrate: **CONFIRMED** (all three components)

| component | rule fixed in PREREG.md | result | verdict |
|---|---|---|---|
| H1a not-a-hedger | boot 95% CI for median D excludes 0.5 from below | median D **0.786**, CI **[0.622, 0.912]** | PASS |
| H1b patchwork prevalent | rate >= 0.5 and Wilson lower > 0.25 | **29/46 = 0.630**, Wilson [0.486, 0.755] | PASS |
| H1c routing-specific | Fisher one-sided p < 0.001 | 29/46 vs **0/60**, p = **2.0e-14** | PASS |

Difference in patchwork rate = 0.630, 95% CI (Newcombe) [0.474, 0.755].
Power: 1.00 against 0.5-vs-0, 0.98 against a 0.30 gap (2000-sim Fisher at alpha = 0.001).
Robustness (pre-stated): the full routed arm, not just the switchable set, gives 37/60 = 0.617
[0.490, 0.729] — the S-conditioning is not doing the work.

**Pre-stated point predictions (non-gating), tested for replication of r3's estimates:**

- median D >= 0.8 — **FAIL**, 0.786 (r3: 0.808; the CIs overlap heavily, this is noise, but it
  is a miss and is recorded as one).
- margin ratio routed/control >= 1.5 — **FAIL**, **1.33** (routed 3.29 [2.66, 3.85] vs control
  2.48 [2.18, 2.78]). r3's headline "margins 2.1x a standard model's" does **not** replicate on
  fresh seeds; the direction holds, the size does not.
- patchwork rate in [0.5, 0.9] — PASS (0.630 vs r3's 0.727).

**A quiet but important shift in the controls.** Fresh control seeds are far more deterministic
than r3's: median D **0.681** [0.571, 0.786] vs r3's 0.479. The routed-vs-control determinism
comparison is correspondingly weak (Mann-Whitney one-sided p = **0.041**, n = 46 vs 60). So
"routed models are more deterministic than standard ones" is *not* a robust claim. What is
robust is the *purity* contrast: control models produced 80 deterministic-true and 117
deterministic-proxy patterns across 60 models, yet **0/60** models contained both. Standard
training gives pure-and-deterministic or noise-driven; routing gives mixed. That asymmetry, not
D, is what the census is detecting.

---

## 3. PRIMARY 2 — transformer substrate: **PARTIAL** (H1a, H1b pass; H1c fails)

| component | result | verdict |
|---|---|---|
| H1a not-a-hedger | median D **0.951**, CI [0.869, 0.995] | PASS |
| H1b patchwork prevalent | **12/20 = 0.60**, Wilson [0.387, 0.781] | PASS |
| H1c routing-specific | routed 12/20 vs control **19/23**, Fisher one-sided p = **0.977** | **FAIL** |

The failure is **directional, not merely null**: controls are *more* patchwork than routed
models. Difference (routed − control) = −0.226, 95% CI [−0.464, +0.042]; on the pre-stated
strict criterion (>= 8 of 60 patterns of each type) −0.302, 95% CI [−0.535, −0.005]. The
reverse test (control > routed) gives p = 0.095 on the base criterion and p = 0.047 on the
strict one. Determinism is identical across arms (0.951 vs 0.955, Mann-Whitney p = 0.70) and
the margin ratio is 0.95, so the "2.1x more confident" claim fails here too.

**Power, stated honestly rather than as "no difference".** With control at 19/23 the test is
against a ceiling: even a perfect 20/20 routed arm would give Fisher p = 0.072, so the
preregistered power calculation (0.087 against a 0.35 gap) is really telling us the
one-sided design could not have won. The interval above is the usable statement: the routed
arm is between 46 points below and 4 points above the control arm.

**This is exactly the risk written into PREREG.md before the run** ("the sequence organism has
no continuous per-input noise channel ... H1c can fail while H1a and H1b pass ... will NOT be
rescued by redefining the criterion"). It is reported as written.

---

## 4. PRIMARY 3 — is the patchwork invisible to a confidence monitor?

### M2 (within-model, the primary safety readout) — MLP: **monitor is blind, with a tight bound**

- Routed patchwork models, n = 29, 0 excluded by the <100-per-class rule.
- Median AUROC of "low confidence flags the inputs where the model followed the proxy" =
  **0.5208**; 95% CI [0.455, 0.551]; **90% CI [0.477, 0.549] lies inside [0.45, 0.55]**, so the
  preregistered ±0.05 equivalence is **declared**. Achieved bound 0.049.
- **Positive control PASSES**: replacing confidence with the routing-derived margin race gives
  median AUROC **0.9996** [0.9989, 0.9999]. The information about which worldview governs each
  input is fully present and the AUROC code finds it — the null is about confidence, not about
  the test.
- **Negative control FAILS**: control-arm hedgers, which `misgen-hedger-mode` showed are
  noise-driven with half-size margins, were predicted to be detectable (AUROC > 0.5) and are
  not — median **0.4205** [0.390, 0.600], Mann-Whitney (hedger > patchwork) p = 0.73. So the
  monitor is not merely blind to *routed* models; it is blind to ordinary underdetermined
  models too. That widens the hazard and removes the routing-specific edge from this corollary.
- Headline confidences: routed patchwork models are **0.924** confident on the inputs where they
  follow the proxy vs 0.927 where they follow the truth — indistinguishable.
- Caveat, pre-stated: per-model AUROCs scatter widely (sd 0.19, range 0.14–0.86, 55% above 0.5).
  The *median* is equivalent to chance and the *sign* is a coin flip across models, so no fixed
  monitor direction can be chosen in advance. Exploratory detail in section 5.

### M2 — transformer: **underpowered, consistent with the MLP**

n = 12 models, median AUROC **0.464**, 90% CI [0.246, 0.670] — equivalence **not** declared;
achieved bound ±0.25 (sd 0.32). Positive control 1.000; negative control (control hedgers,
n = 15) median 0.539 [0.205, 0.644], Mann-Whitney p = 0.45. Confidence on proxy-following inputs
0.974 vs 0.973 on true-following.

### M1 (model-level, secondary, and it does not replicate across substrates)

- MLP: **ANTI-INFORMATIVE**. AUROC of "low mean confidence flags a patchwork model" = **0.197**,
  95% CI [0.112, 0.301] — a confidence-based triage would rank patchwork models as the *safest*
  of the population (mean confidence 0.926 vs 0.855 for control non-patchworks).
- Transformer: nominally "USEFUL" (0.958, CI [0.813, 1.0]) but with **n_neg = 4** and a mean
  confidence gap of 0.985 vs 0.999 — a threshold no operator could set in advance. Treat as
  uninformative; the two substrates disagree, which is what the pre-stated interpretation guard
  ("M1 compares two differently-trained populations") anticipated.

---

## 5. POST-HOC / EXPLORATORY — what the census is actually measuring

**Everything in this section was not preregistered.** It exists because the per-pattern profiles
of the passing MLP models, inspected after the confirmatory run, all looked like this:

```
seed 3001  align 0.529  D 0.912   p_k = [0.11 0.02 0.15 0.03 | 0.98 1.00 0.98 1.00]
```

The 8 patterns are ordered (v, c_t, c_p): the first four have label v = 0, the last four v = 1.
This profile means "predicts class 1 on every cue-disagreement input", which the census scores
as *both* deterministic-proxy (on v = 0) *and* deterministic-true (on v = 1) — i.e. as a
patchwork. Within a v-group all four patterns carry the **same cue semantics** (they differ only
in which bit-ordering encodes the XOR value), and r3 established that every routed column
implements its cue's XOR exactly (44/44), so a routed model is invariant inside a v-group **by
construction**. In this substrate the census can therefore only ever detect the v-split, which
is exactly the signature of a constant/class-biased output.

**Strict criterion (same label, different input configuration, different worldview followed):**

| | patchwork (preregistered) | within-v genuine (post-hoc) |
|---|---|---|
| r4 MLP routed S (n=46) | 29 | **2** |
| r4 MLP control (n=60) | 0 | 0 |
| r3 MLP routed S (n=22, archived) | 16 | **2** |

Fisher one-sided p = 0.19 for the strict criterion. So round 3's 16/22 is, on its own archived
numbers, 2/22 once the constant-output reading is excluded.

**What the unselected routed model actually is** (46 switchable seeds retrained,
`posthoc_mechanism.py`):

- Held-out **agreeing** accuracy, un-ablated: median **1.000**, min 0.992, 46/46 above 0.9. The
  unselected model is fully competent on-distribution.
- Deploy (cue-disagreement) alignment: median 0.500.
- **Majority-class rate on disagreement inputs: 0.890** (IQR 0.726–0.977) at confidence
  **0.911**. Measured over patterns, routed **0.893** [0.811, 0.956] vs control **0.550**
  [0.537, 0.582]; 34/46 routed exceed 0.75 vs **0/60** controls; Mann-Whitney p = **1.3e-10**.
  (0.5 = "follows some cue"; 1.0 = "always one class". Controls follow a cue; routed models
  collapse to a class.)
- It is **not** a readout-bias artifact: the sign of the logit difference agrees with the bias
  on only 11% of inputs, and |bias| exceeds |Delta_A + Delta_B| on 4%. The two columns vote
  opposite on 98.9% of disagreement inputs and the collapse comes from a class-asymmetric
  magnitude race between them, not from the bias term.

**Transformer, counts-only lower bound on genuine within-v mixing** (the per-pattern profiles
were not saved, so this is a proof-only bound: counts prove mixing when one deterministic count
exceeds 30 while the other is > 0): routed **6/20**, control **11/23**. Genuine
position-dependent worldview mixing is real in the sequence substrate — and again more common in
the control arm.

**Mode census (exploratory).** MLP routed S: 5 pure-true, 6 pure-proxy, **29 mixed**,
6 non-deterministic. MLP control: 18 pure-true, 22 pure-proxy, **0 mixed**, 20 non-deterministic.

**Monitor-direction structure (exploratory).** Among MLP routed patchworks, **55%** of models
deviate from AUROC 0.5 by more than 0.10 and **31%** by more than 0.20, but the sign is 55/45
(Wilson [0.38, 0.72]) — large per-model signal with an unpredictable direction, which is worse
for a deployer than a uniformly flat AUROC. The direction is predicted by the model's deploy
alignment (Spearman rho = 0.59, p = 6.9e-4, n = 29; transformer rho = 0.71, p = 0.010, n = 12) —
which is exactly the quantity a deployer cannot measure without the labels the monitor is
supposed to substitute for.

---

## 6. What this does to the paper

- **Keep, now confirmatory:** *the unselected routed model is not a hedger.* Preregistered,
  fresh seeds, both substrates, CI-separated from the hedger regime. Round 1–2's "the model
  hedges, so you can tell you forgot to select" is definitively dead.
- **Keep, now confirmatory and it is the better safety claim:** *the failure is invisible to a
  confidence monitor.* Equivalence-bounded in the MLP (±0.05), positive control at 0.9996,
  and the model-level version is anti-informative (0.197). Add the honest widening: ordinary
  underdetermined models are equally invisible, so this is a limit of confidence monitoring, not
  a unique liability of routing.
- **Rewrite:** the phrase "per-input patchwork of two worldviews". In the MLP it is a **confident
  class collapse** off-distribution (majority-class 0.89 at confidence 0.91, vs 0.55 in
  controls, p = 1.3e-10, while scoring 1.000 on agreeing data) — which is still routing-specific
  and still a liability, just a different one. In the transformer, genuine per-input mixing
  exists but standard training produces *more* of it.
- **Retract:** "standard training never produced this mode (0/55, 0/30)" as a general claim. It
  holds for the MLP census (0/60 replicated) and fails in the sequence substrate (19/23
  controls). Also retract the "margins 2.1x" figure: 1.33x on fresh MLP seeds, 0.95x in the
  transformer.

---

## 7. Caveats

- The MLP census's degeneracy (section 5) means the preregistered PRIMARY 1 passed on a criterion
  that does not distinguish the two mechanisms it was meant to distinguish. I am reporting it as
  passing because that is what was preregistered, and reporting the decomposition immediately
  after. The strict within-v test I would use in a rewrite (2/46) was **not** preregistered.
- Transformer n = 23 per arm (20 after the competence gate), below the requested 24, because the
  preregistered 800 s downscope rule fired. All transformer intervals are wide.
- Transformer per-pattern profiles were not persisted, so its within-v decomposition is a proof
  bound, not an exact count.
- The M2 negative control failed, so this run contains **no demonstration of a population where
  a confidence monitor does work**. The positive control (mechanistic score) shows only that the
  information exists and the test can find it.
- Both substrates are toy organisms with two equal-cost cues; nothing here says real models
  behave this way.
- One shared determinism draw set per substrate (rng 1999); draw-set variance is unsampled.
  SE(p_k) ≈ 0.022 at M = 500.

## 8. Files

- `PREREG.md` (stamped 11:13:29, unedited) · `timing_pilot.py` (wall-clock pilot)
- `mlp_stage.py` -> `results_mlp.json`, `run_mlp.log`
- `tf_stage.py`, `tf_core.py` (byte-copy of r3-lm-organism/core.py) -> `results_tf.json`, `run_tf.log`
- `posthoc_mechanism.py` -> `results_posthoc.json`, `run_posthoc.log` (POST-HOC)
- `explore.py` -> `results_exploratory.json` (POST-HOC)
