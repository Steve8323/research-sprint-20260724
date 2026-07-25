# FINDINGS — A localizer that EXPLOITS the cue-strength asymmetry (round 4)

Prereg: `PREREG.md`, stamped **Sat Jul 25 11:15:18 EDT 2026**, before any code here was executed.
Run: `experiment.py` (808.1 s wall, single process, 2 torch threads, CPU) -> `results.json`;
`analyze.py` -> `analysis.json`; `posthoc.py` -> `posthoc.json` (everything there is POST-HOC and
labelled). `r3_core.py` is a byte-identical copy of round 3's `core.py`
(md5 25358d008b8525bc876ca9566daa0c19); no prior-round file was modified.

## HEADLINE

**The retracted "oracle-free localization" claim comes back — off the symmetric point, and only
with a localizer built for the asymmetry.** A new localizer (L-CCM) that reads the *signed
eigen-spectrum* of the class-conditional second-moment difference recovers the true concept split
**40/40** at margin ratio r=2.05 and makes the worldview post-hoc selectable **38/50 (76%)**, versus
**26/50 (52%)** for round 3's equivariant ICA localizer run contemporaneously
(**Fisher p = 0.021**, risk difference +0.24, 95% CI [+0.05, +0.41]) and **0/30** for standard
training (p = 2.0e-12). All three preregistered PRIMARY conditions passed:
**DECISION = "THE ASYMMETRY LOCALIZER WORKS".** At the symmetric point r=1.00 the same localizer
collapses to 14/50, exactly as preregistered — the win is bought with asymmetry, not with cleverness.

## PRIMARY (preregistered)

| contrast | counts | Fisher p |
|---|---|---|
| **`r205_ccm` vs `r205_ica`** (PRIMARY) | 38/50 vs 26/50 | **0.0213** |
| `r205_ccm` vs `r205_control` (floor) | 38/50 vs 0/30 | 2.0e-12 |
| `r205_ccm` vs round-3 `axis_ica` (stored 13/30) | 38/50 vs 13/30 | 0.0043 |
| `r205_ccm` vs round-3 `rot_ica` pooled (stored 31/60) | 38/50 vs 31/60 | 0.0103 |
| `r205_ccm` vs round-3 broken-split floor (stored 18/60) | 38/50 vs 18/60 | 1.5e-6 |

Conditions: (1) p<0.05 with CCM higher — PASS; (2) CCM >= 35/50 — PASS (38); (3) vs control p<0.01 —
PASS. Preregistered power for the primary was 0.92 against a true 0.85-vs-0.53 gap at n=50/50,
alpha=0.05 (4000-rep simulation done before PREREG.md was written).

## All confirmatory arms (rot basis, seeds 700..700+n-1)

| arm | r | localizer | n | success | lands true/ent/neither |
|---|---|---|---|---|---|
| `r205_ccm` | 2.05 | **L-CCM** | 50 | **38 (0.76)** [0.63,0.86] | **50 / 0 / 0** |
| `r205_ica` | 2.05 | L-ICA (round-3) | 50 | 26 (0.52) | 29 / 20 / 1 |
| `r205_control` | 2.05 | standard training | 30 | **0** | — |
| `r100_ccm` | 1.00 | L-CCM | 50 | 14 (0.28) | 1 / 2 / 47 |
| `r100_ica` | 1.00 | L-ICA | 30 | 12 (0.40) | 14 / 16 / 0 |
| `r205_oracle` | 2.05 | true subspaces | 30 | 26 (0.87) | — |
| `r100_control` | 1.00 | standard training | 30 | **0** | — |
| `r110_ccm` | 1.10 | L-CCM | 30 | 23 (0.77) | 26 / 0 / 4 |
| `r125_ccm`, `r205_eigprobe` | — | — | — | **DROPPED by the preregistered 700 s start guard** | — |

`r100_ica` 12/30 reproduces round 3's `rot_ica` (16/30; pooled 31/60) — Fisher on 12/30 vs 31/60
is non-significant, so the one deliberate deviation (the `sigma_minclass` replacement-noise rule,
applied identically to every learned arm) did not move the anchor. `r100_control` = 0/30 reproduces
round 3 exactly.

## The mechanism, verified deterministically (S7)

`D := E[xx'|y=1] - E[xx'|y=0]` is **exactly diagonal** in (p,p',q,q') with spectrum
`(-2mA^2, +2mA^2, -2mB^2, +2mB^2)` and zero on the noise directions. Measured at n=2e5, r=2.05:
predicted `(-8, +8, -33.62, +33.62)`, observed `(+33.673, +7.998, 0.000, -0.001, -0.002, -8.000,
-33.606)` — a match to three digits, plus three exactly-null noise directions.

Exact enumeration of the orthogonal symmetry group of the labelled centres, at all 16 margin ratios
in the grid: **|G| = 64 with orbit-of-the-true-split = 2 at r = 1.00; |G| = 16 with orbit = 1 at
every r != 1.00** (including r = 0.4878, 1.01, 1.02, ..., 3.00). This independently reproduces round
3's audit correction and locates the obstruction at a single point in r.

So the localizer is not a trick: when mA != mB the four eigenvalues are distinct, the eigenvectors
are unique up to sign, and **|lambda| itself names which cue a direction belongs to**. L-CCM pairs
directions by (opposite eigenvalue sign) + (matched |lambda|). At mA = mB the eigenspaces are 2-fold
degenerate and the rule is uninformative by construction.

## S1-S2: localization is the sharp measurement (n=40 seeds, no training)

Lands on the true split (fidelity > 0.95), N=800:

| r | 1.00 | 1.025 | 1.05 | 1.075 | 1.10 | 1.15 | 1.25 | 1.50 | 2.05 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|---|
| **L-CCM** | 1/40 | 7/40 | 15/40 | 26/40 | 34/40 | **40/40** | 40/40 | 40/40 | 40/40 | 40/40 |
| eig+probe pairing | 0/40 | 1/40 | 3/40 | 8/40 | 7/40 | 11/40 | 15/40 | 15/40 | 18/40 | 19/40 |
| L-ICA (round 3) | 23/40 | — | 23/40 | — | — | — | 25/40 | — | 27/40 | — |
| ICA + asymmetry pairing | 14/40 | — | 13/40 | — | — | — | 15/40 | — | 18/40 | — |

L-CCM at r=2.05: 40/40, Wilson-95 [0.912, 1.000], binomial vs 0.5 p = 9.1e-13; vs L-ICA 27/40,
Fisher p = 7.6e-5. **S2 threshold (preregistered definition: smallest grid r whose Wilson-95 lower
bound exceeds 0.90) = r* = 1.15 at N = 800.** Note L-CCM never lands on the *entangled* split at any
r — its failures are "neither", i.e. a random split inside a degenerate eigenspace, which is a
qualitatively different failure mode from ICA's structured coin flip.

**SWEEP-C:** at r = 0.4878 and r = 0.80 (proxy *weaker* than the true cue) L-CCM lands true 40/40.
Only |mA - mB| matters, not which cue is stronger.

## S3: the threshold obeys the predicted 1/sqrt(N) law

Preregistered prediction: the threshold is set by eigenvalue resolution, gap `2 mA^2 (r^2-1)` versus
sampling error `~1/sqrt(N)`, so critical `(r*^2 - 1)` should scale as `N^{-1/2}`.

* Preregistered threshold rule (first grid r with rate >= 0.90): r*(800) = 1.15, r*(3200) = 1.07.
  Ratio of `(r*^2-1)`: **0.449**, inside the preregistered window [0.35, 0.71] (point 0.5) — **PASS**.
* **N = 200 is RIGHT-CENSORED**: the SWEEP-B grid stops at r = 1.25 where the rate is 0.875 < 0.90,
  so the preregistered threshold is undefined. This is a preregistration miss (grid too short), and
  the N=200 leg is reported as **inconclusive**, not as a failure. The censored bound
  `(r*(200)^2-1)/(r*(800)^2-1) > 1.74` does lie inside the predicted [1.4, 2.8].
* **POST-HOC** (labelled): using the interpolated 50%-crossing instead, r* = 1.1227 / 1.0625 /
  1.0286 at N = 200 / 800 / 3200, giving ratios **2.021** and **0.450** against predicted 2.0 and
  0.5. Suggestive, but this threshold definition was chosen after seeing the censoring.

## S4: the preregistered mechanism prediction FAILED (and the preregistered falsifier fired)

Prediction was "the pairing rule is the whole story: L-CCM ~ ICA+asymmetry >> eigen+probe ~ L-ICA".
Observed at r = 2.05: **L-CCM 40/40, eigen+probe 18/40, ICA+asymmetry 18/40, L-ICA 27/40.**
Neither ingredient helps alone — both single-ingredient cells sit at or *below* the
no-ingredient cell (18/40 vs 27/40, risk-diff 95% CI [-0.41, -0.01], power 0.49 against a 20-point
difference). It is a pure **interaction**: you need the D-eigenvectors *and* the asymmetry pairing
rule. The stated falsifier ("if ICA+asymmetry does not recover the true split, the claim that
whitening alone destroyed the asymmetry is wrong") fired, so the corrected mechanism is:
**whitening does not merely erase the scale information — it also moves the directions off the
D-eigenvectors, and the asymmetry statistic is only meaningful on those eigenvectors.** The two
pooled "main-effect" tests are numerically the identical table (58/80 vs 45/80, p = 0.047), which is
itself a symptom of a design with no main effects.

## S5: the negative control behaves exactly as preregistered

`r100_ccm` = 14/50 (0.28) vs `r205_ccm` 38/50, Fisher **p = 2.8e-6**. Its landing profile is
true 1 / entangled 2 / **neither 47** — the preregistered signature of a random split drawn inside a
degenerate eigenspace. Its success rate is statistically indistinguishable from round 3's honest
broken-split floor (18/60): risk-diff 95% CI **[-0.18, +0.15]**, power 0.55 against a 20-point
difference — an equivalence bound, not a claim of equality. It is *below* round 3's `rot_ica` pooled
31/60 (p = 0.019, CI [-0.40, -0.05]), i.e. at exact cue symmetry L-CCM is worse than ICA, which is
what PREREG S5 predicted directionally; against `r100_ica` (12/30) alone the test is
non-significant (CI [-0.33, +0.09], power 0.37), so only the bound is claimed.

## S6: selectability against margin ratio

r = 1.00 -> 14/50 (0.28) | r = 1.10 -> 23/30 (0.77) | r = 2.05 -> 38/50 (0.76). r = 1.25 was dropped
by the runtime guard. Selectability is already saturated at r = 1.10 (`r110` vs `r205`: risk-diff
95% CI [-0.19, +0.18], power 0.69 against a 20-point difference). The transition in *selectability*
therefore sits between r = 1.00 and r = 1.10, consistent with the localization threshold r* = 1.15
at N = 800 measured independently.

## Nulls, all with bounds and power (PREREG power discipline)

| null | counts | Fisher p | risk-diff 95% CI | power vs 20 pt |
|---|---|---|---|---|
| L-CCM vs oracle ceiling at r=2.05 | 38/50 vs 26/30 | 0.387 | [-0.263, +0.083] | 0.50 |
| L-CCM vs L-ICA *given* landing true | 38/50 vs 25/29 | 0.387 | [-0.260, +0.092] | 0.78 |
| r100_ccm vs round-3 broken-split floor | 14/50 vs 18/60 | 0.837 | [-0.184, +0.150] | 0.55 |
| r100_ccm vs r100_ica | 14/50 vs 12/30 | 0.327 | [-0.326, +0.086] | 0.37 |
| ICA landing rate r=2.05 vs r=1.00 | 27/40 vs 23/40 | 0.489 | [-0.109, +0.297] | 0.39 |
| r110_ccm vs r205_ccm | 23/30 vs 38/50 | 1.000 | [-0.194, +0.184] | 0.69 |

No "no difference" is asserted anywhere; each row is an equivalence bound with its power.

## POST-HOC (labelled, kept separate)

* **Where the remaining loss lives.** Conditional on landing on the true split, success is 0.76
  (L-CCM 38/50), 0.86 (L-ICA 25/29) and 0.87 (oracle 26/30) — mutually indistinguishable. L-ICA
  scores 1/21 when it does *not* land true. So at r = 2.05 L-CCM has removed the localization loss
  entirely and what is left (~13-24%) is the **routing/training stage**, which no localizer can fix.
* **ICA is not exactly a coin flip.** Its landing rate is 23/40, 23/40, 25/40, 27/40 at
  r = 1.00, 1.05, 1.25, 2.05 — flat-to-slightly-rising, ~0.58-0.68, mildly above 0.5 and nowhere
  near 1. Round 3's "ICA coin-flips regardless of margin" is directionally right but the level is
  ~0.6, not 0.5.

## What this does to the paper

1. **The retracted round-2 claim returns, narrowed and defensible.** "Oracle-free localization" is
   real: a localizer that uses only x and the training label y, with no coordinate privilege and no
   knowledge of where the cues live, recovers the true split 40/40 and delivers 38/50 selectability
   in a randomly rotated basis. The correct statement is now **"unsupervised localization of two
   entangled concepts is impossible at exactly equal cue strength and easy just off it, provided the
   localizer is built to use the asymmetry."**
2. **The impossibility claim must be restricted to a measure-zero point.** |G| = 64 / orbit 2 holds
   only at r = 1.00; |G| = 16 / orbit 1 at every other r we enumerated. The knife-edge is now
   measured, not asserted: at N = 800 the true split becomes reliably identifiable at r* = 1.15, and
   the threshold shrinks as 1/sqrt(N) (confirmed at N = 3200; N = 200 censored).
3. **The prescription changes.** Do not reach for a generic unsupervised method (ICA, and by
   extension anything that whitens first). Whitening is exactly the step that destroys the resource
   the asymmetry provides, and re-pairing whitened components does not repair it (18/40). Estimate
   the label-conditional second-moment operator and read its signed spectrum.
4. **Nothing here rescues the other retractions.** This is a localization result only. Round 3's
   retraction of tamper-resistance (a masked switch flips at zero cost; benign fine-tuning erases
   selectability in 100 steps) and the confidently-inconsistent-patchwork hazard are untouched.

## Caveats (honest limits)

* **The localizer is tailored to this generative family.** Exact diagonality of D in the
  (p,p',q,q') frame follows from the XOR-parity cue plus isotropic Gaussian noise; a cue whose
  label signal lives in a *third* moment, or non-isotropic noise, would break it. The transferable
  lesson is the principle (asymmetry is the resource; find the statistic in which it appears as a
  spectral gap), not the estimator.
* **7-dimensional linear toy, one small MLP.** No LM organism, no learned features. Round 3's
  `mix` (non-orthogonal) basis was not tested here at all — L-CCM's equivariance argument covers
  orthogonal T only, and D's eigenvectors are *not* covariant under a non-orthogonal mixing. That
  is the obvious next control and its omission is a real gap.
* **Two arms were dropped by the runtime guard** (`r125_ccm`, `r205_eigprobe`). The
  eigen+probe-pairing cell survives only at the localization level (18/40), not the selectability
  level, so the S4 2x2 is complete for landings and incomplete for selectability.
* **S3's N=200 leg is right-censored**; the clean 1/sqrt(N) numbers (2.02, 0.45) come from a
  post-hoc threshold definition.
* **The oracle comparison is a bound, not equality**: L-CCM is within [-26, +8] percentage points of
  the ceiling with only 50% power against a 20-point gap.
* **`r205_ccm` came in at 0.76, below the preregistered point prediction of 0.85**, because the
  routing stage — not localization — caps the arm; the localization side hit its point prediction
  exactly (40/40 predicted, 40/40 observed; 50/50 in the training arm).
