# PREREG — A localizer that EXPLOITS the cue-strength asymmetry (round 4)

Stamped: **Sat Jul 25 11:15:18 EDT 2026** (`date` output, written before *any* code in this directory was executed)

## Background (prior rounds are READ-ONLY; nothing there is modified)

In the ALIGN toy two equal-cost XOR cues agree on every training example: cue A (input columns
[0,1], the true objective) and cue B (columns [2,3], a proxy that flips at deploy); columns [4,5,6]
are noise. Routing the two worldview-consistent gradient components into disjoint hidden blocks
makes the worldview post-hoc selectable by ablation. Round 3 (`r3-entangled-localization/`) showed:

* Routing is **basis-free** given the true subspaces (rot_oracle 26/30, mix_oracle 28/30).
* **Localization is not.** A basis-agnostic (equivariant) localizer coin-flips in every basis:
  axis_ica 13/30 vs axis_pair 25/30 (p=0.0028); rot_ica 31/60 pooled; ICA lands on the true split
  39/90; success 87% when it lands true vs 26% when it lands entangled (p=8.4e-9).
* At **exactly equal cue strength** there is a proven obstruction: the labelled distribution has an
  orthogonal symmetry group of order 64 whose orbit on the true split has exactly two members.
* Round 3's audit found this obstruction is **knife-edge**: off the symmetric point the group drops
  to order 16 and the orbit collapses to one, so the true split *is* identifiable — yet ICA still
  coin-flips there even at n=200,000. **That makes it a localizer-design problem, not an
  impossibility.** This experiment attacks exactly that gap.

## Definition of the margin ratio

Cue A margin is fixed at `mA = 2.0` (round-1..3 value). The margin ratio is `r = mB / mA`, so
`r = 1.00` is round 3's symmetric point and `r = 2.05` means `mB = 4.10`. (Round 3's report phrase
"margin ratio 2.05" is ambiguous between `mB = 2.05` and `r = 2.05`; the grid below spans both:
`r = 1.025` is `mB = 2.05` and `r = 2.05` is `mB = 4.10`.) Everything else — generator, 7->8->8->2
MLP, Adam(5e-3, wd 2e-3), 2000 full-batch steps, N=800, H=8, sigma=0.6, deploy set (n=4000, gen seed
50001), held-out agreeing set (n=4000, gen seed 60001), success thresholds 0.8/0.2/0.9, scoring under
the designated OR swapped mapping — is inherited byte-identically from round 3 (`r3_core.py` is a
copy of `r3-entangled-localization/core.py`, md5 25358d008b8525bc876ca9566daa0c19). All arms use the
`rot` basis (per-seed Haar-random orthogonal T), i.e. the hardest round-3 condition.

## The new localizer (L-CCM) and its preregistered mechanism

Write p=(x0+x1)/sqrt2, p'=(x0-x1)/sqrt2, q=(x2+x3)/sqrt2, q'=(x2-x3)/sqrt2. The labelled centres are
y=0: (+-sqrt2 mA, 0, +-sqrt2 mB, 0) and y=1: (0, +-sqrt2 mA, 0, +-sqrt2 mB). **Claim (to be verified
numerically as a deterministic control):** `D := E[xx'|y=1] - E[xx'|y=0]` is exactly diagonal in
(p,p',q,q') with spectrum `(-2mA^2, +2mA^2, -2mB^2, +2mB^2)` and zero on the noise directions.

**L-CCM** = (i) eigenvectors of D, top 4 by |lambda|; (ii) pair them by the **asymmetry rule**:
first require both pairs to be sign-opposed (one direction active in each class — the XOR
signature), then minimise within-pair mismatch of |lambda|. The sign requirement kills the
"(p,q),(p',q')" pairing; the |lambda|-match requirement kills the entangled pairing "(q,p'),(p,q')"
**iff mA != mB**. L-CCM uses only x and the training label y; no cue indices, no coordinate
privilege; it is rotation-equivariant (D -> QDQ', eigenvectors covariant, eigenvalues invariant).
At mA == mB the two eigenspaces are 2-fold degenerate and the rule is uninformative **by
construction** — that is the negative control, not a bug.

**Why ICA failed** (preregistered explanation, tested by the 2x2 below): whitening rescales every
direction to unit variance and therefore *destroys* the |lambda| asymmetry, and ICA's pairing rule
("both pairs predict the label") is satisfied by the true and the entangled split alike.

### One deliberate deviation from round 3, applied IDENTICALLY to every learned-localizer arm
Round 3's replacement-noise scale was `std(|s|)`. Here it is `sigma_minclass` = min over classes of
the within-class std of the source (exactly sigma for a true cue direction, since a cue direction is
pure noise in the class where its cue is inactive). This is still data-only. It is applied to
**ccm, ica_probe, ica_scale and eig_probe alike**, so every arm contrast remains a pure localizer
contrast. `r100_ica` doubles as the check that this deviation did not move round 3's anchor.

## Arms — confirmatory training runs (all `rot`; seeds 700..700+n-1)

Executed in this priority order. **Runtime guard (preregistered):** no new training arm is STARTED
after t = 700 s of total elapsed script time; any arm not started is reported as DROPPED.

| # | arm | r | localizer | n | role |
|---|---|---|---|---|---|
| P1 | `r205_ccm` | 2.05 | **L-CCM** | 50 | **PRIMARY** |
| P2 | `r205_ica` | 2.05 | L-ICA (round-3 primary) | 50 | **PRIMARY comparator** |
| P3 | `r205_control` | 2.05 | none (standard training) | 30 | floor |
| P4 | `r100_ccm` | 1.00 | L-CCM | 50 | **NEGATIVE CONTROL — must fail** |
| P5 | `r100_ica` | 1.00 | L-ICA | 30 | round-3 replication anchor (expect ~16/30) |
| P6 | `r205_oracle` | 2.05 | true subspaces (conjugated occlusion) | 30 | ceiling |
| P7 | `r100_control` | 1.00 | none | 30 | round-3 anchor (expect 0/30) |
| P8 | `r110_ccm` | 1.10 | L-CCM | 30 | selectability threshold |
| P9 | `r125_ccm` | 1.25 | L-CCM | 30 | selectability threshold |
| P10 | `r205_eigprobe` | 2.05 | CCM directions + round-3 probe pairing | 30 | mechanism |

## Arms — localization-only sweeps (no training; seeds 800-839, n=40, `rot`)

* **SWEEP-A** `r in {1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.25, 1.50, 2.05, 3.00}`, N=800,
  localizers `{ccm, eig_probe}`; localizers `{ica_probe, ica_scale}` at `r in {1.00,1.05,1.25,2.05}`
  only (ICA is ~20x slower). Outcome = fraction landing on the true split (fidelity > 0.95, round-3
  criterion), plus the entangled and "neither" fractions.
* **SWEEP-B (scaling law)** `ccm` only, training-set size `N in {200, 800, 3200}`,
  `r in {1.00,1.01,1.02,1.03,1.05,1.07,1.10,1.15,1.25}`.
* **SWEEP-C** `ccm` at `r in {0.4878, 0.80}`, N=800 (does a *weaker* proxy work as well as a
  stronger one — is only |mA-mB| what matters?).

## PRIMARY test + decision criterion

**PRIMARY:** two-sided Fisher exact on selection-success counts, `r205_ccm` vs `r205_ica`
(50 vs 50), alpha = 0.05.
Power (4000-rep simulation, run as a design calculation before this file was written): 0.92 for a
true 0.85 vs 0.53 contrast at n=50/50, alpha=0.05 (0.78 at alpha=0.01, which is why alpha=0.05 is
preregistered here rather than 0.01).

**THE ASYMMETRY LOCALIZER WORKS** iff **all three** hold:
1. Fisher p < 0.05 with `r205_ccm` > `r205_ica`;
2. `r205_ccm` >= 35/50 (0.70) — comfortably above round 3's equivariant baseline 13/30 = 0.43 and
   above the honest broken-split floor 18/60 = 0.30;
3. `r205_ccm` vs `r205_control` Fisher p < 0.01.

Otherwise **THE ASYMMETRY IS NOT EXPLOITABLE BY THIS LOCALIZER** — reported as a clean negative and
NOT rescued post hoc.

## Secondary tests (all preregistered)

* **S1 (localization, the sharp measurement):** `ccm` lands-true rate at r=2.05 (n=40), exact
  binomial vs 0.5 one-sided; and Fisher `ccm` vs `ica_probe` lands-true at r=2.05.
* **S2 (margin-ratio threshold):** smallest r in SWEEP-A whose lands-true Wilson-95 lower bound
  exceeds 0.90, at N=800. Reported as the threshold, with the full curve.
* **S3 (scaling law, quantitative prediction):** the threshold is set by eigenvalue resolution:
  the |lambda| gap is `2 mA^2 (r^2 - 1)` and the sampling error of an entry of D scales as
  `1/sqrt(N)`. **Prediction:** the critical value of `(r*^2 - 1)` scales as `N^{-1/2}`, i.e.
  `(r*(200)^2-1)/(r*(800)^2-1)` in **[1.4, 2.8]** (point 2.0) and
  `(r*(3200)^2-1)/(r*(800)^2-1)` in **[0.35, 0.71]** (point 0.5). Threshold defined as the smallest
  grid r with lands-true >= 0.90 (point estimate) at that N.
* **S4 (2x2 mechanism):** lands-true at r=2.05 for {ccm, ica_scale} vs {eig_probe, ica_probe}.
  **Prediction:** the PAIRING RULE is the whole story — `ccm` ~ `ica_scale` >> `eig_probe` ~
  `ica_probe` ~ 0.5. Tested by Fisher on the pooled 2x2 (pairing rule x lands-true), n=80 vs 80.
* **S5 (negative control):** Fisher `r100_ccm` vs `r205_ccm` (must be significant, ccm worse at
  r=1.00); and `r100_ccm` vs `r100_ica` — **prediction: r100_ccm <= r100_ica**, because at exact
  degeneracy L-CCM returns a random split inside the two degenerate eigenspaces ("neither"), which
  round 3 showed is worth ~30% (rot_pair 18/60), whereas ICA's coin flip lands on a *structured*
  split half the time (~53%).
* **S6 (selectability threshold):** success rate for `ccm` at r in {1.00, 1.10, 1.25, 2.05}.
* **S7 (deterministic control):** exact enumeration of the orthogonal symmetry group of the labelled
  centres and of the orbit of the true split, at every r in the grid. **Prediction: |G| = 64 and
  orbit 2 at r = 1.00; |G| = 16 and orbit 1 at every r != 1.00.** Plus a numerical check that D's
  population spectrum is `(-2mA^2, +2mA^2, -2mB^2, +2mB^2)`.

## Preregistered point predictions (so a hit cannot be claimed after the fact)

`r205_ccm` ~ 0.85 (42/50) | `r205_ica` ~ 0.53 | `r205_control` = 0/30 | `r100_ccm` ~ 0.30 |
`r100_ica` ~ 0.53 (16/30) | `r205_oracle` ~ 0.87 | `r100_control` = 0/30 | `r205_eigprobe` ~ 0.53 |
lands-true: ccm@2.05 = 40/40, ica_probe@2.05 ~ 20/40, ica_scale@2.05 ~ 40/40, eig_probe@2.05 ~ 20/40.

## POWER DISCIPLINE

No "no difference" statement will be made from a bare non-significant test. Every null reported gets
BOTH (a) a 95% Newcombe score CI on the risk difference (the equivalence bound) and (b) the
simulated power of that exact test against a 20-percentage-point difference. Specifically, if
`r205_ccm` vs `r205_oracle` is non-significant it will be reported as "equivalent to within
[lo, hi] percentage points", never as "the same".

## Controls

* Positive: `r205_oracle` (routing ceiling with the true subspaces at this margin ratio).
* Negative: `r205_control`, `r100_control` (standard training, expect 0/30); `r100_ccm` (the
  localizer at the symmetric point, where it must fail).
* Replication: `r100_ica` should reproduce round 3's rot_ica (16/30; pooled 31/60).
* Deterministic: S7 group enumeration + D-spectrum check.
* Falsifier for the mechanism story: if `ica_scale` does NOT recover the true split at r=2.05, the
  claim "whitening is what destroyed the asymmetry, the pairing rule is the fix" is wrong.

## Exploratory (declared in advance, reported separately, never used to change anything above)

`explore.py` runs only: the S7 group enumeration, the population/sample D-spectrum check, per-run
timing for budgeting, and a 4-seed localizer fidelity smoke test on seeds **796-799** (disjoint from
every confirmatory seed). Anything else discovered later is labelled POST-HOC in FINDINGS.md.

## Compute budget

Single process, torch 2 threads, CPU, no cloud. Sweeps first (~90 s), then training arms in priority
order under the 700 s start guard. Target total <= 15 min.
