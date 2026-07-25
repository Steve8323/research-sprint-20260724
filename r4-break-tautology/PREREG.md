# PREREG — Selectability without structurally disjoint columns (round 4)

Stamped, before any confirmatory code was run:

```
Sat Jul 25 11:10:15 EDT 2026
```

Disclosure of everything already run at stamp time: `core.py` (the harness) and
`timing_pilot.py`, which trains **one** seed (700) per arm and prints **only wall-clock
seconds**. No accuracy, no success flag, no weight statistic was computed or displayed. The
pilot's sole output was per-seed runtime (0.57-1.66 s), used to fix `n` at 60. Nothing about
any hypothesis below could be informed by it.

---

## 1. Background and the exact thing being attacked

Round-1 (`routing-worldview-select/`) trains a 2-hidden-layer MLP on the ALIGN toy where two
equal-cost XOR cues agree on every training example (cue A = true objective, input cols [0,1];
cue B = misaligned proxy, cols [2,3]; cols [4,5,6] noise). Routing the two worldview-consistent
gradient components into two hidden unit groups makes the worldview post-hoc selectable by
ablation: 22/30 at column width 4, 30/30 at width 8, vs 0/30 for standard training.

**Both round-3 audits flagged the same defect.** In every arm that produced a strong number:

1. the two routed groups are *structurally disjoint subnetworks* — `zero_cross_blocks` zeroes
   the hidden-to-hidden cross blocks at init and forward unit-ablation gives them identically
   zero gradient, so a gradient trace finds **0 shared weights** (r3-multiway: 0/108, 0/144,
   0/24; only the 2 output-bias scalars are shared); and
2. the **training mask equals the evaluation mask** — each group is trained under exactly the
   ablation it is later scored under.

Under (1)+(2) selectability is close to guaranteed by construction. Round 3's verdict was
therefore "report it as a demonstration, never as an effect size."

**This experiment breaks (1) and (2) and measures what is left.**

## 2. Hypothesis

**H1 (the effect is not purely structural).** Post-hoc worldview selectability survives when the
routed groups share parameters and/or when the training mask differs from the evaluation mask,
at a rate significantly above the standard-training floor.

**H0 (the null this is designed to be able to accept).** Selectability is an artifact of disjoint
columns plus mask identity: every tautology-breaking variant collapses toward the 0/n floor.

Both outcomes are publishable and the FINDINGS will state whichever occurs. If H0 holds, the
paper must retract "selectability" as an effect size and restate it as a construction.

## 3. Manipulations (three axes, six variants)

Everything not listed is verbatim round-1: `INPUT=7, MARGIN=2.0, NOISE=0.6, N=800, LR=5e-3,
WD=2e-3, EPOCHS=2000` full-batch Adam, CrossEntropy, routed loss = mean of the two
occluded+masked views, selection = zero the other group's units.

**Column width is held at w = 8 for every variant and for the ceiling**, because round 3 showed
the round-1 misses were width-4 column-trainability failures, not conceptual ones. Any drop is
therefore attributable to the manipulation and not to per-column capacity.

| arm | H | S_A | S_B | manipulation |
|---|---|---|---|---|
| `anchor_disjoint_w4` | 8 | 0-3 | 4-7 | none — exact round-1 replication anchor |
| `ceiling_disjoint_w8` | 16 | 0-7 | 8-15 | none — disjoint ceiling |
| `floor_control` | 16 | 0-7 | 8-15 | standard training (no routing, no occlusion) — floor |
| `overlap25` | 14 | 0-7 | 6-13 | axis 1: groups share 2/8 units (25%) |
| `overlap50` | 12 | 0-7 | 4-11 | axis 1: groups share 4/8 units (50%) |
| `overlap75` | 10 | 0-7 | 2-9 | axis 1: groups share 6/8 units (75%) |
| `shared_trunk` | 16 | 0-7 | 8-15 | axis 2: hidden layer 1 fully shared (no mask at L1); routing acts only on hidden layer 2 |
| `soft25` | 16 | 0-7 | 8-15 | axis 3: TRAIN mask leaks (other group at 0.25); EVAL mask hard 0/1 |
| `soft50` | 16 | 0-7 | 8-15 | axis 3: TRAIN mask leaks at 0.50; EVAL mask hard 0/1 |

Cross-block handling generalizes round 1: `l2[i,j]` is zeroed at init iff no group contains both
i and j. For `shared_trunk` nothing is zeroed (layer-1 output is unmasked, so every layer-2 unit
legitimately reads the whole trunk). For the soft arms the cross blocks are zeroed at init but
receive nonzero gradient thereafter (eps > 0), which is the point.

**Second evaluation axis (train-mask != eval-mask, applied to every routed arm at no training
cost):** a *learned* eval mask. Score each unit u by `sensA(u) - sensB(u)`, where `sensC(u)` is
the mean absolute change in u's unmasked pre-mask activation when cue C's input columns are
resampled as noise; keep the top-w units to select worldview A and the bottom-w to select B.
This uses the cue-column oracle the whole line already assumes, but **not** the training unit
partition, so the eval mask is estimated and may differ from the train mask. Scored at hidden
layer 1 for arms masked at L1, hidden layer 2 for `shared_trunk`.

## 4. Success criterion (verbatim round 1, unchanged)

Deploy set = fixed n=4000, gen seed 50001 (proxy flipped). Held-out agreeing set = n=4000, gen
seed 60001. Per-seed selection success under the **designated** mapping:

`acc(deploy | keep A) > 0.8` AND `acc(held | keep A) > 0.9` AND
`acc(deploy | keep B) < 0.2` AND `acc(held | keep B) > 0.9`.

Both directions must work. `floor_control` is scored **generously** (designated OR swapped
mapping counts), as in round 1.

## 5. n, seeds, and power

n = 60 seeds (700-759) for `floor_control`, `ceiling_disjoint_w8` and all six variants.
n = 30 (seeds 700-729, round-1's own seeds) for `anchor_disjoint_w4`, whose only job is
replication.

Design calculation, done before the run: with the floor at 0/60, a two-sided Fisher exact test
reaches p < 0.05/6 = 0.00833 at k >= 8 successes. Power of that test:

| true variant rate | 0.20 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|
| power (n=60) | 0.933 | 0.999 | 1.000 | 1.000 |

So a variant whose true rate is 20% or more will be detected with >=93% probability; a
non-significant floor test at n=60 is therefore genuinely informative against rates >= 0.2, and
this is stated in advance so that no null is reported without its power.

Ceiling comparison power (Fisher, n=60 vs 60, alpha=0.05, simulated): 1.00 for 0.95 vs 0.60,
0.96 for 0.95 vs 0.70, **0.63 for 0.95 vs 0.80, 0.32 for 0.95 vs 0.85**. Because that test
cannot support "no difference" against small gaps, **any claim that a variant matches the
ceiling must be made as an equivalence bound, never as p > 0.05** (see 6.S1).

## 6. Tests and decision rules (all fixed now)

**PRIMARY (six, one per variant), family-wise Bonferroni alpha = 0.05/6 = 0.008333.**
`P_v`: two-sided Fisher exact on per-seed success counts, variant v vs `floor_control`, 60 vs 60.

Per-variant verdict, decided by this rule alone:
- **SURVIVES** — `P_v` passes AND the Wilson 95% lower bound on v's rate exceeds 0.20.
- **WEAKENED** — `P_v` passes but the Wilson 95% lower bound is <= 0.20.
- **BROKEN** — `P_v` does not pass.

Headline rule stated in advance: if **all six** variants are BROKEN, the finding is that the
selectability numbers were structural, and FINDINGS.md will say so as the headline. If at least
one variant SURVIVES, the headline is the surviving effect size (rate + Wilson CI), positioned
between the disjoint ceiling and the standard-training floor.

**SECONDARY S1 (distance from the ceiling).** For each variant: Fisher exact v vs
`ceiling_disjoint_w8` at alpha = 0.05 (uncorrected, labeled secondary) **and** a Newcombe
hybrid-score 95% CI for (ceiling rate - variant rate). "Indistinguishable from the disjoint
ceiling" may be asserted only if that CI's upper bound is below the pre-stated equivalence
margin **delta = 0.20**. Otherwise the CI is reported as a bound.

**SECONDARY S2 (overlap dose-response).** Cochran-Armitage trend test on success counts across
overlap fractions {0, 25, 50, 75}% (arms `ceiling_disjoint_w8`, `overlap25/50/75`), one-sided
(decreasing) at alpha = 0.05. Prediction: monotone decrease.

**SECONDARY S3 (did the manipulation actually break disjointness?).** Empirical gradient trace:
for one seed per arm, backward each view separately and record the number of parameters
receiving nonzero gradient from **both** views. Pre-stated expectation: `anchor_disjoint_w4` and
`ceiling_disjoint_w8` share exactly 2 parameters (the output bias); every variant shares strictly
more. **A variant whose shared count is 2 does not count as a tautology break and its primary is
void**, regardless of its p-value. Also reported: final max |w| over the disallowed L2 region.

**SECONDARY S4 (learned eval mask).** For each routed arm, the same success criterion under the
learned masks. Family of 7 tests (each arm vs `floor_control` under the learned mask), Holm
correction within the family at alpha = 0.05. Also reported per arm: exact-match rate and mean
Jaccard between the learned keep-A mask and the true S_A.

**ANCHOR A0.** `anchor_disjoint_w4` on seeds 700-729 must reproduce round-1's stored per-seed
`selA_align`, `selA_train`, `selB_align`, `selB_train` with max |diff| < 1e-6 and the identical
per-seed success booleans (22/30). If it does not, the run is reported as **not anchored** and
every number is labeled as coming from a re-implementation rather than a replication. The anchor
is *not* an outcome test and cannot be used to rescue or discard any variant.

## 7. Things that would make me wrong, declared now

- A variant could fail simply because its total width H is smaller (overlap arms have H = 14/12/10
  vs the ceiling's 16). Column width is matched at 8, which is the quantity round 3 identified as
  load-bearing, but total width is not. **Pre-stated diagnostic:** for every failing seed, record
  whether either group missed the competence gate `acc(held|keep) > 0.9`. If failures are
  dominated by competence misses the honest reading is "capacity", not "sharing", and FINDINGS
  will say so. This is a diagnostic, not a rescue: the primary verdict stands as computed.
- `floor_control` is the only non-routed control. Round 1's `occlusion_only` (0/30) and
  `routing_no_occlusion` (6/30) arms are not re-run here; the floor comparison is therefore
  "vs standard training", not "vs every ablation of the method".
- Toy scale throughout: 7-dim inputs, <=1.2k-parameter MLPs, searched-for equal-cost XOR cues.

## 8. Analysis is fully specified

`experiment.py` writes `results.json` containing every per-seed row plus every test above,
computed exactly as written here. Anything reported in FINDINGS.md that is not in this file is
labeled **POST-HOC** and is written to `posthoc.json` by a separate script.
