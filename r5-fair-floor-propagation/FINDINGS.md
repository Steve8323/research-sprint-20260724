# Propagating the fair-floor correction across the worldview-routing portfolio — round 5

**Slug:** `r5-fair-floor-propagation` · **Status:** completed · **Date:** 2026-07-25
**PREREG.md** stamped `Sat Jul 25 13:53:58 EDT 2026`, before any code was written.
Confirmatory runtime **357.8 s** wall, single process, 2 torch threads (MPS for the LM cell).
All 6 stages ran; nothing was dropped for budget.

---

## Headline

**The floor inflation is real, general, and *width-gated* — it eats the k-way results and leaves
the narrow-column results standing.**

Re-scoring every routing-free control in the line under one identical data-chosen-mask procedure
(FMP), the published floor moves from a pooled **2/282 to 100/282** (distinct control model sets
only, so shared controls are counted once). But the movement is almost
entirely confined to cells whose ablation groups are **8 units wide**:

| group width | cells | OLD floor (pooled) | FAIR floor (pooled) |
|---|---|---|---|
| w = 8 | multiway k=2,3,4 (w8) + r4 reference | **2/120 (1.7%)** | **88/120 (73.3%)** |
| w = 4 | round-1, round-2, multiway k=2,3,4 (w4), rotated, mixed | **0/150 (0%)** | **12/150 (8.0%)** |
| transformer (2 heads + 64 MLP units) | LM organism | **0/12** | **0/12** |

(Fisher on the pooled w8-vs-w4 fair floors: p = 2.5e-30. POST-HOC.)

Mechanism, in one sentence: *the fair floor is high exactly when a data-chosen half of a normally
trained network is wide enough to compute the XOR by itself* — under FMP the median column
held-out accuracy is 0.99 with 80–93% of columns above the 0.9 competence gate at w=8, versus
0.82–0.87 with 28–42% above it at w=4 (cross-cell Pearson r between "fraction of columns above
the gate" and the fair floor = **+0.76**, POST-HOC).

So round 4's discovery was not a one-off, but it was also not universal. **It generalises to the
wide cells and only to the wide cells.** The k-way selection results — the ones the paper leaned
on hardest — are the casualties.

---

## 1. The procedure is verified to be round 4's, exactly

- **A1 (FMP anchor).** FMP applied to `r4-break-tautology/floor_control`, seeds 700–709:
  max |diff| vs r4's stored `lrn_*` accuracies = **0.000000000**; the chosen 8/8 masks are
  **identical on 10/10 seeds**. FMP is r4's learned-mask rule, generalised to k groups by a
  greedy-margin assignment that provably reduces to r4's top-w rule at k=2, w=H/2.
- **A0 (re-training anchor).** Every re-trained control reproduces its published per-seed
  numbers: max |diff| = **0.0** for the H=8 cells, the H=16 cell and the transformer;
  **5e-05** for the multiway cells — which is the rounding precision of the stored values
  (`round(x, 4)`), and every per-seed success boolean and every published floor count
  (1/20, 0/20, 0/20, 0/30, 0/12) is reproduced exactly. *Deviation disclosed:* PREREG demanded
  < 1e-6; 5e-05 is the best achievable against 4-dp stored values, so A0 is scored as passing at
  stored precision for the multiway cells.
- **NC1 (random-split negative control).** Random group partitions of the same models succeed
  **21/3060** overall (0.7%), including **10/400** at k=2/w=8 where the fair floor is 17/20 and
  **0/400** at k=3 and k=4. FMP is not "any split works"; it is finding a specific split.
- **NC2.** **0** degenerate masks anywhere: every FMP mask is a genuine full k-way partition.

## 2. The corrected table

Routed/method counts are verbatim from the published `results.json` files (re-scoring changes only
the control). `D_old` = published effect; `D_new` = corrected effect; CI = Newcombe hybrid-score
95% on `D_new`; Holm over the 13 confirmatory cells at family-wise α = 0.05.

| # | published cell | method arm | **OLD floor** | **FAIR floor** [Wilson 95%] | D_old | **D_new [95% CI]** | Fisher p (routed vs fair) | Holm p | **VERDICT** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `routing-worldview-select` (r1) | routed 22/30 | 0/30 | **2/30** [0.019, 0.213] | +0.733 | **+0.667 [0.436, 0.801]** | 1.5e-07 | 2e-06 | **SURVIVES** |
| 2 | `r2-routing-switch-learned` | learned_L1 23/30 | 0/30 | **2/30** [0.019, 0.213] | +0.767 | **+0.700 [0.471, 0.825]** | 3.5e-08 | <1e-06 | **SURVIVES** |
| 3 | `r3-multiway` k=2, w=4 | routed 16/20 | 0/20 | **2/20** [0.028, 0.301] | +0.800 | **+0.700 [0.405, 0.839]** | 1.7e-05 | 1.5e-04 | **SURVIVES** |
| 4 | `r3-multiway` k=3, w=4 | routed 13/20 | 0/20 | **1/20** [0.009, 0.236] | +0.650 | **+0.600 [0.314, 0.774]** | 1.4e-04 | 1.1e-03 | **SURVIVES** |
| 5 | `r3-multiway` k=4, w=4 | routed 6/20 | 0/20 | **1/20** [0.009, 0.236] | +0.300 | **+0.250 [0.008, 0.473]** | 0.092 | 0.183 | **DIES** |
| 6 | `r3-multiway` k=2, w=8 | routed 20/20 | 1/20 | **17/20** [0.640, 0.948] | +0.950 | **+0.150 [−0.038, 0.360]** | 0.231 | 0.231 | **DIES** |
| 7 | `r3-multiway` k=3, w=8 | routed 20/20 | 0/20 | **11/20** [0.342, 0.742] | +1.000 | **+0.450 [0.200, 0.658]** | 1.2e-03 | 6.1e-03 | **SHRINKS** |
| 8 | `r3-multiway` k=4, w=8 | routed 20/20 | 0/20 | **9/20** [0.258, 0.658] | +1.000 | **+0.550 [0.287, 0.742]** | 1.5e-04 | 1.1e-03 | **SURVIVES** |
| 9 | `r4-kway-learned` k=3 | learned 20/20 | 0/20 | **11/20** [0.342, 0.742] | +1.000 | **+0.450 [0.200, 0.658]** | 1.2e-03 | 6.1e-03 | **SHRINKS** |
| 10 | `r4-kway-learned` k=4 | learned 20/20 | 0/20 | **9/20** [0.258, 0.658] | +1.000 | **+0.550 [0.287, 0.742]** | 1.5e-04 | 1.1e-03 | **SURVIVES** |
| 11 | `r3-entangled` rotated | rot_ica 16/30 | 0/30 | **4/30** [0.053, 0.297] | +0.533 | **+0.400 [0.163, 0.583]** | 2.2e-03 | 6.5e-03 | **SURVIVES** |
| 12 | `r3-entangled` mixed | mix_ica 19/30 | 0/30 | **2/30** [0.019, 0.213] | +0.633 | **+0.567 [0.336, 0.722]** | 6.2e-06 | 6.2e-05 | **SURVIVES** |
| 13 | `r3-lm-organism` (transformer) | routed 12/12 | 0/12 | **0/12** [0.0, 0.243] | +1.000 | **+1.000 [0.657, 1.0]** | 7.4e-07 | 8e-06 | **SURVIVES** |
| — | `r4-break-tautology` (reference) | ceiling 60/60 | 1/60 | **51/60** [0.739, 0.919] | +0.983 | **+0.150 [0.058, 0.261]** | 2.8e-03 | (anchor) | (published r4) |

Paired McNemar exact on floor lift (same models, two scorings): significant for the four w=8 cells
(p = 3.1e-05, 9.8e-04, 3.9e-03, 1.8e-15) and **not** significant anywhere else (p ≥ 0.125) — the
narrow cells' small lifts (1–4 seeds) are not established.

*Deviation disclosed:* PREREG fixed the Holm family at the 10 cells 1–10 with the three w=4
multiway cells as conditional extras. All ran, so Holm was applied over 13 — strictly more
conservative. Recomputing at m = 10 changes **no** verdict. Also disclosed: cells 1/2 and 7/9 and
8/10 share the *same trained controls* (verified byte-identical), so the family contains
correlated tests; this too only makes Holm conservative.

## 3. One line per claim

- **"Routing makes a worldview post-hoc selectable; standard training gives 0/30."** — **SURVIVES.**
  Fair floor 2/30, corrected effect +0.667 [0.436, 0.801]. The headline number was inflated by
  7 percentage points, not by 6×.
- **"Oracle-free localization + routing reaches 23/30 vs a 0/30 floor" (round 2).** — **SURVIVES**,
  +0.700 [0.471, 0.825].
- **"k-way (k=3,4) worldview selection: 20/20 vs 0/20" (round 3, w=8).** — **SHRINKS at k=3**
  (+0.450 [0.200, 0.658]) and survives-but-halved at k=4 (+0.550 [0.287, 0.742]). The claim
  "standard training gives no k-way switch" is **false**: a data-chosen partition of an ordinary
  network already gives one on 11/20 (k=3) and 9/20 (k=4) seeds.
- **"2-way switch at width 8 is 20/20 vs 1/20" (round 3).** — **DIES.** Fair floor 17/20
  [0.640, 0.948]; routing's advantage is +0.150 [−0.038, 0.360]. Not "no effect" — the CI still
  admits up to +36 points — but the published +95 points is refuted, and at n = 20 nothing in
  [−0.04, +0.36] is distinguishable.
- **"Learned k-way routing beats the floor 20/20 vs 0/20" (round 4 k-way cell).** — **SHRINKS**
  (k=3) / **SURVIVES halved** (k=4); same numbers as the round-3 cells because the controls are
  the same models.
- **"The switch survives an entangled/rotated input basis" (round 3).** — **SURVIVES.**
  Rotated +0.400 [0.163, 0.583]; mixed +0.567 [0.336, 0.722]. Fair floors 4/30 and 2/30.
- **"k=4 at width 4 shows the switch degrading but still present (6/20 vs 0/20)."** — **DIES** at
  Holm-corrected α. Equivalence bound: the corrected effect is +0.250, 95% CI [0.008, 0.473];
  with n = 20 this cell was never powered to establish an effect this small. Report it as
  unestablished, not as absent.
- **"The switch transfers to a transformer: 12/12 vs 0/12" (round 3).** — **SURVIVES intact.**
  The fair floor does not move at all (0/12). Under FMP the two data-chosen head+MLP groups of a
  standard-trained transformer land *mid-alignment on both sides* on 9/12 seeds
  (median deploy alignment ≈ 0.50 for both groups) even though 79% of groups pass the competence
  gate — competence without commitment. In a shared-residual model the two cue circuits are simply
  not separated along the head/MLP-unit axis, so no ablation mask can select between them.
- **Round 4's own headline (+0.150 [0.058, 0.261]).** — Unchanged; reproduced bit-exactly on the
  10-seed anchor.

## 4. What the paper's table should now say

1. The routing-free floor **must be reported under a data-chosen mask** in every cell. The old
   fixed-partition floor is a statement about one arbitrary coin toss, and at w=8 it is wrong by
   up to 80 percentage points.
2. The clean, surviving claims are the **narrow-column** ones: rounds 1 and 2 at w=4, the rotated
   and mixed-basis cells, and the transformer. Their corrected effects are +0.40 to +1.00.
3. The **k-way-at-width-8 claims must be restated**. "Routing creates a k-way switch" becomes
   "routing makes an already-present k-way switch reliable": 20/20 vs 11/20 (k=3) and 9/20 (k=4).
4. The correct general statement of the whole line, after two rounds of floor correction:
   *given the cue oracle, ordinary training already leaves oppositely-committed subnetworks that
   ablation can select between — whenever the subnetworks are wide enough to be individually
   competent. Routing's contribution is reliability and free labelling, and it is largest exactly
   where the network is too narrow for spontaneous splitting.*

## 5. Limits, stated plainly

- **FMP grants the control the cue oracle.** For cells 2, 11 and 12 the *method* arm was
  oracle-free (learned localizers) while the fair floor is oracle-informed — those corrected
  effects are therefore conservative (the floor is if anything too strong).
- A fair floor computed **without** the oracle is a different, harder question and was not run.
- The reference cell's fair floor (51/60) is quoted from round 4, not recomputed at n = 60; only
  the 10-seed FMP anchor was recomputed here.
- Two cells return DIES at n = 20; both are reported with equivalence bounds and neither is
  claimed as a null. No cell in this report claims "routing has no effect".
- Re-scoring cannot detect inflation in a *routed* arm. If a method number is wrong for some other
  reason, this round is blind to it.
- Toy scale throughout (≤1.2k-param MLPs; one 68k-param transformer), searched-for equal-cost XOR
  cues, and a single deploy distribution.
- The width-gating mechanism (section H1/H2/H3 of `posthoc.json`) is **POST-HOC**: it was read off
  the confirmatory numbers, not predicted in PREREG.md.

## 6. Files

- `PREREG.md` — stamped 2026-07-25 13:53:58 EDT.
- `experiment.py` — FMP + all 6 stages; `lm_core.py` — copy of the LM organism's model code.
- `results.json` — every per-seed row, both scorings, every test; `run.log`.
- `posthoc.py` / `posthoc.json` — the width-gating analysis, labeled post-hoc.
