# PREREG — Propagating the fair-floor correction across the worldview-routing portfolio (round 5)

Stamped, before any confirmatory code was written or run:

```
Sat Jul 25 13:53:58 EDT 2026
```

Disclosure of everything already done at stamp time: I read (read-only) the prior rounds'
`experiment.py` / `core.py` / `PREREG.md` / `results.json` / `FINDINGS.md` for
`routing-worldview-select`, `r2-routing-switch-learned`, `r3-multiway-switch`,
`r3-entangled-localization`, `r3-lm-organism`, `r4-kway-learned-plus-hygiene`,
`r4-break-tautology`. No new code was executed. Two structural facts were established by
comparing stored per-seed arrays and are used below to save compute (they are facts about the
existing files, not outcomes of this experiment):

- `routing-worldview-select/control` and `r2-routing-switch-learned/control` per-seed selection
  accuracies agree to max |diff| = 0.0 -> they are the SAME trained models.
- `r3-multiway-switch/k{3,4}_w8_control` and `r4-kway-learned-plus-hygiene/k{3,4}_control`
  per-column held-out accuracies agree to max |diff| = 0.0 -> SAME trained models.

---

## 1. The defect being corrected

Round 4 (`r4-break-tautology`) found that the routing-free FLOOR in this line was scored by
ablating **one arbitrary fixed unit partition** of the control network (units 0..w-1 vs w..2w-1,
etc.) — a single coin toss among C(16,8) = 12,870 possible 8/8 splits for its own H=16 control.
Re-scoring the *identical* standard-trained models under a mask chosen **from data** moved that
floor from 1/60 to **51/60**, and routing's honest effect from ~+0.98 to **+0.150 [0.058, 0.261]**.

Every earlier table in the line (rounds 1, 2, 3, and the k-way round-4 cell) reports its floor
under the same arbitrary-fixed-partition scoring, and therefore inherits the same inflation.
This round re-scores every one of those routing-free controls under one fixed, identical,
pre-specified fair-mask procedure and recomputes each published routing effect.

**Fair-control principle being enforced:** the control must be granted the same freedom to choose
*which units to ablate* that the routed arm gets for free (the routed arm knows its own training
partition). The control is not granted any freedom the routed arm lacks: it uses the same cue
oracle, it uses only agreeing-distribution data, and its mask is fixed by a single deterministic
rule — never by searching over outcomes on the deploy/probe set.

## 2. Hypotheses

- **H1 (inflation is general).** The fair floor is materially above the published floor in every
  re-scored cell, not just in `r4-break-tautology`.
- **H0 (r4 was a special case).** In the other cells the fair floor stays near the published floor,
  so the published routing effects stand as reported.

Both are publishable. Per-cell verdicts follow the rule in section 6 regardless of direction.

## 3. THE FAIR-MASK PROCEDURE (FMP) — identical in every cell

Let M be a routing-free (standard-trained) control model from a cell; let the cell have k cues
with cue-occlusion operators `occ_1..occ_k`; let the cell's ablation groups have width w over an
ablation-unit set of size k*w.

1. **Data.** X_held = the cell's own held-out **agreeing** evaluation set (the same object the
   cell already uses for its competence gate: gen seed 60001 for the MLP cells, the LM cell's
   held set). **No deploy set, no probe set, and no disagreeing data of any kind enters mask
   selection.** Occlusion noise is drawn from a fixed generator seeded 99991 in every cell.
2. **Occlusion operator.** `occ_i` is the *same oracle-faithful* cue occlusion the cell's routed
   arm uses: replace cue i's input columns with fresh ambient noise (axis-basis MLP cells);
   `T o occlude_cols o T^-1` (rotated / mixed-basis cells); re-randomise cue i's 2-token chunk
   (LM cell).
3. **Sensitivity.** a0 = pre-mask activations of M on X_held at the cell's ablation layer;
   a_i = same on `occ_i(X_held)`; `sens_i(u) = mean |a0(u) - a_i(u)|`, mean over examples and over
   any sequence positions / layers the cell's mask is shared across.
4. **Margin.** `margin_i(u) = sens_i(u) - max_{j != i} sens_j(u)`.
5. **Greedy assignment.** Sort all (u, i) pairs by `margin_i(u)` descending; walk the list assigning
   u to group i whenever u is unassigned and group i holds fewer than w units. This is exactly
   r4's "top-w by (sensA - sensB) for A, top-w by (sensB - sensA) for B" when k = 2 and w = H/2,
   and is its unique natural k-way extension.
6. **Multiple unit types** (LM cell: attention heads and MLP hidden units) run steps 3-5
   independently within each type, at that type's own width.
7. **Scoring.** The resulting masks are plugged into the **cell's own published success criterion,
   verbatim and unchanged**, including that cell's own generosity for controls (designated-OR-
   swapped for k=2 cells; best-permutation for k>2 cells) and its competence gate.

Nothing in FMP is tuned per cell. If a cell cannot accept FMP as written, it is reported as
NOT RE-SCORABLE rather than re-scored with a variant.

## 4. Cells to re-score, and the source of each routed number

Routed/method arm counts are taken verbatim from the published `results.json` files and are NOT
re-run (re-scoring changes only the control). The published OLD floor is likewise read from those
files.

| # | cell | control arm | published floor | published method arm | n |
|---|---|---|---|---|---|
| 1 | `routing-worldview-select` (r1) | `control` | 0/30 | `routed` 22/30 | 30 |
| 2 | `r2-routing-switch-learned` | `control` (= cell 1's models) | 0/30 | `learned_L1` 23/30 | 30 |
| 3 | `r3-multiway-switch` k=2,w=8 | `k2_w8_control` | 1/20 | `k2_w8_routed` 20/20 | 20 |
| 4 | `r3-multiway-switch` k=3,w=8 | `k3_w8_control` | 0/20 | `k3_w8_routed` 20/20 | 20 |
| 5 | `r3-multiway-switch` k=4,w=8 | `k4_w8_control` | 0/20 | `k4_w8_routed` 20/20 | 20 |
| 6 | `r4-kway-learned-plus-hygiene` k=3 | `k3_control` (= cell 4's models) | 0/20 | `k3_learned` 20/20 | 20 |
| 7 | `r4-kway-learned-plus-hygiene` k=4 | `k4_control` (= cell 5's models) | 0/20 | `k4_learned` 20/20 | 20 |
| 8 | `r3-entangled-localization` rotated | `rot_control` | 0/30 | `rot_ica` 16/30 | 30 |
| 9 | `r3-entangled-localization` mixed | `mix_control` | 0/30 | `mix_ica` 19/30 | 30 |
| 10 | `r3-lm-organism` (transformer) | `control` | 0/12 | `routed` 12/12 | 12 |
| 11 | `r4-break-tautology` (reference) | `floor_control` | 1/60 | `ceiling_disjoint_w8` 60/60 | 60 |

Cell 11 is the reference: its fair floor (51/60) is already published and is re-derived here on a
10-seed subsample **only as an anchor for FMP** (section 7), not as a new number.

Additional cells re-scored **if and only if** compute remains after cells 1-11:
`r3-multiway-switch` k=2,3,4 at w=4 (floors 0/20, 0/20, 0/20). If not reached they are reported
as NOT RUN, never as null.

## 5. What is re-trained, and n

No model checkpoints were saved by any prior round, so every control must be re-trained from its
published seed. Training code, hyperparameters, seeds and evaluation sets are copied from each
cell's own source file so that re-training is bit-reproducible. n per cell = the published n
(30 / 20 / 12 / 60-subsampled-to-10), never smaller, so no cell loses power relative to what it
originally claimed.

## 6. Tests and decision rules (fixed now)

For each cell c: k_routed/n from the published file; k_fair/n from FMP.

- **PRIMARY(c):** two-sided Fisher exact, routed vs fair floor. Family = the 10 confirmatory cells
  (1-10; cell 11 is an anchor, not a test). **Holm** correction at family-wise alpha = 0.05.
- **Effect size:** corrected effect `D_new = rate_routed - rate_fair`, with a **Newcombe
  hybrid-score 95% CI**. Published effect `D_old = rate_routed - rate_oldfloor`.
- **Per-cell verdict, decided by this rule alone:**
  - **DIES** — PRIMARY(c) fails Holm. (Reported with a TOST-style equivalence bound: the Newcombe
    upper bound on D_new, so the null is never claimed beyond its resolution.)
  - **SHRINKS** — PRIMARY(c) passes Holm but `D_new < 0.5 * D_old`.
  - **SURVIVES** — PRIMARY(c) passes Holm and `D_new >= 0.5 * D_old`.
- **Fair-floor lift** per cell: `rate_fair - rate_oldfloor` with a McNemar exact test on the
  paired per-seed success flags (same models, two scorings), alpha = 0.05, reported as secondary.

**Power statement, made in advance.** The two smallest cells are n=12 (LM) and n=20 (k-way).
Fisher at n=12 vs 12 cannot reject a 12/12 vs 8/12 gap at Holm-corrected alpha; at n=20 vs 20 it
cannot reject 20/20 vs 16/20. Therefore **no cell will be reported as "routing has no effect"**;
DIES is reported strictly as "the published effect is not established at this n against the fair
floor", together with the Newcombe upper bound on D_new.

## 7. Controls on the re-scoring procedure itself (prestated, all cells)

- **A0 (anchor).** For every re-trained control the OLD fixed-partition scoring must reproduce the
  published per-seed numbers to max |diff| < 1e-6 and identical per-seed success booleans. A cell
  failing A0 is reported as a **re-implementation**, and its fair floor is labeled as such.
- **A1 (FMP anchor on the reference cell).** FMP applied to `r4-break-tautology/floor_control`
  seeds 700-709 must reproduce r4's stored `lrn_*` accuracies to max |diff| < 1e-6. If it does
  not, FMP is not r4's procedure and every number here is labeled non-comparable to r4.
- **NC1 (random-split negative control).** Per cell, 20 random group partitions per seed (5 for the
  LM cell) scored by the same criterion. If random splits score near the fair floor, FMP is
  measuring nothing and the fair floor is really "any split works" — which is a *different*
  and stronger indictment; either way it is reported.
- **NC2 (degeneracy check).** Every FMP mask must be a genuine full k-way partition into groups of
  exactly w units. Any degenerate mask voids that seed's fair score.

## 8. Declared limitations

- Cell 11's fair floor is quoted from r4, not recomputed at n=60 (compute).
- Re-scoring cannot change the routed arms; if a routed arm's own number is inflated for some
  other reason, this round does not detect it.
- FMP uses the cue oracle. A fair floor *without* the oracle would be a different (harder)
  question and is out of scope; this is stated so the corrected effect is read as
  "routing vs oracle-informed ablation of a normally-trained net", not "vs an uninformed attacker".
- Compute cap: ~15 min total on a shared machine. Cells are run in the order listed and any not
  reached are reported as NOT RUN.

## 9. Analysis is fully specified

`experiment.py` writes `results.json` with every per-seed row, both scorings, and every test
above. Anything in FINDINGS.md not specified here is labeled **POST-HOC**.
