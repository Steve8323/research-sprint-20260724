# VERIFY_numbers.md — independent numbers audit of `PAPER.md`

**Auditor role:** numbers auditor. **Date:** 2026-07-25.
**Inputs:** `/Users/steven/research_sprint_20260724/paper/PAPER.md`, `.../paper/LEDGER.md`,
`.../paper/CLAIMS_TABLE.md`, and the `results.json` / `analysis.json` / `posthoc.json` /
`task_control.json` / `hygiene_results.json` / `results_tf.json` files under
`/Users/steven/research_sprint_20260724/`.

**Method.** (1) Every fraction (`a/b`) and every decimal token in `PAPER.md` was extracted
programmatically and matched against `LEDGER.md` + `CLAIMS_TABLE.md` after normalising Unicode
minus/dash/comma. (2) 40 numeric claims — every abstract number, every headline number, and every
number in the §4.2.3 effect table, the §4.7 regime table and the §6 retraction table — were read out
of, or recomputed from, the on-disk JSON with `python3`. Recomputations performed from raw per-seed
data: C7 competence-conditional rates, the transformer per-half competence recount, C10 random-split
pooling, C24 flip structure, C29 slot-order counts, C38 orbit-conditional Fisher, C41 matched
shipped-subnet outcome + Fisher p-values, C55 Fisher, C22 non-switchable truth-table count, R8
ungated transformer counts, N5/N17/C12 Newcombe intervals, C56 per-seed count.

---

## VERDICT: **FAIL**

Two substantive defects, one of them a wrong number stated twice in the paper's own text as an
"operational caveat that must ship with any k-way claim". Everything else reproduced. Detail below;
the count of clean reproductions is high (see §5), so this is a **targeted-fix FAIL, not a
systemic one**.

Blocking:

- **B1** — `14/20` k = 4 slot-order match is **6/20** on disk.
- **B2** — the abstract asserts "both halves are competent" for the 0/30 transformer arm; the arm's
  own per-seed data gives 16/30, and the paper's own Figure 2 note already says so.

---

## 1. Numbers in the paper that are ABSENT from the ledger, or disagree with it

| # | Paper location | Number | Status |
|---|---|---|---|
| A1 | §4.9, Figure 5 caption | "The transformer monitor arm exists on disk (**median 0.4637, n = 12**)" | **ABSENT from LEDGER.md and from CLAIMS_TABLE.md.** No claim ID, and — unlike every other off-ledger number in the paper — **not** marked `[LEDGER GAP]`. This violates LEDGER §0 Rule 1 ("if a sentence needs a number that is not in §2, it does not go in the paper"). *The number itself is correct*: `r4-patchwork-prereg/results.json → PRIMARY3_monitor_transformer.M2_input_level.{median_auroc_conf = 0.4637, n_models = 12}`. **Fix: add a ledger row (e.g. C60b), or delete the parenthetical.** |
| A2 | §4.3.1, Figure 2 caption | "gives **16/30** seeds with both halves competent, and **0/16** conditional selectable" | Absent from the ledger, but **correctly disclosed in-text as `[LEDGER GAP]`** and correct on disk (recomputed from `r5-latent-switch-standard/results.json → arms['TF-A'].per_seed`, `learned.tA ≥ 0.9 AND learned.tB ≥ 0.9` → 16, of which 0 succeed). Acceptable as flagged, but see B2 — the paper does not propagate the correction upward. |

**Rounding-only differences (all acceptable, listed for completeness):** 0.114↔0.1135 (abstract),
3.8e-6↔3.81e-6, 7.8e-16↔7.77e-16, 2.5e-30↔2.53e-30, 1.45e-7↔1.447e-7, 0.521/0.549↔0.5208/0.5486,
0.203↔0.2028, 0.026↔0.0257, 0.065↔0.0648, −2.75↔−2.751, 0.786/0.951↔0.7857/0.9506, 0.962↔0.9623,
0.98↔0.9828.

**No other number in the paper is missing from the ledger.** The automated sweep over every fraction
and decimal in `PAPER.md` returned exactly the two rows above plus the rounding list.

---

## 2. Numbers that DISAGREE with the underlying results file

### B1 — **C29, k = 4 slot-order match: paper says 14/20, disk says 6/20** (BLOCKING)

- Paper §3.6 item 2: "It also returns groups in an arbitrary slot order, matching the oracle on 10/20
  seeds at k = 3 and **14/20** at k = 4 (C29)."
- Paper §4.6, "Operational caveat that must ship with any k-way claim": "matching the oracle on only
  **10/20** seeds at k = 3 and **14/20** at k = 4 [C29]."
- LEDGER C29 says the same, citing `r4-kway-learned-plus-hygiene/results.json →
  localization.k3.n_order_matches_oracle, .k4`.

On disk:

```
localization.k3.n_order_matches_oracle = 10   (per-seed recount: 10 of 20)   ✓
localization.k4.n_order_matches_oracle = 6    (per-seed recount:  6 of 20)   ✗  paper/ledger say 14
```

Provenance of the error: `REPORT4.md` §21 and §227 both write "10/20 (k = 3) and 14/20 (k = 4)"; the
ledger copied REPORT4 rather than the file it cites, and the paper copied the ledger. The direction
of the error is *conservative* — the true rate is worse than claimed, which strengthens the caveat —
but the number as printed is wrong. **Fix: 14/20 → 6/20 in PAPER.md §3.6 and §4.6, and in LEDGER C29.**

### B2 — abstract asserts full competence for the 0/30 transformer arm (BLOCKING)

- Abstract (iii): "in one 68.5k-parameter transformer **both halves are competent**, yet no
  axis-aligned mask we built selects (0/30, Wilson upper 0.114)."
- §1.3: same conjunction. §5.1: "In the transformer both halves are competent (12/12 under the
  units-only mask…) and the axis-aligned mask still selects 0/30."

The 0/30 comes from `arms['TF-A']` (n = 30). That arm's own summary records
`failure_clauses.either_half_incompetent = 14`, and my per-half recount gives **16/30 seeds with both
halves competent, 0/16 selectable**. C17's "0 competence failures" is `n_competence_fail = 0`, which
is the **full-model** gate, not the per-half gate. The "12/12 both halves competent" figure is C18's
**units-only** granularity arm — n = 12, a different mask family, not the arm that produced 0/30.

The paper already knows this: the Figure 2 caption carries a `[LEDGER GAP]` saying "The author must
reconcile C17's wording with the per-half recount." That reconciliation was not propagated to the
abstract, §1.3, §4.3.2 or §5.1. As written, the paper's single strongest surviving claim (a
dissociation of competence from selection) is stated on a conjunction that its own appendix
contradicts for the arm that carries the null.

**Fix:** in the abstract and §1.3, either (a) attribute the competence leg explicitly to the
units-only arm — "*in a units-only mask giving each worldview 64 MLP units, both halves are competent
12/12, and selection is still 0/12; under the head+MLP mask family selection is 0/30*" — or (b) state
the honest per-half number (16/30 competent, 0/16 selectable) alongside 0/30.

### B3 — §3.1 misattributes the 92 zeroed weights to routing (non-blocking, factual)

Paper §3.1: "The smallest configuration has 154 parameters, **of which routing zeroes 92**, leaving 62
live (C53)."

Routing zeroes only the layer-2 cross-blocks. For H = 8, k = 2 that is 2 × (4 × 4) = **32** weights,
not 92. Verified against `r4-break-tautology/results.json → grad_trace.anchor_disjoint_w4`
(`total_params = 154`, `shared_params = 2`) and the per-tensor shapes. 62 is the parameter count of
**one half** (7×4 + 4 + 4×4 + 4 + 2×4 + 2 = 62), and 92 = 154 − 62 is what is dead **after
hard-deleting the unselected half** — which is exactly how LEDGER C53 words it ("Under jitter the 92
zeroed weights revive, so the switch trains 154 parameters against the control's 62"). The three
numbers are all in the ledger; the sentence attaches them to the wrong operation.

**Fix:** "…has 154 parameters; routing zeroes the 32 layer-2 cross-block weights, and the shipped
*committed* switch — one half hard-deleted — has 62 live parameters of 154 (C53)."

### B4 — §5.4 "400 grid cells" vs "100/100" (non-blocking, n-mismatch)

Paper §5.4: "In simulation over **400 grid cells** × 2000 reps it lands **100/100** cells within 15%
of the closed form … improving bias in 397/400 and RMSE in 393/400 (C63)."

On disk, `r3-selection-inflation-curve/results.json`: `partA.P1.n_cells = 100`
(`n_within_15pct = 100`, criterion "≥90% of **100** cells"); `partA.P2.n_cells = 400`
(`n_bias_ok = 397`, `n_rmse_ok = 393`). The 100/100 is over a 100-cell primary, not the 400-cell
grid. The sentence as written implies 100 of 400. Inherited verbatim from LEDGER C63.
**Fix:** "in simulation the closed-form check lands 100/100 cells within 15% (P1, n = 100 cells), and
over a 400-cell grid × 2000 reps it improves bias in 397/400 and RMSE in 393/400 (C63)."

### B5 — §5.4 "−2.751 deflates to −0.070" mixes two fields (non-blocking, provenance)

Both numbers are on disk but come from different keys of `r3-selection-inflation-curve/results.json`:

```
partB.winner_value            = -2.7512773937649193   ← the "-2.751" quoted
partB.V1.c_selected_original  = -2.723818708349157    ← what V1 actually deflates
partB.V1.prediction           = -0.06989311276079713  ← the "-0.070" quoted
```

The deflation −0.070 is computed from −2.724, not from −2.751 (a 0.027 discrepancy inside the source
file). The conclusion is unaffected but the arrow "−2.751 → −0.070" is not a single computation.
**Fix:** quote the pair from one field, or add "(V1 operates on the run's re-centred selected value,
−2.724)".

### B6 — §4.6 drops the `w = 8` scope on the leakage floor (non-blocking, scope)

Paper §4.6: "Cross-worldview leakage sits at the eval set's zero-interference floor (median worst-case
leak **0.0175**, identical at k = 2, 3, 4; k=3-vs-k=2 MWU p = 0.35)". LEDGER C30 carries "(w = 8)".
On disk (`r3-multiway-switch/results.json → secondary.S3_interference`) the statement holds at w = 8
(0.0175 / 0.0175 / 0.0175, k3-vs-k2 p = 0.3489) but **not** at w = 4, where the medians are
0.018 / 0.0182 / 0.019 and `k4_vs_k2_w4` gives **p = 0.0048**. Restore "(w = 8)".

---

## 3. Claims stated without their n or their test

None of these is a numeric error; each is a reporting-hygiene miss against the paper's own §3.8
standard ("Power stated in advance", "Standard test choices") and LEDGER §0 Rule 2.

| # | Location | Claim | What is missing |
|---|---|---|---|
| C1 | §4.2.3 table, row "Round 2, w = 4, learned localizer" | 23/30 vs 2/30, **D = +0.700 [0.471, 0.825]** | **No test.** Every other row in the table carries a Fisher or Holm p; this one carries none, and LEDGER C11 does not supply one either. Either add the Holm-adjusted p (`cells.cell2_*.holm_p_adj = 0.0` on disk, i.e. below the printing floor) or mark it "shares controls with row 1; counted once, not separately tested". |
| C2 | Abstract, clause (ii) | "+0.667–0.700 at 4 units; +1.000 in the transformer" | No n. The three cells are n = 30, n = 30 and n = 12; the n = 12 cell is doing a lot of work in the abstract's framing and the reader cannot see it. |
| C3 | §1.2 | "additivity residual at most 3.8e-6 and cross-block max\|w\| = 0.0 on 30/30 seeds, **against 0.882 in a routing-free control**" | No n on the control (it is n = 30, `r3-switch-mechanism/results.json → control.per_seed`, 30 seeds). |
| C4 | §1.2 | "The composite behaves as a **margin race** between the two columns (C23)" | No number, no n, no test at the point of statement. The supporting 95.7% / n = 22 / POST-HOC label appears only in §4.5, 800 lines later, and the §1.2 sentence does not carry the POST-HOC label the ledger requires (Rule 3). |
| C5 | §1.3 | "median determinism 0.786 MLP, 0.951 transformer, CIs excluding the hedger regime (C58)" | No n (46 and 20). The transformer arm at n = 20 is the weaker of the two and reads as equal weight here. |
| C6 | §4.5 | "Row mass on the own cue **0.865** vs **0.012** on the other (control 0.469/0.456)" | Descriptive medians with no n and no test. (n = 44 routed columns / 60 control columns on disk.) |
| C7 | §4.5 | "Selection is ≥90% complete by **median α = 0.596**" | No n (n = 22 switchable seeds). |
| C8 | §4.3.2 | "**79%** of data-chosen groups pass the competence gate" (C19) | Percentage without a denominator in the text. On disk it is `frac_above_0p9 = 0.7917` over the 24 groups of 12 seeds — i.e. **19/24**, which is materially more informative than "79%". |
| C9 | §4.7, UNLABELLED row | "against Bonferroni **α = 0.0042**" | Family not stated. The prereg derives it as 0.0167/4 over 4 rules inside a 3-primary Bonferroni. (Note: LEDGER C46's gloss "over **6** rule×arm tests" is inconsistent with 0.0042 = 0.05/12; the paper is safe only because it omits the family. Fix the ledger.) |
| C10 | Abstract, first line of the scope box vs §3.8 heading | Abstract: "**Five preregistered rounds**". §3.8 heading: "Preregistration and power protocol (**adopted after round 3**)". | Internal inconsistency, not a missing n. Rounds 1–3 did carry preregs (C1, C21, C25 are marked preregistered), but the *protocol* the paper describes started at round 4. Reword to "five rounds, all preregistered; from round 4 under the uniform protocol of §3.8." |

---

## 4. Remaining `[LEDGER GAP]` and `[CITE]` markers

**18 `[LEDGER GAP]` markers** and **7 `[CITE]` markers** remain in `PAPER.md`.

`[LEDGER GAP]` — lines 277, 369, 417, 435, 586, 611, 739, 815, 846, 893, 926, 930, 989, 1062, 1239,
1387, 1390, 1466:

| Line | Subject | Class |
|---|---|---|
| 277 | non-orthogonal `mix` localizer cells | *Genuinely absent* — confirmed: no `mix` arm exists in `r4-asymmetry-localizer/results.json` (only `sweepA`, `sweepB_scaling`, `sweepC_weaker_proxy`, `arms`). Keep. |
| 369 | optimizer / LR / epochs / train-set size for MLP arms; which H is 154 params | *Recoverable* — `r3-switch-mechanism/results.json → config` has `H=8, LR=0.005, WD=0.002, EPOCHS=2000, N=800`; `grad_trace.anchor_disjoint_w4.total_params = 154` fixes H = 8. **This gap can be closed from disk today.** |
| 417 | exact estimator for determinism D | Recoverable from `r4-patchwork-prereg/` code; keep or close. |
| 435 | forward-mask vs backward-mask implementation | Implementation detail; keep. |
| 586 | mtime verification of prereg stamping | Keep. |
| 611 | design rule n ≥ (a_k·σ_w/δ)² | Keep. |
| 739 | routed arms rescored under a data-chosen mask only for 2 cells | Correct and load-bearing (D_new is an upper bound elsewhere). Keep. |
| 815 | C17 per-half recount (16/30, 0/16) | **See B2** — must be propagated, not just flagged. |
| 846 | transformer mask-finder positive control 4/4 | *Genuinely absent* — confirmed not in `r5-latent-switch-standard/{results,posthoc}.json`. Keep. |
| 893 | paired McNemar p = 0.017 for the localizer primary | *Genuinely absent* — confirmed: `grep -ril mcnemar r4-asymmetry-localizer/` returns nothing. Keep. |
| 926, 930 | axis-vs-rotated 20/20; `mix` degradation 13/20, 17/20 | Genuinely absent. Keep. |
| 989 | "plateau extends to α ≤ 0.3" | Genuinely absent from the stored summary (the REPORT3 narrative asserts it; only `alpha_90_median = 0.5958` is stored). Keep. |
| 1062 | corrected UNKNOWN-TARGET cost 1,923,200 | Correct: stored is `total_cost_3changes = 1842400` (57.6%). Keep. |
| 1239 | determinism-matched triage 0.282 | Genuinely absent. Keep. |
| 1387, 1390 | `mix` arm; axis-vs-rot | Duplicates of 277/926. Keep. |
| 1466 | no on-disk instrument check for the transformer null | Keep — this is the single most important scope caveat in the paper. |

`[CITE]` — lines 225, 233, 243, 258, 280, 296, 314. All are bibliography placeholders in §2
(simplicity bias / Rashomon-set; gradient-routing author list and venue; Casper et al. + lottery-ticket;
Lubana et al.; steering-vector and persona-vector work; Tamirisa et al. + Zou et al.; prereg-in-ML
reference). **None of them is a numeric claim**, and each already carries an instruction to verify the
author list before citing. They must be resolved before submission but they do not affect any number.

---

## 5. Spot-checks performed against the results files (all PASS unless marked)

40 claims checked; 38 reproduce exactly or to the stated rounding.

| Claim | Paper value | On disk | File → key | Result |
|---|---|---|---|---|
| C1 | 27/27/46, mean 0.509, dip p = 0.00145, boot 0.0008, train acc 1.000 min 1.000 | nA 27, nB 27, nmid 46, score_mean 0.50884, p_dip 0.0014539, p_dip_boot 0.0008, train_acc_min 1.0, holm_reject true | `misgen-bimodality-honest/results.json → replication_cell` | **PASS** |
| C2 | 12 seeds, held-out 1.000, deploy 0.000–0.933, std 0.293, χ²(11) = 15212.8 | min 0.0, max 0.9333, std 0.2933, chi2 15212.7657, dof 11, p_homogeneity 0.0 (underflow) | `r3-lm-organism/results.json → primary1_seed_lottery`, `config.params = 68546` | **PASS** |
| C3 | token-count std 0.0, Bayes 0.500, occlusion 0.98 positions, multiset unchanged | 0.0 / 0.0, 0.5 / 0.5, 0.9828, true | `r3-lm-organism/task_control.json` | **PASS** |
| C4 | 1/60 → 51/60, Wilson [0.739, 0.919], margin −0.018 [−0.097, +0.061] → +0.910 | successes 1, learned_successes 51, [0.7389, 0.919]; margin_mean −0.0184, ci [−0.0972, 0.0609], learned 0.9103 | `r4-break-tautology/results.json → arms.floor_control.summary`; `posthoc.json → H2_margin.floor_control` | **PASS** |
| C6 | 2/30→25/30→29/30→30/30, z = 7.9726, p = 7.77e-16, McNemar 1.0 / 2.7e-4 / 2.4e-7 / 1.5e-8 | k [2,25,29,30], z 7.9726, p 7.7716e-16, p 1.0 / 2.747e-4 / 2.384e-7 / 1.490e-8 | `r5-latent-switch-standard/results.json → tests.S1_width_trend` | **PASS** |
| C7 | competent 5/26/29/30; conditional 2/5, 25/26, 29/29, 30/30 | **recomputed from per-seed `tA,tB ≥ 0.9`: 5/26/29/30; 2/5, 25/26, 29/29, 30/30** | `r5-latent-switch-standard/results.json → arms['MLP-W'].H*.per_seed` | **PASS (recomputed)** |
| C8 | 2/282→100/282; w8 2/120→88/120 (73.3%); w4 0/150→12/150 (8.0%); Fisher 2.53e-30 | 88/120 rate 0.7333, 12/150 rate 0.08, fisher_p 2.53427e-30; pools 120 + 150 + 12 (LM) = 282 | `r5-fair-floor-propagation/posthoc.json → H1_width_gating` | **PASS** |
| C9 | Pearson r = +0.764, 10 non-independent cells | pearson_r 0.7642, n_cells 10 | `…/posthoc.json → H3_competence_vs_fairfloor` | **PASS** |
| C10 | 18/4260 = 0.42% (21/5660 = 0.37%) | **recomputed by summing `NC1_random_split_succ`/`_trials` over 13 cells and removing the 3 shared-control cells: 18/4260 and 21/5660** | `r5-fair-floor-propagation/results.json → cells.*` | **PASS (recomputed)** |
| C11 | +0.667 [0.436, 0.801], Holm 2e-6; +0.700 [0.471, 0.825] | D_new 0.6667 [0.4362, 0.8005] holm 2e-6; 0.7 [0.471, 0.8251] | `…/results.json → cells.cell1_*, cells.cell2_*` | **PASS** |
| C12 | 60/60 vs 51/60, +0.150 [+0.058, +0.261], Fisher 0.0028 | D_new 0.15 [0.0584, 0.2611]; `p_raw` 0.0028275; **Newcombe recomputed [0.0584, 0.2611]** | `…/cells.cell11_*`; `r4-break-tautology/results.json → S4_learned_mask` | **PASS** |
| C13 | rot +0.400 [0.163, 0.583] Holm 0.0065; mix +0.567 [0.336, 0.722] Holm 6.2e-5 | 0.4 [0.1628, 0.5829] 0.006457; 0.5667 [0.3359, 0.7222] 6.2e-5 | `…/cells.cell_entangled_{rot,mix}` | **PASS** |
| C14 | 12/12 vs 0/12, +1.000 [0.657, 1.000], Holm 8e-6 | D_old 1.0, D_new 1.0, [0.6571, 1.0], 8e-6 | `…/cells.cell_lm_organism` | **PASS** |
| C15 | 13 cells: 9 survive / 2 shrink / 2 die | VERDICT tally over the 13 non-REFERENCE cells: 9 SURVIVES, 2 SHRINKS, 2 DIES | `…/cells.*.VERDICT` | **PASS** |
| C16 | 60/60, 60/60, 58/60, 60/60, 60/60, 59/60, 2/60; z = −11.73; shared 2/434, 28/352, 53/278, 130/434, 328/434, 396/434, 93/212; data-chosen 60/50/27/0 of 60 | all seven counts exact; `S2_trend.z = -11.7333`; grad_trace shared/total pairs exact; `S4_learned_mask` 60/50/27/0 | `r4-break-tautology/results.json → primary, S2_trend, grad_trace, S4_learned_mask` | **PASS** |
| C17 | 0/30, Wilson upper 0.1135; best-of-10 0/30, per-draw 0/300; fixed 0/30; distractor 0/30; McNemar 1.0; held-out min 1.000; decision ABSENT | learned_k 0, [0.0, 0.1135], rand_best_k 0, rand_per_draw 0/300, fixed_k 0, distractor_k 0, p 1.0, held_full_min 1.0, decision "ABSENT" | `r5-latent-switch-standard/results.json → arms['TF-A'].summary`, `tests.PRIMARY_TF` | **PASS** (but see B2 for the competence clause) |
| C18 | 64 units/worldview, both halves competent 12/12 at 1.000/1.000, margin 0.073, 0/12; heads-only 0/12, 9/12, 0.114 | as stated | `r5-latent-switch-standard/posthoc.json → tf_granularity.summary` | **PASS** |
| C19 | 79% pass the gate; both mid on 9/12; 4/12 contiguous; medians ≈0.50 / 0.44 | frac_above_0p9 0.7917; n_both_selections_mid 9; n_seeds_recovering_published_split 4; **recomputed medians 0.5005 / 0.442** | `r5-fair-floor-propagation/posthoc.json → H2…, H4_lm` | **PASS** |
| C21 | cross-block 0.0 on **30/30**, additivity max 3.81e-6, control 0.882 | **recomputed over all 30 routed seeds: max cross_block = 0.0, max additivity = 3.814697e-6; control max = 0.882107** | `r3-switch-mechanism/results.json → routed.per_seed`, `control.per_seed` | **PASS (recomputed)** |
| C22 | 44/44 vs control 13/60; agreement 0.998 vs 0.500; flip 0.000 vs 0.249; mass 0.865/0.012 vs 0.469/0.456; non-switchable 8/16 | 44/44, 13/60, 0.9981/0.4997, 0.0/0.2493, 0.8652/0.0118 vs 0.4691/0.456; **non-switchable recount = 8/16** | `…/SECONDARY1_closed_form`; per-seed `colA/colB.tt_match` | **PASS (recomputed)** |
| C23 | 95.7% of draws; within-pattern 0.957 | race_decides_output_median 0.9569; race_determinism_median 0.9567 | `r3-switch-mechanism/posthoc.json → aggregate` | **PASS** |
| C24 | W = 0.2894 [0.2759, 0.3068], monotone 22/22, Spearman −0.9997, 99.96%, 10–90 spread 0.285, α = 0.596, gates 0.289 / 0.440 | median_W 0.2894, boot [0.2759, 0.3068], monotone_frac 1.0, spearman −0.9997, **recomputed flip ratio median 0.99958**, spread 0.285, alpha_90_median 0.5958, contribution_gate 0.4402 | `…/PRIMARY_graded_control`, `secondary_sweeps`, `posthoc.json → per_seed` | **PASS (recomputed)** |
| C25 | 20/20 vs 0/20, Fisher 1.45e-11, Wilson [0.839, 1.0]; grad trace 0/108, 0/144, 0/24 | p 1.45089e-11 both, [0.8389, 1.0] | `r3-multiway-switch/results.json → primary` | **PASS** |
| C26 | floors 11/20 and 9/20 → D_new +0.450 / +0.550; k2/w8 fair floor 17/20, +0.150 [−0.038, 0.360], Holm 0.231 | D_new 0.45 / 0.55 / 0.15, [−0.0384, 0.3604], holm 0.230769 | `r5-fair-floor-propagation/results.json → cells.cell_mw_k{2,3,4}_w8` | **PASS** |
| C27 | 17/30 at H = 24, 28/30 at H = 48; McNemar 1.5e-5 and 7.5e-9 | b_method_only 17 / 28; p 1.5259e-5 / 7.4506e-9 | `r5-latent-switch-standard/results.json → tests.S3_multiway` | **PASS** |
| C28 | 20/20 (chance 1/1260, p 9.8e-63) and 20/20 (1/17325, p 1.7e-85); null 0/20; margins 0.451/0.446 | 20/20 both; chance_M 1260 / 17325; p 9.8308e-63 / 1.6847e-85; null_n_exact 0; 0.4513 / 0.4463 | `r4-kway-learned-plus-hygiene/results.json → localization` | **PASS** |
| **C29** | **10/20 (k = 3) and 14/20 (k = 4)** | **10 and 6** (stored and per-seed recount agree at 6) | `…/localization.k{3,4}.n_order_matches_oracle` | **FAIL — see B1** |
| C30 | median leak 0.0175 identical at k = 2,3,4; MWU 0.35 | 0.0175 / 0.0175 / 0.0175 at **w = 8**; p 0.34895 | `r3-multiway-switch/results.json → secondary.S3_interference` | **PASS at w = 8; scope dropped — see B6** |
| C31 | \|G\| = 64, orbit 2 at r = 1.00; \|G\| = 16, orbit 1 at the other 15 | exactly that, over 16 ratios | `r4-asymmetry-localizer/analysis.json → S7.group` | **PASS** |
| C32 | 40/40, binom 9.1e-13; 38/50 vs 26/50 Fisher 0.0213, RD +0.24 [+0.052, +0.406]; vs standard 38/50 vs 0/30 p 2.02e-12; spectrum (−8.000, +7.998, −33.606, +33.673) | 40/40, 9.0949e-13; p 0.021295, ci [0.0518, 0.4062]; 2.0237e-12; observed spectrum matches | `…/analysis.json → PRIMARY, S1_localization_r2.05, S7.spectrum` | **PASS** |
| C33 | r* 1.15 → 1.07, excess 0.3225 → 0.1449, ratio 0.449, window [0.35, 0.71]; N = 200 leg failed | identical; `PASS_N200: false` | `…/S3_scaling_law`, `S2_margin_threshold_N800` | **PASS** |
| C34 / N14 | 14/50; landings 1 / 2 / 47; vs ICA 14/50 vs 12/30, p 0.327 | 14/50; true 1 ent 2 neither 47; p 0.32691 | `…/S5_negative_control` | **PASS** |
| C35 | 40/40, 18/40, 18/40, 27/40; vs best single ingredient p = 7.6e-5 | ccm 40/40, ica_scale 18/40, eig_probe 18/40, ica_probe 27/40; p 7.6368e-5 | `…/S4_2x2_mechanism_r2.05`; `posthoc.json → POSTHOC_S4_interaction` | **PASS** (arm→label mapping in the paper is correct) |
| C36 | 40/40 at r = 0.4878 and 0.80, fidelity 0.9986 / 0.9932 | identical | `…/results.json → sweepC_weaker_proxy` | **PASS** |
| C37 | axis 25/30, rot 26/30, mix 28/30; axis_ica 13/30 (p 0.0028), rot_ica 16/30, rot_dict 8/30, mix_ica 19/30; rot_pair 10/30, pooled 18/60 | all exact; Fisher 0.0027797; pooled 10 + 8 = 18/60 | `r3-entangled-localization/results.json → arms, secondary` | **PASS** |
| C38 | 34/39 vs 13/50, Fisher 8.39e-9; ICA lands true 39/90 | **recomputed from orbit table: 34/39, 13/50, 39/90; Fisher 8.3894e-9** | `…/analysis.json → posthoc_success_by_orbit_member` | **PASS (recomputed)** |
| C39 | 7/14/21/25/24/28/27 and 13/14/16/21/22/24/24; z 6.6902 / 4.1709; p_perm 5e-5 | identical | `r4-symmetry-breaking-exchange/analysis.json → prereg` | **PASS** |
| C40 | 8/30 vs 25/30, Fisher 2.03e-5 | 8 vs 25, p 2.02794e-5 | `…/secondary.S2_S3_other_sources.agree_ctrl` | **PASS** |
| C41 | routing 16/21/25/25/28 vs augmentation 14/16/24/30/30; m = 8 p = 0.052, m = 32 p = 0.49 | **recomputed from per-seed `selA_align ≥ 0.8 AND selA_train ≥ 0.9`: 16/21/25/25/28; augmentation summaries 14/16/24/30/30; Fisher 0.0522 and 0.4915** | `…/results.json → arms.sup_dep_m*.per_seed`, `arms.augment_m*.summary` | **PASS (fully recomputed)** |
| C42 | 30/30 from m ≥ 8, mean align 0.961 → 0.992; routing best 28/30; augmentation 0/30 switchable | 30/30, 0.9614 → 0.9916, successes_switch 0 | `…/arms.augment_m{8,32}.summary` | **PASS** |
| C43 / N10 | 7/17/16/16/17; z 2.07; Wilson UB 0.726 | identical; z 2.0748; [0.392, 0.7262] | `…/secondary.S2_S3_other_sources.unlab` | **PASS** |
| C44 / N8 | 60/60 vs 59/60, McNemar 1.0, TOST δ = 0.15 passes, 90% CI [0.000, 0.050]; occlusion 60/60 | rout 60, ord 59, p_exact 1.0, tost ci [0.0, 0.05] equivalent true; `candidate_marginals.ord.occB.TRUE = [60,60]` | `r5-optionality-case/analysis.json → S1_primary` | **PASS** |
| C45 | 240/240 at 3.20 M; FT-MIX k = 100 240/240 at 1.842 M (57.6%); ord 234/240 at 1.60 M; FT-PURE 48/240, held-out 0.624 | 3,200,000; 1,842,400; 234/240 at 1,600,000; ftpure_k100 48/240, hold_final_mean 0.6239 | `…/S2_primary`, `cost_ledger` | **PASS** |
| C46 | 29/38/26 vs 25/23/28; self-train 8/60 at 0.4987; R2 0.633 p = 0.052 | identical; mean_align 0.4987 (from 0.4924); p_vs_half 0.0519 | `…/S3_primary` | **PASS** |
| C47 | 60/60; blind 4/3/13/34/52/52; random tie-break 51 at m = 16/32; McNemar m = 8 +0.433 [0.317, 0.567] p 3e-8; 57/60 exist, mean 6.4 | ord_prereg_TRUE [4,3,13,34,52,52]; ord_rt_TRUE 51 at 16/32; diff 0.4333 [0.3167, 0.5667] p 2.98e-8; **recount: 57 of 60 seeds have n_good_TRUE > 0**, mean 6.417 | `…/S1b_blind`; `posthoc.json → summary`, `per_seed` | **PASS (recomputed)** |
| C49 | +1.60 M (+100%); routed 0/60 committed, mean 0.505; ord 3/60, mean 0.492; 4000 unlabelled | 1,600,000 / 3,200,000, 100.0%; 0 and 0.5047; 3 and 0.4924; "4000 unlabelled train-mode inputs" | `…/cost_ledger`, `ADDENDUM_posthoc` | **PASS** |
| C50 | 23/23, max\|diff\| 0.0, all below `lo`, cross-block L1 0.0; 60/60 in r4 | max_abs_hardset_vs_masked 0.0, mask_flip_all_below_lo true, max_cross_block_L1 0.0, 23 per-seed rows | `r3-switch-tamper/results.json → structural` | **PASS** |
| C51 / N1 | AUC 0.767 vs 0.786, d −0.0195 [−0.0736, +0.0345], p_perm 0.351, δ* 0.0648 = 36.5% of 0.1776; positive control 0.178 at p_perm 5e-5; split-half 0.118/0.104 | mean1 0.76665, mean2 0.78618, diff −0.01953, ci95 [−0.07355, 0.03448], p_perm 0.350832, delta_star 0.06475; control diff −0.17759, p_perm 5e-5. **0.06475/0.17759 = 0.3646 → 36.5%** | `r4-tamper-power-fix/results.json → claims.C1_*`, `controls.POS_*` | **PASS** |
| C52 | d −0.0789 [−0.124, −0.033] p 0.00135 Holm 0.0095; d −0.0804 [−0.124, −0.037] p 0.00335 Holm 0.0201; competence 26.7% / 16.7% / 68.3% | −0.07887 [−0.12432, −0.03342] 0.00135 → 0.00945; −0.08044 [−0.12385, −0.03702] 0.00335 → 0.0201; frac_still_competent 0.2667 / 0.1667 / 0.6833 | `…/claims.C3_*, C4_*, holm`; `posthoc.json → collateral` | **PASS** |
| C53 | +0.0933 [+0.034, +0.153], p_perm 0.0311, Holm 0.1555; AUC9 0.560/0.467 vs 0.767/0.786; bare full-FT (−0.159, −0.251); 154 vs 62 | 0.0933 [0.03352, 0.15307], 0.031098, 0.15549; mean_fullftjit_auc9 0.5604 / 0.4671; best LoRA cell `r4_lr0.1` 0.7666 / 0.7862; **recomputed bare full-FT gaps: 0.3761 − 0.5346 = −0.1585 and 0.2164 − 0.4670 = −0.2506** | `…/claims.C6_*`, `holm`, `descriptive` | **PASS (recomputed)** |
| C54 / N5 | 11/60 vs 9/60, Fisher 0.807, Levene 0.775, SD ratio 1.05; vs corrected 0/60 p 5.9e-4; Newcombe [−0.102, +0.168] | 11 / 9, 0.807066, 0.774676, 1.052, 0.00059052; **Newcombe recomputed [−0.1023, +0.1682]** | `r4-tamper-power-fix/posthoc.json` | **PASS (recomputed)** |
| C55 | 53/60 vs 25/60, Fisher 1.05e-7 | flipped_fullftjit "53/60" and "25/60"; **Fisher recomputed 1.0460e-7** | `…/descriptive` | **PASS (recomputed)** |
| C56 | 44/60 (73%, [0.610, 0.829]) | k 44, n 60, rate 0.7333, wilson [0.6099, 0.8287]; **per-seed recount of `S2_ftmix_k20_still_switchable` = 44** | `r5-optionality-case/analysis.json`, `results.json → per_seed` | **PASS (recomputed)** |
| C57 | 23/23 → 12/23 → 8/23 → 10/23 → 7/23 → 7/23; cross-block 0.0 → 5.588 → 7.145; 0.869 vs 0.125, sign preserved 23/23 | k = 23,12,8,10,7,7; mean_cross_block 0.0, 5.58848, …, 7.14486 | `r3-switch-tamper/results.json → drift.D1_selectability.summary` | **PASS** |
| C58 | MLP n = 46 median 0.7857 [0.622, 0.912]; TF n = 20 0.9506 [0.869, 0.995]; margin ratios 1.33 / 0.95 | identical; boot95 [0.8694, 0.9948]; 1.3273 / 0.9487 | `r4-patchwork-prereg/results.json → PRIMARY1_mlp, PRIMARY2_transformer` | **PASS** |
| C59 | 1.000 (min 0.992, 46/46); deploy 0.500; majority 0.893 [0.811, 0.956] vs 0.550 [0.537, 0.582], MWU 1.28e-10; conf 0.911; 34/46 vs 0/60; bias 11%, opposite 98.9% | 1.0 / 0.9923 / frac 1.0; 0.5001; 0.8929 [0.811, 0.956] vs 0.55 [0.537, 0.5817], p 1.2781e-10; 0.9107; frac_gt_0p75 0.7391 × 46 = 34, control 0.0; 0.1104; 0.9895 | `…/POSTHOC.class_collapse`, `POSTHOC.unablated_routed_mode` | **PASS** |
| C60 / N15 | n = 29, median 0.5208, 90% CI [0.4773, 0.5486], bound 0.0486; mechanistic 0.9996 [0.9989, 0.9999]; 0.924 vs 0.927 | identical; 0.924 / 0.9269 | `…/PRIMARY3_monitor_mlp` | **PASS** |
| C61 | 0.1966 [0.1115, 0.3012], n_pos 29 / n_neg 60; 0.926 vs 0.855 | identical; 0.9258 / 0.8546 | `…/M1_model_level` | **PASS** |
| C62 / N16 | median 0.4205, MWU 0.731, `passes: false` | 0.4205, 0.73083, false | `…/M2_negative_control_control_hedgers` | **PASS** |
| C63 | 100/100 within 15%; bias 397/400; RMSE 393/400 | P1 n_cells 100, n_within_15pct 100; P2 397/400, 393/400 | `r3-selection-inflation-curve/results.json → partA` | **PASS on values; see B4 on the n** |
| C64 | ρ̂ 0.026, a_k 1.947, R_winner 1.19; −2.751 → −0.070; fresh +0.22 [−0.72, +1.16]; RMSE 1.223 → 0.415 | rho_hat 0.025660, a_k 1.94690, R_winner 1.19382, winner_value −2.75128, V1.prediction −0.069893, fresh_r1_obs 0.22, ci [−0.72, 1.16], V2 rmse_naive 1.22340 / rmse_eb 0.41529 | `…/partB`, `partB.V1`, `partB.V2` | **PASS on values; see B5 on provenance** |
| C65 / N23 | W = 9, p = 0.125, fails; V2b W = 0 p = 0.0039; V3 p = 0.54; V4 −0.798 in [−0.918, −0.367] | V2 W 9.0, p 0.125, pass_ false; V2b W 0.0 p 0.00390625; V3 p 0.54118; V4 grand mean −0.79761, ci [−0.91773, −0.36721] | `…/partB.V2, V2b, V3, V4` | **PASS** |
| C66 | b_4 1.4647 vs a_4 1.0294, factor 1.423; zero verdict flips; 5.04→3.61, 2.50→1.79 | b_k_exact 1.4647, a_k_exact 1.0294, ratio 1.4229; R_gap 5.0379→3.6085, 2.5003→1.7909; `verdict_changes: false` | `r4-kway-learned-plus-hygiene/hygiene_results.json → H1_constants, H1_recomputed_R_gap` | **PASS** |
| C67 | 0.203 / 0.028 / 0.112; AUC 0.725 | 0.2028 / 0.0280 / 0.1119; n_thetas 1770 / 1595 / 1711; 0.725 present | `…/H2_modularity_families.rows` | **PASS** |
| C68 | anchors max\|diff\| = 0.0; exceptions 0.0255 and 5e-5 | `r4-break-tautology.anchor_A0` 0.0 with round1 22 = my 22; `r5-fair-floor.anchors.A1` maxdiff 0.0, masks_identical 10; `r5-optionality.anchors.A0` 0.0, masks match; `r3-multiway.S4_round1_anchor` routed 0.0255, control 5e-5; `r3-switch-mechanism.replication_anchor` 0.0 | multiple | **PASS** |
| R7 | 19/23 transformer controls | control patchwork recount = 19 of 23 | `r4-patchwork-prereg/results_tf.json → per_seed.control` | **PASS (recomputed)** |
| R8 / N17 | gate removed exactly 3 (3103, 3109, 3110); ungated 15/23 vs 19/23 p = 0.157; strict 9/23 vs 15/23 p = 0.070; interval [−0.464, +0.042] | `competence_gate.excluded = [3103, 3109, 3110]`; **recount 15 / 19 and 9 / 15; one-sided Fisher 0.15705 and 0.06962; Newcombe recomputed [−0.4636, +0.0419]** | `…/results_tf.json` | **PASS (fully recomputed)** |
| §3.5 | "12,870 possible 8/8 splits" | C(16,8) = 12870 | arithmetic | **PASS** |

---

## 6. Fix list, in priority order

1. **§3.6 and §4.6:** `14/20` → `6/20` for the k = 4 slot-order match. Fix LEDGER C29 in the same
   edit. (B1)
2. **Abstract, §1.3:** stop asserting "both halves are competent" for the 0/30 arm; attribute the
   competence leg to C18's units-only arm (12/12, n = 12) or quote the per-half recount
   (16/30 competent, 0/16 selectable). Then delete the now-resolved `[LEDGER GAP]` at Figure 2. (B2)
3. **§4.9 Figure 5 caption:** add a ledger row for `median 0.4637, n = 12`, or delete it. (A1)
4. **§3.1:** "routing zeroes 92" → routing zeroes the 32 layer-2 cross-block weights; 92 dead / 62
   live describes the hard-deleted committed switch. (B3)
5. **§5.4:** separate the 100-cell P1 from the 400-cell P2. (B4)
6. **§4.6:** restore "(w = 8)" on the leakage floor. (B6)
7. **§5.4:** disambiguate −2.751 (`winner_value`) from the −2.724 that V1 actually deflates. (B5)
8. **Reporting hygiene (§3):** supply the missing n / test for the ten items in §3 above; the two
   worth doing regardless of length pressure are C1 (an effect size with no test in the headline
   table) and C8 (`79%` → `19/24`).
9. **Before submission:** resolve the 7 `[CITE]` markers. The 18 `[LEDGER GAP]` markers are mostly
   correct disclosures and should stay — except line 369 (MLP optimizer / LR / epochs / which H is
   154 params), which is recoverable from `r3-switch-mechanism/results.json → config` and
   `r4-break-tautology/results.json → grad_trace` and should simply be closed.

**One thing worth saying in the paper's own favour:** of ~40 numeric claims traced to source, 38
reproduced exactly, including every abstract number, every cell of the §4.2.3 effect table, every
cell of the §4.7 regime table, and all six audit anchors. Both defects that survived are in the
*labelling* layer — a number copied from a narrative report instead of the file it cites, and a
competence clause attached to the wrong arm — which is the same failure mode C68 reports for the
sprint itself.
