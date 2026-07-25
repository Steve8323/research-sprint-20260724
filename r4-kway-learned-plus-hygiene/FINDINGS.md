# k-way selection with a data-only localizer, plus portfolio hygiene — round-4 findings

**Directory:** `/Users/steven/research_sprint_20260724/r4-kway-learned-plus-hygiene/`
**PREREG.md stamped:** Sat Jul 25 11:09:33 EDT 2026, before any code here was written.
**Runtime:** experiment 200.7 s (single process, 2 torch threads, CPU), hygiene 2.9 s. No downscoping.

---

## One-line result

The round-2 data-only pair probe scales to k = 3 and k = 4 — it recovers the k true cue groups on
**20/20 seeds at both k** (chance 1/1260 and 1/17325; binomial p = 9.8e-63 and 1.7e-85) and the
end-to-end pipeline `probe → route → post-hoc select` gives **20/20 clean k-way selection vs 0/20
for standard training** (Fisher p = 1.45e-11 at both k, both primaries PASS) — **but this joins
k-way to round 2's localizer at round 2's coordinate privilege, and therefore does not restore the
retracted "oracle-free" claim.** The one genuinely new fact is negative-control-shaped: routing with
a *wrong* disjoint partition scores **0/20**, so the localizer is load-bearing after all — the k-way
number is not tautological in its group content, only in its column disjointness.

---

## PART 1 — the experiment

### What was and was not being tested

`r3-multiway-switch/` reached k = 4 using the round-1 **oracle** `cue_cols(k) = [[2i, 2i+1]]`.
`r2-routing-switch-learned/` removed the oracle at k = 2 with a data-only probe. Those two results
could not appear in the same sentence. This run makes them one sentence.

It does **not** re-open the retraction. `r3-entangled-localization/` showed that the round-2 probe
works only because it is handed a basis in which the concepts already occupy disjoint input columns,
and that coordinate privilege *is* oracle information. That privilege is unchanged here and is stated
in PREREG.md §"Therefore this experiment does NOT claim to restore oracle-free localization."
**The correct sentence is: "k-way selection with a data-only localizer, in a coordinate system where
the k concepts are already column-separated."**

### Localization (PRIMARY family L3/L4, Bonferroni α = 0.025 each)

| k | input dims | candidate pairs | exact recovery | Wilson 95% | chance p₀ | binomial p | decision |
|---|---|---|---|---|---|---|---|
| 3 | 9 | 36 | **20/20** | [0.839, 1.000] | 1/1260 = 7.9e-4 | 9.8e-63 | **SUPPORTED** |
| 4 | 11 | 55 | **20/20** | [0.839, 1.000] | 1/17325 = 5.8e-5 | 1.7e-85 | **SUPPORTED** |

The probe separates cleanly and the separation does **not** degrade with k: every true cue pair
scores accuracy 1.000, the best non-true pair scores ≈ 0.55, giving a per-seed margin of
**median 0.451 (min 0.430) at k = 3** and **median 0.446 (min 0.433) at k = 4** — S1's pre-stated
"clean separation" bar (> 0.2) is met on 40/40 seeds. Partial recovery (S2) is vacuous: mean true
pairs found = 3.0/3 and 4.0/4.

**Negative control (S4, permuted labels, no training):** exact recovery **0/20 at both k**, top
accepted-pair accuracy median 0.551 / 0.550 — i.e. exactly the level of the best *non-true* pair on
real labels. The probe finds structure only when the labels carry it.

### Selectability (PRIMARY P_k3 / P_k4, Bonferroni α = 0.025 each)

| k | arm | successes | Wilson 95% | median per-column held-out acc |
|---|---|---|---|---|
| 3 | **learned** (routed on discovered groups) | **20/20** | [0.839, 1.000] | 0.9985 |
| 3 | control (standard training, scored over all 3! perms) | 0/20 | [0.000, 0.161] | 0.834 |
| 3 | **broken** (cross-cue partition, scored over all 3! perms) | **0/20** | [0.000, 0.161] | **0.504** |
| 4 | **learned** | **20/20** | [0.839, 1.000] | 0.9987 |
| 4 | control | 0/20 | [0.000, 0.161] | 0.848 |
| 4 | **broken** | **0/20** | [0.000, 0.161] | **0.510** |

- **P_k3:** Fisher [[20,0],[0,20]], p = 1.45e-11, 20/20 ≥ 14/20 → **SUPPORTED**.
- **P_k4:** identical table, p = 1.45e-11 → **SUPPORTED**.
- **S3 (broken-split floor):** learned 20/20 vs broken 0/20 at both k, Fisher p = 1.45e-11,
  Newcombe 95% CI on the rate difference [0.772, 1.000].

### The localizer's slot order is arbitrary, and that is the practical catch

Exact recovery is 20/20, but the *order* in which the k groups come out of the greedy probe matches
the oracle's index order on only **10/20 seeds at k = 3** and **6/20 at k = 4** (observed maps
include (0,2,1), (2,0,1), (1,2,0) at k = 3 and (0,2,3,1), (2,3,1,0), (3,0,1,2) at k = 4). Selection
still succeeds 20/20 because the scoring uses the **localizer-induced mapping** — column *s* must
follow the cue whose dims the probe placed in slot *s*. Operational consequence: with a learned
localizer the column↔worldview labelling is set by the probe's output order, not by any intrinsic
index. **You must store the localizer's group assignment with the checkpoint or the switch is
unusable** — you would know a switch exists and not which position is which.

### Controls and anchors

- **S5 oracle anchor / identity:** on the seeds where the discovered slot order *does* match the
  oracle (10 at k = 3, 6 at k = 4), the routed model is trained on numerically identical arguments to
  `r3-multiway-switch`'s oracle run, and every per-seed metric (`full_held`, `held[]`, `probe[][]`)
  reproduces it at **max |diff| = 0.000000** (tolerance was 1e-3; r3's own rerun determinism was
  5e-5). The code copy is verbatim and the pipeline is byte-faithful.
- **Negative control (standard training):** 0/20 at both k, matching r3's control cells.
- **Negative control (broken split):** 0/20 at both k — see the power note below for what this does
  and does not license.
- The `broken` failure mode is **competence, not direction** (POST-HOC descriptive from stored data):
  median per-column held-out accuracy is 0.504 / 0.510, i.e. chance. Cross-cue groups such as {0,3}
  contain one bit of cue 0 and one bit of cue 1, and the occlusion step destroys the partner bit of
  each, so no column can learn its XOR at all. The right reading is *"the discovered groups must be
  complete, sufficient cues"* — not *"a wrong partition selects the wrong worldview."*

### Power discipline

- Every routed arm sits at 20/20 and every non-routed arm at 0/20, so **both primaries are at the
  design's ceiling and floor simultaneously**: the Fisher p is the smallest this design can produce
  and the tests have no dynamic range. They confirm the pipeline runs; they measure nothing about
  effect magnitude.
- **learned vs the r3 oracle cell (20/20 vs 20/20) is reported as an identity, not as equivalence.**
  On order-matching seeds it is literally the same computation (max |diff| = 0.000000). Treated as a
  statistical comparison it would be uninformative: the Newcombe 95% interval on the rate difference
  is **[−0.161, +0.161]**, so this design cannot exclude a localizer penalty smaller than
  **16 percentage points**. No "the learned localizer is as good as the oracle" claim is made from n = 20.
- Same bound applies to any "no k-specific degradation" reading: at n = 20/arm with both arms at 20/20,
  degradations below ~16 points are invisible.

### Pre-stated deflations (carried forward verbatim from PREREG.md, and they still apply)

1. **Not an effect size.** The k routed columns are structurally disjoint subnets sharing only the
   2-element output bias, and each group is trained under exactly the ablation mask it is later
   evaluated with. Selectability at these numbers is close to guaranteed by construction. This is a
   demonstration. — *Partial pushback earned by S3:* the **column disjointness** is imposed, but the
   **group content** is not. Routing on a wrong disjoint partition gives 0/20, so at least one
   ingredient of the 20/20 is discovered rather than assumed. That is a narrower tautology than r3's
   audit assumed, not the absence of one.
2. **Coordinate privilege.** This does not restore oracle-free localization (r3-entangled-localization).
3. **Toy scale.** ≤ 11-dim synthetic inputs, ≤ 1.2k-param MLPs, searched-for equal-cost XOR cues.

### Post-hoc items (labeled)

- The competence-vs-direction decomposition of the `broken` arm's 0/20 (median column accuracy 0.504)
  was read off stored data after seeing the result. PREREG stated only the pass/fail interpretation.
- The slot-order/permutation observation was measured as a pre-stated S5 input (`order_matches_oracle`)
  but its *interpretation* as an operational requirement is post-hoc.
- Nothing else beyond PREREG.md was run.

---

## PART 2 — portfolio hygiene

Full deliverable: **`SELECTION_AUDIT.md`** (single markdown table file), numbers in
`hygiene_results.json`, script `hygiene.py`. Analysis-only; nothing outside this directory modified.
Headlines:

1. **Two-sided constant, corrected.** `b_k = E[max_j |N_j|]` computed exactly by quadrature and
   MC-verified (max err 8e-4): **b₄ = 1.4647** (REPORT3 quoted 1.47), against a₄ = 1.0294 — a factor
   **1.423**. Recomputing `r3-selection-inflation-curve`'s four `R_gap` values: 5.04 → **3.61**,
   2.50 → **1.79**, 5.20 → **3.73**, 7.55 → **5.41**. **No verdict flips**, but the two headline
   "resistance arrives at the endpoint" rows drop from ~5× to ~3.6–3.7×, just above the ROBUST
   boundary. The cell-level `R` values are **not** affected — their directions were pre-specified.
   Recommended closed form: `b_k ≈ Φ⁻¹((1+q)/2)`, `q = (k−0.375)/(k+0.25)`, within 1% for k ≥ 3.
2. **Connectivity modularity is family-dependent, as REPORT3 said.** Max-over-θ label-permutation
   null, 1000 perms, seed 0: all 1770 θ → **p = 0.2028**; the published window θ ∈ [0.0015, 0.090]
   (k = 1595) → **Q = 0.199, p = 0.0280**; θ with ≥ 60 edges (k = 1711) → **Q = 0.060, p = 0.1119**.
   The all-θ null is dominated by degenerate near-empty graphs. Restate as family-dependent, not
   "no community structure at any threshold" and not "withdrawn". The threshold-free primary
   (AUC = 0.725, node-perm p ≤ 1e-4) is **IMMUNE** and unchanged.
3. **routing-hybrid-knob T3: the demotion is right, the stated reason is wrong.** REPORT3 asks to
   demote T3 *because ρ̂ = 0.00 on the quality axis*. That ρ̂ is an artifact: Part C computed the
   **marginal** across-seed SE, while `r3-selection-inflation-curve/PREREG.md` Part B explicitly
   prescribes the **interaction** SE ("Use the interaction (item × cell) SE, not the marginal one").
   The hybrid-knob design is fully paired and **72.8% of the variance is a seed main effect**, so the
   marginal SE is 2× too large. With the prescribed SE, **ρ̂ = 0.680 and R = 2.49 (BORDERLINE, not
   SUSPECT)**, and the paired contrast the T3 criterion actually requires — retain(α = 0.5) −
   retain(α = 1) = **+0.0024 [+0.0008, +0.0040], 5/6 seeds, exact p = 0.042** — excludes zero, with
   `R_T3 = 4.52` after correcting for taking the best of 3 interior alphas.
   **T3's criterion is met.** What should be demoted is the wording: *one* interior setting is
   non-dominated, its quality gain is 0.24 retain points on an axis spanning 0.6 points, and the
   p = 0.042 does not survive Bonferroni over the three interior alphas (p = 0.13). Publish
   "statistically resolvable, practically negligible — take α = 1", which is what
   `routing-hybrid-knob/FINDINGS.md` already says in prose.
   Applying the same SE fix to the other three hybrid cells moves verdicts in **both** directions
   (p_waste 0.9 retain: BORDERLINE → SUSPECT), which is the signature of an honest correction.
4. **Selection audit for all 52 headline numbers** across REPORT.md / REPORT2.md / REPORT3.md, with
   `k, n, s, S, ρ̂, R`. **Every number in the switch line is k = 1 and therefore IMMUNE** — 22/30,
   23/30, 11/12, 20/20 at k = 3 and k = 4, and this round's 20/20 are per-seed success counts at a
   single prespecified configuration tested by Fisher exact, with no maximum over cells anywhere.
   The selection-exposed numbers are all outside the switch line (audit rows 15–24).
   **One target audited here for the first time:** `routing-lora-attack`'s "attacker-best −0.066"
   takes the max-AUC LoRA cell over a 6-cell grid per arm — k = 6, n = 6, s(interaction) = 0.0125,
   S = 0.222, ρ̂ = 0.997, **R = 18.1 (routed) / 7.4 (baseline) → ROBUST**. The grid is so
   over-dispersed relative to seed noise that cell selection contributes essentially nothing.
   Four rows of residual audit debt are listed in `SELECTION_AUDIT.md` §7; none touches the switch line.

---

## Files

- `PREREG.md` — hypotheses, arms, primary tests, decision rules, power discipline, deflations, budget
  and downscope order. Stamped 2026-07-25 11:09:33 EDT before any code was written.
- `experiment.py` — Part 1. Data/model/routing/thresholds/eval sets copied verbatim from
  `r3-multiway-switch/experiment.py`; `localize_L1` copied verbatim from
  `r2-routing-switch-learned/experiment.py`. Copied rather than imported so no bytecode is written
  into a read-only directory.
- `results.json` — per-seed localization (groups, margins, permuted-label null), per-seed metrics for
  all 6 cells, primary and secondary tests, S5 anchor.
- `hygiene.py`, `hygiene_results.json` — Part 2 (H1 constants, H2 modularity families, H3 T3,
  H4 marginal-vs-interaction SE [post-hoc], H5 lora attacker-best audit).
- `SELECTION_AUDIT.md` — the hygiene deliverable: constants table, three corrections, the 52-row
  selection audit, residual debt.

Read-only inputs: `r3-multiway-switch/`, `r2-routing-switch-learned/`, `r3-selection-inflation-curve/`,
`routing-hybrid-knob/`, `routing-lora-attack/`, `misgen-connectivity-graph/`,
`~/moral_framework_misgen/stats/results.json`.
