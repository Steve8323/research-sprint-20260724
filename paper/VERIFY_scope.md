# VERIFY_scope.md — hostile scope audit of PAPER.md

**Auditor role:** scope creep only — does any sentence generalize beyond the width / basis /
substrate / regime it was measured in?
**Inputs:** `PAPER.md` (1575 lines), `LEDGER.md` (§1 global scope, §4 nulls, §5 quarantine,
§6 forbidden sentences), `CLAIMS_TABLE.md`, `figures/NOTES.md`, and the underlying
`results.json` / `analysis.json` / `posthoc.json` for every number I re-derived.
**Date:** 2026-07-25

## VERDICT: **FAIL** (narrow)

Nothing needs retracting and no new experiment is required. But four reader-facing
statements — two of them in the abstract, one in the contributions list — assert more than the
data behind them, and the paper's own `CLAIMS_TABLE.md` and `figures/NOTES.md` already say so.
A paper whose selling point is scope discipline cannot ship a contradiction between its abstract
and its own figure note. Nine repairs below; S1–S4 are blocking.

---

## 1. What the paper gets right (so the FAIL is read correctly)

Checked and clean:

- **Frontier / scale.** Zero violations of FORBIDDEN 1–6. The scope banner (L3–13), the abstract's
  last sentence, §1.5, §2.7, and §5.5(1) all disclaim it. §5.6 explicitly says "nothing here
  motivates a frontier-model deployment proposal."
- **Safety framing.** Zero violations of FORBIDDEN 7–17. §5.2 is titled "Why this is not a safety
  result." L142 states it in the introduction. C54 is stated as "not *shown* to be a routing
  property" — correctly weaker than the ledger's own C54 headline, which itself trips FORBIDDEN 10.
- **Retracted claims.** R1–R14 appear only inside "What changed" boxes or §6. No retracted claim is
  stated as live anywhere.
- **Quarantined numbers (LEDGER §5).** All thirteen appear only inside explicit `[LEDGER GAP]`
  brackets that name them as not-on-disk and decline to cite them. This is exactly right.
  (One exception: item 8 below.)
- **Near-tautology.** Conceded at §1.5, §4.3.2, §4.6, §4.7 (BLIND "near-definitional"), §4.5
  ("imposed at init … not discovered"), §4.9 (positive control "near-tautological"), and
  §5.5(6). Train-mask == eval-mask is named.
- **Most nulls.** N1, N5, N6, N7, N8, N10, N11, N13, N14, N15, N16, N17, N19, N20, N22, N23 all
  ship with a bound, a power figure, or a prestated reading. §3.8 states the rule.
- **Round history.** §6 is in the body, not an appendix, and the closing paragraph names the
  single recurring defect. Not smoothed.

## 2. Findings, most damaging first

### S1 (BLOCKING) — the abstract asserts a competence premise the paper's own figure note refutes

`PAPER.md:31-32` — "in one 68.5k-parameter transformer **both halves are competent**, yet no
axis-aligned mask we built selects (0/30, Wilson upper 0.114)."

C17's "0 competence failures" is the **full-model** count (`n_competence_fail: 0`, `held_full_min:
1.0`). Recomputed from `r5-latent-switch-standard/results.json → arms['TF-A'].per_seed` with the
paper's own per-half gate (`learned.tA ≥ 0.9 AND learned.tB ≥ 0.9`):

- both halves competent: **16/30**, not 30/30
- conditional selectable: **0/16**, Wilson 95% upper **0.194** (not 0.114)
- the arm's own summary corroborates: `failure_clauses.either_half_incompetent = 14`

`figures/NOTES.md:17-18`, the Figure 2 caption (`PAPER.md:815-820`), and `CLAIMS_TABLE.md:44` and
`:154` all record this and say "the author must reconcile C17's wording with the per-half recount."
The reconciliation was never done in prose. The strong form still appears at **L31-32 (abstract),
L118-119, L166-169 (contribution 2), L322-325, L822 (section heading), L1291-1293, L1504-1505**.

L118-119 is the worst of them because it states its own refutation: "both data-chosen halves *are*
competent — zero competence failures **at the full-model level**". Full-model competence does not
establish per-half competence; that is the whole point of the per-half gate defined at §3.2 leg 3.

L1291-1293 compounds it by arm-mixing: "both halves are competent (12/12 under the units-only mask,
64 MLP units per worldview) and the axis-aligned mask still selects 0/30". The 12/12 is C18's
**units-only** family at n = 12 (whose own selection is 0/12); the 0/30 is C17's **head+MLP**
family. Two mask families and two sample sizes, presented as one measurement.

The dissociation survives — in the weaker form "0/16 conditional at Wilson upper 0.194, plus 0/12
at 12/12 competence in a second, post-hoc, n = 12 mask family." That is what should be written.

### S2 (BLOCKING) — width and basis are confounded in the MLP portfolio; the paper states a different, weaker confound

Contribution 2 (L166-169) and §4.3 (L839) claim "**two independent conditions**, dissociated."
Enumerating every cell in `r5-fair-floor-propagation/results.json`:

| basis | w = 4 cells (fair floor) | w = 8 cells (fair floor) |
|---|---|---|
| axis-aligned MLP | 2/30, 2/20, 1/20, 1/20 | 51/60, 17/20, 11/20, 9/20 |
| rotated / non-orthogonal MLP | 4/30, 2/30 | **none exist** |

`r3-entangled-localization/results.json → config` is `H = 8` (w = 4) for every entangled arm.
The w = 8 pool that carries the entire 8% → 73% lift (`posthoc.json → H1_width_gating`:
88/120 vs 12/150) is **four axis-aligned MLP cells and nothing else**.

So the paper's stated confound at L728-729 — "the pooled cells differ in k and **in input basis**" —
is true of the pooled set as a whole and **false of the w = 8 pool**, where basis does not vary at
all. Every cell with a non-trivial fair floor is axis-aligned; every entangled cell sits at the
width where the floor is near zero in all three bases.

Partial defense the paper should make and does not: at w = 4 the floor is 2/30 (axis), 4/30 (rot),
2/30 (mix) — basis has no measured effect *at that width*. What is missing is any entangled cell at
w ≥ 8. The consequence is that the only evidence separating the width condition from the basis
condition is the transformer null — one organism, a different substrate, one mask family, no
on-disk instrument check, and the competence premise of S1. "Two independent conditions" is not
established by the MLP data; it rests entirely on that one null.

### S3 (BLOCKING) — the largest surviving effect size in the paper has a demonstration in its numerator

The +1.000 transformer row appears in the **abstract** (L30-31), L98-99, the §4.2.3 effect table
(L757), and §5.1 (L1297) — always as an effect size, never with a demonstration label.

`r5-fair-floor-propagation/results.json → cells.cell_lm_organism` has `routed_rate: 1.0`,
`n_routed: 12` — that numerator is C20's **ungated 12/12**, which LEDGER C20 classifies
`DEMONSTRATION` with the instruction "**Report as an existence proof, never as an effect size**"
(each routing group is trained in isolation under exactly the ablation mask it is later evaluated
with). FORBIDDEN 24 forbids treating the transformer success rate as an effect size.

The paper honors this at §4.3.2 (L850-853, "a demonstration, not an effect size") and at §5.5(6),
then quietly violates it four times by pushing the derived D_new = +1.000 forward without the link.
Note also that the table uses the ungated 12/12; the competence-gated figure is 11/12.

This is a ledger-internal tension (C14 SOLID vs C20 DEMONSTRATION + FORBIDDEN 24) that the paper
inherits silently rather than resolving. It must be resolved in the text, in the direction the
ledger's own instruction points.

### S4 (BLOCKING) — a null stated as an absence, with its bound dropped

`PAPER.md:1049` (regime table) and `L132`, `L1310`: "**Nobody wins, by equivariance.**"

LEDGER C46's scope condition — "Only R1 meets the stricter δ = 0.15 chance-equivalence criterion;
the rest are bounded within ±0.19–0.24 of chance" — is carried in `CLAIMS_TABLE.md:73` and dropped
from PAPER.md entirely. Recomputed from `r5-optionality-case/analysis.json → S3_primary`: five of
the six arm×rule cells have `equivalent_to_chance_d15: false`; `rout_R2` is 38/60 with Wilson
[0.507, 0.744] and p = 0.052.

"Nobody wins" is precisely the null-as-absence that LEDGER rule 4 and §3.8 of the paper forbid.
It is also the only one of the four regime cells whose verdict word is not backed by a bound —
NO-RETRAIN carries its TOST, UNKNOWN-TARGET is a directional loss, BLIND carries three caveats.

"by equivariance" is a second, smaller problem: it asserts a principled impossibility. Nothing in
C46 or C31 licenses an equivariance argument for unlabelled *deployment rules*; C31's symmetry
result is about equivariant *localizers* on the generative family. Either derive it or delete it.

### S5 — a forbidden floor used as the comparison

`PAPER.md:886` — "against standard training, **38/50 vs 0/30**, p = 2.02e-12 [C32]".

FORBIDDEN 18 bars "any use of the 0/30, 0/20, or 1/60 floors as the comparison." The paper enforces
this correctly two places later (L1019-1020: "That end-to-end contrast uses the fixed-partition
floor and inherits the correction above") and at L946-947 ("the honest floor … is
routing-with-a-broken-split, not standard training's 0/30"). L886 gets no such marker. The
correction is small in magnitude here — C8 gives a w = 4 fair floor of ~2/30 — but the paper cannot
enforce the rule selectively.

### S6 — the abstract omits the single most consequential scope condition

The abstract concedes toy scale (L34-35: "Toy scale throughout; nothing transfers to frontier
models by default") — good, and it is the last sentence, where it will be read.

It does **not** concede G4: the data-chosen mask procedure **consumes the training-time cue
oracle**. A reader of the abstract alone concludes that 51/60 ordinary models are discoverably
switchable. They are not — not without knowing which input dimensions carry which cue. The banner
at L8-10 states it, but abstracts travel alone and the banner does not.

Same omission for the abstract's post-hoc content: 51/60 (the fair-floor rescoring was not the
run's preregistered primary) and +0.150 (C12: "post-hoc to the run's own primaries") both appear
unlabelled. LEDGER rule 3 requires the label "in the text where they are stated."

### S7 — C12's post-hoc status is never stated where the number is stated

The +0.150 is called "the number to quote whenever the paper says routing's effect size" at L94-95
and L759. C12's prereg column reads: "The comparison is post-hoc to the run's own primaries (which
were vs the fixed-mask floor)." That appears nowhere in PAPER.md. §4.2.1 discloses the adjacent
fact (the rescoring was not the primary) at L700-701, which is close but not the same statement,
and it is 55 lines away from the table row.

### S8 — an audit recomputation cited as if stored

`PAPER.md:1218` — "a strict within-label criterion is met by **2/46** (2/22 in round 3's own
archived data)". LEDGER §5 row 13: the 2/22 rescoring "is not stored as a key. **State it as an
audit recomputation.**" The paper states it as a fact in a parenthetical.

### S9 — contribution 7 sells a determinism check as reproduction

L192-196: "Not one primary statistic failed to reproduce; anchors sit at max|diff| = 0.0 across
rounds." §5.4 concedes the deflation honestly — L1443-1444: "the audits' own anchors are
re-executions of the stored procedures, **which is why they come back at max|diff| = 0.0**" — but
contribution 7, G6, and §3.9's opening do not. Re-executing a stored deterministic procedure on
stored seeds is expected to return 0.0; that is not evidence of measurement discipline, it is
evidence the code is deterministic. The genuinely load-bearing version is C5's distinction, which
the paper does carry at L515-516: "an independent re-derivation, **not** a fresh-seed replication."

Related, smaller: "six **independent** audits" (L192, L310, L622, L1432, and G6) is never defined.
Same sprint, same tooling, plausibly the same model. A reviewer will ask, and the paper leans on
that word in its meta-finding.

### Minor wording (non-blocking)

- L26 "The lottery reproduces at **two scales**" → two *substrates*. 154 params vs 68.5k is not a
  scale ladder, and the paper elsewhere insists calling the transformer "at scale" would be false.
- L53 "It is not an artifact of tiny MLPs" — a universal negative from one organism at n = 12.
  Defensible form: "it also occurs in a from-scratch transformer where the cue is a circuit."
- L1050 the BLIND table cell reads "Routing 60/60 at **every m ≥ 1**". The m = 1 cell is a known
  fixed-example artifact and FORBIDDEN 26 permits the win only "from two labels onward." The
  correction arrives 15 lines later (L1065-1067) and on the figure; the table is the quotable
  object and should carry the marker inline.
- L28-29 abstract: "Ordinary training already contains the switch" is a universal topic sentence
  for a claim that is width-conditional (2/30 at w = 4) and basis-conditional. Item (iii) does
  repair it, but the topic sentence is what gets quoted.
- L291-294 (§2.5) generalizes the capacity-matching lesson to "any evaluation comparing a
  sparsified defense to a small dense baseline." This one is fine — it is a deductive point about
  live parameter counts (154 vs 62 under jitter), not an empirical extrapolation. Noted so the
  next auditor does not re-flag it.

## 3. The single most damaging objection a hostile reviewer will raise

> **"Strip the three self-corrections and what is left? You claim two conditions, dissociated.
> The dissociation does not exist in your MLP data — every cell with a non-trivial fair floor is
> axis-aligned and every entangled cell sits at w = 4, so width and basis are perfectly confounded
> in the direction that matters, and you never say so. The only thing separating them is a null
> from one 68.5k transformer, in one mask family, with no positive control on disk, whose
> 'both halves are competent' premise is 16/30 by your own figure note — and whose conditional
> Wilson upper is 0.194, not the 0.114 you put in the abstract. So your one non-tautological
> positive contribution is a failed search you cannot show the instrument for. Everything else is
> either a demonstration by construction (train mask == eval mask, disjoint columns) or a
> restatement of the fact that you put the two cues in disjoint input coordinates and then chose
> the mask by input-cue sensitivity."

### Does the paper as written answer it?

**Partially — and not on the two limbs that matter.**

| Limb | Answered? |
|---|---|
| One organism, one mask family | **Yes**, repeatedly and unprompted: L124-125, L843-848, §5.5(4), §5.6 |
| No on-disk instrument check | **Yes**, four times, including in the contribution and the limitation |
| Rates 0.10–0.40 not excluded | **Yes**, N6 quoted at L123 and L845 |
| Demonstrations are not effect sizes | **Yes in principle** (§1.5, §5.5(6)) — **no in practice** for the +1.000 row (S3) |
| Competence premise is 16/30, conditional upper 0.194 | **No.** Known, recorded in three places, absent from all seven prose statements including the abstract (S1) |
| Width and basis are confounded; no entangled cell at w ≥ 8 | **No.** The paper states a *different* confound (w with H = k·w) and asserts basis diversity in the pooling that the w = 8 pool does not have (S2) |
| "Latent switch" may be a restatement of coordinate separation | **Not stated as an alternative anywhere.** The negative controls (random splits 0.42%, noise masks 0/10) do not rebut it — the deflationary account predicts both. C9's r = +0.764 is *consistent* with it. The transformer null is *predicted* by it |

The paper is one honest paragraph away from answering the whole objection, and that paragraph does
not exist. It is not a paragraph of new evidence — the deflationary reading is compatible with
everything measured, and saying so costs nothing except the word "independent" in contribution 2.

## 4. Required repairs

**Blocking**

1. **Abstract L31-32** → "in one 68.5k-parameter transformer both halves are competent on 16/30
   seeds and no axis-aligned mask we built selects on any of them (0/16 conditional, Wilson upper
   0.194; 0/30 unconditional)." Propagate to L118-119, L166-169, L322-325, L822 (heading),
   L1291-1293, L1504-1505. Stop merging C18's n = 12 units-only competence with C17's n = 30 null.
2. **L728-729** → replace "the pooled cells differ in k and in input basis" with "all four w = 8
   cells are axis-aligned MLPs; the three entangled cells all sit at w = 4. The portfolio contains
   no fair-floor measurement in an entangled basis at w ≥ 8, so width and basis are confounded in
   the pooled contrast." Add the same sentence to §5.5(7).
3. **Contribution 2 (L166-169) and L839** → drop "independent". The conditions are not shown to be
   separable in the MLP; the separation rests on one transformer null in a different substrate.
4. **L1049 / L132 / L1310** → "No rule beats chance at n = 60; only routed-R1 is chance-equivalent
   at δ = 0.15, the other five cells are bounded within ±0.19–0.24 of chance." Delete or derive
   "by equivariance."
5. **Abstract L30-31, L98-99, L757, L1297** → mark the +1.000 row: its routed arm is C20's
   demonstration (train mask == eval mask), so the cell is a floor measurement with a demonstration
   numerator, not an effect size. Prefer quoting the floor result (0/12 old = 0/12 fair, the only
   cell the correction does not move) over the derived D_new.

**Should fix before circulation**

6. **L886** → attach the fixed-partition marker used at L1019-1020, or drop the "vs standard
   training 0/30" comparison.
7. **Abstract** → add the oracle condition to item (ii) ("under a data-chosen mask *whose
   derivation consumes the training-time cue oracle*"), and label 51/60 / +0.150 post-hoc.
8. **L94-95 and L759** → add C12's prereg status: post-hoc to the run's own primaries.
9. **L1218** → "2/22 in round 3's archived data, as an audit recomputation (not stored)".
   **L192-196** → add §5.4's concession that the anchors are re-executions of stored procedures.
   Define "independent" on first use of "six independent audits" (L192).

**Cosmetic:** L26 "two scales" → "two substrates"; L53 universal negative; L1050 BLIND cell "from
m = 2"; L28-29 abstract topic sentence.

## 5. One sentence for the author

The paper is more careful than its own contributions list: every blocking finding above is a case
where the prose is weaker than the abstract, or where the figure note is more honest than the claim
it illustrates — so the fix is to promote what is already written down, not to measure anything new.
