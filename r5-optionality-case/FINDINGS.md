# When is optionality actually worth it? — round-5 findings

**Slug:** `r5-optionality-case` · **Status:** completed · **Date:** 2026-07-25
**PREREG.md** stamped `Sat Jul 25 13:55:32 EDT 2026`, before any code in this directory was
written. Confirmatory run 126.0 s, post-hoc extension 158.6 s, single process, 2 torch threads,
n = 60 seeds (700–759), everything paired by seed.

---

## Headline

**Optionality wins in exactly one of the three regimes, and it is not either of the two the
line has been implicitly assuming.** Routing buys nothing in the no-retrain regime and loses on
cost in the changing-target regime; it wins only when the *deployment-time* actor has lost the
cue oracle that routing consumed at *training* time.

| regime | routed model | best non-routing option (same resources) | verdict |
|---|---|---|---|
| **S1 no-retrain**, cue oracle at deployment, 8 labels | **60/60** | ORD + selection **59/60**; ORD + proxy-cue occlusion **60/60** | **TIE** (equivalent at δ=0.15) |
| **S1a** same, **zero** labels, designated intervention | **60/60** | prereg'd designation (`maskA_w8`) 53/60; best zero-label designation (`occB`) **60/60** | **TIE** under a fair designation |
| **S1b no-retrain, no cue oracle at deployment** | **60/60** from m = 1 | blind mask search over 200 splits: 34/60 at m=8, saturating **52/60** at m=16–32 | **ROUTING WINS** |
| **S2 three target changes** | 240/240 stages, **3.20 M** ex-fwd | ordinary + mask selection 234/240 at **1.60 M**; ordinary + 100-step fine-tune **240/240** at **1.84 M** | **ROUTING LOSES** (same success, 58 % of the cost) |
| **S3 unlabelled only** | every rule at chance | every rule at chance | **NOBODY WINS** (symmetry obstruction) |

One sentence for the paper: **routing is not a way to create a switch, it is a way to *record*
the training-time cue oracle in the architecture so that a deployment-time actor who no longer
has that oracle can still select; wherever the deployment-time actor keeps the oracle, or keeps
gradient access, a strictly cheaper non-routing option matches it exactly.**

---

## 0. Anchors (all pass)

| check | result | target |
|---|---|---|
| **A0** my sensitivity score vs `core.learned_masks` | max abs diff **0.0**, masks identical in 60/60 | exact |
| **A1** ordinary model, data-chosen mask, round-4 bidirectional criterion | **51/60** (either-mapping *and* designated) | r4's 51/60 |
| **A2** routed model, true partition | **60/60** | r4's 60/60 |
| **NEG** one random 8/8 mask, no selection | **2/60** [0.009, 0.114] | r4's 1/50 |

A1 reproduces round 4's corrected floor exactly, so this experiment is measuring against the
same denominator round 4 established, not a fresh one. Note A1 designated == either == 51: the
control's successes are *not* an artifact of generous either-mapping scoring.

## 1. PRIMARY 1 — S1 no-retrain (m = 8 labels, cue oracle, target TRUE)

Candidate sets: ORD {none, maskA_w8, maskB_w8, maskA_w4, maskB_w4, occB, occA};
ROUT the same 7 plus its true partition. Same selection rule, same labels, same evaluation set.

| arm | TRUE | PROXY |
|---|---|---|
| ROUT + select | **60/60** [0.940, 1.0] | **60/60** [0.940, 1.0] |
| ORD + select | **59/60** [0.911, 0.997] | 58/60 [0.886, 0.991] |

- **TRUE:** paired diff **+0.017, 95 % CI [0.000, 0.050]**, exact McNemar (1 discordant pair)
  **p = 1.0**. **TOST at δ = 0.15 PASSES** (90 % CI [0.000, 0.050]) — this is an *equivalence*
  result, not an underpowered null.
- **PROXY:** diff +0.033 [0.000, 0.083], McNemar p = 0.50, TOST at δ = 0.15 passes.
- **PREREGISTERED DECISION: NO DIFFERENCE (equivalent at δ = 0.15).** Routing is bounded above
  by +5 points in the no-retrain regime with the cue oracle.

**What carries the control.** Per-candidate marginals on the *ordinary* model (60 seeds):

| candidate | TRUE | PROXY | mean deploy alignment |
|---|---|---|---|
| `none` (as-is) | 2/60 | 1/60 | 0.492 |
| `maskA_w8` / `maskB_w8` | 53/60 | 55/60 | 0.955 / 0.045 |
| `maskA_w4` / `maskB_w4` | 37/60 | 32/60 | 0.871 / 0.152 |
| **`occB` / `occA`** (replace the *other* cue's input columns with noise) | **60/60** | **60/60** | **0.993 / 0.006** |

The strongest no-gradient intervention is not a mask at all: replacing the proxy cue's input
columns with noise at inference repairs **every** ordinary model, using zero labels, zero
gradient steps and the same cue oracle the mask needs. The one ORD miss at m = 8 is a *selection*
miss, not a model miss — a correct candidate existed in **60/60** seeds
(`ADDENDUM.S1_ord_exists_TRUE`) and the preregistered tie-break preferred `maskA_w8` (chosen 48
times) over `occB` (7 times).

**S1a (zero labels, designated candidate) — and a fair-control correction against myself.**
The prereg designated `maskA_w8` for ORD: 53/60 vs routed 60/60, diff **+0.117 [0.050, 0.200]**,
McNemar p = 0.0156 — a real gap *for that designation*. But **POST-HOC**: designating `occB`
instead — same information (cue oracle), same zero labels, same prereg'd candidate list — gives
**60/60, zero discordant pairs with the routed arm, p = 1.0**. The +12-point "routing advantage"
at m = 0 is entirely a consequence of which zero-label intervention the control is allowed to
use. This is round 4's lesson recurring inside round 5's own preregistration, and the honest
reading is the second one.

**S1c (TOY-ARTIFACT baseline).** Negating the logits: 2/60 (TRUE), 1/60 (PROXY) — it fails,
because ordinary models mostly do *not* commit at the whole-network level (mean deploy alignment
0.492; only 3/60 outside [0.2, 0.8]). The toy's exact-complement deployment labels did **not**
hand out a free win, so nothing in S1/S2 rests on that degeneracy.

## 2. S1-b — the regime where routing genuinely wins (no cue oracle at deployment)

ORD candidate set: {none} + **200 random 8/8 unit splits** (the control gets the search freedom;
routing gets only {none, trueA, trueB}). Selection by m labels, evaluation disjoint.

| m labels | ROUT blind TRUE | ORD blind TRUE | paired diff (95 % CI) |
|---|---|---|---|
| 1 | 60/60 | 4/60 | +0.933 [0.867, 0.983] |
| 4 | 60/60 | 13/60 | +0.783 [0.683, 0.883] |
| 8 | 60/60 | 34/60 | +0.433 [0.317, 0.567] |

All three McNemar p < 3e-8. **POST-HOC label-exchange extension** (m up to the full 32-example
pool, random tie-break):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ORD blind TRUE | 2/60 | 4/60 | 13/60 | 31/60 | 51/60 | **51/60** |
| ORD blind PROXY | 0/60 | 6/60 | 17/60 | 37/60 | 47/60 | **54/60** |
| ROUT blind TRUE | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 |

The blind control **saturates at ~52/60 and does not catch routing within 32 labels**. The
failure is identification, not existence: a *correct* random mask exists in **57/60** seeds
(mean 6.4 good masks out of 200). So in the blind regime routing's contribution is real and
worth roughly "more than 32 labelled deployment examples".

## 3. PRIMARY 2 — S2 unknown target, sequence [TRUE, PROXY, TRUE, PROXY] (3 changes)

| arm | stage successes | rate [Wilson 95 %] | grad cost/change | **total 3-change cost** |
|---|---|---|---|---|
| ROUT + mask selection | 240/240 | 1.000 [0.984, 1.0] | 0 | **3.20 M** ex-fwd |
| **ORD + mask selection** | 234/240 | 0.975 [0.947, 0.989] | **0** | **1.60 M** |
| ORD + FT-MIX k = 20 | 218/240 | 0.908 [0.865, 0.939] | 16.2 k | 1.648 M |
| **ORD + FT-MIX k = 100** | **240/240** | 1.000 [0.984, 1.0] | 80.8 k | **1.842 M** |
| ORD + FT-PURE k = 100 (8 examples only) | 48/240 | 0.200 | 0.8 k | 1.602 M |

- **k\* (natural reading, "reaches the ROUT rate") = 100**, at **1.84 M vs routing's 3.20 M**
  example-forwards — 58 % of the cost, **zero discordant seeds** vs the routed arm (McNemar
  p = 1.0). **PREREGISTERED DECISION: ROUTING LOSES S2.**
- **Disclosed prereg defect.** The prereg also wrote k\* as "smallest k whose Wilson-95 lower
  bound ≥ the ROUT point estimate". With the routed arm at exactly 1.000 that rule is
  *unsatisfiable* (a Wilson LB is never 1), and taken literally it returns "ROUTING WINS S2".
  Both readings are recorded in `analysis.json`
  (`DECISION_literal_wilsonLB_rule` vs `DECISION`); the natural reading is used because it is
  the conservative one — it can only make routing lose — and because k = 100 matches the routed
  arm exactly, seed for seed.
- **The cheapest arm is not routing and not fine-tuning.** Ordinary training plus a data-chosen
  ablation mask switches target with **zero gradient steps** at 97.5 % over 240 stages and half
  routing's training cost. Optionality does not require routing.
- FT-PURE (fine-tune on the 8 deployment examples alone) is the one thing that fails: 48/240 and
  held-out accuracy collapses to 0.624 by the last stage. The mixed set is what makes cheap
  switching work.
- **Prereg'd secondary FALSIFIED (in the interesting direction):** after the full 4-stage
  fine-tuning sequence at k = 20, **44/60 (73 %, [0.610, 0.829])** of the models are *still*
  switchable under a data-chosen mask. Round 4 reported 0/30 switchable for augmented models —
  but scored under the fixed arbitrary partition, i.e. the very artifact round 4 itself
  identified for the floor. **"Fine-tuning commits the model and destroys the option" is not
  established**; with a data-chosen mask most fine-tuned models retain both worldviews.

## 4. PRIMARY 3 — S3 unlabelled only (m = 0)

| rule | routed | ordinary |
|---|---|---|
| R1 coin | 29/60 (0.483) p = 0.90 | 25/60 (0.417) p = 0.25 |
| R2 self-consistency with the un-ablated model | 38/60 (0.633) **p = 0.052** | 23/60 (0.383) p = 0.093 |
| R3 mean max-softmax confidence | 26/60 (0.433) p = 0.37 | 28/60 (0.467) p = 0.70 |
| R4 entropy-min self-training (100 steps, gradient, no labels) | — | **8/60 (0.133)** |

- **PREREGISTERED DECISION: no unlabelled rule beats chance** (Bonferroni α = 0.0042 over the
  4 rules; the largest, routed R2 at 0.633, gives p = 0.052 across 6 rule×arm tests).
- **There is a symmetry argument for why this must be so**, and it was preregistered: the two
  cue blocks are generated by identical calls to `xor_block`, so swapping input columns [0,1]
  with [2,3] maps both the training and the deployment distribution to themselves with the
  worldview label swapped. R2 and R3 are equivariant under that swap, hence exactly at chance
  for the *position* question. Only R1 meets the strict equivalence bound (δ = 0.15,
  CI [0.362, 0.607]); the others are bounded within roughly ±0.19–0.24 of chance at n = 60 and
  are reported as "not distinguishable", not as zero.
- **The only better use of unlabelled data we found is worse than nothing.** Entropy
  minimization on 2000 unlabelled deployment inputs leaves mean deploy alignment at 0.4987
  (0.4924 before): it sharpens each example without committing the network to either worldview,
  and success drops to 8/60.
- **What optionality *is*, separated from exercising it:** a correct candidate exists in
  **60/60** routed and **53/60** ordinary seeds (mask pair only; **60/60** with the full
  cue-oracle candidate set). The option is present in ordinary models too — it just cannot be
  cashed without ≥ 1 label, for either arm.

## 5. Cost ledger (three currencies, never collapsed)

| arm | gradient (example-forwards) | labels | access needed at deployment |
|---|---|---|---|
| ordinary training | 1.60 M | — | — |
| **routing premium** | **+1.60 M (+100 %)** | — | decision to route must precede training |
| mask derivation | 0 | 0 | cue oracle + 4000 unlabelled train-mode inputs |
| input-occlusion repair (`occB`) | 0 | 0 | cue oracle |
| blind mask search (200 splits) | 0 | 8–32 | nothing |
| FT-MIX per target change | 80.8 k at k = 100 | 8 | gradient access + the training set |

The routed model also pays a standing price: it **never commits** — 0/60 routed models are
usable un-ablated (mean deploy alignment 0.505) versus 3/60 ordinary. Optionality is not free
even before you count the compute.

## 6. Honest scoping — and the caveat that matters most

The regime where routing wins (S1-b) is the one where the deployment-time actor **lacks the cue
oracle**. That is arguably the *realistic* regime: in a real model the two "cues" are not
separate input columns, so `occB` — the intervention that carries the control everywhere else —
would not exist, and neither would the sensitivity mask. Two things must be said in the same
breath:

1. **The win is real but it is a transport result, not a creation result.** Routing needed the
   cue oracle at *training* time to build the columns. It converts training-time interpretability
   into a deployment-time capability. It does not manufacture knowledge nobody had.
2. **The blind control's ceiling is set by our candidate family.** 200 of the 12 870 possible 8/8
   splits were searched; a correct one existed in 57/60 seeds. A larger family raises the
   existence ceiling but also makes m-label selection easier to overfit — we did not measure that
   trade-off, so "more than 32 labels" is a statement about *this* family.

Other caveats: one 7-input toy, one architecture (H = 16), exactly two worldviews, exactly two
targets in the S2 sequence, and a cue oracle available at training in every arm. The S1-b m = 1
PROXY cell was a degenerate tie-break (0/60 under the prereg's order-based tie-break, 32/60 ≈
chance under a random tie-break — the post-hoc check); the same tie-break costs the ORD arm its
one S1 miss. Fine-tuning reuses the same 8 labelled inputs at every stage (relabelled for the new
target); they are disjoint from all evaluation data.

## 7. Deviations, downscopes, and what is post-hoc

- **No downscopes.** The full preregistered grid ran (n = 60, m ∈ {1,4,8} blind, k ∈ {0,5,20,100}
  × {FT-MIX, FT-PURE}, all 4 stages, all 4 S3 rules). Runtime 126 s + 159 s post-hoc, inside the
  ~15-minute budget.
- **Disclosed prereg defect:** the S2 Wilson-LB k\* rule is unsatisfiable at rate 1.0 (§3); both
  readings reported.
- **POST-HOC (labelled everywhere they appear):** (i) the m ∈ {2,16,32} blind label-exchange
  extension and the random tie-break robustness check (`posthoc.py`, `posthoc.json`);
  (ii) designating `occB` instead of `maskA_w8` as the zero-label control in S1a; (iii) the
  per-candidate marginals were preregistered as reportable, their use to explain *which*
  candidate carries the control is post-hoc interpretation.
- **Pilot disclosure** (also in PREREG §preamble): one seed's alignment numbers were computed
  before the prereg was written and are reported there.

## 8. What this does to the line

1. **The round-4 provisional claim survives, and hardens.** Disagreeing data spent on ordinary
   training beats data spent on routing — and now: *the same is true of compute, and of
   deployment-time interventions.* Routing's honest niche is one regime, not three.
2. **"Routing buys optionality" needs a qualifier.** Ordinary training already carries the
   option (53–60/60 of seeds), fine-tuned models mostly *keep* the option (44/60), and the
   option cannot be exercised without at least one label in any arm. What routing uniquely buys
   is a **pre-labelled** option, and that only matters to an actor who cannot re-derive the
   labelling.
3. **A publishable scoping sentence, which is what the brief asked for if the data said so:**
   *optionality is worth paying for exactly when the deployment-time actor has neither gradient
   access nor the interpretability access that routing consumed at training time; in every other
   cell of this toy a strictly cheaper non-routing option matches it.*
