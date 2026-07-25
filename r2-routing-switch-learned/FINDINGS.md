# Worldview switch WITHOUT the oracle — round-2 findings

**One-line result.** Replacing round-1's oracle cue-localization with a *learned* localizer — a
data-only XOR-pair probe over input columns — makes the worldview just as selectable by post-hoc
ablation as the oracle: **learned-L1 23/30 vs oracle 22/30 vs standard baseline 0/30** (learned vs
control Fisher two-sided **p = 1.7e-10**; learned vs oracle **p = 1.0**). The oracle is removable in
this toy.

## What round-1 needed and what round-2 removes
Round-1 (`routing-worldview-select/`) routed the two worldview-consistent gradient components into
disjoint hidden columns by *occluding* the other cue's input columns — but it was **told** which
columns carry which cue (`CUE_A=[0,1]`, `CUE_B=[2,3]`). Round-2 keeps the entire routing/training
machinery byte-for-byte and changes only the source of those column sets: they are now **learned**,
with no knowledge of which inputs are cues.

## Preregistered arms (PREREG.md, stamped 2026-07-24 23:10:41 EDT, before any confirmatory run)
n = 30 seeds/arm (700-729). Scoring identical to round-1; because "which learned cue pair is the
true objective vs the proxy" is genuinely unidentifiable from agreeing data, all routed arms are
scored under the designated-OR-swapped mapping (exactly how round-1 scored its non-routed arms).

| arm | selection success | Wilson 95% CI | full-model align hi/lo/mid | localizer matched oracle |
|---|---|---|---|---|
| **learned_L1** (data probe) | **23/30 (77%)** | [0.591, 0.882] | 0 / 1 / 29 (hedged, mean 0.480) | **30/30 (100%)** |
| oracle (round-1 routed) | 22/30 (73%) | [0.556, 0.858] | 0 / 1 / 29 (hedged, mean 0.491) | — (given) |
| control (standard) | 0/30 | [0.000, 0.114] | 7 / 4 / 19 (seed lottery, mean 0.539) | — |
| learned_L2 (model coupling) | 17/30 (57%) | [0.392, 0.726] | 1 / 1 / 22 | 24/30 (80%) |

- **PRIMARY A — learned beats floor:** Fisher(learned_L1, control) = [[23,7],[0,30]], **p = 1.74e-10**.
  Decision rule (p < 0.001 AND >= 20/30): **PASS** (23/30). Hypothesis SUPPORTED.
- **PRIMARY B — learned ~= oracle:** Fisher(learned_L1, oracle) = [[23,7],[22,8]], **p = 1.0**.
  No detectable deficit; learned localization is equivalent to the oracle here (indeed +1 seed).
- Oracle reproduces round-1 exactly: **22/30 under designated-only scoring** (round-1's headline);
  22/30 either-mapping here (the two commit-directions never both fire, so the numbers coincide).

## The learned localizer (primary, L1) — why it works and that it is genuinely "learned"
For every pair of input columns (i,j), fit a 2-D logistic probe on features `[x_i, x_j, x_i*x_j]`
(the interaction term is what lets a linear probe read an XOR) to predict the label on the training
set; greedily take the top-2 *disjoint* pairs. On **all 30 seeds** this returns exactly the two
oracle cue pairs: probe accuracy **1.00** for {0,1} and {2,3}, next-best pair **<= 0.54**. It uses
the data's XOR structure, not any label about which cue is which — so it recovers the two cue
*locations* without the oracle. (It does not, and cannot, label which is the true objective vs the
proxy from agreeing data — that stays a separate, deploy-dependent question, which is why scoring
allows either mapping.)

Two things the exploration ruled out (`explore_localization.py`, exploratory / not confirmatory):
- **Input gradient-attribution is useless at the minimum** — the converged control model fits
  perfectly so `|dL/dx| ~= 0` on every dim. A naive attribution localizer would find nothing.
- **Single-dim label probes are at chance** (~0.51 every dim) — XOR hides in pairs, so you must
  probe pairs. This is the load-bearing design choice.

## Secondary: a model-based localizer (L2) and its honest failure mode
L2 reads the control model's first-layer weights: input coupling `C = |W1|^T|W1|` block-structures
cleanly into {0,1} and {2,3} (high within-pair, low cross-pair coupling), and a spectral 2-way split
recovers them. But L2 **fails on 6/30 seeds**, all by the same mechanism: on a *committed* control
model (one that latched onto a single cue), weight decay has **pruned the columns of the cue it
ignores**, leaving only 2 signal dims — so the localizer cannot see two cues and refuses (routing
does not run -> scored as a miss). Result: **17/30**, matching-oracle 24/30. Fisher(L2, oracle)
p = 0.28 (not significantly worse), Fisher(L2, control) p = 6.2e-7 (still far above floor). This is
a clean negative-control lesson: **a model-based localizer inherits the very seed-commitment the
method is trying to defer** — the committed model erases the evidence of the road not taken. The
data-only probe (L1) has no such dependence and is the right primary.

## Controls and anchors
- **Positive control (oracle):** 22/30 designated-only — reproduces round-1.
- **Negative control (control/standard):** 0/30; and full-align on the published deploy set matches
  round-1's control arm per-seed exactly, **max |diff| = 0.000000** (30 seeds).
- **Reproducibility spot-check** (`verify_rerun.py`): learned_L1, oracle, learned_L2 re-run from
  scratch for seeds 700-702 match `results.json` per-seed metrics, **max |diff| = 0.000000**.
- **Cost of selectability reproduces:** both routed arms turn the control's seed lottery (7 hi / 4 lo)
  into hedgers (0-1 hi, ~29 mid) — the un-ablated routed model must be selected before deployment.

## Interpretation
Round-1's headline caveat was that the router needed oracle knowledge of which input dims carry
each cue. In this toy that oracle is **not** needed: a principled, learned concept-localization step
(probe the data for the disjoint feature groups that jointly predict the label) recovers the routing
masks perfectly and yields identical selectability. The deferred-selection mechanism does not depend
on being handed the concept coordinates — provided the concepts are *localizable* from the data.

## Caveats / scope
- Same toy scale as round-1 (7-dim synthetic inputs, <=770-param MLPs, searched-for equal-cost XOR
  cues). Nothing here is evidence about frontier models.
- L1 works because the two cues live in **disjoint, discoverable input-column groups** and the
  concept is a low-order interaction (XOR pair). In a real model, cue directions are distributed and
  entangled in activation space, not axis-aligned in inputs; the input-probe route would not
  transfer directly. The honest scaled analog is: *can you learn a localization of two entangled
  concept subspaces well enough to route them?* — L2's failure previews why that is hard (committed
  representations hide the alternative).
- Still unidentifiable from agreeing data: **which** learned worldview is "aligned." Selectability !=
  knowing which switch position to pick; that needs deploy/anti-proxy data or an external label.
  Scoring reflects this by allowing either mapping (as round-1 did for its non-routed arms).
- n = 30 seeds/arm; single fixed deploy (n=4000, seed 50001) and held-out agreeing (n=4000, seed
  60001) sets; thresholds (0.8/0.2/0.9) inherited from round-1.
- The 7 learned_L1 misses are the same column-trainability failures as round-1's oracle misses (a
  width-4 column stuck at the partial-XOR optimum), NOT localization errors — L1 localized correctly
  on all 30. (Round-1 post-hoc showed width-8 columns remove these; not re-run here — would be
  post-hoc.)

## Post-hoc items (labeled)
- The L2 failure-mechanism narrative (committed models prune the ignored cue) was diagnosed *after*
  seeing the 6 failures; the localizer's refusal-on-<4-signal-dims was preregistered as a stated
  risk, but the specific "pruning on committed seeds" explanation is post-hoc (supported by the
  stored `signal_dims` and by exploratory seed 701 deploy-align 0.03).
- No other analyses beyond PREREG.md were run.

## Files (all under /Users/steven/research_sprint_20260724/r2-routing-switch-learned/)
- `PREREG.md` — hypothesis, arms, primary tests, decision rule (stamped before running).
- `experiment.py` — all 4 arms, both learned localizers, verbatim round-1 routing/data/scoring, tests.
- `results.json` — per-seed metrics, per-seed localization outputs, Fisher tests, replication anchor.
- `verify_rerun.py` — reproducibility spot-check (max |diff| = 0.000000).
- `explore_localization.py`, `explore_method2.py` — EXPLORATORY only (chose the localizers); NOT
  part of the confirmatory run.
- Read-only sources: `../routing-worldview-select/{experiment.py,results.json}` (round-1),
  `~/moral_framework_misgen/alignment.py` (data + regime).
