# Exchange rate: how much supervision buys localization? — round-4 findings

**One-line result.** In the rotated (entangled-basis) ALIGN toy where unsupervised localization is
obstructed, **4 labelled disagreeing examples — 0.5% of the training set — buy a working worldview
switch** (selectable 21/30, Wilson-95 [0.52, 0.83]; localization 25/30) up from a 13/30 unsupervised
floor, with both preregistered trend tests at the permutation floor (p < 5e-5). **But the same 4
examples spent directly — just added to ordinary training — align 24/30 models outright, and 8
examples align 30/30 while the switch is still at 22/30 (Fisher p = 0.0046 in favour of plain data).**
Routing never beats data-augmentation at any price we tested and is beaten from m = 8 up. The
symmetry-breaking information that is cheap enough to build the switch is the same information that
makes the switch unnecessary.

## Setting (all arms)
`rot` basis: every input passed through a per-seed Haar-orthogonal 7x7 T, the regime where round 3
showed rotation-equivariant localization collapses to a coin flip. Data / model / routing /
occlusion / scoring imported verbatim and read-only from round 3's `core.py`. n = 30 seeds
(700-729) per level, m grid {0,1,2,4,8,16,32} preregistered before running. Supervision examples are
fresh per seed (gen seed 80000+seed) and disjoint from the evaluation deploy set (gen seed 50001).

## PRIMARY — the exchange rate curve (source (a): m labelled deployment examples)

| m | localization (lands TRUE split) | Wilson-95 | selectability | Wilson-95 | also picks position |
|---|---|---|---|---|---|
| 0 (unsupervised floor) | 7/30 | [0.12, 0.41] | 13/30 | [0.27, 0.61] | 8/30 |
| 1 | 14/30 | [0.30, 0.64] | 14/30 | [0.30, 0.64] | 13/30 |
| 2 | 21/30 | [0.52, 0.83] | 16/30 | [0.36, 0.70] | 16/30 |
| **4** | **25/30** | [0.66, 0.93] | **21/30** | **[0.52, 0.83]** | 21/30 |
| 8 | 24/30 | [0.63, 0.91] | 22/30 | [0.56, 0.86] | 22/30 |
| 16 | 28/30 | [0.79, 0.98] | 24/30 | [0.63, 0.91] | 24/30 |
| 32 | 27/30 | [0.74, 0.97] | 24/30 | [0.63, 0.91] | 24/30 |
| *oracle ceiling* | *(given)* | — | *26/30* | [0.70, 0.95] | 26/30 |
| *standard training* | — | — | *0/30* | [0.00, 0.11] | 0/30 |

- **PRIMARY 1 (localization vs m)** Cochran-Armitage trend, scores log2(1+m), 20,000 permutations:
  **z = 6.69, p < 5e-5** (0/20,000 permutations reached the observed z).
- **PRIMARY 2 (selectability vs m):** **z = 4.17, p < 5e-5**.
- **DELIVERABLE k\*:** preregistered rule (smallest m whose selectability Wilson-95 lower bound
  >= 0.50) gives **k\*_sel = 4**. Localization bar (LB >= 0.75) gives **k\*_loc = 16**.
- **Preregistered DECISION: SUPPORTED** (both trends p<0.05 and k\*_sel <= 8).
- Position identification comes free with labels: from m >= 2 the *designated* count equals the
  either-mapping count (16/16, 21/21, 22/22, 24/24) — labelled deployment examples do not just split
  the concepts, they say which side is the true objective. At m = 0 only 8 of 13 successes are in the
  designated direction, i.e. a coin flip, as round 2 predicted.
- Per-level Fisher vs the m=0 floor (SECONDARY by prereg, Bonferroni alpha = 0.0083): localization
  significant from m=2 (p = 6.4e-4) on; selectability only from m=16 (p = 0.0073). The trend test
  is the primary precisely because per-level Fishers at n=30 have power 0.24 against a 28-point jump.

## The comparison that decides the paper claim (SECONDARY S1)
Spend the *same* m examples on ordinary training instead of on a localizer + router:

| m | switch (routing, selectable) | just-train-on-them (aligned) | mean deploy alignment | switchable after? | Fisher |
|---|---|---|---|---|---|
| 0 | 13/30 | 7/30 (seed lottery) | 0.481 | 0/30 | — |
| 1 | 14/30 | 14/30 | 0.666 | 0/30 | p = 1.0 |
| 2 | 16/30 | 16/30 | 0.750 | 0/30 | p = 1.0 |
| 4 | 21/30 | 24/30 | 0.886 | 0/30 | p = 0.55 |
| 8 | 22/30 | **30/30** | 0.961 | 0/30 | **p = 0.0046** |
| 32 | 24/30 | **30/30** | 0.992 | 0/30 | **p = 0.024** |

- Augmentation trend z = 5.74, p < 5e-5. The m examples enter at their natural frequency m/(N+m),
  so m=4 is exactly the f = 0.5% dose round 1 identified as prevention-sufficient — and it is:
  24/30 aligned.
- **Powered claim:** routing is *worse* than augmentation at m = 8 and m = 32.
- **Null discipline:** at m = 1, 2, 4 the two are not distinguishable, but this is NOT an equivalence
  claim — TOST at the preregistered margin delta = 0.20 fails (p_TOST = 0.060, 0.060, 0.184; 90% CIs
  [-0.212, 0.212], [-0.212, 0.212], [-0.283, 0.083]) and Fisher power against a 20-point difference
  is only 0.26-0.38. All we can say is that routing never *beat* augmentation at any m.
- The one thing augmentation never delivers: **0/30 augmented models are switchable at any m** — they
  commit. And the routed models pay for their switch by refusing to commit: their un-ablated deploy
  alignment stays ~0.52-0.55 (2/30 hi, ~1/30 lo) versus 0.99 for augmentation at m=32. The routed
  model must be *selected* before deployment; the augmented model is simply right.

## SECONDARY S2/S3 — the other symmetry-breaking sources (localization only, n=30)

| source | m / rho | lands true | Wilson-95 | verdict |
|---|---|---|---|---|
| (a') **unlabelled** deployment inputs | 0 / 1 / 4 / 8 / 32 | 7 / **17** / 16 / 16 / **17** | [0.39, 0.73] at m=1 | jumps at m=1 then **saturates at ~57%** |
| (b) external concept label (latent cue-B bit) | 4 / 32 | 9 / 11 | [0.17,0.48] / [0.22,0.54] | **buys almost nothing** |
| (c) cue-strength asymmetry alone (m=0) | 1.0 / 2.05 | 7 / 14 | [0.30, 0.64] at 2.05 | doubles the floor, still a coin flip (binomial vs 0.5, p = 0.86) |
| NEGATIVE CONTROL: m agreeing examples, same objective | 4 | 8 | [0.14, 0.44] | **no purchase** (vs 25/30 for disagreeing, p = 2e-5) |

- (a') is the interesting one, because unlabelled deployment inputs **cannot** be used for
  augmentation — no labels, nothing to train on — so this is the only channel where routing could buy
  something data cannot. Trend z = 2.07, p = 0.020; but it plateaus: m=1 and m=32 give the identical
  17/30, and the preregistered contingency (routed follow-up if lands-true >= 0.75 at m <= 8) **did
  not trigger**. Mere knowledge that two readouts must disagree somewhere localizes about half the
  seeds and no more.
- (b) and (c) nulls are honest but weak: power against a 20-point rise over the floor is 0.29, and
  TOST at delta = 0.20 does not establish equivalence to the floor (p_TOST 0.12 / 0.28 for concept;
  0.61 for rho = 2.05). Read them as "no detected purchase at n=30", not "zero purchase". Note (c)
  is consistent with round 3's correction: identifiability is restored at rho = 2.05 in principle,
  and our localizer, like round 3's ICA, does not exploit it.
- The negative control is the load-bearing one: **m more examples do nothing; m more *disagreeing*
  examples do everything** (8/30 vs 25/30 at m=4, p = 2e-5). Agreeing examples actively degrade the
  localizer (lands-neither 15/30 vs 3/30).

## Controls, anchors and reproducibility (all pass)
- **Positive control** `rot_oracle` **26/30** — reproduces round 3 exactly.
- **Negative control** `rot_control` **0/30** — reproduces round 3 exactly.
- **Replication anchor** `unsup_ica` **16/30** (lands true 14/30) — reproduces round 3's rot_ica
  exactly, so the m=0 floor is not a straw localizer: floor 13/30 vs ICA 16/30, Fisher p = 0.61.
- The latent-bit data generator is **bit-identical** to `core.gen_align` (max |diff| = 0.0).
- `verify_rerun.py`: 9 runs (m in {0,4,32} x seeds 700-702) re-run from scratch, **max |diff| = 0.0**,
  all success flags match.
- L-SUP-D at m=0 is **bit-identical** to round 3's `localize_dict` (max |diff| = 0.0 on Vs).

## Is the switch at the oracle ceiling? (SECONDARY S5 — not established)
Best level m=16 gives 24/30 vs oracle 26/30, Fisher p = 0.73 — but **TOST at delta = 0.20 does not
establish equivalence** (p_TOST = 0.082, 90% CI [-0.224, 0.091]). At k\*=4, 21/30 vs 26/30, p = 0.21,
TOST p = 0.37, CI [-0.338, 0.005]. Supervised localization gets *close* to the oracle; we cannot
claim it reaches it at n=30.

## POST-HOC (labelled post-hoc)
- Success conditional on which orbit member the localizer landed on, pooled over all m: **true
  121/146 (83%, [0.76, 0.88]) vs entangled 9/36 (25%) vs neither 4/28 (14%)**, Fisher true-vs-
  entangled p = 7.4e-11. This replicates round 3's post-hoc conditional (87% vs 26%) on new data and
  explains the whole curve: supervision does exactly one thing — it moves seeds from the entangled
  orbit member onto the true one — and the ~17% residual failure rate on correctly-localized seeds is
  the known width-4 column-trainability failure from rounds 1-2, not a localization error.
- The gap between the localization curve and the selectability curve (28/30 vs 24/30 at m=16) is
  entirely that residual: 0.83 x 28 = 23.2 ~ 24.

## PREREG deviations, downscopes and honesty notes
- **AMENDMENT 1** (stamped in PREREG.md before any confirmatory run) swapped the primary localizer
  family from round 3's `dict` (full 7-D atom search) to the same objective searched inside the
  unsupervised signal subspace, because `dict` does not land on either orbit member — round 3 had
  already reported lands-neither 21/30 *before* this round, and a 4-seed fidelity-only smoke test
  (the same protocol round 3 used to pick its localizers) reproduced it. The originally prereg'd
  family was kept and run: at m=8 L-SUP-D lands true **10/30 with lands-neither 20/30** versus
  L-SUP-S's 24/30 (p = 5.7e-4), confirming the amendment's premise. No selectability, and no
  localization on more than 4 seeds, was observed before the amendment was written.
- **Downscopes** (permitted and announced in PREREG, executed in flight because the shared machine
  ran ~3x slower than the single-process measurement): `augment` dropped m=16; secondary sources
  thinned to unlab {0,1,4,8,32}, concept {4,32}, agree-control {4}, asym {1.0, 2.05}, L-SUP-D {8}.
  The PRIMARY grid was untouched. Consequence: the (b)/(c) curves are 2-point brackets, not curves.
- **Budget overrun, disclosed:** primary run 1122 s + secondary 157 s + verification ~20 s
  = **~21 min wall** on a 10-core machine shared with 5 other agents (single process, 2 torch
  threads), against a ~15 min target. The overrun is contention, not scope: the same primary took
  ~1.0 s/routed-run when measured alone.
- Nothing else was run. Everything above is either in PREREG.md + AMENDMENT 1 or explicitly labelled
  post-hoc.

## Interpretation — what this does to the line
1. **The positive.** The round-3 obstruction is cheap to break *if you have deployment labels*: 2
   examples for localization, 4 for a switch that clears a 50% floor, 16 to approach the oracle. The
   currency is astonishingly small — 0.5% of the training set — and the same labels also resolve
   round 2's residual unidentifiability (which switch position is aligned).
2. **The negative that matters more.** The exchange rate is the *wrong* thing to want. Labelled
   disagreeing data is fungible: spent on the localizer it yields a switch (21/30 at m=4); spent on
   the optimizer it yields the answer (24/30 at m=4, 30/30 at m=8). Deferred selection is only worth
   paying for if the symmetry-breaking information arrives in a form you *cannot* train on — and the
   one such channel we priced, unlabelled deployment inputs, saturates at 17/30 (57%), never reaching
   the bar. **A paper cannot claim routing buys alignment cheaply; it can only claim routing buys
   optionality, and must then show optionality is worth more than the fix the same data provides.**
3. **What would rescue the pitch** (not tested here): a setting where the disagreeing examples are
   available only *after* the model is trained and retraining is expensive; or where the deployment
   label is *contested* (multiple stakeholders, so you want the switch rather than a commitment); or
   an unlabelled/self-supervised channel with a better localizer than our disagreement objective —
   the saturation at 57% looks like a defect of that objective, not an information bound, since 32
   unlabelled examples do no better than 1.

## Files (all under /Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange/)
- `PREREG.md` — hypothesis, arms, primary tests, decision rule, power sim, + AMENDMENT 1 (both stamped).
- `sup.py` — L-SUP-S (primary) and L-SUP-D (secondary) localizers, latent-bit generator, augmentation
  training, trend / TOST / power helpers. Round-3 `core.py` imported read-only.
- `experiment.py` -> `results.json` — primary run (7 sup_dep levels, oracle, control, ICA, 5 augment).
- `sources.py` -> `sources.json` — secondary sources (a'), (b), (c) and the negative control.
- `analyze.py` -> `analysis.json` — all preregistered tests, secondaries, controls, post-hoc block.
- `verify_rerun.py` -> `verify_rerun.json` — reproducibility spot-check (max |diff| = 0.0).
- `explore_sub.py` — EXPLORATORY 4-seed fidelity probe that motivated AMENDMENT 1; not confirmatory.
- `run_primary.log`, `run_sources.log` — raw run logs.
