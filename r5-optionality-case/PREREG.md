# PREREG — When is optionality actually worth it? (round 5)

Stamped, before any confirmatory code was written or run:

```
Sat Jul 25 13:55:32 EDT 2026
```

**Full disclosure of everything already executed at stamp time.** A timing pilot on the single
seed 700 that (a) trained one `floor_control` and one `ceiling_disjoint_w8` model with the
read-only round-4 harness, (b) printed wall-clock (0.99 s / 0.77 s), and (c) — beyond pure
timing — printed four accuracies for that one seed: ordinary model un-ablated deploy alignment
0.601, ordinary model under the data-chosen (`core.learned_masks`) mask 0.998 / 0.002, routed
model under the true partition 0.998 / 0.002. This is one seed of a quantity round 4 already
published (`r4-break-tautology`: floor under data-chosen mask 51/60). It is disclosed here
because it is not zero information about H1 below; it cannot have selected the hypothesis,
which is fixed by the round-4 report and the round-5 brief. No other quantity, no other seed,
and nothing about scenarios 2 or 3 was computed before this file was written.

---

## 1. The question

Round 4 (`r4-symmetry-breaking-exchange`, UNAUDITED) reported that m labelled disagreeing
deployment examples spent on *ordinary training* align 30/30 models at m = 8, while the same m
spent on a localizer + router yield a switch selectable in only 22/30. If that holds, gradient
routing does not buy alignment; it buys **optionality** — the ability to choose the worldview
after training. This experiment asks whether optionality has a regime where it genuinely wins,
i.e. where the routed model reaches the deployment target at lower total cost than any
non-routing alternative given the *same* resources.

Three deployment regimes, one per preregistered primary:

* **S1 NO-RETRAIN** — the model exists, gradient updates are forbidden. Post-hoc selection is
  all you have. Is a *routed* model better than an *ordinarily trained* one under the same
  no-gradient toolbox?
* **S2 UNKNOWN-TARGET** — you learn which worldview you want only at deployment, and it changes
  >= 3 times. Is switching-by-ablation cheaper than switching-by-fine-tuning?
* **S3 UNLABELLED-ONLY** — you have deployment inputs, no labels. Round 4 says routing saturates
  at 17/30 here. Verify, and probe whether unlabelled data has any better use.

**FAIR-CONTROL DISCIPLINE (round 4's lesson, binding on this experiment).** Every control gets
the *same* deployment-time search freedom as the routed arm: the same candidate set size where
possible, the same number of labelled selection examples, and the same selection rule. Where the
routed arm gets a candidate the control cannot construct (the true routing partition), that
asymmetry is stated explicitly and a separate blind comparison is run in which the control is
allowed to search a large family of masks.

## 2. Setup (inherited verbatim, read-only)

`sys.path` import of `/Users/steven/research_sprint_20260724/r4-break-tautology/core.py`
(`sys.dont_write_bytecode = True`; nothing is written outside this directory). ALIGN toy:
INPUT = 7 (cue A = true objective cols [0,1], cue B = proxy cols [2,3], cols [4,5,6] noise),
MARGIN 2.0, NOISE 0.6, N = 800, full-batch Adam LR 5e-3, WD 2e-3, EPOCHS 2000, H = 16.
Training data: cues agree. Deployment data: cues disagree (proxy label a = 1 - v).

Two base models per seed, matched architecture:

* `ORD`  = `core.train_arm(seed, "floor_control")` — ordinary training. Cost 2000 steps x 1 loss
  eval on 800 examples = **1.6 M example-forwards**.
* `ROUT` = `core.train_arm(seed, "ceiling_disjoint_w8")` — routed (round-4 ceiling arm, 60/60).
  Cost 2000 steps x 2 loss evals = **3.2 M example-forwards**, i.e. routing's upfront premium
  is +1.6 M (+100%).

**n = 60 seeds, 700-759** (matching round 4's floor/ceiling n, all comparisons paired by seed).

Evaluation sets, fixed and disjoint: deploy-eval `gen_align(4000, 50001, "deploy")`; held-out
competence `gen_align(4000, 60001, "train")`; selection pool `gen_align(32, 50777, "deploy")`
(the first m examples of it are the labelled budget; never used for evaluation).

**Targets.** `TRUE` = act on cue A; `PROXY` = act on cue B. SUCCESS(target = TRUE) iff
deploy alignment > 0.8 and held-out accuracy > 0.9. SUCCESS(target = PROXY) iff deploy alignment
< 0.2 and held-out accuracy > 0.9. (Thresholds inherited: ALIGN_HI 0.8, ALIGN_LO 0.2,
TRAIN_OK 0.9.)

**No-gradient interventions** (all forward-only, available to whichever arm can construct them):

| id | intervention | needs |
|---|---|---|
| `none` | use the model as-is | — |
| `maskA_w8`, `maskB_w8` | zero all but the top-8 units by (cue-A minus cue-B) activation sensitivity, either sign | cue oracle |
| `maskA_w4`, `maskB_w4` | same at width 4 | cue oracle |
| `occB`, `occA` | at inference, replace the *other* cue's input columns with noise | cue oracle |
| `trueA`, `trueB` | zero the other routed unit group | ROUT only (free from training) |
| `rand_i`, i < 200 | a random 8/8 unit split, both signs | nothing |
| `flip` | negate the logits | nothing (TOY ARTIFACT, see S1-c) |

`maskA_w8` reuses `core.learned_masks` unchanged; an anchor check asserts my re-implementation
of the sensitivity score reproduces it bit-exactly.

**Selection rule (identical for every arm).** Given a labelled budget of m deployment examples
and a candidate set C, pick argmax over C of accuracy-toward-target on those m examples; ties
broken by the candidate's fixed prereg order (as listed above). Evaluation is on the disjoint
4000-example deploy set. m = 0 means no selection is possible: a single *designated* candidate
must be named in advance (`maskA_w8` for ORD, `trueA` for ROUT, for target TRUE; the B versions
for PROXY).

## 3. PRIMARY 1 (S1, no-retrain)

**Candidate set (cue oracle available at deployment):**
ORD: {none, maskA_w8, maskB_w8, maskA_w4, maskB_w4, occB, occA} (7).
ROUT: the same 7 plus {trueA, trueB} (9).
Labelled budget m = 8. Target = TRUE.

**H1.** Routing beats ordinary training in the no-retrain regime: SUCCESS(ROUT) > SUCCESS(ORD).
**H0.** They are the same; routing buys nothing that post-hoc mask search on an ordinary model
does not already give.

**Test.** Exact McNemar (binomial on discordant pairs, two-sided) on the 60 paired seeds; 95% CI
on the paired difference by 10,000-resample seed bootstrap (percentile).
**Decision.** ROUTING WINS S1 iff McNemar p < 0.0167 (Bonferroni 3 primaries) **and** the
bootstrap CI lower bound on (ROUT - ORD) > 0. ROUTING LOSES S1 iff the CI upper bound < 0. If
the CI straddles 0, report TOST at margin **delta = 0.15** and state explicitly whether
equivalence is established or the test is merely underpowered.

**Prereg'd secondaries of S1** (reported, alpha = 0.05 uncorrected, labelled secondary):
* **S1-a (m = 0, designated candidate)** — ROUT `trueA` vs ORD `maskA_w8`, both targets. This is
  the direct round-4 replication under designated (not either-mapping) scoring.
* **S1-b (blind, no cue oracle at deployment)** — ORD candidate set {none} + 200 random 8/8
  splits (seeded per model, both signs); ROUT candidate set {none, trueA, trueB}. m in
  {1, 4, 8}. This is the regime where routing should win, because routing *records* the
  training-time cue oracle in the architecture. Same McNemar + bootstrap machinery.
  Also reported: the oracle-over-candidates ceiling (does a good random mask *exist*), which
  separates "the option is absent" from "the selector could not find it".
* **S1-c (toy-artifact baseline)** — `flip`. Because the deployment proxy label is the exact
  complement of the true label in this toy, negating the logits of a proxy-committed model is a
  perfect no-gradient repair. Reported and labelled TOY ARTIFACT; it is not counted in any
  primary and does not generalize past two exhaustive, exactly anti-correlated worldviews.
* Per-candidate marginal success rates for every intervention, so the reader can see which
  candidate carried each arm.

## 4. PRIMARY 2 (S2, unknown target, 4 targets = 3 changes)

Target sequence **[TRUE, PROXY, TRUE, PROXY]** — three changes, as the brief requires.

* **ROUT arm:** at each stage select the mask (candidate set as in PRIMARY 1, m = 8 labels of the
  *current* target). Gradient cost per change = **0**.
* **ORD + fine-tune arm (the control):** start from `ORD`; at each stage fine-tune the *current*
  model for k steps, fresh Adam (LR 5e-3, WD 2e-3), on the augmented set = the 800 original
  training examples + the m = 8 labelled deployment examples of the current target
  (`FT-MIX`, primary) or on the 8 examples alone (`FT-PURE`, secondary). k ladder, preregistered:
  **{0, 5, 20, 100}**. Cost per change = k x 808 (or k x 8) example-forwards.

**Primary quantity:** per-stage SUCCESS rate over 60 seeds x 4 stages (240 stage-successes per
arm), and the smallest k in the ladder whose Wilson-95 lower bound over stages >= the ROUT arm's
point estimate.
**Decision.** ROUTING WINS S2 iff no k <= 100 reaches the ROUT rate, or if reaching it costs more
example-forwards in total (initial training + 3 changes) than routing's 3.2 M.
ROUTING LOSES S2 iff some k reaches the ROUT rate at total cost < 3.2 M example-forwards.
**Prereg'd secondary:** degradation across stages (does repeated flipping erode held-out accuracy
or make later flips harder?) — stage-index trend in SUCCESS and in held-out accuracy, and
whether the FT'd model remains *switchable* afterwards (it should not).

## 5. PRIMARY 3 (S3, unlabelled only, m = 0)

No labels; deployment inputs only (4000 unlabelled deploy inputs). Preregistered unlabelled
selection rules, applied to ROUT's {trueA, trueB} and to ORD's {maskA_w8, maskB_w8}:

* **R1 coin** — seed-determined coin flip (reference floor, expected 0.5).
* **R2 self-consistency** — pick the mask whose predictions agree more with the *un-ablated*
  model's predictions on the unlabelled deploy inputs.
* **R3 confidence** — pick the mask with the higher mean max-softmax on the unlabelled deploy
  inputs.
* **R4 self-training (ORD only, uses gradient but no labels)** — 100 steps of entropy
  minimization on the unlabelled deploy inputs, then score as-is. Tests whether unlabelled data
  has a *better* use than selection.

**H0 (expected, and there is a symmetry argument for it).** Cue A and cue B blocks are
exchangeable by construction: `xor_block` is called with identical parameters for both, so
swapping columns [0,1] with [2,3] maps the training distribution to itself and the deployment
distribution to itself with the worldview label swapped. Any rule that does not see a label and
is equivariant under that swap must be at chance for the *position* question. R2 and R3 are
equivariant. So H0 predicts 0.5.
**Test.** Exact binomial vs 0.5 per rule, Bonferroni over the 4 rules (alpha = 0.0167/4 =
0.0042 for the primary decision).
**Decision.** ROUTING WINS S3 iff some rule is significantly above 0.5 *and* only available to
the routed arm. Otherwise report TOST equivalence to chance at **delta = 0.15** with its margin,
never a bare "no difference".
**Prereg'd secondary:** *option existence* — the fraction of seeds for which a correct candidate
is present in the arm's candidate set at all (i.e. what one single label would buy). This is the
honest measure of what optionality is, separated from the ability to exercise it.

## 6. Cost ledger (reported for every arm, no test)

Three currencies, reported side by side and never collapsed into one number:
1. **gradient cost** = example-forwards during training + fine-tuning;
2. **label cost** = number of labelled deployment examples consumed;
3. **access requirements** = {gradient access?, cue oracle at deployment?, decision known before
   training?}.

## 7. Power

n = 60 paired seeds. Exact McNemar with 60 pairs detects a 20-point paired difference with
power > 0.9 when the discordance rate is ~0.25 (round 4's floor/ceiling gap), and a 15-point
difference with power ~0.7. Underpowered cells will be reported with TOST at delta = 0.15 and
their achieved power, not as nulls. For S2, 240 stage-observations per k give Wilson half-widths
of ~0.06 (they are not independent across stages within a seed; the CI is reported as nominal
and the seed-level bootstrap is used for any decision).

## 8. Downscope rules (declared in advance)

If the run exceeds ~15 minutes of CPU: drop, in this order, (i) `FT-PURE`, (ii) the m = 1 and
m = 4 levels of S1-b, (iii) the random-mask family from 200 to 50, (iv) n from 60 to 40 seeds
(keeping the *lowest* seeds, 700-739, so the anchor still holds). PRIMARY 1 at m = 8, PRIMARY 2
at k in {0, 20, 100}, and PRIMARY 3 are never dropped. Any downscope executed is reported in
FINDINGS.md.

## 9. Anchors and controls

* **A0** my sensitivity-mask re-implementation reproduces `core.learned_masks` exactly (max abs
  diff on the score vector = 0) at w = 8.
* **A1** ORD under `maskA_w8`/`maskB_w8` with round-4's *either-mapping* scoring reproduces
  `r4-break-tautology`'s 51/60 on seeds 700-759 (models are deterministic given the seed).
* **A2** ROUT under the true partition reproduces 60/60.
* **NEG** random 8/8 masks, scored *without* selection (a single random mask per seed), should
  sit near round 4's 1/50.

Anything not in this file is post-hoc and will be labelled POST-HOC in FINDINGS.md.
