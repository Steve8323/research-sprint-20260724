# 4. Discussion

Everything below is about ≤11-dimensional synthetic cue tasks, 154–1.2k-parameter MLPs, and one
68,546-parameter from-scratch transformer (G1). No sentence here is licensed to travel to a frontier
model, and several of the sentences we would have written after round 1 are now retracted (§3, §4.4).

## 4.1 What the results mean

The structural picture first. Train a network on data where a true rule and a proxy rule always agree,
and it does not build one blended decision surface. Under some conditions it lays down two nearly
unit-disjoint groups, each of which can compute a full rule on its own, and each committed to a
different worldview off-distribution. Zero one group and the other's worldview is what comes out. That
structure is present in **ordinary** training. Routing does not put it there.

The evidence for the "already there" half is the correction that reorganised this whole line. Scored
under one arbitrary fixed unit partition, standard-trained models look unselectable (1/60). Scored
under a mask chosen from cue sensitivity on held-out **agreeing** data, the same 60 models give 51/60
(0.85, Wilson [0.739, 0.919]), with a selection margin of +0.910 against −0.018 for the fixed partition
(C4). The mask is not fitting noise: re-derived on 10 independently retrained models it lands 8/10, all
under the designated mapping, against 1/50 for random 8/8 splits and 0/10 for noise-dimension masks
(C5). Random splits of the same models succeed 18/4260 = 0.42% pooled across the portfolio (C10).
Applied uniformly to every routing-free control in the line, the fair floor moves 2/282 → 100/282 (C8).

Two conditions govern whether the latent halves exist, and they are separate.

**Capacity per half.** The data-chosen mask selects 2/30 → 25/30 → 29/30 → 30/30 at H = 8/16/32/64
(Cochran–Armitage z = 7.97, p = 7.8e-16; C6). The gate is per-half **competence**: conditional on both
halves clearing the 0.9 held-out bar, the mask selects 2/5, 25/26, 29/29, 30/30 (C7, POST-HOC; the H = 8
cell has effective n = 5, so the honest reading of that width is "the halves are too small to answer the
question", N7). The same relation holds across cells: column-competence fraction vs fair-floor rate
r = +0.764 over 10 non-independent cells (C9, POST-HOC-DESCRIPTIVE).

**Basis.** In the transformer both halves are competent (12/12, held-out 1.000/1.000, 64 MLP units per
worldview, 4× the width at which the MLP saturates) and the axis-aligned mask still selects 0/30
(Wilson upper 0.1135; C17, C18). Competence is satisfied and selection still fails. Width satisfies one
condition; it does not satisfy the other.

Because the floor moved, routing's effect size has to be restated by regime rather than quoted as one
number. At w = 8 it is **+0.150** [+0.058, +0.261], p = 0.0028 (C12) — the number to quote whenever the
paper says "routing's effect" without qualification. At w = 4, rounds 1 and 2's actual configuration, it
is +0.667 [0.436, 0.801] and +0.700 [0.471, 0.825] (C11). In the shared residual stream the correction
changes nothing at all, 0/12 → 0/12, and the effect is +1.000 (C14) — the only cell in the portfolio
where that is true. Of 13 rescored cells, 9 survive, 2 shrink, 2 die, and "DIES" means unestablished at
n = 20, not demonstrated absence (C15, N19, N20).

So what does routing sell? A **label**. The routed net is literally two isolated ~62-parameter subnets
plus a summing readout (cross-block max|w| = 0.0 on 30/30 seeds, additivity residual 3.8e-6; C21), and
each column implements its own cue's rule in closed form (44/44 on switchable seeds; C22 — a property of
the 2-bit-XOR generator, G2, not a discovery about worldviews). The thing routing adds over ordinary
training is that you know **which column is which**. That the label is the product, not the switch, is
visible in the k-way case: the localizer recovers the k true groups 20/20 but its group→worldview slot
order matches the oracle on only 10/20 seeds at k = 3, so the labelling has to ship with the checkpoint
or the switch is unusable (C28, C29).

The four deployment regimes make the same point operationally (C44–C47). With the cue oracle and no
gradients, routing 60/60 **ties** ordinary training plus an 8-label mask selection, 59/60, TOST passing
at δ = 0.15 with an achieved 90% CI of [0.000, 0.050] — the only genuine equivalence in the ledger (N8).
With gradients available and three target changes, routing **loses**: 100 fine-tuning steps per change
match it at ~58–60% of the cost, and ordinary training plus mask selection reaches 234/240 with zero
gradient steps at half the cost (C45). From unlabelled deployment inputs alone, nobody beats chance
(C46). Routing wins decisively in exactly one regime: when the deployment actor no longer has the cue
oracle, routing is 60/60 while a fair 200-mask blind search on ordinary models goes 3/60 at m = 2 and
saturates at 52/60 by m = 32 (C47; state the win from m = 2 — the m = 1 cell is a fixed-example
artifact). That win is near-definitional, since subnet A is trained by occluding cue B; the empirical
content is the *size* of the blind control's identification problem, and failure there is identification,
not existence (a good random mask exists in 57/60 seeds). The one-sentence reading: **routing records the
training-time cue oracle in the architecture for an actor who no longer has it** (C48). It costs +100%
example-forwards, and routed models are never usable un-ablated (0/60 committed either way; C49).

Against the obvious alternative — just collect a few disagreeing examples and train on them — routing
does not win. On a matched shipped-subnet outcome, routing is 16/21/25/25/28 of 30 at m = 1/2/4/8/32
against augmentation's 14/16/24/30/30; the m = 8 gap is p = 0.052 and m = 32 is p = 0.49 (C41, POST-HOC
rescoring). What survives is that augmentation reaches a ceiling routing does not: 30/30 with deploy
alignment 0.961→0.992 from m ≥ 8 (C42). And the thing that buys the break is **disagreement, not data**:
agreeing examples at the same m land on the true split 8/30 vs 25/30 (C40).

## 4.2 Why this is not a safety result

We went looking for a safety story and did not find one. Three separate reasons, each measured.

**The label is the fragile part.** An adversary holding the weights flips a masked switch at 0 gradient
steps and 0 data, 60/60 (C50). Benign fine-tuning, contrary to what we said in rounds 3 and 4, leaves
44/60 models still switchable under a data-chosen mask (73%, [0.610, 0.829]; C56) — our own strongest
argument against shipping the switch died to the same control-scoring artifact as R1 and R2, on its
third occurrence. What benign fine-tuning does destroy is the block-diagonal structure and therefore the
labelling: under the fixed partition 23/23 → 7/23 at 2000 steps, cross-block mass 0.0 → 7.145, while the
aligned side still leads 0.869 vs 0.125 with the sign preserved 23/23 (C57, stated jointly with C56).
You lose the name, not the thing.

**Attackers with matched budgets.** Under the realistic functional attacker, a hard-deleted switch is
indistinguishable from a routing-free model with the same live parameter count: d = −0.0195,
95% CI [−0.0736, +0.0345], with an achieved equivalence bound δ* = 0.065 = 36.5% of the run's total
defense effect of 0.178 (C51, N1). That is a bound, not an equality. Under the weaker preregistered LoRA
attacker the switch does beat both controls (d = −0.079, p_perm = 0.0014, Holm 0.0095; C52), but that is
the regime where the attack leaves a broken model — only 26.7% of attacked switches are still competent
at step 100. Under full fine-tuning plus 1% jitter the sign flips (+0.093, Holm-adjusted 0.156, not
surviving Holm), and that cell is **not** capacity-matched: jitter revives the 92 zeroed weights, so the
switch trains 154 parameters against the control's 62 (C53, R14). Dead parameters are an attack surface.
The resistant tail is 11/60 switch vs 9/60 control, Fisher p = 0.807, Newcombe [−0.102, +0.168] — the
tail could be up to twice as common in the switch and the data do not separate them (C54, N5, POST-HOC).

**The monitor hazard cuts the other way.** An unselected routed model is not a hedger: median
determinism 0.786 [0.622, 0.912] in the MLP and 0.951 [0.869, 0.995] in the transformer, both CIs
excluding the D ≤ 0.5 hedger regime (C58, retracting R9). What it is instead is a confident
off-distribution **class collapse** while remaining fully competent on-distribution: held-out accuracy
median 1.000, majority-class rate on disagreement inputs 0.893 [0.811, 0.956] vs 0.550 for controls
(C59, POST-HOC). Per-input confidence does not flag the inputs on which such a collapsed model is wrong
— median AUROC 0.5208, 90% CI [0.4773, 0.5486], achieved bound 0.0486 against a ±0.05 margin, while a
mechanistic ablation-derived readout of the same inputs scores 0.9996 (C60, N15; the equivalence is
knife-edge, and 31% of individual models deviate > 0.20 from chance). Model-level triage is
anti-informative, AUROC 0.197 [0.112, 0.301], partly a composition artifact (C61).

The scope correction matters more than the result. Those 29 models have median majority-class rate
0.962 and 62% emit one class on > 95% of deploy inputs. The defensible claim is about **collapsed**
models, not about models that have switched worldview (R11). And the preregistered negative control
failed: control-arm hedgers, predicted detectable, score median AUROC 0.4205 (C62, N16). The run
contains no demonstrated population where a confidence monitor works, which makes this a limit of
confidence monitoring rather than a routing-specific liability. We also retract outright the claim that
standard training never produces this mode — 19/23 transformer controls do (R7) — and we drop the
follow-on claim that routed models are more prone to it, which was a one-armed competence gate (R8:
ungated 15/23 vs 19/23, p = 0.157).

## 4.3 The localization boundary, and the whitening prescription

Routing needs to be told which units carry which cue. That, not routing, is where the method breaks.

Given the true subspaces, routing is basis-free: axis 25/30, random rotation 26/30, dense non-orthogonal
mixing 28/30 (C37). Swap in a basis-agnostic localizer and it collapses in every basis, including the
unrotated one (axis_ica 13/30 vs axis_pair 25/30, p = 0.0028). So the scaling question is a localization
question.

There is an exact obstruction, and it is knife-edge. At exactly equal cue strength the orthogonal
symmetry group of the labelled centres has order |G| = 64 with an orbit of 2 on the true split, so any
equivariant localizer must coin-flip; at all 15 other margin ratios |G| = 16 and the orbit is a point
(C31). Off that point the problem is solvable: a localizer reading the signed eigen-spectrum of
D = E[xx′|y=1] − E[xx′|y=0] recovers the true split 40/40 at r = 2.05 and lifts selectability to 38/50
against ICA's 26/50 (Fisher p = 0.0213; C32). Recoverability has an onset that moves with sample size:
r* = 1.15 at N = 800 and 1.07 at N = 3200, excess ratio 0.449 inside the preregistered N^(−1/2) window
(C33 — but the N = 200 leg is right-censored and FAILED, so this rests on one ratio, not a fitted
exponent). Only the magnitude of the asymmetry matters, not which cue is stronger (C36). At r = 1.00 it
collapses exactly as preregistered, 14/50, landing profile true 1 / entangled 2 / neither 47 (C34).

The prescription that generalises: **do not reach for a generic component analysis on this problem.**
The preregistered 2×2 at r = 2.05 gives eigen-directions + asymmetry pairing 40/40, ICA directions +
asymmetry pairing 18/40, eigen-directions + probe pairing 18/40, plain ICA 27/40 — a pure interaction in
which both single-ingredient cells sit at or below the no-ingredient cell (C35). Whitening, ICA's first
step, is what destroys the resource the localizer uses, and re-pairing whitened components does not
repair it. The run's own preregistered falsifier ("the pairing rule is the whole story") fired.

Three scope conditions travel with all of it. The localizer is validated for **orthogonal** entanglement
only; the general invertible case is untested in the artifact
[LEDGER GAP: no non-orthogonal `mix` arm exists in the localizer results on disk, so the reported
degradation under non-orthogonal mixing may not be quoted]. Rotation-invariance is an analytic property
of the estimator (exact orthogonal equivariance), not something we measured
[LEDGER GAP: the axis-vs-rotated comparison is not on disk]. And it is a **model-based** estimator:
D's diagonality follows from the XOR-parity-plus-isotropic-noise family, so 40/40 is a numerical
confirmation of exact algebra that embodies designer knowledge, oracle-free per instance but not
assumption-free. Against the oracle it is bounded, not matched (38/50 vs 26/30, RD CI [−0.263, +0.083],
power 0.502; N13).

The cheap escape is supervision. A handful of labelled **disagreeing** examples buys localization and a
working switch in the basis where unsupervised localization is obstructed, with both preregistered trends
at the permutation floor and a deliverable of k* = 4 examples, 0.5% of the training set (C39).

## 4.4 Limitations

Stated flatly, because a reader will find them anyway.

1. **Scale.** G1. 154–1.2k-parameter MLPs and one 68.5k-parameter transformer, 12–30 seeds per cell,
   synthetic cues. Nothing here constrains what a trained-at-scale system does.
2. **The worldview is a 2-bit XOR.** Both cues are sign-product rules on disjoint input coordinate
   pairs, searched to be equal-cost (G2). The closed-form column readability (C22) is a property of that
   generator. "154 parameters and here is the rule" is an existence proof for a worldview that *is* a
   parity bit.
3. **One deploy distribution.** "Deployment" is the single held-out set where the two cues disagree
   (G3). We have no evidence about graded, multi-axis, or shifting disagreement.
4. **One transformer, one mask family.** C17's null is over axis-aligned masks shared across both
   layers, 4 heads and 128 MLP units. Rates in 0.10–0.40 are not excluded (N6), and the mask-finder's
   positive control is not on disk
   [LEDGER GAP: no on-disk instrument check for the transformer null; the paper must either recompute it
   or state the null without an instrument check].
5. **The cue oracle is consumed nearly everywhere.** Every mask-selection procedure, including the fair
   floor, uses the training-time knowledge of which dimensions or positions carry which cue (G4). Only
   the pair probe and the eigen-localizer are oracle-free, each with its own conditions (C28, C32).
6. **Near-tautology where columns are disjoint and train mask == eval mask.** The k-way 20/20 vs 0/20
   and the transformer 11/12 vs 0/12 are **demonstrations**, not effect sizes (C20, C25). Each routing
   group is trained in isolation under exactly the ablation mask it is later evaluated with, and the
   block structure is imposed at init and preserved by zero gradient rather than discovered (C21).
7. **The width law is post-hoc.** The trend was preregistered; reading it as a law was not, and w is
   confounded with total hidden size H = k·w while the pooled cells differ in k and in input basis
   (C8). The competence-conditional diagnostic was implemented for the transformer and omitted for the
   MLP, then recomputed (C7).
8. **Several headline claims changed sign or magnitude across rounds.** Three separate effect sizes were
   inflated by the same control-scoring artifact (R1, R2, R3). The exchange-rate verdict reversed (R4).
   A "never" claim was retracted with 19 counterexamples (R7). A reverse-direction claim was dropped as
   a one-armed gate artifact (R8). The hedging story died (R9), the patchwork characterisation was
   restated as class collapse (R10), the monitor claim was rescoped to collapsed models (R11), and the
   tamper picture reversed twice (R12, R14).
9. **The success criterion is a composite with hand-set thresholds** (hi = 0.8, lo = 0.2,
   train_ok = 0.9; G5). Cells that "die" under it are unestablished, not empty (N19, N20).
10. **Our own selection inflation is k = 1, with one exception.** The switch line's counts are per-seed
    successes at prespecified configurations tested by Fisher exact, with no max-over-k — except that
    w = 8 is an inherited selection: round 3's oracle at w = 4 gives 13/20 (k = 3) and 6/20 (k = 4).
11. **Cells are not all independent.** Rounds 1 and 2's cells share control models, the two k-way cells
    share controls with the multiway w = 8 cells (C10, C11), and C9's correlation is over 10
    non-independent cells with no CI.
12. **Some tests had no dynamic range.** Cross-worldview leakage sits at the eval set's zero-interference
    floor, identical at k = 2, 3, 4 (C30), so "no interference at higher k" is uninformative.
13. **C41's matched numbers are a round-5 rescoring** computed from per-seed data; they are not stored
    as a summary in any results file.

## 4.5 The methods contribution

Three things here are reusable outside this line.

**A deflation rule for sweep-selected numbers.** When you report the best of k cells, the reported value
carries a selection premium. The rule shrinks it toward the sweep mean by the fraction of cell-to-cell
spread that is not noise: θ̂ = D̄ + ρ̂(D_sel − D̄) with ρ̂ = max(0, 1 − s²/S²), where s is the
within-cell SE and S the across-cell SD. In simulation over 400 grid cells × 2000 reps it lands 100/100
cells within 15% of the closed form a_k·s·√(1−ρ), improving bias in 397/400 and RMSE in 393/400 (C63) —
validated against its own generative model. Applied to a published 24-layer noise-probe sweep it
returns ρ̂ = 0.026: there was essentially no layer-to-layer signal to select on. The published selected
contrast of −2.751 nats deflates to −0.070 against a fresh preregistered truth of +0.22 [−0.72, +1.16],
and 8-layer out-of-sample RMSE drops 1.223 → 0.415 (C64). The correct statement is that **no
layer-to-layer signal was demonstrated**, not that the layer effect is absent; and at ρ̂ ≈ 0 the
calibrated rule is empirically indistinguishable from shrinking all the way to the mean.

Its preregistered real-data primary **failed**: exact Wilcoxon over 8 layers, W = 9, p = 0.125, at ~53%
power even if the rule is exactly true (C65, N23). That is uninformative rather than disconfirming, and
it belongs in the same sentence as the simulation result. Two smaller corrections travel with it: for
max-of-|·| statistics the constant is b_k = E[max|N(0,1)|], not a_k (b_4 = 1.4647 vs a_4 = 1.0294, a
factor of 1.42, with zero verdict flips in the affected portfolio; C66), and a max-over-θ modularity
claim turns out to be family-dependent rather than simply true or withdrawn (C67).

**The fair-control lesson.** Score the control with the same degrees of freedom you gave the treatment.
Our routed arm got to choose its partition; the control got one arbitrary 8/8 split out of 12,870, and
that single asymmetry inflated three separate headline claims across three rounds (R1, R2, R3) before we
caught it. The fix costs nothing: derive the control's mask from the same held-out agreeing data, and
add a search-matched negative control so the reader can see what the freedom is worth (random splits
0.42%, C10; best-of-10 random masks as the comparator in the width sweep, C6).

**The meta-finding.** Across six independent audits in rounds 4 and 5, **not one primary statistic
failed to reproduce.** Anchors sit at max|diff| = 0.0 for round 1 vs round 4 (22/30 identical booleans),
round 4 vs round 5 masks (identical 10/10), the optionality masks (60/60), the tamper trainer, the
exchange-rate generator, and the round-3 rerun files (C68; two exceptions must be stated — a 0.0255
multiway anchor that was an undeclared override of the prereg's own rule, and a 5e-5 storage-precision
gap where the per-seed booleans are exact). Every defect the audits found lived one layer up, in
interpretation:

- an **asymmetric outcome definition** — routing scored two-sided, its comparator one-sided (C41, R4);
- a **gate applied to one arm** — a competence filter that removed 3 routed models, all 3 positive, and
  no controls (R8);
- a **control given a different scoring rule** — the fixed-partition floor (C4, R1–R3);
- a **null promoted to an equivalence** — reported as a tie at ~53% power (N23), and more generally the
  distinction §4 exists to enforce;
- a **diagnostic implemented for one substrate only** — the competence gate run on the transformer and
  omitted for the MLP (C7).

The practical claim is narrow and cheap to act on: measurement discipline held and framing discipline did
not, so the audit budget should be spent on re-reading outcome definitions, gates, and control-scoring
rules rather than on re-running the code. Three of our five rounds were adversarial toward our own best
result. Each was mostly a rescoring of runs that already existed — the audits' own anchors are
re-executions of the stored procedures, which is why they come back at max|diff| = 0.0 — and each moved
a headline number: the routing effect (R1), the k-way claim (R2), the fine-tuning claim (R3), the
exchange-rate verdict (R4). That is a low price for corrections we would otherwise have shipped. We
would rather publish the history than a clean-looking line a reader has to reconstruct. The claim "every
result reproduced and every claim held" is not one we can make; "every primary statistic reproduced and
several interpretations did not" is.

## 4.6 What follows

Three questions are actually implied by the results, and one is not (nothing here motivates a
frontier-model deployment proposal).

**What is the second condition in sequence models?** The dissociation is sharp: in the MLP, competence
of both halves is nearly sufficient for selection (2/5, 25/26, 29/29, 30/30; C7), and in the transformer
both halves are competent, with 4× the width the MLP needs, and selection is 0/30 (C17, C18). Something
other than capacity is binding. The obvious candidates are the mask family (we tested one: axis-aligned,
shared across layers, over heads and MLP units) and the shared residual stream itself. The experiment is
to enlarge the family — per-layer masks, non-axis-aligned directions, subspace rather than unit
ablation — and see whether the 0/30 survives. It should be run with a positive control on disk, which is
what C17's null currently lacks.

**Non-orthogonal localization.** C32 is validated for orthogonal entanglement only, C35 rules out the
generic first move (whitening), and the general invertible case is untested in the artifact. The open
question is whether the signed-eigen resource has an analogue that survives an arbitrary invertible
mixing, or whether the supervised escape (k* = 4 disagreeing examples, C39) is the only route once the
basis is genuinely unknown.

**Do latent separable halves exist in real models?** This is the honest version of the scaling question,
and it is a measurement, not a proposal. The recipe is already specified by C4–C7: find a naturally
underdetermined cue pair, derive a mask from cue sensitivity on held-out **agreeing** data, gate on
per-half competence, and compare against a search-matched random-mask control. A negative result there
is as informative as a positive one, and either way it does not depend on anyone adopting routing.

Two smaller items: the recoverability scaling exponent needs a fitted version, since the N = 200 leg was
right-censored and the claim currently rests on one ratio (C33); and the tamper equivalence deserves a
rerun powered for the preregistered 0.05 margin rather than the 0.065 it achieved (C51, N1).
