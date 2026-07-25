# 3. Results

Everything below is toy scale: synthetic inputs of ≤11 dimensions, MLPs of 154–1.2k parameters, and
one from-scratch transformer of 68,546 parameters (G1). No claim here licenses a statement about
frontier models. Several headline numbers in this line changed sign or magnitude between rounds; each
such change is flagged in a **What changed** aside at the point where the final number appears.

Bracketed IDs (C4, N6, R1, …) are entries in the claims ledger.

## 3.0 What "selectable" means, and what every number inherits

Unless stated otherwise, a model counts as **selectable** when a single mask over hidden units splits
it into two halves such that ablating one half gives >0.8 deploy alignment toward the target worldview,
ablating the other gives <0.2, and both halves still score >0.9 on held-out *agreeing* data (G5). The
last leg matters more than it looks: most of the results below turn on it.

Three conditions ride on every number and are not repeated:

- **The worldview is a 2-bit XOR.** Both cues are sign-product rules on disjoint input coordinate pairs,
  searched to be equal-cost (G2). Closed-form readability of a column (§3.5) is a property of that
  generator, not a discovery about worldviews.
- **One deploy distribution.** "Deployment" is the single held-out set on which the two cues disagree.
  Training and held-out data always agree (G3).
- **Mask selection consumes the training-time cue oracle.** Every mask-finding procedure here, including
  the corrected "fair" floor, uses knowledge of which input dimensions or token positions carry which
  cue (G4). The two oracle-free localizers (§3.4, §3.6) carry their own conditions.

## 3.1 Underdetermination reproduces at both scales

**Structural picture.** Two rules label the training data identically. Nothing in the loss distinguishes
them. Training therefore does not pick between them on the merits; the seed picks.

**MLP (n = 100 fresh seeds, equal-cost regime, width 8).** Train accuracy 1.000 (min 1.000). Deploy score
mean 0.509, splitting **27 aligned / 27 misaligned / 46 hedgers**. Hartigan dip p = 0.00145 (bootstrap
0.0008), surviving Holm over the 25-regime grid [C1]. *Scope:* this is the only jitter-robust well-fit
bimodal cell of 20 in that grid. Five of the other rejections are the underfit W = 2 row and three are
discreteness artifacts, so bimodality is a property of this regime, not of the grid.

**Transformer (n = 12 control seeds, 68.5k params, d = 64, 2 layers, 4 heads).** All 12 at held-out
accuracy 1.000, with deploy alignment spanning **0.000–0.933** (std 0.293). Homogeneity χ²(11) = 15212.8,
p < 1e-300 [C2]. Preregistered primary (χ² p < 0.001 and ≥3 seeds committed).

**The transformer's cue is a circuit, not a token statistic.** Token-count standard deviation across
examples is exactly 0.0 in both train and deploy, so a bag-of-tokens model has Bayes accuracy **0.500**;
occlusion changes a mean of 0.98 positions and leaves the token multiset unchanged [C3]. The lottery is
therefore over which *circuit* the model builds, not over which surface statistic it latches onto.

## 3.2 The routing-free floor was mis-scored, and the corrected floor is width-gated

**Structural picture.** Ablating "half the network" requires choosing which half. Rounds 1–3 scored the
routing-free control under one arbitrary fixed unit partition — a single coin toss over 12,870 possible
8/8 splits — while the routed arm was scored under the partition it was trained with. Give the control
the same freedom to choose its mask from data, and much of routing's apparent effect is already sitting
in ordinary training.

### 3.2.1 The correction

At the round-4 configuration (H = 16, w = 8 units per worldview, MLP, axis-aligned cues), n = 60
identical standard-trained models score **1/60** under the arbitrary fixed partition and **51/60**
(0.85, Wilson [0.739, 0.919]) under a mask chosen from cue sensitivity on held-out *agreeing* data only.
Selection margin −0.018 [−0.097, +0.061] fixed vs **+0.910** data-chosen [C4]. The fair-floor rescoring
was not the run's preregistered primary; it is the run's self-refutation, disclosed as such.

The mask is measuring something specific rather than fitting noise. Re-derived independently on 10
retrained control models it succeeds **8/10**, all under the designated (not swapped) mapping, every mask
a genuine full 8/8 split; random 8/8 splits succeed **1/50** (Fisher p = 1.5e-7) and masks built on noise
dimensions **0/10** (p = 7.1e-4) [C5]. *Scope:* seeds 700–709 retrained from scratch, an independent
re-derivation rather than a fresh-seed replication.

Random splits of the *same* models almost never work: **18/4260 = 0.42%** over distinct model sets
(21/5660 = 0.37% if cells sharing controls are counted twice) [C10].

### 3.2.2 The width law (Figure 1)

**Structural picture.** The data-chosen mask can only succeed if each half, alone, has enough units to
compute the rule. Widen the layer and the floor climbs.

Data-chosen mask succeeds **2/30 → 25/30 → 29/30 → 30/30** at H = 8/16/32/64 (4/8/16/32 units per
worldview), n = 30 seeds per width. Cochran–Armitage **z = 7.9726, p = 7.77e-16**. Per-width McNemar
against best-of-10 random masks: 1.0 / 2.7e-4 / 2.4e-7 / 1.5e-8 [C6]. *Scope:* MLP, k = 2, axis-aligned
cues, no corrective data; the random-mask control gets K = 10 draws to the method's 1. The **trend** was
preregistered; **reading it as a law is post-hoc**.

Applying one identical fair-mask procedure to every routing-free control in the line moves the pooled
floor **2/282 → 100/282**, and the movement is concentrated where the ablation groups are wide:
**w = 8: 2/120 → 88/120 (73.3%)** vs **w = 4: 0/150 → 12/150 (8.0%)**, Fisher **p = 2.53e-30** [C8].
The rescoring itself was preregistered (13 cells, per-cell verdict rule fixed in advance) and verified
bit-identical to round 4's rule (max|diff| = 0.0, masks identical 10/10). **The w = 8 vs w = 4 contrast
is POST-HOC**, w is confounded with total hidden size H = k·w, and the pooled cells differ in k and in
input basis. Descriptive only.

Post-hoc corroboration: across cells, fair-floor rate tracks column competence at Pearson **r = +0.764**;
median column held-out accuracy is 0.99 with 80–93% of columns above the 0.9 gate at w = 8, against
0.81–0.87 with 28–42% above it at w = 4 [C9]. *Scope:* **POST-HOC**, n = 10 non-independent cells (three
pairs share byte-identical control models), no confidence interval.

### 3.2.3 What routing is left with

| Configuration | Routed | Fair floor | D_new | Source |
|---|---|---|---|---|
| Round 1, w = 4, H = 8, axis | 22/30 | 2/30 | **+0.667** [0.436, 0.801], Fisher 1.45e-7 | [C11] |
| Round 2, w = 4, learned localizer | 23/30 | 2/30 | **+0.700** [0.471, 0.825] | [C11] |
| Round 4, w = 8, H = 16 | 60/60 | 51/60 | **+0.150** [+0.058, +0.261], Fisher 0.0028 | [C12] |
| Rotated basis, w = 4 | 16/30 | 4/30 | **+0.400** [0.163, 0.583], Holm 0.0065 | [C13] |
| Non-orthogonal mixing, w = 4 | 19/30 | 2/30 | **+0.567** [0.336, 0.722], Holm 6.2e-5 | [C13] |
| Transformer, shared residual stream | 12/12 | 0/12 | **+1.000** [0.657, 1.000], Holm 8e-6 | [C14] |

Rounds 1 and 2 share control models and count once. The +0.150 at w = 8 is the number to quote whenever
the paper says "routing's effect size" without qualification [C12]. The two entangled-basis rows are
**conservative**: their routed arms used oracle-free learned localizers while their floors got the cue
oracle [C13]. The transformer row is the only cell in the portfolio where the fair-floor correction moves
the number by exactly zero [C14].

Of the 13 rescored cells, **9 survive, 2 shrink, 2 die** under a verdict rule fixed before any code was
written [C15]. "DIES" is not "no effect": k = 4 / w = 4 dies at D_new = +0.250 with a Newcombe CI
[0.008, 0.473] that excludes zero and a Holm p of 0.183 — unestablished at n = 20, and the tension
between the two summaries should be stated rather than resolved [N19]. k = 2 / w = 8 dies at +0.150
[−0.038, 0.360], Holm 0.231; the original 20/20-vs-1/20 claim is refuted outright, and the residual
effect is not established in either direction [N20]. In only two configurations were the *routed* arms
also rescored under a data-chosen mask, so elsewhere D_new is an **upper bound** [C15].

> **What changed.** Rounds 1–3 published "routing makes the worldview selectable, 22/30 (later 60/60)
> vs 0/30 — a ~+0.98 effect" [R1]. That floor was one arbitrary partition. The same artifact inflated
> two further headline claims in later rounds (§3.6, §3.8). Nothing in the underlying measurements moved:
> the round-1 anchor reproduces at max|diff| = 0.0 on 22/30 identical booleans.

**Selectability needs private capacity, and tolerates a lot of sharing short of that.** Under the
training partition: ceiling (2/434 shared params) 60/60; 25% unit overlap 60/60; 50% overlap 58/60;
fully shared first hidden layer 60/60; soft train-mask leak 0.25 60/60; leak 0.50 (91% of params shared)
59/60; **75% overlap collapses to 2/60**, Cochran–Armitage over the overlap dose z = −11.73 [C16]. The
collapse is not a competence failure: 0 of 58 misses are competence misses and median held-out accuracy
is 1.000. The break is also partial — the shared-trunk arm shares zero gradient in the later layers, so
selection still acts on disjoint columns, and the soft-leak arms break the leak level rather than the
partition (learned masks recover the training partition at Jaccard 1.000/0.996/0.979). Sharing and
private width are confounded: there is no width sweep at fixed overlap.

## 3.3 The gate is per-half competence — and the transformer dissociates it from selection

**Structural picture.** Two things have to be true for post-hoc selection to work: each half must be able
to compute a rule alone (competence), and the two halves must have committed to *different* rules
(commitment). Width buys the first. The transformer shows the first without the second.

### 3.3.1 In the MLP, competence is the gate (Figure 2)

Both data-chosen halves clear the 0.9 held-out bar in **5 / 26 / 29 / 30** of 30 seeds at H = 8/16/32/64.
Conditional on both being competent, the mask selects **2/5, 25/26, 29/29, 30/30** [C7]. *Scope:*
**POST-HOC** — the preregistration committed to reporting competence-dominated failures, and the
diagnostic was implemented for the transformer but omitted for the MLP until this analysis. The H = 8
conditional cell has effective n = 5 (Wilson [0.076, 0.788]).

This is why the unconditional 2/30 at H = 8 must never be quoted as "no latent switch at round 1's
width". The defensible statement is: **at H = 8 the halves are too small to answer the question** [N7].

### 3.3.2 In the transformer, competent halves still do not select

n = 30 transformer control seeds, data-chosen head+MLP-unit mask: **0/30** selected (Wilson 95% upper
**0.1135**). Best-of-10 random balanced masks 0/30 (per-draw 0/300); fixed partition 0/30; distractor
mask 0/30; McNemar vs random p = 1.0. **Zero competence failures**, held-out minimum 1.000. The
preregistered decision **ABSENT** was applied by code [C17].

Capacity is excluded. A units-only mask hands each worldview **64 MLP units per layer** — 4× the width
at which the MLP saturates. Both halves are competent 12/12 at held-out 1.000/1.000, mean selection
margin 0.073, and selection is still **0/12**. Heads-only: 0/12, 9/12 competent, margin 0.114 [C18].
*Scope:* **POST-HOC**, n = 12, same single mask family.

An independently retrained transformer set says the same thing descriptively: under the fair-mask
procedure 79% of data-chosen groups pass the competence gate, yet both groups land mid-alignment on
9/12 seeds (median deploy alignment ≈ 0.50 A-side, 0.44 B-side), and only 4/12 seeds recover a contiguous
head split [C19]. *Scope:* **POST-HOC**, n = 12, one organism.

**Two independent conditions, and width satisfies only one.** Competence is necessary and width buys it;
commitment along an axis-aligned mask family is separate, and in a shared residual stream we could not
find it.

*Limits of this null.* One organism, one mask-granularity family (axis-aligned, shared across both
layers, over 4 heads and 128 MLP units). The prestated power table is 0.15/0.52/0.87/0.98/1.00 against
true rates 0.40/0.50/0.60/0.70/0.85 — so **rates above ~0.11 are excluded; rates in 0.10–0.40 are not**
[N6]. [LEDGER GAP: the mask-finder's positive control on routed transformers — reported elsewhere as
4/4 — is not on disk, so this null currently has no instrument check. Either recompute it or state the
null without one.]

Routing does make the worldview selectable in that same transformer: competence-gated **11/12 vs 0/12**,
Fisher p = 9.61e-6 (ungated 12/12 vs 0/12, p = 7.40e-7), with every routed seed at selA_deploy 1.000 and
selB_deploy 0.000 [C20]. This is a **demonstration, not an effect size**: each routing group is trained
in isolation under exactly the ablation mask it is later evaluated with. Occlusion alone gives 3/12
(p = 0.217, underpowered) [N22].

## 3.4 Localization: an exact obstruction, and a narrow way around it

**Structural picture.** Routing needs to be told which subspace is which. If the cues sit in known input
columns, that is free. If the input basis is entangled, someone has to find the split — and at exactly
equal cue strength there is a symmetry that makes finding it impossible for a whole class of methods.

### 3.4.1 The obstruction is exact, and it is knife-edge

Exact enumeration over 16 cue-margin ratios: at r = 1.00 the orthogonal symmetry group of the labelled
class centres has order **|G| = 64** and its orbit on the true split has **2** members, so any equivariant
localizer must coin-flip between them. At all 15 other ratios **|G| = 16** with orbit **1** [C31]. This
is a statement about equivariant localizers on this generative family. It does not say no localizer can
exist away from the symmetric point.

Measured behaviour matches: at r = 1.00 the signed-eigen localizer lands on the true split **14/50**,
with a landing profile of true 1 / entangled 2 / **neither 47** — degenerate eigenspaces produce a random
split [C34]. Against ICA at the same point, 14/50 vs 12/30 (Fisher p = 0.327, risk-difference CI
[−0.326, +0.086], power 0.365): both are coin flips there, and the landing profile rather than the rate
is the informative statistic [N14].

> **What changed.** Round 3 concluded "no unsupervised localizer can exist / the degeneracy is not
> knife-edge in cue strength" [R6]. Exact enumeration showed the obstruction holds *only* at exactly
> equal cue strength.

### 3.4.2 What the signed-eigen localizer buys (Figure 3)

Read the signed eigen-spectrum of D = E[xx′|y=1] − E[xx′|y=0]. Off the symmetric point it recovers the
true concept split **40/40** at r = 2.05 (binomial vs 0.5, p = 9.1e-13), with predicted spectrum
(−8, +8, −33.62, +33.62) against observed (−8.000, +7.998, −33.606, +33.673). Downstream selectability
**38/50 vs 26/50** for round 3's ICA localizer on the same seeds, Fisher **p = 0.0213**, risk difference
+0.24 [+0.052, +0.406]; against standard training, 38/50 vs 0/30, p = 2.02e-12 [C32].

*Scope, load-bearing:* **orthogonal (rotated) entanglement only**, N = 800, r ≥ 1.15. It is a
**model-based estimator** — D's diagonality follows from the XOR-parity-plus-isotropic-noise family — so
40/40 is a numerical confirmation of exact algebra. It is oracle-free per instance while embodying
designer knowledge of the generative family.

[LEDGER GAP: the paired McNemar version of this primary (reported elsewhere as p = 0.017) is not on
disk; the Fisher p = 0.0213 above is the on-disk value.]

Only the **magnitude** of the asymmetry matters, not which cue is stronger: at r = 0.4878 and r = 0.80,
where the proxy is the stronger cue, the localizer still lands true 40/40 (fidelity 0.9986 / 0.9932)
[C36].

Recoverability has a measured onset that moves with sample size: r* (Wilson lower bound > 0.90) = **1.15
at N = 800** and **1.07 at N = 3200**, so the excess (r*² − 1) falls 0.3225 → 0.1449, ratio **0.449**,
inside the preregistered N^(−1/2) window [0.35, 0.71] [C33]. *Caveat:* the N = 200 leg is right-censored
by a too-short grid and **failed** its own check, so the scaling claim rests on one ratio, not a fitted
exponent.

Against the oracle at r = 2.05 the localizer is 38/50 vs 26/30 (Fisher p = 0.387), risk-difference CI
[−0.263, +0.083], power 0.502 against a 20-point difference: **bounded, not matched** [N13].

### 3.4.3 Rotation invariance, non-orthogonal mixing, and the whitening warning

Invariance to orthogonal change of basis is **free by construction**: the estimator is exactly
orthogonally equivariant, so a rotated problem is the same problem. This is an analytic property of the
estimator, stated as such. [LEDGER GAP: the empirical axis-vs-rotated comparison (reported elsewhere as
20/20 = 20/20) is not on disk and is not cited here.]

Under general non-orthogonal mixing the guarantee goes away, and this case is **untested in the
artifact**. [LEDGER GAP: the reported degradations to 13/20 at r = 2.05 and 17/20 at r = 3.00 are
auditor-run and not on disk; the scope limit is real and stated without them.]

**The whitening warning.** ICA's first step destroys exactly the resource this localizer uses. A
preregistered 2×2 at r = 2.05: eigen-directions + asymmetry pairing **40/40**; ICA directions + asymmetry
pairing **18/40**; eigen-directions + probe pairing **18/40**; plain ICA **27/40** [C35]. Both
single-ingredient cells land at or *below* the no-ingredient cell, so this is a pure interaction
(vs best single ingredient, p = 7.6e-5). The run's own preregistered falsifier — "the pairing rule is
the whole story" — fired. Prescription: do not reach for a generic component analysis on this problem;
re-pairing whitened components does not repair the damage.

### 3.4.4 Routing is basis-free; localization is what the basis breaks

Given the true subspaces, routing works in any basis: axis 25/30, random rotation 26/30, dense
non-orthogonal mixing 28/30. Swap in a basis-agnostic localizer and it collapses in every basis:
axis_ica 13/30 vs axis_pair 25/30 (Fisher p = 0.0028, data unrotated), rot_ica 16/30, rot_dict 8/30,
mix_ica 19/30 [C37]. The honest floor for a routed arm here is routing-with-a-broken-split
(rot_pair 10/30, pooled 18/60), not standard training's 0/30 — the run's own agent volunteered this.

Where the localizer lands predicts whether selection works, which locates the failure in localization
rather than in routing: pooled over ICA arms, success is **34/39** when it lands on the true split vs
**13/50** when it lands on the entangled split, Fisher p = 8.39e-9 (ICA lands true 39/90 overall) [C38].
*Scope:* **POST-HOC** and selection-confounded — seeds are not randomised into landing groups.

> **What changed.** Round 2 published "oracle-free localization: a data-only probe removes the oracle
> (23/30 ≡ oracle 22/30)" [R5]. Round 3 showed the probe works only in a coordinate system where the
> concepts already occupy disjoint columns, and that coordinate system *is* oracle information. The
> oracle-free claim was withdrawn, then partially restored in the narrow form of §3.4.2.

## 3.5 Mechanism: what the switch is, exactly

**Structural picture.** In the routed net there is nothing to interpret. The cross-blocks are zero at
initialization and receive identically zero gradient, so they stay zero. The trained object is literally
two isolated ~62-parameter subnets plus a summing readout.

**Exact decomposition.** Cross-block max|w| = **0.0 on 30/30 seeds**; additivity residual max **3.81e-6**
for Δ_full = Δ_A + Δ_B + bias. In the routing-free control the same nominal split reaches cross-block
weight 0.882, so the split there is arithmetic but not causal [C21]. The structure is *imposed* at init
and *preserved* by zero gradient, not discovered.

**Each column is a closed-form rule.** On switchable seeds, **44/44** columns match the 4-corner XOR
truth table read straight off the weights with no fitting (control 13/60). Agreement with the cue on
deploy data: median 0.998 (control 0.500). Flip rate when the other cue's input block is resampled:
median 0.000 (control 0.249). Row mass on the own cue 0.865 vs 0.012 on the other (control 0.469/0.456)
[C22]. Non-switchable seeds give 8/16. Readability is a property of the 2-bit-XOR generator (G2), not a
general result about worldview modules.

**Composition is a margin race.** In the un-ablated model the column with the larger |contribution|
predicts the full model's output on **95.7%** of draws (median over 22 seeds), and the race outcome is
near-deterministic within an input pattern (median 0.957) [C23]. *Scope:* **POST-HOC**, k = 2, MLP.

**Partial ablation is a per-input threshold, not a blend.** The population transition width is
**W = 0.2894**, bootstrap 95% [0.2759, 0.3068] — above the preregistered GRADED cut of 0.25, monotone
22/22, median Spearman −0.9997. But per input (**POST-HOC**; the population graded-vs-toggle test was
the preregistered primary, the flip structure was not), **99.96%** (median over seeds) of the inputs that
change flip **exactly once**, and the 10–90 spread of per-input flip points is 0.285 against a population
W of 0.289 [C24]. The population gradedness is dispersion of individual thresholds, not hedging: "α = 0.7
buys 70% of cases judged by rule A", not a model that half-believes each rule. Selection is ≥90% complete
by median α = 0.596. *Note:* the gate choice matters (unit gate 0.289 vs contribution gate 0.440).
[LEDGER GAP: the claim that the plateau extends to α ≤ 0.3 fully selected is not in the ledger; the
on-disk summary is the median α = 0.596 figure above.]

## 3.6 k-way

**Structural picture.** Nothing about the construction is specific to two worldviews. With k disjoint
column groups you get a k-way labelled selector — and, again, standard training already contains a
weaker version of it.

Routed k = 3 and k = 4 at w = 8: **20/20 vs 0/20** for standard training, scored generously over all k!
permutations, Fisher **p = 1.45e-11** at both, Wilson [0.839, 1.0] [C25]. This is a **demonstration**,
not an effect size: the columns are structurally disjoint subnetworks sharing only the 2-element output
bias (gradient trace 0/108, 0/144, 0/24) and the train mask equals the eval mask.

Under the fair mask the k-way floors are **11/20 (k = 3, w = 8)** and **9/20 (k = 4, w = 8)**, giving
D_new = +0.450 and +0.550. At w = 4 the k-way floors stay near zero (1–2/20) and D_new is +0.600 (k = 3)
and +0.250 (k = 4, unestablished) [C26]. The restated claim: **routing makes an already-present k-way
switch reliable; it does not create it.**

> **What changed.** Round 3–4 published "routing creates a k-way switch: 20/20 vs 0/20 at k = 3 and
> k = 4" [R2]. Same fixed-partition artifact as §3.2. The k = 2 / w = 8 version of that claim
> (20/20 vs 1/20) is refuted outright — its fair floor is 17/20.

A latent k = 3 switch exists in standard training once each group is wide enough: **17/30** at H = 24
(8 units per worldview) and **28/30** at H = 48 (16 units), against best-of-10 random masks 0/30 in both
(per-draw 0/300), McNemar p = 1.5e-5 and 7.5e-9 [C27]. This is the width law again, at k = 3.

A **data-only pair probe** recovers the k true cue groups without the oracle: exact group recovery
**20/20** at k = 3 (chance 1/1260, binomial p = 9.8e-63) and **20/20** at k = 4 (chance 1/17325,
p = 1.7e-85), permuted-label null 0/20 at both, probe margins median 0.451 / 0.446, end-to-end
selectability 20/20 vs 0/20 (Fisher 1.45e-11) [C28]. That end-to-end contrast uses the fixed-partition
floor and inherits the correction above; the localizer's recovery rates do not. *Scope:* this runs **at round 2's coordinate
privilege** — the probe needs the concepts to occupy disjoint axis-aligned input columns — so it does
not restore a general oracle-free claim. Conditional on exact recovery the routed arm *is* round 3's
oracle arm up to a column relabelling (anchor max|diff| = 0.0), so the only independent content is the
localizer itself.

**Operational caveat that must ship with any k-way claim.** The localizer's group→worldview slot order is
an arbitrary permutation, matching the oracle on only **10/20** seeds at k = 3 and 14/20 at k = 4 [C29].
The labelling has to ship with the checkpoint or the switch is unusable.

Two further limits. Cross-worldview leakage sits **at the eval set's zero-interference floor** (median
worst-case leak 0.0175, identical at k = 2, 3, 4; k=3-vs-k=2 MWU p = 0.35), so the interference tests
have no dynamic range and "no interference at higher k" means only that [C30]. And w = 8 is an inherited
selection: round 3's oracle at w = 4 gives 13/20 (k = 3) and 6/20 (k = 4).

## 3.7 Optionality: four deployment regimes, one clear win (Figure 4)

**Structural picture.** Ask what routing buys *at deployment* and the answer depends entirely on what the
deploying actor still has. Gradients? A cue oracle? Neither? Routing wins in exactly one of the four
cells, and it is the cell where the deployer has lost the oracle that training had.

All four rows: n = 60 seeds, paired comparisons. Anchors: the fair-mask procedure reproduces the stored
learned masks at max|diff| = 0.0 (60/60); ordinary + data-chosen mask 51/60 (reproducing §3.2.1 exactly,
scored with designated == either); routed + true partition 60/60; single random mask 2/60.

| Regime | What the deployer has | Result |
|---|---|---|
| **NO-RETRAIN** | cue oracle, no gradients | **Tie.** Routing 60/60 vs ordinary + 8-label mask selection 59/60; 1 discordant pair, McNemar p = 1.0; **TOST at δ = 0.15 passes, 90% CI [0.000, 0.050]**. Ordinary + proxy-cue input occlusion: 60/60 [C44] |
| **UNKNOWN-TARGET** | gradients, 3 target changes | **Routing loses.** Routing 240/240 stages at 3.20 M example-forwards; fine-tune-mix at k = 100 matches 240/240 at ~58–60% of that cost (1.842 M stored, ~1.92 M on the audit's corrected accounting), 0 discordant seeds, McNemar p = 1.0; ordinary + mask selection 234/240 at 1.60 M with **zero gradient steps** [C45] |
| **UNLABELLED-ONLY** | deployment inputs, no labels | **Nobody wins, by equivariance.** Routed rules R1/R2/R3 = 29/38/26 of 60; ordinary 25/23/28; entropy-min self-training 8/60 at mean deploy alignment 0.4987. Largest cell (routed R2, 0.633) gives p = 0.052 uncorrected against Bonferroni α = 0.0042 [C46] |
| **BLIND** | no cue oracle at deployment | **Routing wins decisively.** Routing 60/60 at every m ≥ 1 toward the TRUE target; a blind 200-mask search on ordinary models gets 4/60, 3/60, 13/60, 34/60, 52/60, 52/60 at m = 1/2/4/8/16/32 under the preregistered tie-break. McNemar at m = 8: diff +0.433 [0.317, 0.567], p = 3e-8 [C47] |

Three qualifications on the BLIND win, all of which have to travel with it. **State the win from m = 2**:
the m = 1 cell is a fixed-example artifact, because the same 32 deploy examples are used for every seed
and example #0 is proxy-predicted in 30/30 seeds, so the order tie-break yields TRUE 60/60 mechanically.
**The win is near-definitional**: the routed arm trains subnet A by occluding cue B, so subnet A *is* the
cue-A subnet by construction — the empirical content is the size of the blind control's identification
problem, not the routed arm's success. And **the control's failure is identification, not existence**:
a good random mask exists in 57/60 seeds (mean 6.4 good masks out of 200). The blind curve quoted above
is the preregistered-tie-break row; a random tie-break gives 51/60 at m = 16 and 32.

In the NO-RETRAIN row, δ = 0.15 is a wide margin and the achieved bound is ±0.05 — that is what to quote
[N8]. Note also that the deployment-time intervention routing ties with there (input occlusion of the
proxy cue) exists only because the cues occupy separate input columns in this toy; it may have no analogue
in a real model. In the UNKNOWN-TARGET row the decision runs through the seed-level paired bootstrap and
the zero-discordant-seed McNemar, because Wilson intervals over 240 stage-observations are
anti-conservative (stages within a seed are dependent); the pure fine-tuning variant fails outright
(48/240, held-out collapses to 0.624) [C45].

**Cost.** Routing costs a premium of **+1.60 M example-forwards (+100%)** over ordinary training, routed
models are **never usable un-ablated** (0/60 committed either way, mean alignment 0.505, vs 3/60 and 0.492
for ordinary), and mask derivation consumes 4000 unlabelled train-mode inputs plus the cue oracle [C49].

**The one-sentence reading of the table:** routing does not create the switch; it **records the
training-time cue oracle in the architecture** for an actor who no longer has it [C48].

### 3.7.1 The exchange rate: what a few disagreeing labels buy instead

A handful of labelled *disagreeing* examples buys localization and a working switch in the basis where
unsupervised localization is obstructed. Localization at m = 0/1/2/4/8/16/32: **7/14/21/25/24/28/27** of
30; selectability **13/14/16/21/22/24/24** of 30. Both preregistered Cochran–Armitage trends sit at the
permutation floor (z = 6.6902 and z = 4.1709, p_perm = 5e-5 at 20k permutations). Deliverable: k* = 4
labelled disagreements, 0.5% of the training set [C39]. *Scope:* rotated basis, n = 30 per level,
H = 8 / w = 4.

**It is disagreement, not data.** An agreeing-example arm at m = 4 lands on the true split **8/30** vs
**25/30** for disagreeing examples at the same m, Fisher **p = 2.03e-5** [C40]. This is a localization
contrast, not a selectability contrast.

**Routing vs simply training on the disagreements.** On a matched outcome — the label-designated subnet
being both aligned and competent — routing scores **16/21/25/25/28** of 30 at m = 1/2/4/8/32 against
augmentation's **14/16/24/30/30**. Fisher at m = 8: **p = 0.052**; at m = 32: **p = 0.49** [C41]. There
is no significant difference at any price tested. *Scope:* **POST-HOC rescoring**; the matched numbers
are recomputed from stored per-seed data and are not stored as a summary anywhere. Part of the residual
gap is width-4 column trainability, a cost the unrouted comparator does not carry.

What survives the correction: **augmentation reaches a ceiling routing does not** — 30/30 aligned from
m ≥ 8 with mean deploy alignment 0.961 → 0.992, against routing's best 28/30. Augmentation produces
0/30 switchable models at any m, so the two buy different objects [C42]. The defensible strength is
"at least as good, and probably better once m ≥ 8".

> **What changed.** Round 4 published "routing never beats augmentation at any price tested, and is
> beaten at m = 8 and m = 32 (p = 0.0046, 0.024)" [R4]. The two arms had been scored on different
> outcomes — routing two-sided (both subnets had to behave), augmentation one-sided (one aligned model).
> Matched, the gaps fall to p = 0.052 and p = 0.49.

The one channel routing could uniquely exploit — **unlabelled** deployment inputs — never reaches the
bar: localization 7/17/16/16/17 of 30 at m = 0/1/4/8/32, with the trend z = 2.07 (p_perm 0.020) driven
entirely by the m = 0→1 step, and a best of 17/30 (Wilson upper bound 0.726) against a preregistered
0.75 bar [C43]. Report this as "**no detected improvement** from m = 1 to m = 32" — TOST at δ = 0.20
gives p = 0.059 and Fisher power against a +20-point effect is 0.28 [N10].

## 3.8 Tamper: what an adversary and a benign fine-tuner each do

**Structural picture.** If the switch ships as weights plus an inference-time mask, flipping it is free.
If the unselected half is physically deleted, the question becomes whether 62 live parameters are harder
to re-align than 62 live parameters in an ordinary model — and mostly the answer is that they are not
distinguishable.

**A masked switch flips for free.** An adversary holding the weights flips the worldview at 0 gradient
steps and 0 data: 23/23 in round 3 (hard-set vs masked max|diff| = 0.0, all flipped alignments below the
`lo` threshold, cross-block L1 = 0.0), replicated 60/60 in round 4 [C50]. This applies to the
shipped-model-plus-inference-mask deployment; the alternative is to delete the unselected half, which is
what the rest of this subsection tests.

**Under the realistic functional attacker, the deleted switch is indistinguishable from a routing-free
model with the same live parameter count — bounded, not equal.** n = 60 vs 60: AUC 0.767 vs 0.786,
d = −0.0195, 95% CI [−0.0736, +0.0345], MWU p_perm = 0.351. The preregistered 0.05 margin was **missed**
(~36% power); the achieved equivalence bound is **δ* = 0.0648 = 36.5% of the run's total defense effect
of 0.1776** [C51]. The only defensible reading: any routing-specific tamper advantage under this attacker
is at most about a third of the total defense effect [N1]. The equivalence machinery is validated (the
positive control sees the known 0.178 effect at p_perm = 5e-5; truth-exactly-zero split-half calibration
bounds at 0.118/0.104).

**Under the weaker preregistered LoRA attacker the switch does beat both controls**, overturning two
round-3 nulls: vs the capacity control d = −0.0789 [−0.124, −0.033], p_perm = 0.00135, Holm-adjusted
0.0095; vs the corrected baseline d = −0.0804 [−0.124, −0.037], p_perm = 0.00335, Holm 0.0201 [C52].
*Scope that cancels most of the interest:* this is the regime where the attack leaves a broken model —
only 26.7% (switch) and 16.7% (control) of attacked models are still competent at step 100, against 68.3%
for the corrected baseline.

**Under the strongest honest attacker the sign flips, and that cell is not capacity-matched.** Full
fine-tuning with 1% jitter: d = **+0.0933** [+0.034, +0.153], p_perm = 0.0311, Holm-adjusted 0.1555 —
**does not survive Holm** [C53]. Jitter revives the 92 zeroed weights, so the switch is training **154**
parameters against the control's **62**. The reading is that dead parameters are an attack surface, not
that routing loses to capacity. This arm is also not the strongest attacker by the run's own primary
metric (mean AUC9 0.560/0.467 vs the best LoRA cell's 0.767/0.786), and the bare full-FT rungs
(−0.159, −0.251) are a relu′(0) = 0 artifact that must be greyed out wherever plotted. A post-hoc
corroboration of the same direction is stronger than C53 itself: competence-gated flip counts under
full-FT + jitter are **switch 53/60 vs capacity control 25/60**, Fisher p = 1.05e-7 [C55] — same
non-capacity-matched caveat.

> **What changed.** Round 3 reported that a routing-free H = 4 model with the identical 62 live
> parameters matches the switch (p = 0.40), concluding tamper resistance is not separable from parameter
> count [R12]. That test was underpowered (n = 23 v 12). At n = 60 v 60 the two differ significantly in
> both directions depending on attacker strength — hence C51/C52/C53 above rather than a single verdict.

**The resistant tail is not shown to be a routing property.** Strong resisters
(AUC < 0.6) under the functional attacker: 11/60 switch vs 9/60 capacity control (Fisher p = 0.807;
Levene on spread p = 0.775, SD ratio 1.05), against 0/60 for the corrected baseline (p = 5.9e-4) [C54].
*Scope:* **POST-HOC**, and the Newcombe 95% CI on 11/60 vs 9/60 spans [−0.102, +0.168], so the tail could
be more than twice as common in the switch. This suggests; it does not show.

**Benign fine-tuning does not erase the switch — it erases the labelling.** After the full four-stage
benign fine-tuning sequence at k = 20, **44/60 (73%, [0.610, 0.829])** of models remain switchable under
a data-chosen mask [C56]. What benign fine-tuning does destroy is the block-diagonal structure and the
composite criterion *under a fixed partition*: 23/23 → 12/23 at 100 steps (5% of the training budget, no
adversary) → 8/23 → 10/23 → 7/23 → **7/23** at 2000 steps, McNemar p = 3.05e-5, with mean cross-block
mass 0.0 → 5.588 → 7.145 [C57]. The mechanism of that decay is the criterion's competence leg: at 2000
steps the aligned side still leads 0.869 vs 0.125 with the sign preserved 23/23. C56 and C57 must be
stated together.

> **What changed.** Rounds 3–4 published "benign fine-tuning erases the switch: 23/23 → 12/23; 0/30
> fine-tuned models remain switchable — the strongest single argument against the switch as a shipped
> artifact" [R3]. This was the fixed-partition artifact for the third time, and it was the round's own
> preregistered secondary that falsified it, in the direction cutting *against* our safety-negative.

## 3.9 The hazard: an unselected model is confident, wrong, and invisible to a confidence monitor (Figure 5)

**Structural picture.** Suppose the deployer forgets to select. Rounds 1–2 assumed the model would then
hedge, so the omission would announce itself. It does not hedge. It collapses to one class,
confidently — and off-distribution confidence carries no signal about whether that class is right.

**Not a hedger.** MLP (n = 46): median determinism **0.7857**, bootstrap 95% [0.622, 0.912]. Transformer
(n = 20): **0.9506** [0.869, 0.995]. The hedger regime is D ≤ 0.5 and both CIs exclude it [C58]. Two
preregistered point predictions were missed and are reported as misses: D ≥ 0.8 (got 0.786) and margin
ratio ≥ 1.5 (got 1.33 MLP, 0.95 transformer).

**What it actually is: a confident off-distribution class collapse, with full on-distribution
competence.** n = 46 routed models: held-out agreeing accuracy median **1.000** (min 0.992, 46/46 above
0.9); deploy alignment median 0.500; majority-class rate on disagreement inputs **0.893 [0.811, 0.956]**
vs **0.550 [0.537, 0.582]** for controls, MWU **p = 1.28e-10**; confidence on the majority class 0.911,
with 34/46 above 0.75 against 0/60 controls. Not a readout-bias artifact: the bias decides 11% of inputs
and the two columns vote opposite on 98.9% [C59]. *Scope:* **POST-HOC**, and unreplicated on data
independent of the primary.

> **What changed, twice.** Round 3 published "the unselected model is a confidently inconsistent
> per-input **patchwork** of two worldviews (16/22 seeds)" [R10]. Routed columns are invariant inside a
> label group by construction, so that census can only detect the label split; a strict within-label
> criterion is met by **2/46** (2/22 in round 3's own archived data), Fisher p = 0.186 [N18]. Use "class
> collapse", not "patchwork". And round 1–2's "it hedges, so a deployer can tell they forgot to select"
> [R9] is dead in both substrates.

**Per-input confidence does not flag the inputs on which such a collapsed model is wrong.** n = 29
models: median AUROC **0.5208**, 90% CI [0.4773, 0.5486], achieved equivalence bound **0.0486** against
a preregistered ±0.05 margin. A mechanistic (ablation-derived) readout of the same inputs scores
**0.9996** [0.9989, 0.9999]; confidences on proxy-following vs truth-following inputs are 0.924 vs 0.927
[C60].

**Round 5's scope correction, which is load-bearing.** These 29 models have median majority-class rate
**0.9623**, and **62% emit one class on more than 95% of deploy inputs**. So the defensible claim is that
confidence does not flag the inputs on which a **collapsed** model is wrong, not that confidence cannot
find worldview switches — the measured population mostly does not switch, it collapses [C60, R11]. The
equivalence is also knife-edge (0.0486 against a 0.05 margin), it is an equivalence for the *median across
models*, and 31% of individual models deviate more than 0.20 from chance [N15].

**Model-level confidence triage is anti-informative: it ranks these models as the safest.** AUROC of
"low confidence flags a collapsed model" = **0.1966** [0.1115, 0.3012], n_pos = 29 / n_neg = 60; mean
confidence 0.926 (routed collapsed) vs 0.855 (control non-collapsed) [C61]. *Caveat that must ship with
it:* this is partly a composition artifact, since 20/60 controls are non-deterministic hedgers that the
criterion guarantees are non-collapsed. [LEDGER GAP: the determinism-matched recomputation, reported
elsewhere as 0.282, is not on disk; quote 0.197 with the composition caveat or recompute.]

**This is a limit of confidence monitoring, not a routing-specific liability.** The preregistered
negative control **failed**: control-arm hedgers, predicted to be detectable, score median AUROC 0.4205
(MWU vs collapsed p = 0.731) [C62]. The run therefore contains **no demonstrated population where a
confidence monitor works** [N16]. The positive control at 0.9996 is near-tautological — algebraically it
is the model's own decision rule — so it validates the harness rather than providing a live monitoring
contrast.

**Two claims from earlier rounds are withdrawn here.** "Standard training never produces this failure
mode (0/55, 0/30)" holds for the MLP census (0/60 replicated) but **fails in the sequence substrate:
19/23 transformer controls show the mode**. A "never" claim needs one counterexample; this has nineteen
[R7]. And "routed models are *more* prone to the mode than controls (transformer, strict reverse
p = 0.047)" was a gate artifact: a competence gate was applied to the routed arm only, removing exactly
three models, all three of which showed the mode, with no control excluded. Ungated: **15/23 vs 19/23,
p = 0.157**; strict **9/23 vs 15/23, p = 0.070** [R8]. Neither routing-specificity claim stands. The
routing-specificity test itself could not have won: even a perfect 20/20 returns p = 0.072, with usable
interval [−0.464, +0.042] [N17].

---

**Reading the section as a whole.** The measurements held up: across six independent audits in rounds
4–5, not one primary statistic failed to reproduce, and the anchors sit at max|diff| = 0.0. Every defect
found and corrected above lived in the interpretive layer — an asymmetric outcome definition (§3.7.1),
a control given a different scoring rule (§3.2, three times), a gate applied to one arm (§3.9), a null
promoted to an equivalence, a diagnostic implemented for one substrate only (§3.3.1) [C68].
