# Routing buys a label, not a switch: capacity and basis conditions for post-hoc worldview selection in underdetermined models

> **Scope, up front and not negotiable.** Everything below is toy scale: synthetic tasks with ≤11-dimensional
> inputs, 154–1.2k-parameter MLPs, and one 68,546-parameter from-scratch transformer (G1). The "worldview"
> is a 2-bit XOR (G2). "Deployment" is a single held-out set on which the two cues disagree (G3). Every
> mask-selection procedure we report, including the fair control, consumes the training-time cue oracle
> (which input dimensions or token positions carry which cue), except two localizers that carry their own
> conditions (G4). "Selectable" is a composite criterion: one ablated half > 0.8 deploy alignment toward the
> target, the other < 0.2, both > 0.9 on held-out agreeing data (G5). **No claim here licenses any statement
> about frontier models, LLMs, or any system trained at scale.**

---

## Abstract

Two rules that agree on all training data and disagree only at deployment leave the choice between them to the
seed. We ask whether that choice can be set after training, by routing each rule's gradient into its own hidden
columns. Five preregistered rounds, three adversarial to our own best result.
(i) The lottery reproduces at two scales: 27 aligned / 27 misaligned / 46 hedgers over 100 MLP seeds, and
deploy alignment 0.000–0.933 across 12 transformer seeds, all at accuracy 1.000 on cue-agreeing data.
(ii) Ordinary training already contains the switch: under a data-chosen mask rather than one arbitrary
partition, 51/60 standard models split into two competent halves committed to opposite worldviews; our
headline effect falls from ~+0.98 to **+0.150** at 8 units per worldview (+0.667–0.700 at 4 units; +1.000 in
the transformer). (iii) Selectability is gated by per-half competence and by basis: in one 68.5k-parameter
transformer both halves are competent, yet no axis-aligned mask we built selects (0/30, Wilson upper 0.114).
(iv) Routing's residual value is bookkeeping — it records the training-time cue oracle for a deployment actor
who no longer has it — not tamper resistance. Toy scale throughout; nothing transfers to frontier models by
default.

---

## 1. Introduction

### 1.1 The structural picture

Put two rules in the data that never disagree. A "true" rule and a "proxy" rule, matched in cost, both perfect
on every training example and on held-out data drawn the same way. Nothing in the loss distinguishes them,
because on the training distribution they are the same function. The optimizer is not choosing badly between
them; it is not choosing at all. Which one the network ends up computing is decided by initialization and batch
order, and you find out only at deployment — the one place the two rules come apart.

That is the whole setup, and it behaves exactly as the picture predicts. Over 100 fresh seeds of a small MLP,
all at training accuracy 1.000, the deploy split is 27 aligned / 27 misaligned / 46 hedgers, with a mean deploy
score of 0.509 and a significant Hartigan dip (p = 0.00145, surviving Holm over a 25-regime grid) — a lottery,
not a compromise (C1; MLP only, and this is the only jitter-robust well-fit bimodal cell of 20, so bimodality is
not generic across the grid). It is not an artifact of tiny MLPs. In a 68.5k-parameter from-scratch transformer
where the cue is a circuit rather than a coordinate, 12 seeds all reach held-out accuracy 1.000 while their
deploy alignments span 0.000–0.933 (χ²(11) = 15212.8, p < 1e-300) (C2). In that task the cue genuinely is a
circuit: the token multiset is constant across examples, so a bag-of-tokens model is at Bayes accuracy exactly
0.500 (C3).

So the question is not "which rule does gradient descent prefer". It is: **can the choice be made explicit, and
set after the fact?**

### 1.2 The intervention, and why it should work

Gradient routing gives an obvious mechanism. Split the hidden layer into two column groups, zero the
cross-blocks at initialization, and mask the gradient so the cross-blocks never receive an update. They stay
zero. The layer then decomposes exactly: Δ_full = Δ_A + Δ_B + bias, with additivity residual at most 3.8e-6 and
cross-block max|w| = 0.0 on 30/30 seeds, against 0.882 in a routing-free control (C21). Each surviving column
is its own cue's rule in closed form — the 4-corner XOR truth table read straight off the weights, no fitting,
44/44 columns on switchable seeds (C22), though that readability is a property of the 2-bit generator (G2), not
a discovery about worldviews. Ablate one group and the other's rule is what the model computes. The composite
behaves as a margin race between the two columns (C23), and the ablation dial, though graded at the population
level (transition width W = 0.289, CI excluding the 0.25 rule), is a per-input threshold: 99.96% of the inputs
that change flip exactly once (C24).

The intended reading was: routing turns an unselectable model into a selectable one. We reported that for three
rounds.

### 1.3 What we found, including the part where we broke our own headline

The routing-free control was scored under **one arbitrary fixed unit partition** — a single coin toss over
12,870 possible 8/8 splits — while the routed arm was evaluated under the partition it was built with. Give the
control the same freedom, and derive its mask from data (cue sensitivity on held-out *agreeing* inputs only),
and 51/60 standard-trained models already contain two unit-disjoint, individually competent halves committed to
opposite worldviews: 1/60 → **51/60**, selection margin −0.018 → +0.910 (C4). The mask is measuring something
specific, not fitting noise: independently re-derived on 10 retrained models it finds 8/10 genuine full splits,
against 1/50 for random 8/8 splits and 0/10 for noise-dimension masks (C5). Random splits of the *same* models
succeed 18/4260 = 0.42% across the portfolio (C10).

Applying one identical rescoring procedure to every routing-free control in the line moves the pooled floor
2/282 → 100/282, and it moves almost entirely where the ablation groups are wide — 88/120 at 8 units per
worldview against 12/150 at 4 (Fisher p = 2.5e-30, POST-HOC, and w is confounded with total hidden size)
(C8). Our previously reported effect of ~+0.98 (R1) becomes **+0.150 [+0.058, +0.261]** at 8
units per worldview (C12) — a sixfold shrink, and that is the number to quote whenever the paper says "routing's
effect size" without qualification. The correction is real but not uniform: of 13 rescored cells, 9 survive, 2
shrink, 2 die (C15). At 4 units per worldview — rounds 1 and 2's actual configuration — routing still
contributes +0.667 and +0.700 (C11), it survives an entangled input basis at +0.400 (rotated) and +0.567
(non-orthogonal mixing) (C13), and in the transformer's shared residual stream the correction changes the number
by exactly zero: 0/12 → 0/12, effect +1.000 (C14).

The same artifact had inflated two other headlines. The k-way claim ("routing creates a 3- and 4-way switch,
20/20 vs 0/20") becomes "routing makes an already-present k-way switch reliable", against fair floors of 11/20
and 9/20; the k = 2 version is refuted outright (C26, R2). And the claim we had treated as the strongest
argument *against* shipping a routed model — "benign fine-tuning erases the switch, 23/23 → 12/23 → 0/30" —
inverts: 44/60 (73%, [0.610, 0.829]) fine-tuned models are still switchable under a data-chosen mask (C56, R3).
What benign fine-tuning destroys is the *labelling*, not the switch; the block-diagonal structure does drift
(C57).

Three headline effect sizes, one artifact, three rounds. We found all three ourselves, and §3 of the claims
ledger — the retraction table — is a section of this paper rather than an appendix.

**What replaces them is a pair of conditions.** Whether ordinary training leaves a post-hoc-selectable switch is
a monotone function of hidden width: 2/30 → 25/30 → 29/30 → 30/30 at H = 8/16/32/64 (Cochran–Armitage z = 7.97,
p = 7.8e-16) (C6). The gate is per-half **competence** — conditional on both data-chosen halves clearing the 0.9
held-out bar, the mask selects 2/5, 25/26, 29/29, 30/30 (C7, POST-HOC, and the H = 8 cell has effective n = 5,
so "no latent switch at H = 8" is a statement about what n = 5 can answer, N7). Width is one condition; basis is
the other, and they dissociate. In the transformer both data-chosen halves *are* competent — zero competence
failures, held-out min 1.000 — and the axis-aligned mask still selects **0/30** (Wilson upper 0.1135;
best-of-10 random masks 0/30, per-draw 0/300; fixed partition 0/30) (C17). Capacity is excluded as the
explanation: a units-only mask giving each worldview 64 MLP units per layer — 4× the width at which the MLP
saturates — leaves both halves competent 12/12 at held-out 1.000/1.000 and still selects 0/12 (C18, POST-HOC).
That is a preregistered primary with a prestated power table (0.87 against a true rate of 0.60), so rates above ~0.11
are excluded and rates in 0.10–0.40 are not (N6). One organism, one axis-aligned mask family, and **the
instrument's positive control is not on disk** — we state the null with that caveat attached rather than with an
instrument check we cannot show.

**So what is routing still for?** We tested four deployment regimes at n = 60, paired. With the cue oracle and no
gradients, routing *ties* ordinary training plus a no-gradient intervention, 60/60 vs 59/60, and this is the one
genuine equivalence in the line (TOST at δ = 0.15 passes, 90% CI [0.000, 0.050]) (C44, N8). With gradients
available and an unknown target, routing *loses*: 100 fine-tuning steps per target change match it at ~58–60% of
the cost, and ordinary training plus mask selection switches target 234/240 with zero gradient steps (C45). From
unlabelled deployment inputs alone, nobody wins, by equivariance (C46). Routing wins decisively in exactly one
regime — when the deployment actor has **no** cue oracle: 60/60 from two labels onward, against a fair 200-mask
blind search that reaches 3/60 at m = 2 and saturates at 52/60 by m = 16 (preregistered tie-break row) (C47).
That win is near-definitional
(subnet A is trained by occluding cue B, so it *is* the cue-A subnet by construction); its empirical content is
the size of the blind control's identification problem, and the failure there is identification, not existence —
a good random mask exists in 57/60 seeds. The honest one-sentence reading: **routing does not create the switch;
it records the training-time cue oracle in the architecture for an actor who no longer has it** (C48). It costs
+100% in example-forwards, and routed models are never usable un-ablated (C49).

**Negative results, stated early.** Routing is not a safety property. Against a hard-deleted switch, a realistic
functional attacker cannot distinguish it from a routing-free model with the same live parameter count —
bounded, not equal, at δ* = 0.065 = 36.5% of the run's own total defense effect (C51, N1). It does beat a
capacity control under a weaker LoRA attacker (d = −0.079, p_perm = 0.0014), but that is the regime where the
attack leaves a broken model (C52). Under the strongest honest attacker the sign flips (+0.093, not surviving
Holm), and that cell is *not* capacity-matched — jitter revives 92 zeroed weights, so the switch trains 154
parameters against the control's 62; it shows dead parameters are an attack surface, not that routing loses to
capacity (C53, R14). And an unselected routed model does not announce itself. It is not a hedger (median
determinism 0.786 MLP, 0.951 transformer, CIs excluding the hedger regime) (C58); it is a confident
off-distribution class collapse while scoring 1.000 on held-out agreeing data (C59, POST-HOC); and per-input
confidence does not flag the inputs on which such a collapsed model is wrong (median AUROC 0.521, 90% CI
[0.477, 0.549] inside a preregistered ±0.05 margin, achieved bound 0.0486 — knife-edge) (C60). Two guards
matter here: this is measured on a population whose median majority-class rate is 0.962, so it is a statement
about *collapsed* models, not about models that have switched worldview (R11); and the preregistered negative
control failed, so the run contains no demonstrated population where a confidence monitor works — this is a
limit of confidence monitoring, not a routing-specific liability (C62, N16). Our earlier claim that standard
training never produces this mode is retracted outright: 19/23 transformer controls do (R7). The follow-on
claim that routed models do it *more* was a gate artifact and is dropped (R8).

### 1.4 Contributions

1. **A rescoring result about controls.** A control given only one arbitrary partition understates the
   routing-free floor (1/60 → 51/60) and inflates the reported effect roughly sixfold at 8 units per worldview
   (~+0.98 → +0.150); the same artifact independently inflated three of our own headlines (C4, C8, C12; R1–R3).
2. **Two conditions, dissociated.** Post-hoc selectability under an axis-aligned mask needs per-half competence
   (the width law, C6/C7) *and* a basis in which the two worldviews are axis-separable. The transformer meets
   the first — full competence, 4× the saturating width — and no axis-aligned mask we built meets the second
   (C17/C18). Two independent conditions; width satisfies only one.
3. **An exact mechanism at toy scale.** Routing makes the two-worldview decomposition causal, not merely
   arithmetic, with a closed-form per-column rule and a per-input threshold ablation dial (C21–C24).
4. **An identifiability obstruction and a partial solution.** At exactly equal cue strength the orthogonal
   symmetry group of the labelled centres has order 64 with a 2-element orbit on the true split, so any
   *equivariant* localizer must coin-flip; at all 15 other margin ratios tested |G| = 16 with a single-point
   orbit (C31). A localizer reading the signed eigen-spectrum of D = E[xx′|y=1] − E[xx′|y=0] recovers the true
   split 40/40 at margin ratio 2.05 and lifts selectability to 38/50 against 26/50 for ICA (Fisher p = 0.021)
   (C32) — for **orthogonal** entanglement, N = 800, r above a measured onset of ~1.15 at N = 800 (C33), and as
   a model-based estimator that embodies designer knowledge of the generative family. Whitening, ICA's first
   step, is precisely what destroys the resource it uses, and re-pairing whitened components does not repair it
   (C35).
5. **An exchange rate, corrected.** A few labelled *disagreeing* examples buy localization and a working switch
   where unsupervised localization is obstructed (k* = 4 examples, 0.5% of the training set) (C39), and it is
   the disagreement rather than the data that does the work — 8/30 vs 25/30, p = 2.0e-5, which is a
   **localization** contrast (landing on the true split), not a selectability one (C40). But on a matched
   shipped-subnet outcome, routing does not significantly beat data augmentation at any
   price tested (m = 8: p = 0.052; m = 32: p = 0.49) — our round-4 claim to the contrary had scored the two arms
   on different outcomes — and augmentation reaches a ceiling routing does not (C41, C42, R4).
6. **A selection-hygiene rule.** A max-of-k deflation estimator, validated in simulation (100/100 cells within
   15% of the closed form) (C63), which diagnoses a published 24-layer sweep at ρ̂ = 0.026 and turns a selected
   −2.75 into −0.07 against a fresh truth of +0.22 [−0.72, +1.16] (C64). Its own preregistered real-data primary
   **failed** (p = 0.125) at ~53% power — uninformative rather than disconfirming (C65, N23).
7. **A meta-finding from six audits.** Not one primary statistic failed to reproduce; anchors sit at max|diff| =
   0.0 across rounds, with two disclosed exceptions. Every defect found lived in the interpretive layer: an asymmetric outcome definition, a
   gate applied to one arm, a control given a different scoring rule, a null promoted to an equivalence, a
   diagnostic implemented for one substrate only (C68). Readers should weight the measurements and discount the
   framings — including ours.

### 1.5 What this is not

It is not evidence that frontier models contain latent selectable worldview switches, and not a method for
post-hoc value selection in production systems. The transformer result is 68.5k parameters, one organism, 12–30
seeds; calling it "at scale" would be false. Several of our own success rates are **demonstrations** rather than
effect sizes — where the training mask equals the evaluation mask and the columns are disjoint by construction,
20/20 and 12/12 are existence proofs and nothing more (C20, C25). Where a number is post-hoc it is labelled
post-hoc in the sentence that states it, not only in a methods appendix. Where a null is only a bound, the bound
is quoted instead of the null.

---

## 2. Related work

### 2.1 Underdetermination and goal misgeneralization

The phenomenon we build a toy of is standard. Goal misgeneralization — a policy that retains capability
out-of-distribution while pursuing the wrong objective, because training could not distinguish objectives —
was characterized by Langosco et al. (2022) and Shah et al. (2022). D'Amour et al. (2020) made the statistical
version general: an underspecified pipeline admits many equally-good predictors that behave differently under
shift, and which one you get is an accident of the training run. Related framings include shortcut learning
(Geirhos et al. 2020), the Rashomon effect (Breiman 2001), and the observation that "spurious" features can be
genuinely predictive rather than merely artifactual (Ilyas et al. 2019). Our contribution to this literature is
narrow: we construct the underdetermination deliberately, verify that the two cues are equal-cost, and then ask
a question that literature does not — not *which* rule is learned, but whether the choice remains recoverable
and settable after training. The seed lottery itself is a replication (C1, C2), not a finding.

[CITE: the "simplicity bias" line — Shah et al. 2020 on the pitfalls of simplicity bias — if the author wants the
mechanistic-preference angle; and Rashomon-set / model-class-reliance work (Fisher, Rudin & Dominici) for the
"many equally good models" formalism.]

### 2.2 Gradient routing and localized capability removal

Gradient routing — masking gradients so that a designated data or token subset updates only a designated part of
the network — is due to Cloud et al. (2024), including the expand–route–ablate pattern our setup is a
two-worldview instance of. [CITE: exact author list and venue for the gradient routing paper.] The unlearning
literature supplies the deployment motivation: representation-level removal such as RMU on WMDP (Li et al. 2024),
concept erasure in activation space (Ravfogel et al. 2020; Belrose et al. 2023), and the general finding that
apparent removal is often shallow. Our results cut across that framing in a specific way. Routing is what makes
the decomposition *causal* rather than arithmetic (C21) — but ordinary training already contains the
unit-disjoint halves whenever each half is wide enough and the cues are axis-separable (C4, C6). Routing buys
the *label* on the halves, reliably, for an actor who cannot re-derive it (C47, C48). We know of no prior work
that scores an unrouted control under a data-chosen mask; the closest analogue is pruning/lottery-ticket work,
where a data-derived mask is the object of study rather than a control.

[CITE: Casper et al., "Eight Methods to Evaluate Robust Unlearning"; and the sparse-mask / lottery-ticket line if
the author wants the mask-as-object framing.]

### 2.3 Mode connectivity and the permutation question

That an unrouted network's units can be split into two coherent halves at all is a claim about basis, and the
mode-connectivity literature is the right prior. Garipov et al. (2018) showed distinct SGD solutions are joined
by low-loss curves; Frankle et al. (2020) established linear mode connectivity as a property of stable training;
Entezari et al. (2022) conjectured that most barriers vanish once permutation symmetry is accounted for, and
Ainsworth et al. (2023) gave practical weight-matching algorithms. Our width law (C6) and our basis dissociation
(C17) are, in that language, statements about when a *permutation-selected* partition carries semantics. We do
not test connectivity between the two ablated halves and make no claim about the loss landscape. Note the
confound we cannot remove: in the pooled portfolio, ablation-group width w is confounded with total hidden size
H = k·w, and the width contrast is post-hoc (C8).

[CITE: mechanistic mode connectivity — Lubana et al. (2023) — for the "same loss, different mechanism" version,
which is closest in spirit to our two cues.]

### 2.4 Concept localization: probes, ICA, sparse autoencoders

Finding *where* a concept lives is the step our whole pipeline turns on. Linear probes (Alain & Bengio 2017) and
their descendants; ICA and its identifiability theory (Comon 1994; Hyvärinen & Oja 2000), including the
non-identifiability of the nonlinear case (Hyvärinen & Pajunen 1999); and sparse autoencoders as an
unsupervised dictionary over activations (Bricken et al. 2023; Cunningham et al. 2023; Templeton et al. 2024).
Our results here are unusually specific about a failure mode of the generic recipe. At exactly equal cue
strength there is an *exact* symmetry obstruction — |G| = 64 with a 2-element orbit — so no equivariant
localizer can do better than a coin flip, and the ICA arm and our arm are both at chance there, with the
informative statistic being the landing profile rather than the success rate (C31, C34, N14). Off the symmetric
point the obstruction lifts (|G| = 16, single-point orbit at all 15 other ratios), and a signed second-moment
localizer recovers the true split (C32) — but its first ingredient is precisely the one ICA discards: whitening
destroys the asymmetry the estimator reads, and re-pairing whitened components does not repair it (C35). The
prescription "just run a component analysis" is the thing we can rule out on this family. Two scope limits are
load-bearing: the localizer is a model-based estimator embodying knowledge of the XOR-parity-plus-isotropic-noise
generator, and the general non-orthogonal-mixing case is **untested in the artifact** — an auditor ran it, the
numbers are not on disk, and we do not quote them. [LEDGER GAP: non-orthogonal `mix` localizer cells; C32 covers
orthogonal entanglement only.]

[CITE: steering-vector and concept-ablation work the author may want here — activation addition (Turner et al.
2023), contrastive activation addition (Rimsky et al. 2024), concept-ablation fine-tuning, persona vectors. Mark
each only if verified.]

### 2.5 Tamper-resistance evaluation

The safety case for any post-hoc switch depends on what an adversary with the weights can do, and the field's
standard is now that removal must survive fine-tuning (Qi et al. 2023). We report against three attacker
strengths and the results do not line up into a story favorable to routing: bounded-not-equal under a realistic
functional attacker (C51), a win under a weaker LoRA attacker in the regime where the attack breaks the model
(C52), a sign flip under the strongest attacker in a cell that is not capacity-matched (C53). The methodological
point we would like to contribute is the last one: **a defense whose mechanism is zeroed weights is not
capacity-matched to a smaller dense control once the attacker adds jitter**, because the zeros come back (154
trainable parameters vs 62). Any evaluation comparing a sparsified defense to a small dense baseline should
state which parameter count is live under the attack.

[CITE: tamper-resistant safeguards (Tamirisa et al. 2024) and circuit-breakers / representation rerouting (Zou
et al. 2024) — both are the natural comparison points for "make removal survive fine-tuning", but verify the
author lists before citing.]

### 2.6 Selection inflation, preregistration, and audit

The line's methods contribution is defensive. A number chosen as the best of k conditions is biased upward by
roughly a_k·s·√(1−ρ), and the fix is to shrink toward the sweep mean by an estimated ρ̂ (C63) — the garden of
forking paths (Gelman & Loken 2013) in a form you can apply to an already-published sweep. Applied to a
published 24-layer probe sweep, it says there was no layer-to-layer signal to select on (C64) — which licenses
"no layer effect was demonstrated", not "the layer effect is absent". Each round carried a preregistration
stamped before its confirmatory run — but not everything we report is preregistered: the width contrast (C8),
the class-collapse characterisation (C59), the matched exchange rescoring (C41), the margin race and flip
structure (C23/C24) and the resistant-tail analysis (C54) are post-hoc, and are labelled post-hoc where they
are stated. Six independent audits across rounds 4–5 found that every primary statistic reproduced while every
defect lived in the interpretation (C68). That asymmetry is the most transferable thing in the paper:
measurement discipline is comparatively easy to enforce and framing discipline is not.

[CITE: preregistration in ML — if the author has a preferred reference for prereg practice in empirical deep
learning, insert here.]

### 2.7 Model organisms, and the honest position of this work

This is toy-scale mechanism work in the model-organism tradition (Hubinger et al. 2024), where a behavior is
constructed deliberately so that its mechanism can be read out completely. The trade is explicit: we get an
exact causal decomposition (residual 3.8e-6), a closed-form per-column rule, an enumerable symmetry group, and
preregistered power tables — at 154 parameters, on a worldview that is a 2-bit XOR. What we cannot offer is any
evidence about scale. The single organism above 1k parameters is 68.5k parameters, one architecture, one mask
family — and it is the organism carrying our strongest surviving claim, which is a null (the axis-aligned mask
finds nothing despite full competence) with no on-disk instrument check. Whether that generalizes to real
shared residual streams is a hypothesis, and we write it as one.
