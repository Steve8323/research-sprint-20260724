# 2. Setup and methods

Everything below is toy scale, and stays toy scale: inputs of at most 11 dimensions, MLPs of
154–1.2k parameters, and one from-scratch transformer of 68,546 parameters (G1). Nothing in this
section, or anything built on it, licenses a claim about a frontier model. Where a design choice
buys the result by construction we say so at the point of the choice, not in a limitations
paragraph at the end.

## 2.1 The task: two rules that agree until they don't

The structural picture first. Imagine a labelling job that two different people could do and never
disagree — until one day they meet a case nobody has seen. During training, the network sees only
the cases they agree on. At deployment it sees only the cases they don't. Which person's rule the
network absorbed is then fully visible, and nothing in training touched it.

Concretely: the input carries two cues. A **true** rule and a **proxy** rule, each a sign-product
(parity) function of a disjoint pair of input coordinates, searched to be **equal-cost** so that
neither is the cheaper thing to learn (G2). The training set and the held-out set are both drawn
from the region where the two rules assign the same label. The **deployment** set is the single
held-out set where they disagree (G3).

Two consequences to keep in view throughout:

- The "worldview" here is a 2-bit XOR. When we later say a routed column's rule can be read off
  the weights in closed form with no fitting (C22: 44/44 columns on switchable seeds match the
  4-corner truth table, control 13/60), that is a property of this generator, not a discovery
  about worldviews (G2).
- There is exactly one deployment distribution. Every deploy number is a repeated measurement on
  one distribution, not a test of generalization across distributions (G3).

### The MLP substrate

A small ReLU MLP, trained to convergence on the agreeing data — training accuracy 1.000, minimum
1.000 across 100 fresh seeds (C1). The smallest configuration has 154 parameters, of which routing
zeroes 92, leaving 62 live (C53); the width-8 configuration has 434 (C16).

The knob we sweep is hidden width H, reported throughout as **w = units per worldview**: H =
8/16/32/64 gives w = 4/8/16/32 (C6). For k = 3 worldviews we use H = 24 and H = 48, i.e. 8 and 16
units per worldview (C27). This distinction is load-bearing — nearly every effect size in the
paper is a function of w — and w is confounded with total hidden size, since H = k·w and no sweep
holds one fixed while moving the other (C8).

[LEDGER GAP: optimizer, learning rate, epoch count, training-set size for the MLP arms, and which
H the 154-parameter count corresponds to, are not recorded in the ledger. The exchange-rate arms
imply a training set on the order of 10³ examples, since m = 4 labelled examples is quoted as 0.5%
of it (C39).]

### The transformer organism, and why bag-of-tokens is at chance

The MLP substrate has an obvious escape hatch: the cue is a coordinate, so any statement about
"finding the cue in the network" might be a statement about finding a coordinate. The transformer
organism closes it. It is a from-scratch transformer, d = 64, 2 layers, 4 heads, 68,546 parameters,
trained on a sequence version of the same agreeing/disagreeing construction (C2).

The task control is what makes this substrate worth having. The sequences are built so that the
**token multiset carries no information at all**: token-count standard deviation across examples is
exactly 0.0 in both the training and the deploy split, so the Bayes accuracy of any bag-of-tokens
model is exactly 0.500 (C3). A model can only be right by reading order — by running a circuit over
positions. The cue-occlusion intervention used later respects the same constraint: occluding a cue
leaves the token multiset unchanged, changing a mean of 0.98 positions (C3). This statement is
about this task construction only; it is not a general claim about sequence tasks.

## 2.2 What "worldview" means operationally

**Deploy alignment** of a model (or of an ablated half of one) toward a target worldview is the
fraction of deployment inputs — the disagreeing ones — on which its output matches the label that
worldview assigns. 1.000 means it follows that rule everywhere the two rules part company; 0.000
means it follows the other one; 0.5 means it is doing neither cleanly.

**Selection succeeds** on a model when all three of the following hold (G5):

1. one ablated half reaches deploy alignment > 0.8 toward the target worldview (`hi = 0.8`),
2. the other half falls below 0.2 (`lo = 0.2`), and
3. **both** halves score > 0.9 on held-out *agreeing* data (`train_ok = 0.9`).

Leg 3 is the competence gate, and it is measured on agreeing data while legs 1–2 are measured on
disagreeing data. That separation is what keeps competence from being scored by the same numbers
that score commitment. The gate is also where the sprint's own worst control error lived, twice: a
competence gate applied to one arm and not the other removed exactly 3 models, all 3 of them
positive cases (R8), and the same gate was implemented as a diagnostic for the transformer and
omitted for the MLP for a full round (C7). When we report a gated number in this paper, the
ungated number is reported beside it.

Two derived descriptors are used in the hazard analysis. A model is **committed** if its un-ablated
deploy alignment clears `hi` or falls below `lo`; routed models are never usable un-ablated on this
criterion (0/60 committed either way, mean alignment 0.505, versus 3/60 for ordinary training,
C49). **Determinism** D summarizes how repeatable a model's off-distribution behaviour is, with
D ≤ 0.5 defined in advance as the "hedger" regime (C58). [LEDGER GAP: the exact estimator for D is
not recorded in the ledger.]

## 2.3 The routing intervention

What physically changes: the second-layer weight matrix is written in block form, rows and columns
grouped by worldview, and the **off-diagonal (cross) blocks are zeroed at initialization and never
receive gradient**. The network stops being one net with a structured weight matrix and becomes
two nets that share only the summing readout.

Zeroing a weight is not enough on its own — an ordinary zeroed weight still gets gradient and
comes back on the next step. Routing keeps the block dead by making the forward pass use the masked
weight, so the derivative with respect to every masked entry is identically zero at every step, not
merely small. What the ledger records is the verified outcome rather than the implementation form:
cross-block max|w| = **0.0 on 30/30 seeds**, and a per-parameter gradient trace over cross-block
parameters reading **0/108, 0/144, 0/24** in the k-way configurations (C21, C25). By contrast the
same blocks in an ordinarily-trained control reach 0.882 (C21).

[LEDGER GAP: whether the mask is applied to the weight in the forward pass or to the gradient in
the backward pass is not recorded; the ledger records only that cross-block weights stay at
exactly 0.0 and cross-block gradients are exactly 0.]

The payoff is that the decomposition becomes exact rather than approximate. For a routed net,
Δ_full = Δ_A + Δ_B + bias holds to a maximum residual of **3.81e-6** (C21) — the two columns are
literally separate ~62-parameter subnets plus a summing readout, so ablating one is a causal
intervention and not an approximation. In the control the same arithmetic split exists but is not
causal.

**Partial-routing variants.** To ask how much sharing selection tolerates, six variants relax the
disjointness at fixed architecture, indexed by shared parameters out of total: ceiling (2/434), 25%
unit overlap (28/352), 50% (53/278), a fully shared first hidden layer (130/434), and soft
train-mask leaks at 0.25 (328/434) and 0.50 (396/434 = 91%), plus a 75%-overlap arm (93/212) that
breaks (C16). Two of these break the *leak level* rather than the partition: `shared_trunk` still
shares zero gradient in the later layers, and the soft-leak arms recover the training partition at
Jaccard 1.000/0.996/0.979 (C16). Any reading of the sharing dose-response has to carry that.

## 2.4 Selection

Selection is: pick a group of hidden units, zero them, measure deploy alignment of what remains;
then do the same for the complement. In the MLP the groups are hidden units. In the transformer
the ablation family is **2 heads + 64 MLP units per group, with the mask shared across both
layers** (C14); the mask search space is axis-aligned over 4 heads and 128 MLP units (C17). Every
transformer null in this paper is a null about *that* mask family — one granularity, one organism
(N6).

The dial can also be turned partially, scaling a group's contribution by α ∈ [0,1] rather than
zeroing it, which is how we test whether selection blends worldviews or switches between them
(C24).

## 2.5 Masks: who gets to choose the partition

This is the methodological core of the paper, and it is where three of our own headline results
were wrong.

### The invalid version

Rounds 1–3 scored the routing-free control under **one arbitrary fixed unit partition** — a single
coin toss over the 12,870 possible 8/8 splits — while the routed arm was scored under the
partition it had been trained with, which is the correct one by construction. The treatment got the
right mask handed to it; the control got a random one. Under that scoring the control scores 1/60
(C4). The same artifact separately inflated the k-way headline (R2) and the benign-fine-tuning
headline (R3).

### The data-chosen mask (the fair-mask procedure, FMP)

The fix gives the control the same freedom the treatment has: let the data choose the partition,
without letting the deployment set anywhere near it. For each hidden unit, measure **cue
sensitivity** — how much the unit responds to the true-cue coordinates versus the proxy-cue
coordinates — using **held-out agreeing data only**, plus 4000 unlabelled train-mode inputs, and
assign each unit to whichever cue it is more sensitive to. The deployment set is used only to score
the resulting mask, never to choose it.

The honest cost, stated once and inherited by every mask number in this paper: **this procedure
consumes the training-time cue oracle** — it needs to know which input dimensions (or which token
positions) carry which cue (G4). It is not an unsupervised discovery procedure. Only the localizers
in §2.6 attempt that.

Under FMP the same 60 control models score **51/60** instead of 1/60, with selection margin +0.910
against −0.018 [−0.097, +0.061] for the fixed partition (C4). One identical FMP was then applied to
every routing-free control in the line: pooled 2/282 → 100/282 (C8). FMP was verified bit-identical
to round 4's rule (max|diff| = 0.0, masks identical 10/10, C8).

### Controls on the mask itself

A mask procedure that finds anything at all is worthless unless a bad mask finds nothing. Four
negative controls, preregistered:

- **Random balanced splits of the same models**: 1/50 in the re-derivation (Fisher p = 1.5e-7,
  C5); pooled over distinct model sets in the portfolio, **18/4260 = 0.42%** (C10).
- **Best-of-10 random masks**, which gives the control K = 10 draws against the method's 1 — the
  method still wins per width at McNemar p = 1.0 / 2.7e-4 / 2.4e-7 / 1.5e-8 (C6).
- **Noise-dimension masks** (masks built from cue-irrelevant input dimensions): 0/10, p = 7.1e-4
  (C5).
- **Distractor mask** in the transformer: 0/30 (C17).

Independent re-derivation on 10 models retrained from scratch reproduces 8/10, all under the
designated rather than the swapped mapping (C5). Note what this is: an independent re-derivation,
not a fresh-seed replication (C5).

## 2.6 Localizers

The FMP consumes the cue oracle. The localization experiments ask what is recoverable without it,
in an entangled input basis where the cues are not coordinates. Four procedures are compared on the
same seeds:

1. **Oracle** — the generator's true coordinate split. Upper reference, not a method.
2. **Round-2 pair probe** — a data-only probe that recovers the k true cue groups exactly, 20/20
   at k = 3 (chance 1/1260) and 20/20 at k = 4 (chance 1/17325), permuted-label null 0/20 at both
   (C28). Its condition is severe and was found only in round 3: it needs the concepts to already
   occupy disjoint axis-aligned input columns, and *that coordinate system is itself oracle
   information* (R5). It also returns groups in an arbitrary slot order, matching the oracle on
   10/20 seeds at k = 3 and 14/20 at k = 4 (C29).
3. **ICA** — round 3's basis-agnostic localizer. Collapses in every basis tested (axis_ica 13/30 vs
   axis_pair 25/30, Fisher p = 0.0028; rot_ica 16/30; rot_dict 8/30; mix_ica 19/30, C37).
4. **Signed eigen-spectrum of D = E[xx′|y=1] − E[xx′|y=0]** — take the class-conditional
   second-moment difference, read its signed eigen-spectrum, take the eigen-directions as the
   candidate concept axes and pair them by the *sign and magnitude asymmetry* of the spectrum
   (C32).

Localizer 4 is evaluated against a swept **cue-margin ratio r** (the ratio of the two cues'
strengths; r = 1.00 is exactly equal cues) at N = 800 and N = 3200. It is a model-based estimator:
D's diagonality follows from the XOR-parity-plus-isotropic-noise family, so it embodies designer
knowledge of the generative family even though it is oracle-free per instance (C32). A preregistered
2×2 separates its two ingredients — directions (eigen vs ICA) × pairing (asymmetry vs probe) —
which is what identifies whitening as the destructive step (C35).

## 2.7 Attacker models in the tamper evaluation

Two deployment stories, and they have different threat models.

**Shipped model plus an inference-time mask.** An adversary holding the weights flips the worldview
by flipping the mask: 0 gradient steps, 0 data, hard-set versus masked at max|diff| = 0.0
(C50: 23/23 in round 3, 60/60 in round 4). This is structural; there is no attacker to model.

**Shipped model with the unselected half hard-deleted.** Here there is an attack, and the
comparison needs a control matched on live capacity. Two controls are used: a **capacity control**
(a routing-free H = 4 model with the identical 62 live parameters) and a **corrected baseline**
(round 3's PRIMARY A1 arm), plus a no-defense arm to calibrate the equivalence machinery (C51,
C52, R12).

Attacker rungs, weakest to strongest:

- **LoRA** — the preregistered weak attacker. Note its regime: it leaves broken models, with only
  26.7% (switch) and 16.7% (control) still competent at step 100, against 68.3% for the corrected
  baseline (C52).
- **Functional** — the realistic attacker, and the one the headline equivalence bound is computed
  under (C51).
- **Full fine-tuning**, and **full fine-tuning + 1% weight jitter** — the strongest honest rung
  (C53).

Outcome metrics: the area under the attack trajectory (mean AUC9 is quoted, e.g. 0.560/0.467 for
full-FT+jitter versus 0.767/0.786 for the best LoRA cell, C53), whether the model has flipped at
step 100 (N2), and capped flip time (N3).

Two artifacts must travel with these numbers. Bare full-FT rungs (−0.159, −0.251) are a
**relu′(0) = 0 artifact** and are greyed out wherever plotted (C53). And the jitter rung is **not
capacity-matched**: jitter revives the 92 zeroed weights, so the switch trains 154 parameters
against the control's 62 (C53).

## 2.8 Preregistration and power protocol (adopted after round 3)

From round 4 on, every confirmatory run followed the same protocol.

- **Prereg written and stamped before the run.** Primaries, decision rules, thresholds, and
  negative controls are fixed in the prereg file; amendments are stamped before the confirmatory
  run they affect (C39). Decision rules are executed *by code* against the prereg, not by a human
  reading the numbers afterwards — the transformer primary's decision "ABSENT" was applied that way
  (C17). [LEDGER GAP: the mtime-based verification of prereg stamping is not itself recorded in the
  ledger.]
- **Power stated in advance, as a table, not as a word.** The transformer primary shipped with a
  prestated power table of 0.15/0.52/0.87/0.98/1.00 against true rates 0.40/0.50/0.60/0.70/0.85
  (C17); the localizer primary shipped with design power 0.92 (C32).
- **Nulls are reported as bounds, never as absences.** Where a comparison comes out null we report
  a TOST or an achieved equivalence bound δ* and the only reading it licenses. Exactly one genuine
  equivalence exists in the whole line — routing versus ordinary training plus a no-gradient
  intervention, TOST at δ = 0.15 passing with 90% CI [0.000, 0.050] (C44/N8). Everything else is a
  bound: the tamper comparison achieves δ* = 0.0648 against a preregistered 0.05 margin it missed
  at ~36% power (C51/N1), and the monitor equivalence is knife-edge at 0.0486 against a 0.05 margin
  (C60/N15). Promoting a null to an equivalence is one of the five defect types the audits found
  (C68).
- **Standard test choices.** Fisher exact for 2×2 counts, McNemar for paired designs, Wilson
  intervals for single proportions, Newcombe for risk differences, Cochran–Armitage for ordered
  trends, permutation p-values where the statistic is not standard (reported at the permutation
  floor, e.g. p_perm = 5e-5 at 20k permutations, C39), and Holm within a declared family (C52: Holm
  over 7 tests).
- **Selection-inflation discipline.** Any number chosen as the maximum over a sweep is deflated by
  θ̂ = D̄ + ρ̂(D_sel − D̄) with ρ̂ = max(0, 1 − s²/S²), validated in simulation against the closed
  form a_k·s·√(1−ρ) at 100/100 cells within 15% (C63). For max-of-|·| statistics the two-sided
  constant b_k = E[max|N(0,1)|] replaces a_k — b_4 = 1.4647 against a_4 = 1.0294, a factor of 1.423
  (C66). The rule's preregistered real-data primary **failed** (exact Wilcoxon W = 9, p = 0.125) at
  ~53% power, which makes it uninformative rather than disconfirming (C65/N23), and that failure is
  stated in the same breath as the simulation success wherever the rule is used.
  [LEDGER GAP: the design rule n ≥ (a_k·σ_w/δ)² used to size sweeps is not recorded in the ledger;
  what is recorded is the closed form a_k·s·√(1−ρ) it derives from (C63).]
- **Anchors on every rerun.** Each round re-ran at least one prior round's cell and reported
  max|diff| against the stored result. Across six audits every anchor came back at max|diff| = 0.0,
  with two disclosed exceptions: a 0.0255 anchor in the k-way run (an undeclared override of that
  prereg's own rule) and a 5e-5 anchor traceable to 4-decimal storage precision, with per-seed
  booleans exact (C68).

## 2.9 What counts as a fair control

This is the lesson the sprint actually produced, and it is worth stating as a rule set rather than
a moral. Across six independent audits in rounds 4–5, **not one primary statistic failed to
reproduce**; every defect found was in the interpretive layer (C68). The measurements were fine.
The comparisons were not.

The five defect types found, each with the claim it damaged:

1. **A control given a different scoring rule than the treatment.** The routed arm was scored under
   the partition it was trained with; the control under one arbitrary partition out of 12,870
   (C4/R1). Cost: three headline effect sizes across three rounds (R1, R2, R3).
2. **An asymmetric outcome definition.** Routing was scored two-sided (both subnets had to behave)
   and augmentation one-sided (one aligned model). Rescored on the matched outcome, the p-values
   moved from 0.0046 and 0.024 to 0.052 and 0.49 (C41/R4).
3. **A gate applied to one arm.** A competence gate removed 3 models from the routed arm and none
   from the control; all 3 were positive cases (R8).
4. **A null promoted to an equivalence.** A bounded non-difference reported as sameness (C68, and
   the §4 rules that now govern every null in this paper).
5. **A diagnostic implemented for one substrate only.** The competence-conditional analysis existed
   for the transformer and not the MLP for a full round (C7).

The operational rules we now apply, each of which would have caught one of the above:

- **Same degrees of freedom.** If the treatment gets to choose its mask, the control chooses one
  too, by the same procedure. If that is impossible, give the control *more* freedom than the
  treatment (best-of-10 random masks against the method's 1, C6) and report the comparison anyway.
- **Same outcome, both arms, written down before scoring.**
- **Same gates, both arms; ungated numbers reported alongside gated ones.**
- **The right floor, not the convenient one.** For a routed arm the honest floor is
  routing-with-a-broken-split (rot_pair 10/30, pooled 18/60), not standard training at 0/30 — the
  round-3 agent volunteered this against its own result (C37).
- **A null gets a bound and a prestated reading, or it does not get reported.**
- **Every diagnostic runs on every substrate it is defined for.**
- **Anchor every rerun** against the stored prior result and publish max|diff|.

One caution about that list: it is a rule set for catching *our* errors, derived from five rounds
on one toy line. It has not been validated as a general protocol, and rule 4 in particular
(the honest floor) required knowing what the honest floor was — which is a research result, not a
checklist item.
