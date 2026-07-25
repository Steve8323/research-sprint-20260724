# Is unlearned behavior underdetermined?

Combines `~/gradient_routing_experiment` (routing+prune unlearning, ablation-aware
variant from `fix_experiment.py`) with the underdetermination framing of
`~/moral_framework_misgen`: after N seeds are trained and unlearned by the *identical*
procedure, is the deployed (pruned) network's behavior on forget-side inputs
seed-dependent (underdetermined) while its behavior on in-distribution retain inputs is
pinned by the data?

## Question

Routing+prune deletes the forget capability (forget acc -> ~0.0, retain ~0.95, per the
routing repo). The unlearning objective specifies *what not to do* on forget inputs but
does not specify *what to do instead*. Hypothesis (misgen-repo analogy): the replacement
behavior is loss-underdetermined, so different seeds commit to different arbitrary
behaviors on forget inputs, while agreeing on retain inputs.

## Pre-registered plan (written BEFORE running the experiment; a 1-seed timing pilot
runs first and its numbers are not used in any test)

### Setting (copied from the routing repo, unchanged)

- MNIST, MLP 784-256-256-10, forget classes {0..4}, retain {5..9}.
- Ablation-aware routing (`fix_experiment.step_aa`, aa=True), random partition,
  p_waste=0.5, mislabel=0.0, epochs=8, Adam lr 1e-3, bs=128, per-seed 6000-example
  train subsample and per-seed waste mask (exactly as in `run_aa`).
- Deployed unlearned model = pruned net (waste weights zeroed).
- Control arm, same seeds: plain training (no routing) on the same per-seed data,
  evaluated unpruned; plus control-pruned (same net, per-seed mask zeroed) as the
  pruning-artifact control.
- N = 8 seeds (0..7). Common fixed eval set for ALL seeds: the full MNIST test set
  (10,000 images; ~5,139 forget-class, ~4,861 retain-class). Boundary set: 600 fixed
  (forget image, retain image) test pairs, x(lam) = lam*forget + (1-lam)*retain,
  lam in {0.25, 0.5, 0.75}.

### Metrics

- Pairwise cross-seed agreement a_ij(S) = fraction of inputs in S where seeds i and j
  predict the same label. Per-seed mean agreement A_i(S) = mean over j != i of a_ij(S).

### PRIMARY TEST (stated before running)

In the unlearned (routed+pruned) arm: exact two-sided Wilcoxon signed-rank across the
N = 8 seeds of A_i(forget-class test inputs) vs A_i(retain-class test inputs).
Prediction: agreement is lower on forget inputs. alpha = 0.05. (Minimum attainable
two-sided p at N=8 is 0.0078.)

### Secondary / descriptive analyses (also pre-stated)

1. Same-inputs cross-arm contrast (isolates the effect of unlearning from the inputs):
   paired Wilcoxon across seeds of A_i(forget inputs) in the unlearned arm vs A_i(forget
   inputs) in the control arm, and vs the control-pruned arm.
2. Chance-corrected agreement on forget inputs (unlearned arm): expected agreement under
   seed-marginal independence E_ij = sum_k p_i(k) p_j(k); report observed - expected.
   obs ~ exp => each seed has an arbitrary global label preference and no shared
   per-input structure; obs > exp => shared input-driven structure.
3. Within-seed coherence ("commitment"): per seed, per true forget class c, modal
   fraction max_k P(pred = k | class c); the seed's class->modal-label 5-vector; number
   of distinct 5-vectors among the 8 seeds; pairwise mapping agreement (fraction of the
   5 classes where two seeds' modal labels coincide).
4. Boundary sweep: mean pairwise agreement vs lam on the 600 interpolation inputs, for
   the unlearned-pruned and control arms.
- All agreement rates reported with 95% bootstrap CIs over inputs (1000 resamples).
- Sanity gates (must hold or the run is reported as failed unlearning, not analyzed
  further): unlearned arm mean forget acc (pruned) < 0.05 and retain acc > 0.90.

Anything not listed above will be labeled POST-HOC.

## Result

Sanity gates pass: unlearned arm forget acc 0.000 (all 8 seeds), retain 0.948 mean;
control forget 0.951, retain 0.923; control-pruned forget 0.934.

**Primary test: significant, in the predicted direction — but the effect is partial,
not misgen-style underdetermination.**

- Cross-seed agreement, unlearned (pruned) arm: **forget inputs 0.807** (95% CI
  [0.800, 0.813], n = 5139) vs **retain inputs 0.960** ([0.956, 0.964], n = 4861).
  Per-seed A_i lower on forget for 8/8 seeds; exact Wilcoxon **p = 0.0078** (the floor
  at N = 8).
- Same-inputs cross-arm (secondary 1): on the identical forget inputs, control nets
  agree 0.955 ([0.951, 0.959]) and control-pruned 0.934 ([0.929, 0.938]); unlearned
  0.807 is below both, p = 0.0078 each (8/8 seeds). So unlearning itself — not the
  inputs, and not pruning per se — multiplies cross-seed disagreement on forget inputs
  by ~4.3x (0.045 -> 0.193 disagreement rate vs control; ~2.9x vs control-pruned).
- BUT the replacement behavior is mostly **convergent, not seed-arbitrary**
  (secondaries 2-3): observed forget agreement 0.807 vastly exceeds the
  marginal-independence chance level 0.252 (mean obs-exp = +0.555, range across the 28
  pairs [0.513, 0.599]). And the coarse replacement mapping is nearly identical across
  seeds: 7/8 seeds share the exact forget-class->modal-predicted-label map
  {0->5, 1->8, 2->8, 3->5, 4->9} (seed 4 differs only at 3->8); 2 distinct maps among
  8 seeds, pairwise map agreement 0.95. Within-seed the mapping is class-coherent
  (mean modal fraction 0.66 vs 0.36 expected from each seed's global label marginal
  alone).
- Boundary sweep (secondary 4): unlearned-arm agreement falls monotonically toward the
  forget side: 0.938 (lam=0.25) -> 0.827 (0.5) -> 0.809 (0.75), n = 600 each. Controls
  show a V (0.873 / 0.742 / 0.874): blends at 0.5 are genuinely ambiguous for intact
  nets, while unlearned nets are *more* consistent there (they can only answer with the
  retain-side class) and less consistent where the forget content dominates.

**Answer to the question:** partially yes, mostly no. Unlearning does leave forget-side
behavior measurably less determined than in-distribution behavior (5x the residual
cross-seed disagreement, unanimous across seeds), but it is nowhere near the worldview
seed-lottery of ~/moral_framework_misgen (where deployment behavior spans [0,1] by
seed). Here the loss landscape plus input geometry pins the *replacement mapping* —
every seed reroutes each deleted digit class to the same perceptually-nearest retain
class — and the seed only decides behavior on a ~19% boundary fringe of forget inputs.
The underdetermination that unlearning creates is per-input, not functional/worldview-
level.

POST-HOC note (not pre-registered): the per-class modal fractions (0.42-0.92) show the
fringe is concentrated in classes 2 and 3, whose modal targets are least dominant —
consistent with the "which nearby retain class wins" tie being the underdetermined
quantity.

## Caveats

- N = 8 seeds; all three Wilcoxon tests sit at the exact-test floor p = 0.0078 (8/8
  consistent ordering). Effect sizes carry the story; input-level 95% bootstrap CIs are
  tight and non-overlapping.
- "Seed" jointly varies init, 6000-example train subsample, batch order, and waste
  mask (exactly as in the source repo's run_aa); the design cannot attribute
  disagreement to any single source.
- Agreement on retain inputs is partly forced by shared correctness (~95% acc). The
  cleaner contrast is the same-inputs cross-arm one (0.807 vs 0.955/0.934), which is
  accuracy-matched in the relevant sense (control nets are ~95% accurate on forget
  inputs; unlearned nets have no accuracy constraint there at all, yet still agree at
  0.807).
- Toy scale: MNIST class-unlearning in a 784-256-256-10 MLP. The convergent replacement
  mapping is plausibly driven by pixel-space similarity between digit classes; in
  domains where forget-adjacent behavior is not dictated by input geometry, more
  functional underdetermination could appear. That is speculation, not a finding.
- Boundary inputs are off-manifold pixel blends, not natural ambiguous examples.
- Control nets use a separate init stream (seed+500000) on the same per-seed data; the
  control-pruned arm shares the unlearned arm's per-seed masks.
- Single machine, single run of the pre-registered pipeline; the 1-seed pilot (seed
  100) was used only for timing and its numbers appear nowhere.
