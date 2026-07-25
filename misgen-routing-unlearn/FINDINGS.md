# Routing+prune vs misaligned worldview

Combines the random-interconnected routing+prune unlearning method of
`~/gradient_routing_experiment` (harness.py / fix_experiment.py, ablation-aware variant)
with the misaligned-worldview MLP setting of `~/moral_framework_misgen`
(alignment.py: equal-cost true-cue vs proxy-cue XOR task, width-8 two-hidden-layer MLP,
154 params, N=800, Adam lr 5e-3, wd 2e-3, 2000 full-batch steps; deployment = proxy flips).

## Question

Can gradient routing during training + pruning at deployment remove the misaligned
worldview (proxy-cue reliance) while retaining task accuracy - and how does it compare to
the published disagreeing-data fine-tuning baseline?

## Pre-registered plan (written BEFORE the full run; a 2-seed mechanics pilot runs first,
its numbers are not used in any test)

### Setting / metrics
- Task and hyperparameters copied from `alignment.py` (`gen_align`); deployment eval set =
  `gen_align(4000, 50001, "deploy")`, in-distribution retain eval set =
  `gen_align(4000, 60001, "train")` (held-out agreeing data).
- **Forget metric** = deployment alignment (accuracy on deploy data where proxy = 1-truth).
  Misaligned worldview removed <=> alignment high. Misaligned: <0.2; aligned: >0.8 (repo thresholds).
- **Retain metric** = accuracy on held-out in-distribution (agreeing) data.

### Arms (n = 20 seeds each, disjoint seed ranges so arms are independent; no per-seed pairing)
1. `control` (seeds 700-719): plain training, f=0. Plus `control_pruned`: same nets with the
   random 50% waste mask zeroed (pruning-artifact control from the routing repo).
2. `route_aa` (seeds 1700-1719): routing + prune, ablation-aware (the routing repo's best
   variant). Training data: N=800 rows, fraction f=5% are disagreeing rows (proxy = 1-truth)
   labeled with the PROXY value (bad-behavior demonstrations) and tagged FORGET; the other 95%
   agree and are tagged RETAIN. Waste = random scattered 50% of the two hidden weight matrices.
   Forget gradients update only waste (full forward); retain gradients update only main
   weights, computed on the pruned forward. Prune waste at deployment. Report pre- and post-prune.
3. `route_naive` (seeds 1700-1719 config, aa=False): same but retain loss on the full forward
   (co-adaptation check from the routing repo).
4. `ft_removal` (continues from `control`, seeds 700-719): the published disagreeing-data
   fine-tuning baseline - 1000 further plain steps on data with f=5% disagreeing rows labeled
   with the TRUTH (geom_persist protocol; published f*=5% gave 100% removal on 30 models).
5. `prevention` (seeds 2700-2719): plain training from scratch with f=5% truth-labeled
   disagreeing rows (published during-training baseline), context arm.
6. `route_oracle` (seeds 3700-3719): routing + prune with an ORACLE FEATURE partition -
   waste = all first-layer weight columns reading the proxy input dims, f=0 (zero disagreeing
   data), ablation-aware. Tests whether feature-identity knowledge substitutes for
   disagreeing data. Clearly labeled oracle (uses ground-truth knowledge of which input dims
   carry the proxy).

Dose matched at f=5% (= published removal-curve f*), but note the supervision differs by
design: routing gets only example TAGS + bad-behavior labels; fine-tuning gets corrected labels.

### Primary test (stated before running)
One-sided Mann-Whitney U on per-seed post-treatment deployment alignment,
`ft_removal` vs `route_aa` (post-prune), n=20 vs 20. H1: ft_removal > route_aa.
Mechanistic prediction: ablation-aware routing trains the retained subnetwork only on
agreeing (worldview-underdetermined) data, so the pruned model should re-flip the seed coin -
its alignment distribution should look like `control`, not like `ft_removal`.

### Secondary tests
- Fisher exact (one-sided) on misaligned (<0.2) counts: route_aa vs ft_removal.
- Two-sided Mann-Whitney: route_aa (post-prune) vs control (does routing move the
  distribution at all?).
- Retain cost: two-sided Mann-Whitney on retain accuracy, route_aa (post-prune) vs control.
- Descriptive Wilson 95% CIs on all fractions; oracle arm descriptive.

Anything not listed above is labeled post-hoc in the results section.

## Result

**Routing+prune fails to remove the misaligned worldview; disagreeing-data fine-tuning
removes it completely at the same dose.** All numbers: n = 20 seeds per arm, run in
`results.json` (181 s total, single process, CPU).

| arm | deploy alignment (mean +/- sd) | misaligned (<0.2) [Wilson 95% CI] | aligned (>0.8) | retain acc |
|---|---|---|---|---|
| control (f=0) | 0.535 +/- 0.300 | 3/20 [.05, .36] | 4/20 | 1.000 |
| control + prune (artifact ctrl) | 0.574 +/- 0.167 | 1/20 | 0/20 | 0.686 |
| route_aa PRE-prune (full net) | 0.034 +/- 0.051 | 19/20 | 0/20 | 0.861 |
| **route_aa POST-prune** | **0.407 +/- 0.397** | **9/20 [.26, .66]** | 6/20 | **0.999** |
| route_naive POST-prune | 0.337 +/- 0.184 | 4/20 | 0/20 | 0.782 |
| **ft_removal (f=5%)** | **0.990 +/- 0.016** | **0/20 [0, .16]** | 20/20 | 0.999 |
| prevention (f=5%) | 0.987 +/- 0.025 | 0/20 | 20/20 | 0.999 |
| route_oracle (feature waste, f=0) | 0.998 +/- 0.000 | 0/20 | 20/20 | 0.999 |

### Pre-registered tests
- **PRIMARY: Mann-Whitney U, ft_removal > route_aa (post-prune): U = 358, p = 1.0e-5.**
  Confirmed. Fine-tuning on 5% truth-labeled disagreeing data removes misalignment in
  20/20 seeds (all > 0.9); routing+prune at the same 5% dose leaves a full coin-flip
  spread [0.002, 0.998].
- Fisher exact (one-sided), misaligned counts route_aa 9/20 vs ft_removal 0/20: p = 6.1e-4.
- Mann-Whitney route_aa (post-prune) vs control: U = 157, p = 0.25 - the post-prune
  distribution is statistically indistinguishable from doing nothing, exactly the
  pre-registered mechanistic prediction (the retained subnetwork re-flips the seed coin).
- Retain cost route_aa vs control: MW p = 1.2e-4 but the magnitude is negligible
  (0.999 vs 1.000); ablation-aware routing preserves task accuracy. (Statistically
  detectable only because both arms sit at ceiling with tiny variance.)

### What the mechanism did and did not do
- Routing genuinely localizes and deletes the *demonstrated* proxy-following behavior:
  the full (unpruned) route_aa net is almost universally proxy-following at deployment
  (0.034, 19/20 misaligned - the waste weights absorbed the 5% proxy-labeled demos and
  generalize proxy-following), and pruning removes that (0.034 -> 0.407). Deletion works,
  exactly as in the MNIST source repo.
- But deleting the demonstrated behavior does not pin the worldview: the retained
  subnetwork was trained (ablation-aware) only on agreeing data, which underdetermines
  the worldview, so the pruned model re-flips the coin (indistinguishable from control).
  This is the underdetermination thesis of moral_framework_misgen biting the unlearning
  method: routing+prune removes a *capability exercised by tagged data*; a misaligned
  worldview in this setting is not exercised by any in-distribution data.
- The controls replicate the source repo inside the new task: pruning a normally trained
  net wrecks the task (retain 0.686 vs 0.999 ablation-aware), and naive (non-ablation-aware)
  routing collapses retain (0.782) via co-adaptation.
- The oracle arm shows the escape hatch: if the waste partition encodes *feature identity*
  (first-layer columns reading the proxy dims), routing+prune yields 20/20 aligned at
  0.998 with retain 0.999 using ZERO disagreeing data. Knowledge of which features carry
  the proxy substitutes for disagreeing labels. This is an oracle - it assumes exactly
  the knowledge that is unavailable in realistic misgeneralization.

### Post-hoc observations (labeled as such)
- route_aa post-prune has *more* misaligned seeds than control (9/20 vs 3/20, two-sided
  Fisher p = 0.082, n.s.). If real, a plausible mechanism is capacity: the retained subnet
  has ~50% of hidden weights, and the repo's capacity result says narrower nets commit
  (rather than hedge) more often. Not established here.
- ft_removal at f=5% matches the published removal curve (geom_persist: f* = 5%,
  30/30 removed).

## Caveats
- Toy scale: 154-parameter MLP, synthetic disentangled XOR cues; "worldview" =
  which of two equally predictive cues is tracked. Inherits all limitations of the
  source repos.
- The comparison is dose-matched (5% disagreeing inputs) but supervision-mismatched by
  design: routing consumes example tags + bad-behavior labels, fine-tuning consumes
  corrected labels. That asymmetry is the point (unlearning methods are meant for the
  tag-only regime), but routing was never given corrected labels; a hybrid was not tested.
- Arms use disjoint seed ranges (independent, unpaired); n = 20/arm. The seed-1700 batch
  could differ from seed-700 by chance; the route_aa-vs-control null (p = 0.25) is
  consistent with that either way.
- One routing config only (p_waste = 0.5, both hidden layers, full-batch, f = 5%);
  no sweep over p_waste or f for the routing arm.
- Single eval draw for deploy/retain sets (n = 4000 each); per-seed accuracy noise
  ~ +/-0.008, negligible vs effect sizes.

## Files
- `experiment.py` - all arms + pre-registered tests (single process, ~3 min)
- `results.json` - full per-seed numbers, CIs, tests (`results_pilot.json` = 2-seed pilot)
- `run.log` - run transcript

## Audit (2026-07-24, post-run verification pass)
- All arm summaries (mean/sd, misaligned/aligned counts, retain) recompute exactly from the
  raw per-seed score lists in `results.json`; all four pre-registered tests and the post-hoc
  two-sided Fisher (p = 0.082) reproduce to the printed precision with an independent script.
- From-scratch reruns of control seed 700, route_aa seed 1700, and route_oracle seed 3700
  reproduce the logged per-seed numbers exactly (deterministic seeding).
- The bit-identical pre-/post-prune scores in `route_oracle` are explained, not a bug:
  with f = 0 no example is forget-tagged, so the oracle waste weights receive zero task
  gradient and only Adam weight decay, collapsing to |w| ~ 1e-40 by end of training -
  pruning them is a no-op. The oracle effect therefore comes from ablation-aware training
  (the retained subnet never reads the proxy dims), and "pre-prune 0.998" should not be
  read as the full net resisting the proxy: the full net has effectively already pruned itself.
