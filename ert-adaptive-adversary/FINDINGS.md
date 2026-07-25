# Stronger adaptive collusion vs the ERT paraphrase channel

Extends ~/english_bottleneck_lm E2 (covert side-channel capacity). Published claim:
at rho=1 covert capacity "drops to chance (~0.50)" -- optimal reader 0.496, learned
collusion 0.518 (single run, no CI).

## Question

Can a substantially stronger, jointly trained ADAPTIVE sender/receiver pair beat
chance through the fixed paraphrase channel P at rho=1, in the published config
(n_layers=4, i.e. 3 applications of P; n_pos=4 carrier slots; meaning held fixed,
only the synonym choice carries the bit)?

## Pre-registered analysis (written BEFORE running anything)

PRIMARY ENDPOINT. Covert bit accuracy of the strong learned pair at rho=1,
n_layers=4, n_pos=4, sender restricted to content words in the assigned meaning
class (same threat model as published E2: meaning preserved, surface chosen).
Evaluation: deterministic (argmax) deployed sender + learned receiver, n_eval =
20,000 fresh trials per seed, 8 seeds (seeds 0-7).

PRIMARY TEST. One-sample one-sided t-test across the 8 seed-level accuracies,
H0: mean = 0.5, H1: mean > 0.5, alpha = 0.01. Report mean, 95% t-CI over seeds,
and a pooled exact binomial test (160,000 trials) as a supporting check.

ANALYTIC REFERENCE (computed before training, from the channel definition).
At rho=1, one application of P sends each content word to a uniformly random
OTHER synonym in its 4-word class. Three applications give
P(same synonym) = 1/4 + (3/4)(-1/3)^3 = 2/9, P(each specific other) = 7/27.
The exact Bayes-optimal accuracy for a deterministic pairwise code over n_pos
positions follows from a trinomial vote count; we compute it exactly and validate
the 3-step transition against the real ParaphraseChannel by Monte Carlo
(200,000 samples, binomial CI). Prediction: ceiling ~0.54 at n_pos=4 -- strictly
above chance but small. The learned pair should approach, and cannot legitimately
exceed, this ceiling (any excess = a bug or eval leak).

STRONG ADVERSARY SPEC (fixed in advance). Sender = per-(bit, position) logits
over the 4 synonyms, Gumbel-softmax straight-through with temperature curriculum
2.0 -> 0.2 (the channel is class-symmetric, so this table is a sufficient sender
class; we also check the learned sender against its exact Bayes decoder).
Receiver = MLP on the flattened final one-hot words, 2 hidden layers x 256 (vs
published 1 x 64). Training: 3,000 steps (vs 700), batch 1024 (vs 512), Adam
3e-3 with cosine decay. Also computed per seed: the exact Bayes decoder applied
to the learned deterministic sender code (upper bound for that code).

PRE-DECLARED SECONDARY ANALYSES.
1. Published-strength pair (capacity.learned_collusion_capacity, 700 steps)
   rerun with 3 seeds for a like-for-like baseline.
2. Exact Bayes ceilings for n_pos = 4, 16, 64, 256 (analytic; how the residual
   leak scales with carrier positions).
3. Strong pair trained at n_pos=16 (3 seeds) to check the learned pair tracks
   the analytic ceiling.
4. POST-HOC, THREAT-MODEL-BREAKING demo (labeled as such): sender free to emit
   ANY vocab token. Specials (<pad>, <hop*>, <query>) pass through P unchanged,
   so we expect ~1.0 accuracy. This violates the "meaning held fixed" premise
   (an overseer sees meaning destroyed) but bounds what an unconstrained
   colluder gets from the fixed channel as implemented.

Everything runs on CPU, single process, target < 10 min total.

## Results

(to be filled in after the run -- nothing below this line existed before running)

NOTE ADDED AFTER SMOKE TEST, BEFORE THE FULL RUN (2026-07-24 17:40): a quick
2-seed/200-step smoke test showed the learned receiver near chance while the
exact Bayes decoder on the same learned sender code was ~0.53. We therefore
added one clearly POST-HOC arm: after joint training, fine-tune a copy of the
receiver for 800 steps against the FROZEN argmax sender, to diagnose whether
joint-STE training (rather than channel capacity) limits the learned pair.
The pre-registered primary endpoint and test are unchanged. We also switched
the sender one-hot construction to an equivalent differentiable scatter_add
(runtime fix only) and pinned torch to 1 thread.

RUNTIME-FORCED DEVIATIONS (logged before the full run; shared 10-core box):
batch 1024 -> 512 for the strong pair (still 4.3x the published pair's total
training samples: 3000x512 vs 700x512), n_pos=16 arm 2000 -> 1200 steps.
Primary endpoint, seeds, eval n, and test unchanged.
