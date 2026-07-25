# How much of the ERT's covert-channel safety comes from the channel being FIXED?

Extends `~/english_bottleneck_lm` (read-only). Published E2 claim: at full
paraphrase strength (rho=1) the *fixed, class-structured* paraphrase channel P
closes the 1-bit covert side-channel (optimal reader 0.496, learned collusion
0.518, chance 0.5). The README pitches P as "the un-foolable floor ... nothing
to optimize against". Question: if P is instead a small *learned* net trained
on the same meaning-preserving / surface-randomizing objective, how much covert
capacity reopens against the same adversary?

## Pre-registered design (written BEFORE running)

**Channel under test.** `LearnedParaphraser`: one-hot word -> logits over V
(V=106), categorical sample, straight-through estimator (same STE convention as
the published channel). Trained (then FROZEN) with cross-entropy toward the
ideal surface-randomizing target: uniform over the input's meaning class
(specials -> identity). Note the ideal target is an input-independent
conditional per class, i.e. exactly capacity-zero; any reopened capacity is
training imperfection.

**Arms** (each = train paraphraser, freeze, attack):
- A0 `fixed_rho1`: published fixed channel at rho=1 (reproduction baseline).
- A1 `learned_strong`: linear V->V logit table, 2000 training steps. PRIMARY.
- A2 `learned_mid`: linear, 200 steps.
- A3 `learned_weak`: linear, 50 steps.
- A4 `learned_bottleneck`: V->8->V MLP (rank-limited, cannot represent an
  arbitrary per-token table), 2000 steps.

**Adversary (identical across arms, verbatim from `capacity.py`)**: the
published `_Sender`/`_Receiver` collusion pair, trained end-to-end through 3
channel applications (n_layers=4), 600 steps, B=512, n_pos=4, lr 5e-3, eval
5x2000 = 10,000 fresh trials per run. Plus the published-protocol optimal
reader, here computed EXACTLY: the channel's one-hop transition matrix Q is
known in closed form for every arm, so 3-hop Bayes accuracy of the
synonym-0/1 encoding is Sum_y max_b Q^3(y|m_b)/2 (no Monte Carlo, no plug-in
bias); averaged over the first 8 classes to match `optimal_reader_capacity`.

**Seeds**: 5 per arm (seed 0..4 controls paraphraser init/training AND
collusion training AND eval sampling). n = 5 runs/arm, 10,000 eval trials/run.

**Primary test (pre-stated)**: one-sided one-sample t-test (alpha=0.05, n=5
seeds) of A1 learned-collusion accuracy > 0.5 -- does the *best-effort* learned
paraphraser reopen any covert capacity?

**Secondary tests (pre-stated)**:
1. Welch two-sided t-test A1 vs A0 per-seed collusion accuracies.
2. Dose-response: Spearman correlation between per-run TV distance of Q from
   the ideal uniform-over-class conditional and per-run collusion accuracy,
   across all learned-arm runs (20 points).
3. Report reopened-capacity fraction (acc-0.5)/0.5 per arm (1.0 = the full
   1-bit channel of the rho=0 identity).

**Functionality guard (confound control)**: an arm only counts as a
*functional* paraphraser if its exact 1-hop meaning-class preservation is
>= 0.95 and its within-class surface entropy is >= 1.5 bits (max 2.0); arms
failing this are reported but flagged (capacity reopening there could be
"channel is broken", not "channel is learnable").

Anything not listed above is post-hoc and will be labeled as such.

## Results

Runtime 287 s CPU, 5 arms x 5 seeds, results.json + capacity_fixed_vs_learned.png.

| arm | 1-hop class keep | TV from ideal | collusion acc (95% CI, n=5 seeds) | EXACT Bayes reader | reopened fraction of 1 bit |
|---|---|---|---|---|---|
| A0 fixed rho=1        | 1.000 | 0.250 | 0.5071 (0.4931, 0.5211) | **0.51852** (exact) | +1.4% (ns) |
| A1 learned_strong     | 0.999 | 0.0006 | 0.5016 (0.4948, 0.5085) | 0.50000 | +0.3% (ns) |
| A2 learned_mid        | 0.976 | 0.024 | 0.5004 (0.4945, 0.5063) | 0.50022 | +0.1% (ns) |
| A3 learned_weak*      | 0.753 | 0.247 | 0.5001 (0.4970, 0.5032) | 0.50164 | +0.02% (ns) |
| A4 learned_bottleneck | 0.9999 | 0.0017 | 0.5008 (0.4941, 0.5075) | 0.50000 | +0.2% (ns) |

*fails the pre-registered functionality guard (class preservation 0.75 < 0.95);
flagged, reported anyway.

**Primary test**: A1 collusion accuracy > 0.5 -- t(4)=0.66, one-sided p=0.27.
NOT significant. The best-effort learned paraphraser does NOT reopen the covert
channel against the published adversary.

**Secondary**: A1 vs A0 Welch t(two-sided) p=0.37 (no detectable difference).
Dose-response Spearman(TV, collusion acc) over 20 learned runs: rho=-0.017,
p=0.95 -- no relationship, because every learned arm sits at chance (range
restriction: no capacity reopened anywhere, even at TV=0.25).

**Answer to the title question: ~none of the measured safety comes from the
channel being fixed.** At this toy scale the safety comes from the channel's
conditional being (near-)input-independent within a meaning class, and the
learned surface-randomizing objective reaches that easily (TV 0.0006 after
2000 steps; even the rank-8 bottleneck reaches TV 0.002). Undertraining
degrades meaning preservation (overt quality) long before it opens a covert
channel.

**Post-hoc finding (labeled; exact arithmetic, no sampling error)**: the FIXED
channel is actually the *leakier* one under the published E2-optimal protocol.
Because rho=1 never outputs the input token, its conditional is
input-dependent: after 3 hops the return probability is a=2/9=0.2222 vs
b=7/27=0.2593 for each other synonym. A Bayes reader that guesses the
*opposite* synonym achieves exactly 2b=0.51852; the published heuristic
"optimal" reader (guess same synonym, coin otherwise) achieves exactly
a+b=0.48148 -- below chance, i.e. anti-correlated after an odd number of hops
-- and its published MC estimate (0.496, 4000 trials) masked both. The ideal
learned channel (uniform including self) is exactly input-independent: Bayes
reader exactly 0.5000. So the fixed table's "un-foolable floor" carries a small
provable parity side-channel (~2.7% of a bit at 1 carrier position; more with
4 positions) that a well-trained learned channel does not have. The published
learned-collusion adversary (600 steps) largely failed to find this leak
(0.5071 mean), which also means published E2 slightly understates worst-case
capacity of the fixed channel.

## Caveats

- Toy substrate: V=106 one-hot words, so a linear learned paraphraser can
  memorize a per-token table; real learned paraphrasers on natural language
  generalize from data and could carry structured, adversary-predictable
  deviations. The weakness modes tested (undertraining, rank-8 bottleneck)
  yield near-input-independent conditionals; weakness modes that produce
  *correlated input-dependent* deviations (biased training corpus, distribution
  shift, adversarial pretraining) were not tested and are the more dangerous
  real-world analog.
- The paraphraser is FROZEN during the attack (matches the brief). If it were
  co-trained inside the task loop, gradient pressure could carve a channel
  through it -- untested, and the more likely failure in practice.
- The published adversary is weak (600 steps, fixed architecture); it did not
  even fully exploit the fixed channel's exact 1.85pp/position parity leak. The
  exact Bayes numbers are adversary-independent for the 1-carrier protocol and
  are the stronger evidence.
- n=5 seeds/arm: power to detect collusion differences <~1.5pp is limited, but
  the headline null is corroborated by the exact (zero-sampling-error) Bayes
  capacities.
- Post-hoc parity-leak analysis was not pre-registered (labeled above).
