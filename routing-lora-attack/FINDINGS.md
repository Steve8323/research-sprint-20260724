# Low-rank (LoRA) tamper attack on routed unlearning

## Question

`~/gradient_routing_experiment` claims routing+prune is far more tamper-resistant than
gradient-ascent-style (GA/KL-to-uniform) unlearning, but that claim was established with a
**full fine-tuning** relearning attack that also **froze + re-zeroed the pruned waste
entries** (a defender-favorable assumption). Does the advantage survive a realistic
**low-rank adapter (LoRA-style) attack**, where the attacker freezes all base weights and
trains only a small dense-but-low-rank additive delta -- which can place mass **on top of
the pruned entries** without touching them?

## Provenance disclosure

When this (final) agent session started, the output directory already contained
`lora_tamper_attack.py`, `results.json`, `recovery_curves.png` from an earlier aborted
attempt at the same task earlier today. The old `results.json` was **not read** before this
pre-registration was finalized; everything below is re-run fresh from scratch.
CORRECTION (post-run): the plan was to compare the fresh run against the old results file
as an incidental replication check, but the fresh run overwrote `results.json` in place
before that comparison was made, so the old numbers are unrecoverable and NO such
comparison is reported. The old script (read after pre-registration was finalized) differed
only in LoRA A-init (N(0,0.01) vs N(0,1/sqrt(in_f))), adapter RNG offset, and its choice of
primary cell (r4 lr1e-2 vs this run's r4 lr1e-3).

## Pre-registration (finalized BEFORE the fresh run; no results seen)

- **Models per seed** (seeds 0-5, n=6, exactly the seeds of the original `tamper_experiment.py`):
  - **routed**: ablation-aware clean-scalpel model (`train_routed("random", seed)` from
    `tamper_experiment.py`, h=128, p_waste=0.5, 8 epochs), then permanently pruned
    (`apply_prune`).
  - **GA-baseline**: normally trained then unlearned via the repo's bounded KL-to-uniform +
    retain-CE procedure (`train_baseline(seed)`), no pruning.
  - Setup check: routed forget_pruned <= 0.05 and retain_pruned >= 0.85 on average;
    baseline forget_post <= 0.05, retain_post >= 0.85 on average.
- **Attack budget** (identical to the original full-FT attack): 200 forget-class training
  examples, Adam, batch 64, 100 steps, probes of forget-class TEST accuracy at
  steps [0, 10, 20, 30, 50, 75, 100].
- **LoRA attack**: base weights entirely frozen (routed model's pruned entries therefore
  stay exactly zero in the base); trainable rank-r adapters W_eff = W + (alpha/r) B A on
  fc1/fc2/fc3 weights, B init 0, A init N(0, 1/sqrt(in_features)), alpha/r = 2.0 fixed
  across ranks. Biases frozen. Primary rank r=4 (~5.2k trainable params, ~4% of the
  ~118k-param MLP), primary lr = 1e-3 (the original attack's lr, for a same-budget match).
- **Primary metric**: recovery AUC = trapezoid of the forget-test-accuracy curve over
  steps 0->100, normalized by 100 (identical to the original script's AUC). Lower = slower
  recovery = more tamper-resistant.
- **Primary test (pre-stated)**: paired difference in recovery AUC, routed minus GA-baseline,
  under the rank-4 lr-1e-3 LoRA attack, n=6 seeds. Exact two-sided sign-flip permutation
  test on the mean paired difference (2^6 = 64 flips; minimum attainable two-sided
  p = 2/64 = 0.03125), significance level 0.05. Also report the mean difference with a
  95% t-CI over the 6 paired differences.
- **Direction**: the repo's claim predicts routed AUC < baseline AUC (negative difference).
  If the difference is ~0 or positive, the tamper-resistance claim does not extend to
  low-rank attacks at matched budget.
- **Secondary analyses (pre-stated, exploratory)**:
  (a) rank x lr grid: r in {1, 4, 16} x lr in {1e-3, 1e-2} for both models -- the lr=1e-2
      column models an attacker who tunes the adapter lr (LoRA's B=0 parameterization can
      make lr=1e-3 artificially slow); "attacker's best cell per model" is a POST-HOC
      selection and will be labeled as such;
  (b) reference arms re-running the ORIGINAL full-FT attack (with its freeze-mask) on the
      same trained models/seeds, to anchor against the published claim;
  (c) mechanism check: masked-LoRA arms on the routed model (r=4, both lrs) where the
      adapter delta is projected off the pruned entries of fc1/fc2 every forward pass --
      if unmasked LoRA recovers but masked LoRA does not, recovery flows specifically
      through the pruned coordinates;
  (d) retain accuracy at each probe (attacker collateral damage).
  Anything not listed here that appears in Results is labeled post-hoc.

## Method notes

- All defended-model training and the full-FT reference attack are imported unchanged from
  the read-only originals (`harness.py`, `fix_experiment.py`, `tamper_experiment.py`).
- Single process, 3 torch threads, MNIST already on disk (no downloads).
- Only the "random" partition is attacked (the repo's flagship scalpel); "structural" is
  out of scope for compute-budget reasons.

## Results

Run: 2026-07-24, 19.3 s total, n=6 seeds (0-5), single process, 3 threads.
All numbers below are means over the 6 seeds; every test is the pre-registered exact
two-sided sign-flip permutation test on paired per-seed differences (min attainable
p with n=6 is 0.03125; every significant result below sits at that floor because all
6 paired differences share a sign).

**Setup check (passed):** routed+pruned forget=0.000 / retain=0.947; GA-baseline
forget=0.038 / retain=0.959.

### Primary (pre-registered): LoRA r=4, lr=1e-3, matched to the original attack budget

| model | recovery AUC (mean +- sd) | forget acc @10 / @30 / @100 steps |
|---|---|---|
| routed+pruned | **0.396 +- 0.058** | 0.000 / 0.204 / 0.725 |
| GA-baseline | 0.763 +- 0.029 | 0.239 / 0.763 / 0.951 |

Paired diff (routed - baseline) = **-0.367**, 95% t-CI [-0.422, -0.312],
exact p = 0.03125, 6/6 seeds negative. **The routed+pruned advantage survives —
and widens — under a matched-budget low-rank attack.** (At the same lr, LoRA is a
*weaker* attack on the routed model than the original full-FT attack: AUC 0.396 vs 0.723.)

### Secondary: full rank x lr grid (recovery AUC, routed / baseline)

| | lr=1e-3 | lr=1e-2 |
|---|---|---|
| r=1 | 0.203 / 0.600 | 0.511 / 0.867 |
| r=4 | 0.396 / 0.763 | 0.699 / 0.892 |
| r=16 | 0.607 / 0.872 | 0.826 / 0.885 |

Routed < baseline in every cell (all p = 0.03125). The gap shrinks monotonically as the
attack strengthens: -0.40 (r1 lr1e-3) down to -0.059 (r16 lr1e-2).

### Post-hoc (labeled): attacker's best cell per model

Best cell for routed = r16 lr1e-2 (AUC 0.826), for baseline = r4 lr1e-2 (AUC 0.892).
Paired diff -0.066, 95% CI [-0.090, -0.042], p = 0.03125. So under attacker-tuned LoRA
the routed advantage **persists in direction in 6/6 seeds but collapses from -0.37 to
-0.07 AUC**; the tuned adapter half-recovers the routed model by step ~10 (0.593 @10,
0.808 @20) — faster than the original full-FT attack (0.014 @10, 0.549 @20), i.e. **a
tuned low-rank adapter is the strongest attack on the routed model found here, beating
the repo's full-FT attack** (routed AUC 0.826 vs 0.723; note the full-FT attack enforces
the freeze-mask, LoRA arms do not — see caveats).

### Secondary: full-FT reference arms (replication anchor)

Baseline full-FT: 0.911 forget by step 10 (README: "relearns to 0.91 in 10 steps" —
replicates exactly). Routed full-FT: 0.014 @10, 0.549 @20, 0.808 @30, AUC 0.723 vs
baseline 0.912 (diff -0.189, p = 0.03125) — qualitatively replicates the published
"routed needs ~20-30 steps to half-recover" claim on these seeds.

### Mechanism check: masked LoRA (delta projected off pruned entries), routed, r=4

- lr=1e-3: masked AUC 0.323 vs unmasked 0.396 (diff -0.073, p = 0.03125): masking slows
  recovery somewhat.
- lr=1e-2: masked AUC 0.758 vs unmasked 0.699 (diff **+0.059**, p = 0.03125): the sign
  FLIPS — masking *speeds* recovery.

So access to the pruned coordinates is **not** the essential recovery channel: a low-rank
delta confined entirely to surviving weights recovers the forget capability just as well
(or better at high lr). The routed model's residual resistance is not primarily
"protected by keeping the pruned entries zero".

### Observation (pre-stated secondary d): attacker collateral

Because the attacker trains only on forget-class data, retain accuracy collapses in every
LoRA arm for both models (final retain 0.00-0.20). This attacker recovers the forget
capability at the cost of the rest of the model — consistent with a pure
capability-extraction threat model, but worth noting.

## Conclusion

The repo's tamper-resistance claim survives a low-rank attack at matched budget (and the
gap actually widens), but an attacker who tunes the adapter (higher rank + higher lr)
compresses the routed model's advantage to ~0.07 AUC and half-recovers the "deleted"
capability within ~10-20 steps of fine-tuning ~4-14% as many parameters — and does so
*without ever touching the pruned coordinates*. Prune-localized deletion slows relearning;
it does not prevent it, and its advantage over GA-unlearning is far smaller against a
tuned low-rank attacker than the full-FT numbers in the README suggest.

## Caveats

- Toy scale: 3-layer MLP on MNIST, class-split as "capability" proxy; existence-proof only.
- n=6 seeds; the exact sign-flip test bottoms out at p = 0.03125, so all "significant"
  results are at the test's floor. CIs are t-based on 6 paired diffs (normality assumed).
- The full-FT reference attack enforces the repo's freeze-mask on the routed model
  (defender-favorable); the LoRA arms do not (attacker-realistic). The routed full-FT vs
  routed LoRA comparison therefore conflates parameterization with the mask assumption —
  the masked-LoRA arm partially controls for this (and shows the mask barely matters).
- The attacker-best-cell comparison is post-hoc (selected on the same data it is tested
  on) and the grid is small (6 cells); a stronger attacker (more steps, more data, lr
  schedules, adapter merging) was not explored.
- "Structural" partition and other unlearning baselines (NPO/RMU) not attacked (budget).
- Provenance: an earlier aborted attempt's results.json was accidentally overwritten
  before the planned replication comparison (see Provenance disclosure above).

