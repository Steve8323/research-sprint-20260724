# PREREG — LoRA tamper-eval as a reusable protocol

Stamp (from `date` run in shell before any experiment ran):
```
Fri Jul 24 23:09:47 EDT 2026
```

## Background / motivation

Round-1 (`../routing-lora-attack/`) showed that a LoRA-style relearning attack (frozen base,
trainable rank-r adapters on fc1/fc2/fc3) is a clean, budget-matched way to stress-test an
unlearning method's tamper-resistance, and that an *attacker who tunes rank+lr* collapses the
routed+prune model's advantage over gradient-ascent (GA) unlearning from AUC ≈ −0.37 to ≈ −0.07.
That was run on exactly TWO defended arms (one routed model + one GA baseline).

Round-2 goal: package the round-1 attack as a **reusable protocol** (a rank × lr grid, an
attacker-best-cell rule, budget-matched steps, retain-tracking, exact paired tests) and apply the
SAME protocol to EVERY unlearning arm in `~/gradient_routing_experiment`, producing a
"provisional (weak attack) vs LoRA-robust (tuned attack)" table.

## Defense arms attacked (each trained from the READ-ONLY repo, unchanged)

Six unlearning configurations, each producing a "deleted-forget" model that the protocol attacks:

- **GA-baseline** — `train_baseline(seed)`: KL-to-uniform + retain-CE gradient-ascent unlearning,
  no routing, no pruning. (the "ordinary unlearning" reference.)
- **routed_p0.5** — `train_routed('random', seed)` (h=128, p_waste=0.5) then `apply_prune`. The
  repo flagship random scattered-waste scalpel.
- **routed_p0.9** — same but p_waste=0.9 (trained via the hybrid trainer with alpha=0, p_waste=0.9,
  which reduces to pure scattered random routing — verified identical mask construction).
- **hybrid_a0.25 / hybrid_a0.5 / hybrid_a0.75** — interior interpolation points between random
  (alpha=0) and structural (alpha=1) routing, p_waste=0.5, via `make_masks_hybrid` reproduced from
  `routing-hybrid-knob/hybrid_knob_experiment.py`. (3 interior alpha points.)

Setup gate (per arm, mean over seeds, must hold or the arm is flagged): forget-after-deletion
≤ 0.06 AND retain-after-deletion ≥ 0.85. Arms that fail the gate are still reported but excluded
from the headline claim.

## Attack protocol (the reusable module `lora_tamper_protocol.py`)

Identical budget to the repo's original full-FT attack: 200 forget-class examples, Adam, batch 64,
100 steps, forget-class TEST accuracy probed at steps [0,10,20,30,50,75,100]; retain-class TEST
accuracy probed at the same steps (attacker collateral).

- **LoRA attack**: base weights frozen (routed pruned entries therefore stay exactly 0 in base),
  trainable rank-r adapters W_eff = W + (alpha_lora/r)·B·A on fc1/fc2/fc3, B=0 init,
  A ~ N(0, 1/in_f), alpha_lora/r = 2.0 fixed across ranks (no rank/scale confound). Biases frozen.
  Grid: **r ∈ {4, 16} × lr ∈ {1e-3, 1e-2}** (4 cells). Downscoped from round-1's 6 cells (dropped
  r=1) to fit 6 arms in the compute budget; r=4/lr=1e-3 (the weak, budget-matched cell) and
  r=16/lr=1e-2 (the strong, tuned cell) are the two poles that carried round-1's story.
- **Reference full-FT attack**: the repo's own `relearn_attack` (freeze-mask enforced on routed
  arms) — the published attack, as an anchor.

## Two attack strengths compared per arm (the "provisional vs LoRA-robust" table)

- **provisional** = the weak, budget-matched attack: LoRA **r=4, lr=1e-3** (round-1's primary).
- **LoRA-robust** = the tuned attack: the **attacker-best LoRA cell** for that arm, i.e. the cell
  in the 4-cell grid with the HIGHEST mean recovery AUC on that arm. (Post-hoc *cell selection* per
  arm — labeled post-hoc — but the *protocol* of "report the arm under its worst-case cell" is
  pre-registered here.)

## Hypotheses

- **H1 (primary, per-arm):** For each routing/hybrid arm, its LoRA-robust recovery AUC is
  *strictly less* than the GA-baseline's LoRA-robust recovery AUC (routed still beats ordinary
  unlearning even against a tuned adapter). Direction: (arm − baseline) < 0.
- **H2 (primary, protocol-level):** For each routing/hybrid arm, the tuned (LoRA-robust) attack
  yields a HIGHER recovery AUC than the provisional attack on the SAME arm — i.e. the provisional
  (weak) attack over-states tamper-resistance. Direction: (robust − provisional) > 0 within-arm.

## Primary tests (pre-stated)

For EACH of the 5 non-baseline arms, TWO pre-registered paired comparisons over n=6 seeds
(seeds 0–5, the repo's own seeds), each an **exact two-sided sign-flip permutation test** on the
mean paired difference (2^6 = 64 sign assignments; minimum attainable two-sided p = 2/64 = 0.03125),
plus the mean paired difference with a 95% t-CI (df=5, t=2.5706):

1. **H1 test** — paired diff in LoRA-robust recovery AUC: (arm − GA-baseline), per seed. The
   baseline's LoRA-robust cell is its own attacker-best cell.
2. **H2 test** — paired diff in recovery AUC on the arm: (arm's LoRA-robust cell − arm's
   provisional cell = r4/lr1e-3), per seed.

Significance level 0.05. There are 5 arms × 2 tests = 10 primary comparisons; I report a
Holm–Bonferroni note but the per-arm test is the pre-registered primary unit (each arm judged on
its own two tests). Nulls (diff ≈ 0, or CI crossing 0) are reported plainly as such.

## Decision criteria

- H1 supported for an arm iff its (arm − baseline) LoRA-robust AUC 95% CI lies entirely below 0.
- H2 supported for an arm iff its (robust − provisional) AUC 95% CI lies entirely above 0.
- The headline is the count of arms passing each, plus the magnitude of the collapse
  (provisional advantage vs robust advantage, arm-vs-baseline).

## Secondary / exploratory (labeled post-hoc if not here)

- Full 4-cell grid AUC per arm (routed/baseline), monotonicity of the gap in attack strength.
- Full-FT reference AUC per arm (replication anchor vs the README claim).
- Retain-accuracy collapse (attacker collateral) at final step, per arm/cell.
- Waste-fraction realized per arm; U = retain−forget after deletion.

## Compute / rigor guardrails

Single process, torch 3 threads, MNIST already on disk (no downloads). n=6 seeds for every number.
Target runtime well under 15 min. Everything not listed above is labeled post-hoc in FINDINGS.
