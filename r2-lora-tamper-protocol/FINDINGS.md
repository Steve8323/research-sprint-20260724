# LoRA tamper-eval as a reusable protocol — round-2 findings

## One-sentence headline

Packaged round-1's LoRA relearning attack into a reusable protocol and ran it against all six
unlearning arms in ~/gradient_routing_experiment; every routing/hybrid arm keeps a small but
directional (6/6-seed) tamper-resistance edge over gradient-ascent unlearning under a tuned
low-rank attacker (arm − baseline AUC −0.04 to −0.14), but that edge is 5–10x smaller than the
weak-attack (provisional) numbers suggest — the provisional attack over-states resistance by
0.43–0.51 AUC on every arm.

## Deliverables

- PREREG.md — hypotheses, per-arm primary tests, decision criteria, stamped `Fri Jul 24 23:09:47 EDT 2026` (before any run).
- lora_tamper_protocol.py — reusable module: lora_attack, fullft_reference, auc, attack_arm (LoRA rank×lr grid + repo full-FT reference), signflip_p, t_ci95. Points at an arbitrary deleted-forget model + mask; heavy pipeline imported unchanged from the read-only repo.
- run_protocol.py — applies the protocol to all 6 arms × 6 seeds and runs the pre-registered tests.
- results.json — full per-seed curves, per-cell AUC, all tests, Holm note. Runtime 57.1 s, single process, 3 threads, no downloads.
- protocol_table.png — provisional vs LoRA-robust vs full-FT AUC per arm.

## The six arms and the deletion gate (all passed)

| arm | how deleted | forget_del | retain_del | gate |
|---|---|---|---|---|
| GA_baseline | KL-to-uniform + retain-CE grad-ascent | 0.038 | 0.959 | ok |
| routed_p0.5 | repo train_routed('random') + prune (round-1-identical) | 0.000 | 0.947 | ok |
| routed_p0.9 | scattered routing p_waste=0.9 + prune | 0.000 | 0.928 | ok |
| hybrid_a0.25 | random↔structural interp α=0.25, p=0.5 | 0.000 | 0.944 | ok |
| hybrid_a0.5 | interp α=0.5, p=0.5 | 0.000 | 0.945 | ok |
| hybrid_a0.75 | interp α=0.75, p=0.5 | 0.000 | 0.945 | ok |

## Protocol table (recovery AUC, mean over 6 seeds; higher = attack recovers faster)

| arm | provisional (r4/1e-3) | LoRA-robust cell | LoRA-robust AUC | repo full-FT |
|---|---|---|---|---|
| GA_baseline | 0.763 | r4/1e-2 | 0.892 | 0.912 |
| routed_p0.5 | 0.396 | r16/1e-2 | 0.826 | 0.723 |
| routed_p0.9 | 0.244 | r16/1e-2 | 0.752 | 0.365 |
| hybrid_a0.25 | 0.398 | r16/1e-2 | 0.821 | 0.721 |
| hybrid_a0.5 | 0.383 | r16/1e-2 | 0.852 | 0.711 |
| hybrid_a0.75 | 0.359 | r16/1e-2 | 0.841 | 0.690 |

Full 4-cell grid (r4/1e-3, r4/1e-2, r16/1e-3, r16/1e-2):

| arm | r4/1e-3 | r4/1e-2 | r16/1e-3 | r16/1e-2 |
|---|---|---|---|---|
| GA_baseline | 0.763 | 0.892 | 0.872 | 0.885 |
| routed_p0.5 | 0.396 | 0.699 | 0.607 | 0.826 |
| routed_p0.9 | 0.244 | 0.567 | 0.402 | 0.752 |
| hybrid_a0.25 | 0.398 | 0.734 | 0.619 | 0.821 |
| hybrid_a0.5 | 0.383 | 0.731 | 0.624 | 0.852 |
| hybrid_a0.75 | 0.359 | 0.752 | 0.605 | 0.841 |

Attacker-best is r16/lr1e-2 for every routing arm; higher rank AND higher lr both help.

## Pre-registered primary tests (n=6, exact two-sided sign-flip, 95% t-CI; floor p = 0.03125)

H1 — routing arm still beats GA-baseline under a tuned attack? (arm − baseline) LoRA-robust AUC:

| arm | mean diff | 95% CI | p | seeds neg | H1 supported |
|---|---|---|---|---|---|
| routed_p0.5 | −0.066 | [−0.090, −0.042] | 0.03125 | 6/6 | yes |
| routed_p0.9 | −0.141 | [−0.176, −0.105] | 0.03125 | 6/6 | yes |
| hybrid_a0.25 | −0.071 | [−0.103, −0.040] | 0.03125 | 6/6 | yes |
| hybrid_a0.5 | −0.040 | [−0.062, −0.019] | 0.03125 | 6/6 | yes |
| hybrid_a0.75 | −0.052 | [−0.069, −0.034] | 0.03125 | 6/6 | yes |

H2 — weak (provisional) attack over-states resistance? (robust − provisional) within-arm:

| arm | mean diff | 95% CI | p | seeds pos | H2 supported |
|---|---|---|---|---|---|
| routed_p0.5 | +0.430 | [+0.371, +0.490] | 0.03125 | 6/6 | yes |
| routed_p0.9 | +0.508 | [+0.454, +0.562] | 0.03125 | 6/6 | yes |
| hybrid_a0.25 | +0.423 | [+0.368, +0.478] | 0.03125 | 6/6 | yes |
| hybrid_a0.5 | +0.469 | [+0.426, +0.512] | 0.03125 | 6/6 | yes |
| hybrid_a0.75 | +0.481 | [+0.419, +0.544] | 0.03125 | 6/6 | yes |

The collapse (advantage over baseline, weak → tuned attack):

| arm | provisional adv | robust adv |
|---|---|---|
| routed_p0.5 | −0.367 | −0.066 |
| routed_p0.9 | −0.519 | −0.141 |
| hybrid_a0.25 | −0.365 | −0.071 |
| hybrid_a0.5 | −0.380 | −0.040 |
| hybrid_a0.75 | −0.404 | −0.052 |

All 5 arms: H1 supported (CI entirely below 0) and H2 supported (CI entirely above 0) per the
pre-registered per-arm decision rule (α=0.05).

## Multiplicity — the honest limitation (pre-stated Holm note)

I pre-registered the per-arm test as the primary unit and a Holm–Bonferroni pass over the 10
p-values as a note. With n=6 the exact sign-flip test's minimum attainable p is 0.03125; the
smallest Holm threshold over 10 tests is 0.05/10 = 0.005. Every individual p sits at the 0.03125
floor, so after a 10-way Holm correction only the least-corrected test (thresh 0.05) formally
rejects — the exact test cannot go below its floor at n=6. The multiplicity-robust reading rests on
the consistency (all 10 tests directional in 6/6 seeds), not per-test FWER. Larger n (≥ ~12 seeds)
is the fix and the obvious round-3 step.

## Round-1 internal replication (positive control)

| quantity | this run | round-1 |
|---|---|---|
| routed_p0.5 provisional (r4/1e-3) AUC | 0.3955 | 0.396 |
| routed_p0.5 r16/1e-2 (robust cell) AUC | 0.826 | 0.826 |
| routed_p0.5 full-FT AUC | 0.723 | 0.723 |
| GA_baseline r4/1e-3 AUC | 0.7628 | 0.763 |
| GA_baseline full-FT AUC | 0.9123 | 0.912 |
| routed_p0.5 robust adv (arm − baseline) | −0.066 | −0.066 |

## Secondary observations (exploratory)

- p_waste=0.9 is the most tamper-resistant routing arm (robust adv −0.141), full-FT AUC 0.365 vs
  0.723 for p_waste=0.5; more waste mass = slower relearning under both attacks. Still, tuned-LoRA
  AUC 0.752 = mostly recoverable.
- Interior α is flat: α=0.25/0.5/0.75 all within 0.03 AUC of the α=0 flagship under the robust
  attack (0.821–0.852 vs 0.826). No hidden interior optimum.
- Attacker collateral: at every arm's robust cell final retain accuracy ≤ 0.01 (pure
  capability-extraction, destroys the rest of the model).
- For routed arms tuned LoRA beats the repo's own full-FT attack (0.826 vs 0.723 at p_waste=0.5)
  without touching pruned coordinates; the published full-FT numbers under-state recoverability.

## Conclusion

Prune-localized / routed unlearning slows relearning more than gradient-ascent unlearning does, and
that ordering holds in 6/6 seeds for every routing configuration even against a tuned low-rank
adapter — but the margin is 5–10x smaller than a weak, budget-matched attack reports, and the
"deleted" capability is 75–85% recoverable (recovery AUC) once the attacker tunes rank+lr. The
provisional-vs-robust gap (H2, +0.43–0.51 AUC on every arm) is the reusable warning: never certify a
tamper-resistance claim on a single un-tuned attack cell.

## Caveats

- Toy scale: 3-layer MLP on MNIST, class-split as capability proxy; existence-proof only.
- n=6 → exact test floors at p=0.03125; Holm over 10 tests does not clear the floor. Claims rest on
  the per-arm pre-registered test + 6/6 directional consistency, not FWER.
- Grid downscoped to 4 cells (dropped r=1) to fit 6 arms in budget; the two poles carrying round-1's
  story (r4/1e-3 weak, r16/1e-2 tuned) retained. Stronger attackers (more steps/data, lr schedules,
  adapter merging, r>16) not explored — robust AUCs are lower bounds on recoverability.
- routed_p0.5 uses repo train_routed('random') (round-1-identical); routed_p0.9 and hybrids use the
  hybrid trainer reproduced from routing-hybrid-knob (statistically identical scattered masks at
  α=0, different RNG draw order so not bit-identical to harness.make_masks). Structural (α=1) and
  non-routing baselines (NPO/RMU) not attacked.
- "LoRA-robust cell" is a post-hoc cell selection per arm (selected on the same seeds it is tested
  on); the protocol of reporting worst-case cell is pre-registered. It is r16/1e-2 for every routing
  arm regardless, which mitigates this.
