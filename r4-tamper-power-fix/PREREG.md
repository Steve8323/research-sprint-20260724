# PREREG — The missing tamper cell, with equivalence bounds (round 4)

Stamp (`date` in the shell, before any arm / metric / test below was evaluated):

```
Sat Jul 25 11:11:13 EDT 2026
```

**Disclosure of what was run BEFORE this stamp** (both are pure engineering probes; neither
evaluates an arm, a metric or a test in this document):

1. `probe_timing.py` — how long one sequential `train_standard` takes on this (loaded) box, and
   whether a vectorised *batched* trainer reproduces the sequential one. Result used for budgeting
   only: sequential 0.655 s/model; batched 0.156 s/model (4.2x); **batched vs sequential deploy
   alignment max|diff| = 0.0 on seeds 3000/3001/3002** (bitwise-identical models). It printed the
   deploy alignment of standard models on seeds 3000–3002, which are *pool-membership* numbers, not
   an arm statistic; those three seeds are inside the round-3 sweep block and their values are
   already public in `r3-switch-tamper/results.json`.
2. `probe_attack.py` — wall-clock of one model x 9 attack cells at two probe grids (23 ms vs 40 ms
   per attack). It did not print or store any AUC / flip cost.

---

## 0. What round 3 got wrong, and the exact repair

Round 3 (`r3-switch-tamper/`, PREREG stamped 10:11:24, results in `results.json`) concluded that the
worldview switch "is not a safety property", resting on two nulls **and one degenerate test**:

| r3 result | r3 number | why it is not evidence of no-difference |
|---|---|---|
| `SEC_auc_switch_vs_h4` (prereg attacker, best-cell AUC) | 0.840 vs 0.834, U=113, p=0.400, n=23 vs 12 | 80% power only at a ~0.10 AUC shift, against a total defense effect of ~0.18. A routing-specific residual of 0.05 is invisible. |
| `posthoc P1_flip_switch_vs_h4` (prereg attacker, flip cost) | median 101 vs 101, p=0.202 | **Degenerate**: 16/23 and 11/12 values sit at the censoring bound 101, and the "working misaligned model" clause (`agree>0.9`) is essentially never satisfied under the prereg attacker. The test cannot reach p<0.05 even if the switch were *perfect*. |
| `PRIMARY A1` (prereg attacker, switch vs corrected) | p=0.133, n=23 vs 12 | reported as a null at low power. |
| **the missing cell** | — | `h4_aligned` was **never run under the Amendment-2 functional attacker**, which is the only attacker under which the switch shows a live, non-degenerate advantage (`AMEND2_func_auc_switch_vs_corrected` p=0.019, `..._flip` p=0.036, only 7/23 censored). The "it's capacity, not routing" conclusion is therefore untested exactly where the effect lives. |

Round 4 does two things and nothing else:

**(A)** runs the missing `h4_aligned` x functional-attacker cell;
**(B)** re-runs every load-bearing tamper comparison at **n = 80 per arm** and reports, for each,
either a **real difference with a CI** or an **equivalence bound with its margin**. The phrase
"no difference" will not be used.

### Scope declaration (what is NOT in round 4)

DRIFT (D1/D2) and STABILITY (S1) are **out of scope**. D1 was a real difference (p=3.05e-5), not an
underpowered null; D2 and the shift retention numbers are left as round-3 reported them. Any
statement about them in FINDINGS.md is a quotation of round 3, not a new test.

---

## 1. Object under test — copied verbatim from round 3

Model / data / thresholds / routing are byte-identical to `r3-switch-tamper/experiment.py`
(INPUT=7, H=8, two hidden layers, 154 params; MARGIN=2.0, NOISE=0.6, N=800, LR=5e-3, WD=2e-3,
EPOCHS=2000; UNITS_A=[0,1,2,3], UNITS_B=[4,5,6,7]; ALIGN_HI=0.8, ALIGN_LO=0.2, TRAIN_OK=0.9;
deploy eval `gen_align(4000, 50001, "deploy")`, held-out agreeing eval `gen_align(4000, 60001,
"train")`). Routing masks come from the round-2 learned data-only localizer L1. The shipped artifact
under attack is the **hard-set switch** (`set_switch`: the non-selected subnet's parameters are
physically zeroed), exactly as in round 3.

**Implementation change, declared in advance:** training is vectorised over models (one batched
tensor of B models trained simultaneously). Adam is elementwise and the per-model loss is the
per-model mean, so this is mathematically the same optimiser trajectory. Verified before the stamp:
**max|diff| = 0.0** in deploy alignment vs the sequential trainer on three seeds. A wider anchor is
preregistered in section 6.

### Arms (all n = 80 unless the sweep cannot reach it)

| arm | definition | source seeds |
|---|---|---|
| `switch_aligned` | routed(L1) selectable model, hard-set to the aligned subnet | routed seeds 700–819, first 80 selectable |
| `h4_aligned` | **capacity control**: routing-free H=4 standard model that landed aligned by chance (same ~62 live parameters as one switch subnet) | 5000–5599, first 80 aligned |
| `corrected_f5` | misaligned-by-chance H=8 standard model corrected by f=5% disagreeing-data fine-tuning (1000 Adam steps, 40 flipped-proxy rows of 800) | 3000–3499, first 80 misaligned |
| `aligned_chance` | H=8 standard model aligned by chance, no defense | 3000–3499, first 80 aligned |
| `switch_misaligned` | routed model hard-set to the **misaligned** subnet | same 80 routed seeds |

Seed blocks are supersets of round 3's (700–729, 3000–3149, 5000–5119), so the round-3 sample is
nested inside the round-4 sample. Setup gate is unchanged (deploy alignment > 0.8 and held-out
agreeing accuracy > 0.9 to enter an attack arm; `switch_misaligned` gate is alignment < 0.2).
Realized n and drops are reported.

## 2. Attack protocol — unchanged from round 3 except probe density

Attacker data, budget, cells, LoRA parameterisation, jitter: identical
(200 proxy-labelled deploy rows `gen_align(200, 91001, "deploy")` with `a = 1-v`; 100 Adam steps,
batch 64; 6-cell prereg grid r in {1,4} x lr in {1e-3,1e-2,1e-1} with alpha/r = 2.0 fixed;
Amendment-1 cells `lora_jit_r4_lr1e-2` and `fullft_jit` at sigma = 0.01; `fullft` anchor).

**Functional attacker (round-3 Amendment 2)**: the same 200 proxy-labelled deploy rows **plus** 200
fresh agreeing rows with true labels (`gen_align(200, 91002, "train")`), same budget. Round 4 runs
the functional attacker on **all four substantive arms including `h4_aligned`** — that is repair (A).

**Only change:** the probe grid goes from 9 coarse points to
`[0..20] + [22,24,...,100]` (61 points). Rationale, stated in advance: flip cost at 9 points has a
resolution of 25 steps near the median, which is a second source of ties on top of censoring. The
denser grid costs 17 ms/attack and changes no other quantity. **The PRIMARY AUC statistic is
computed on the round-3 9-point subgrid** so it is numerically comparable to round 3; the dense-grid
AUC is reported alongside as a secondary.

### Metrics

- **AUC(9pt)** — trapezoidal integral of `1 - deploy_align(t)` over the round-3 probe set, /100. PRIMARY.
- **AUC(61pt)** — same on the dense grid. Secondary.
- **flip cost** — first probe step with `deploy_align < 0.2` **and** `agree_acc > 0.9`; censored -> 101.
- **flipped@100** — binary: did the model ever satisfy the flip condition within 100 steps.
  *This metric cannot be censored and is the designated repair for the degenerate r3 flip test.*
- **capped flip time** `min(flip, 101)` — bounded continuous (restricted-mean style). Secondary.
- **collateral** — held-out agreeing accuracy at step 100.
- **attacker-best cell**: arm-level, the cell with the highest **mean** AUC(9pt) over the 6
  preregistered LoRA cells (identical rule to round 3).

## 3. Preregistered claims, and for each one the test

Notation: `d` = mean(arm1) - mean(arm2). **Achieved equivalence bound** `delta*` = the 90% two-sided
Welch CI bound on |d| (i.e. `|d| + t_{0.95,df} * SE_d`); TOST at margin `delta` rejects non-equivalence
at alpha=0.05 exactly when `delta* <= delta`.

### Preregistered equivalence margins (fixed here, before any run)

Anchor for the scale (round-3 numbers, functional attacker, quoted not re-tested): no-defense
`aligned_chance` best-cell AUC 0.951 vs `switch_aligned` 0.756. **Total defense effect = 0.195 AUC.**

- **delta_AUC = 0.05** = 25.6% of the total defense effect. Justification: a routing-specific residual
  smaller than a quarter of what plain deletion already buys does not make the switch a safety
  property. One margin only; no second margin will be introduced.
- **delta_flip = 10 steps** on capped flip time = 25% of the no-defense -> switch effect (10 -> 50).
- **delta_prop = 0.15** on the difference in `flipped@100` proportions.

### The claims

| id | claim under test | arms | metric | attacker | type |
|---|---|---|---|---|---|
| **C1** | **the missing cell.** Is the switch's advantage under the functional attacker separable from parameter count? | `switch_aligned` vs `h4_aligned` | AUC(9pt), best cell | **functional** | difference-or-equivalence |
| **C2** | same, on the repaired flip metric | `switch_aligned` vs `h4_aligned` | `flipped@100` and capped flip time | **functional** | difference-or-equivalence |
| **C3** | round-3's load-bearing null, re-run at power | `switch_aligned` vs `h4_aligned` | AUC(9pt), best cell | prereg | difference-or-equivalence |
| **C4** | round-3 PRIMARY A1, re-run at power | `switch_aligned` vs `corrected_f5` | AUC(9pt), best cell | prereg | difference-or-equivalence |
| **C5** | the live advantage, re-estimated | `switch_aligned` vs `corrected_f5` | AUC(9pt), best cell | **functional** | difference-or-equivalence |
| **C6** | strongest honest attacker (jitter defeats the relu'(0)=0 artifact) | `switch_aligned` vs `h4_aligned` | `fullft_jit` AUC(9pt) | **functional** | difference-or-equivalence |

**Decision rule, applied identically to C1–C6:**

1. Two-sided exact/permutation Mann–Whitney U on the metric.
2. If **p < 0.05** -> **REAL DIFFERENCE**. Report the Hodges–Lehmann shift with a 95% CI, the mean
   difference with a 95% Welch CI, and the rank-biserial correlation.
3. If **p >= 0.05** -> report the **achieved bound** `delta*`, and label
   **EQUIVALENT-AT-0.05** iff `delta* <= delta_AUC` (resp. `delta_flip`, `delta_prop`), otherwise
   **BOUNDED-AT-`delta*`**. The words "no difference" are not an available verdict.
4. For proportions, step 1 is Fisher's exact and step 3 uses the Newcombe hybrid-score 90% CI on the
   difference.

Every claim also carries: n per arm, the two arm means with 95% bootstrap CIs, and its own bound.

### Sample size justification

Round-3 per-model SD of best-cell AUC: `switch_aligned` ~0.134, `h4_aligned` ~0.121 (from the
reported bootstrap CIs). Take sigma_w = 0.13.

- **Round-3's own selection-inflation design rule**, `n >= (a_k * sigma_w / delta)^2` with k = 6 selected
  cells (`a_6 = 1.267` from `r3-selection-inflation-curve`): `n >= (1.267*0.13/0.05)^2 = 10.9`.
  Satisfied by a wide margin at n=80 — i.e. attacker-best-cell selection inflation is not the
  binding constraint.
- **TOST power is the binding constraint.** At n=80/80, `SE_d = 0.13*sqrt(2/80) = 0.0206`, so the 90%
  CI half-width is 0.034. TOST at delta=0.05 therefore declares equivalence iff |d| <= 0.016, giving
  ~57% power when the true difference is exactly 0. **This is stated in advance**: n=80 is enough to
  *bound* the difference at roughly +/-0.034 but is not guaranteed to reach the 0.05 verdict. The
  bound is the deliverable; the verdict is a bonus. Compute budget (15 min, shared box) does not
  permit n=200, which is what ~90% TOST power at delta=0.05 would need.

### Dynamic-range check (the repair for the degenerate r3 test) — preregistered

For each flip metric I will report the p-value the test **would** return under the extreme
configuration "switch resists perfectly (`flipped@100` = 0/80) vs h4 behaves like no-defense
(80/80)". A metric whose extreme-configuration p is not < 0.05 is declared unusable and is not
reported as evidence. (Round 3's censored median flip test fails this check by construction; Fisher
on `flipped@100` at 80/80 does not.)

## 4. Controls

- **Positive control for the readout**: `switch_misaligned` under the prereg attacker must show
  AUC ~ 1 and flip cost 0 (it starts misaligned). If not, the attack readout is broken.
- **Positive control for the equivalence machinery**: `switch_aligned` vs `aligned_chance` under the
  functional attacker must come out **REAL DIFFERENCE** (known effect ~0.195). If the procedure
  calls that equivalent, the procedure is broken.
- **Negative/calibration control for the equivalence machinery**: split each of `switch_aligned` and
  `h4_aligned` into two random halves (seeded split, n=40/40, true difference exactly 0) and run the
  identical pipeline. Reports the bound the procedure achieves when the truth is 0.
- **Replication anchor**: routed selectability rate on seeds 700–729 must reproduce round 3's 23/30
  under the batched trainer, and the standard-model base rates must be within sampling error of
  round 3's (0.333 misaligned / 0.235 aligned / 0.182 H=4 aligned). Reported as a
  protocol-equivalence check; a mismatch is reported, not repaired.
- **`fullft` on `switch_aligned` remains an artifact anchor**, not evidence: with exactly-zero dead
  units `relu'(0) = 0` blocks their revival. The jitter cells (C6) are the honest version.

## 5. Multiplicity

Six preregistered claims (C1–C6). Each is judged on its own stated rule. A Holm–Bonferroni note over
C1–C6 is reported for the difference-direction tests. Nulls/bounds are reported plainly. No claim
will be rescued post-hoc; anything not in this file is labeled POST-HOC in FINDINGS.md and lives in
a separate file.

## 6. Verification and budget

- **Protocol-equivalence anchor**: re-train one `h4_aligned` and one `aligned_chance` seed with the
  original sequential `r3-switch-tamper` code path and assert max|diff| in deploy alignment
  vs the batched trainer. Reported as a number, whatever it is.
- Single process, torch 2 threads, CPU, no downloads. Budget estimate: training ~185 s, attacks
  ~6000 runs x 40 ms ~ 240 s, total ~7 min against a 15 min cap.
- **Preregistered drop order if the run exceeds 660 s before the attack phase finishes**:
  (1) `aligned_chance` under the *prereg* attacker, (2) the `switch_misaligned` positive control,
  (3) the split-half calibration control. Any drop is stated in FINDINGS.md. C1–C6 are never dropped.

---

## 7. Amendment 1 — n reduced 80 -> 60, with full smoke disclosure

Stamp (before the confirmatory run; after a reduced-budget smoke run):

```
Sat Jul 25 11:18:44 EDT 2026
```

**What happened.** A SMOKE run (`smoke.py`: n=6/arm, 12 routed seeds, 60/80 sweep seeds, routed
EPOCHS=300, correction 200 steps) was executed to check the pipeline executes end to end. It took
75 s and showed the shared box is running ~2.2x slower than the pre-stamp timing probe measured
(0.35 s/model batched vs 0.156 s; 93 ms/attack vs 40 ms — five other agents are running). Extrapolated
to the preregistered design, n=80/arm costs ~930 s, over the 15-minute hard cap.

**Change.** `N_ARM` 80 -> 60; routed seed block 700–819 -> 700–789; sweep chunk 100 -> 60. Nothing
else changes: same arms, same metrics, same margins, same decision rules, same claims C1–C6, same
controls. n=60 is the floor named in the round-4 brief. This change can only *reduce* power; it
cannot rescue any claim.

**Updated power statement (replaces the n=80 numbers in section 3).** At n=60/60 with
sigma_w = 0.13: `SE_d = 0.13*sqrt(2/60) = 0.0237`, 90% CI half-width 0.039. TOST at delta=0.05
therefore declares equivalence iff |d| <= 0.011, i.e. **~36% power when the true difference is
exactly 0**. This is stated in advance and is why the *achieved bound* `delta*`, not the verdict, is
the deliverable. Round-3's design rule `n >= (a_6*sigma_w/delta)^2 = 10.9` is still satisfied ~5x over,
so attacker-best-cell selection inflation remains not the binding constraint.

**Full disclosure of what the smoke run showed (n=6/arm, 300-epoch routed models — NOT the round-4
numbers, and used nowhere).** C1 d=+0.051 p=0.49; C2a d=0.00 p=1.0; C2b d=-7.2 p=1.0; C3 d=-0.011
p=0.94; C4 d=-0.084 p=0.014; C5 d=-0.078 p=0.24; C6 d=+0.119 p=0.48; positive control REAL
DIFFERENCE; readout control mean AUC 0.995, flip 0/6 seeds resisted; split-half controls bounded at
0.243/0.356; dynamic-range check confirmed `flipped@100` is usable (p=0.0022 under the extreme
configuration) and the round-3 censored median-flip test is not (p=1.0). **No decision rule, margin,
arm, metric or claim was altered after seeing these.** The two things they did change are recorded
above: n, and nothing else.

**Also disclosed:** in the smoke run `E.EPOCHS=300` did not take effect for standard/H=4 models
(Python binds default arguments at definition time), so those were trained at the full 2000 epochs;
only routed training and the correction step were shortened. Irrelevant to the confirmatory run,
which uses the module constants directly.
