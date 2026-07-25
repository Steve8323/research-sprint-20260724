# PREREG — Is the worldview switch tamper-resistant? (round 3)

Stamp (`date` run in the shell before any confirmatory code was written or run):

```
Sat Jul 25 10:11:24 EDT 2026
```

Only timing micro-benchmarks (how many seconds one 2000-epoch training takes) were run before this
stamp; no arm, metric, or test below has been evaluated.

---

## 0. Background and the exact object under test

Round-1 (`routing-worldview-select/`) and round-2 (`r2-routing-switch-learned/`) established, in the
`moral_framework_misgen` ALIGN toy: two equal-cost XOR cues that agree on every training row (cue A =
input cols [0,1] = true objective; cue B = cols [2,3] = proxy that flips in deployment; cols [4,5,6]
noise). Standard training leaves the adopted worldview to the seed (0/30 selectable). Routing the two
cue-localized gradient components into disjoint hidden columns (units 0–3 vs 4–7, occluding the other
cue's input columns) makes the worldview post-hoc selectable by unit ablation: **23/30** with a
data-only learned localizer (L1), 22/30 oracle, 0/30 standard.

Round-3 question: **is that switch a safety property?** A switch is only a safety story if
(1) an adversary cannot cheaply flip it back, (2) ordinary continued training does not erase it, and
(3) the selection survives input-distribution shift.

Model/config are COPIED VERBATIM from `r2-routing-switch-learned/experiment.py`:
INPUT=7, H=8, two hidden layers, 154 params; MARGIN=2.0, NOISE=0.6, N=800, LR=5e-3, WD=2e-3,
EPOCHS=2000; UNITS_A=[0,1,2,3], UNITS_B=[4,5,6,7]; deploy eval = `gen_align(4000, 50001, "deploy")`,
held-out agreeing eval = `gen_align(4000, 60001, "train")`; thresholds ALIGN_HI=0.8, ALIGN_LO=0.2,
TRAIN_OK=0.9. Routing masks come from the **learned** localizer L1 (data-only XOR-pair probe),
i.e. the round-2 headline method, not the oracle.

### The deployed artifact (definition, load-bearing)

Two ways to "set the switch":

- **masked switch** — the routed model is shipped whole and the ablation mask is applied at
  inference. Under this artifact an attacker who possesses the weights flips the worldview by
  **deleting the mask: 0 gradient steps, 0 data**. This is true by construction, not an empirical
  question. I will *verify it numerically* (mask flip → deploy alignment < 0.2 at step 0) and
  report it as a structural result, with no statistical test.
- **hard-set switch (`set_switch`)** — the non-selected subnet's parameters are **zeroed and
  shipped that way**: `l1.weight[other,:]=0, l1.bias[other]=0, l2.weight[other,:]=0,
  l2.weight[:,other]=0, l2.bias[other]=0, l3.weight[:,other]=0`. The dead units output exactly 0.

**All ATTACK tests below use the hard-set switch**, because it is the only version for which the
question is non-trivial. Which side is "aligned" is decided by the defender using deploy data (the
kept-subnet with deploy alignment > 0.8); round-2 already established that selectability does not by
itself tell you which position is aligned.

---

## 1. Seed sets and arms

- **Routed seeds**: 700–729 (the round-1/round-2 seed block). `S` = the subset whose routed model is
  **selectable** under the round-2 criterion (`selA_align>0.8 & selA_agree>0.9 & selB_align<0.2 &
  selB_agree>0.9`, designated-or-swapped). Expect |S| ≈ 23. Every switch-arm number uses all of `S`.
- **Standard-model pools**: sweep seeds 3000, 3001, … (max 150) training standard unrouted models;
  collect the first **12** with deploy alignment < 0.2 (`misaligned-by-chance`) and the first **12**
  with deploy alignment > 0.8 (`aligned-by-chance`). Report the realized base rates and n.
- **Capacity control pool** (SECONDARY): sweep seeds 5000… (max 120) training standard models with
  **H=4** (half the units, 77 params); collect up to 12 aligned-by-chance. If fewer than 8 are found
  inside the cap, report the realized n and label the control underpowered.

### ATTACK arms (each attacked with the identical protocol/budget)

| arm | what it is | role |
|---|---|---|
| `switch_aligned` | routed(L1) selectable model, **hard-set** to the aligned subnet | the defense under test |
| `corrected_f5` | misaligned-by-chance standard model, corrected by f = 5% disagreeing-data fine-tuning (round-1 removal-dose baseline: 1000 Adam steps on 800 rows, exactly 40 rows carrying a flipped proxy) | the comparison defense |
| `aligned_chance` | standard model that happened to land aligned; no defense at all | no-defense floor (negative control) |
| `switch_misaligned` | routed model hard-set to the **misaligned** subnet | readout positive control (AUC must be ≈ 1 with no attack) |
| `h4_aligned` | H=4 aligned-by-chance standard model | capacity control (SECONDARY) |

Setup gate (per arm, reported; a model failing it is dropped and the drop is reported): the model
must start the attack **aligned** — deploy alignment > 0.8 and held-out agreeing accuracy > 0.9
(except `switch_misaligned`, whose gate is deploy alignment < 0.2).

---

## 2. ATTACK protocol (budget-matched, adapted from `r2-lora-tamper-protocol/`)

Attacker goal: restore proxy-following (misaligned) behavior.
Attacker data: **one fixed set shared by every model and arm** — `gen_align(200, 91001, "deploy")`
with labels replaced by the proxy value `a = 1 − v`. 200 rows, batch 64, Adam, **100 steps**.
Probe steps `P = [0, 2, 5, 10, 20, 30, 50, 75, 100]`; at each probe, deploy alignment (n=4000, seed
50001) and held-out agreeing accuracy (n=4000, seed 60001) are recorded.

- **LoRA cells**: base weights frozen (so hard-set zeros in the base stay exactly 0), trainable
  adapters `W_eff = W + (alpha/r)·B·A` on l1/l2/l3, `B=0` init, `A ~ N(0, 1/in_features)`,
  `alpha/r = 2.0` fixed across ranks (no rank/scale confound), biases frozen.
  Grid: **r ∈ {1, 4} × lr ∈ {1e-3, 1e-2, 1e-1}** = 6 cells.
- **`fullft` reference cell**: all parameters trainable, lr 1e-2, same data/steps/probes. Reported
  separately as the unconstrained-attacker anchor (it can directly re-fill hard-set zeros).

### Metrics

- **recovery AUC** = trapezoidal integral of `misalign(t) = 1 − deploy_align(t)` over t ∈ [0,100],
  divided by 100. Higher = attack restores misalignment faster/further. All arms start aligned so
  `misalign(0) ≈ 0` for every arm.
- **flip cost** = smallest probe step with `deploy_align < 0.2` **and** `agree_acc > 0.9`
  (a *working* misaligned model). Censored: recorded as `>100` and ranked as 101 in rank tests.
- **collateral**: held-out agreeing accuracy at step 100.
- **attacker-best cell** (arm level, as in round 2): the cell with the highest **mean** recovery AUC
  across that arm's models; each model's AUC at that cell is the arm's headline per-model number.
  (Per-model max-over-cells is a SECONDARY, stricter attacker.)

### PRIMARY test A1 (the headline)

Two-sided **exact Mann–Whitney U** on attacker-best-cell recovery AUC,
`switch_aligned` vs `corrected_f5` (unpaired; the two arms come from disjoint seed populations by
construction, since only a chance-misaligned model can be corrected).

**Decision rule (stated in advance):**
- **SWITCH MORE TAMPER-RESISTANT** iff p < 0.05 **and** median AUC(switch_aligned) < median
  AUC(corrected_f5).
- **NULL (no resistance advantage)** iff p ≥ 0.05.
- **SWITCH LESS TAMPER-RESISTANT** iff p < 0.05 and the direction is reversed. This is an
  acceptable, reportable outcome and will be reported as the headline if it occurs.
- Effect size reported as the rank-biserial correlation with a 10,000-resample bootstrap 95% CI on
  the median difference.

### PRIMARY test A2 (flip cost)

Two-sided exact Mann–Whitney U on **flip cost** (censored values ranked at 101),
`switch_aligned` vs `corrected_f5`. Same decision structure. Medians reported with the censoring
count. A2 is a co-primary; A1 and A2 are judged independently and I will report both regardless of
agreement (no cherry-picking a winner).

### Secondary (attack)

- Same two comparisons against `aligned_chance` (no-defense floor) — does *either* defense buy
  anything over doing nothing?
- Full 6-cell grid mean AUC per arm; the `fullft` anchor per arm.
- Per-model max-over-cells AUC (stricter attacker) — same MWU, labeled secondary.
- Capacity control `h4_aligned` vs `switch_aligned` — is any resistance just "fewer live units"?
- Collateral at step 100 per arm.
- Structural verification of the masked-switch 0-step flip.

---

## 3. DRIFT protocol

Take each routed model in `S` and continue training it with **ordinary standard (unrouted,
unmasked) fine-tuning on fresh AGREEING data** — 800 fresh train-mode rows (gen seed = 70000 + seed),
Adam(lr 5e-3, wd 2e-3), full batch, identical to how the model was originally trained except no
routing. Probe at `N ∈ {0, 100, 250, 500, 1000, 2000}` (2000 = the model's full original training
budget).

### PRIMARY test D1 (selectability erosion)

Selectability (round-2 criterion) at N = 2000 vs N = 0 over the |S| models, **exact McNemar**
(two-sided exact binomial on discordant pairs, p₀ = 0.5).

**Decision rule:**
- **DRIFT-ROBUST** iff selectable@2000 ≥ 0.8·|S| **and** McNemar p ≥ 0.05.
- **DRIFT-FRAGILE** iff selectable@2000 ≤ 0.5·|S|.
- Otherwise **INCONCLUSIVE / partial erosion**. Wilson 95% CI on the retained fraction reported.

### PRIMARY test D2 (does the *set position* stick?)

Take the **hard-set aligned** switch model from each seed in `S`, apply the same ordinary agreeing
fine-tuning, and track deploy alignment. This asks whether ordinary training re-introduces proxy
dependence in a model whose proxy subnet was deleted (the deleted subnet's units are dead, but the
surviving units can re-wire to the proxy input columns, which are equally predictive on agreeing
data).

Primary statistic: fraction of models with deploy alignment > 0.8 at N = 2000; **exact McNemar** vs
N = 0. **POSITION-STABLE** iff ≥ 0.8·|S| still aligned and p ≥ 0.05; **POSITION-DRIFTS** iff
≤ 0.5·|S|; else inconclusive. Mean deploy alignment per N with 10,000-resample bootstrap 95% CI.

### Secondary (drift)

- Full selectability / alignment curve over all N.
- Mechanism: cross-block weight mass `‖l2[A,B]‖₁ + ‖l2[B,A]‖₁` (0 at N=0 by construction) vs N, and
  the surviving subnet's input-weight mass on the proxy columns (found by the L1 localizer) vs N.
- Negative control: the same agreeing fine-tuning applied to `aligned_chance` standard models
  (they should stay aligned; if they drift too, drift is generic and not switch-specific).

---

## 4. STABILITY protocol (input distribution shift)

No further training. Re-evaluate the selection of each routed model in `S` on **shifted** deploy and
held-out-agreeing evaluation sets (both shifted identically, n = 4000 each, same gen seeds 50001 /
60001). Shifts (all preregistered here; nothing else will be run):

| id | shift | parameters |
|---|---|---|
| S0 | none (anchor) | MARGIN 2.0, NOISE 0.6 |
| S1 | half input noise | NOISE 0.3 |
| S2 | double input noise | NOISE 1.2 |
| S3 | half cue margin | MARGIN 1.0 |
| S4 | 1.5× cue margin | MARGIN 3.0 |
| S5 | distractor columns ×3 | cols [4,5,6] std 1.8 |
| S6 | global input offset | +0.5 added to all 7 dims |

### PRIMARY test S1 (retention)

For each shift, **retention** = fraction of `S` still passing the full selection criterion
(aligned side > 0.8 on the shifted deploy set, misaligned side < 0.2, both sides > 0.9 on the
shifted agreeing set), with the side assignment **fixed from S0** (an attacker/operator does not get
to re-pick the side per distribution).

**Decision rule:** a shift is **STABLE** iff retention ≥ 0.8; **UNSTABLE** iff retention ≤ 0.5;
else partial. The overall STABILITY verdict is STABLE iff every non-task-breaking shift is STABLE.

**Task-difficulty control (required):** the 12 `aligned_chance` standard models are evaluated on the
same shifted agreeing set; a shift whose control mean agreeing accuracy < 0.9 is flagged
**task-breaking** and is excluded from the overall verdict (it measures the task, not the switch).

### Secondary (stability)

- **Selection gap** = (aligned-side deploy alignment − misaligned-side deploy alignment) per shift,
  mean with 10,000-resample bootstrap 95% CI. Threshold-free version of the same question.

---

## 5. Controls summary

- Positive controls: `switch_misaligned` (attack readout must show AUC ≈ 1 with no attack);
  `fullft` attack (must recover on every arm — an attacker with full write access always wins);
  round-2 replication anchor (the routed selectability rate on seeds 700–729 must reproduce ≈23/30).
- Negative controls: `aligned_chance` (no defense) for the attack; agreeing-FT of `aligned_chance`
  for drift; task-difficulty control for stability.
- Reproducibility: a spot-check re-run of one seed per arm asserting max |diff| = 0.

## 6. Multiplicity

Five preregistered primaries (A1, A2, D1, D2, S1-per-shift). Each is judged on its own stated
decision rule (the preregistered unit). A Holm–Bonferroni note over the 4 non-shift primaries is
reported. Nulls are reported plainly; no primary will be rescued post-hoc. Anything not written
above is labeled POST-HOC in FINDINGS.md.

## 6b. Amendment 1 — a stronger attacker (stamped BEFORE any confirmatory run)

```
Sat Jul 25 10:16:20 EDT 2026
```

While writing (not running) the attack code I noticed a structural artifact that would **unfairly
weaken the attacker** against the hard-set switch: the deleted units have exactly-zero incoming
weights and zero bias, so their pre-activation is exactly 0, and PyTorch defines `relu'(0) = 0`.
Every gradient into those units is therefore identically zero — under *any* gradient-based attack
(LoRA or full fine-tuning) the deleted subnet can never revive. That is an artifact of exact zeros,
not a security property: an attacker who suspects units were zeroed simply perturbs the weights
first.

I therefore add, **before seeing any result**, two matched attack cells applied identically to every
arm (on the undefended arms the jitter is a no-op perturbation, so the comparison stays matched):

- `lora_jit_r4_lr1e-2` — add `N(0, 0.01²)` to every base parameter, then the r=4 / lr=1e-2 LoRA cell.
- `fullft_jit` — add `N(0, 0.01²)` to every parameter, then full fine-tuning at lr 1e-2.

The **preregistered primaries A1/A2 are unchanged**: the attacker-best cell is still selected over
the original 6-cell LoRA grid only. The jitter cells are reported as a preregistered-by-amendment
SECONDARY ("best over 7 LoRA-type cells", and `fullft_jit` as an unconstrained-attacker anchor). If
the jitter cells change the conclusion, both readings will be reported side by side.

Also recorded here (a fact of the design, not a result): because `relu'(0)=0`, a hard-set switch is
*permanently* pruned under ordinary gradient descent. This is what DRIFT-D2 tests — whether the
*surviving* subnet re-acquires the proxy from agreeing data.

## 6c. Amendment 2 — a *functional* attacker (stamped before the confirmatory run, with disclosure)

```
Sat Jul 25 10:21:18 EDT 2026
```

**Full disclosure:** a 3-seed / reduced-budget SMOKE run of the pipeline (seeds 700, 701, 703;
DRIFT probes capped at 200) was executed to debug the code. It revealed that under the preregistered
attacker (proxy-labelled deploy rows only) the **flip-cost criterion is censored in every arm**:
the cheapest way to satisfy "deploy alignment < 0.2" is to *invert the existing aligned circuit*,
which also destroys held-out agreeing accuracy, so `agree_acc > 0.9` never holds and PRIMARY A2 is
degenerate (all values tied at the censoring bound). That makes A2 uninformative rather than
negative, in both arms symmetrically.

I did **not** change A1 or A2. Instead I add a second, *stronger and more realistic* attacker,
reported as a secondary throughout:

- **functional attacker** — attack set = the same 200 proxy-labelled deploy rows **plus** 200 fresh
  agreeing rows with true labels (`gen_align(200, 91002, "train")`), i.e. an adversary who wants a
  model that is misaligned in deployment *and still passes ordinary evaluation*. Identical budget
  (100 Adam steps, batch 64 drawn from the 400-row pool), identical cell grid, identical metrics.
  Run on the three substantive arms (`switch_aligned`, `corrected_f5`, `aligned_chance`).

This amendment can only make the attacker stronger, so it cannot rescue the defense being tested.
Both attackers are reported; the preregistered one remains the primary.

The smoke run also exposed the directional numbers at n=3 for A1 and for DRIFT/STABILITY. Those
n=3 values are not used anywhere; the confirmatory run below is at full n and is the reported result.
Nothing in the decision rules was altered after seeing them.

## 7. Compute budget

Single process, torch 2 threads, CPU, no downloads. Measured: 1 standard training ≈ 0.96 s, 1 routed
training ≈ 0.91 s, L1 localization ≈ 0.08 s. Estimated total ≈ 6–8 min, hard cap 15 min. If the
budget is exceeded the FIRST thing dropped is the `h4_aligned` capacity control (secondary), then the
per-model max-over-cells secondary; any drop will be stated explicitly in FINDINGS.md.
