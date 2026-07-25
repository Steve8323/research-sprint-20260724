# Portfolio selection audit + three corrections

**Produced:** 2026-07-25 (round 4) · **Scope:** analysis-only on saved artifacts. Nothing outside
`r4-kway-learned-plus-hygiene/` was modified. All recomputed numbers are in `hygiene_results.json`;
the generating script is `hygiene.py` (runtime 2.9 s).

Reading key for `k`: **the number of cells the reported number was selected from.** `k = 1` means the
number was pre-specified (or the statistic contains no max/argmax over cells) and is therefore
**IMMUNE** to selection inflation. `s` = per-cell standard error, `S` = SD across cell means,
`rho_hat = max(0, 1 - s²/S²)`, `R = |winner − mean| / (c_k · s)` where `c_k = a_k` for a
direction-pre-specified selection and `c_k = b_k` for an argmax-of-|·| selection.
Verdict scale (inherited from `r3-selection-inflation-curve/PREREG.md`): `R < 1.5` SUSPECT,
`1.5 ≤ R < 3` BORDERLINE, `R ≥ 3` ROBUST.

---

## 1. The constant: one-sided `a_k` vs two-sided `b_k`

`a_k = E[max_j N_j]` is the right null scale only when the **direction** of selection was fixed in
advance. For a statistic selected as `argmax_j |·|` the null scale is `b_k = E[max_j |N_j|]`.
Exact values by quadrature (`∫₀^∞ 1 − (2Φ(x) − 1)^k dx`), cross-checked against 400 k MC draws
(max abs error 8·10⁻⁴):

| k | a_k (one-sided, exact) | a_k (Blom, what r3 used) | **b_k (two-sided, exact)** | b_k (Blom-style plug-in) | b_k / a_k |
|---|---|---|---|---|---|
| 2 | 0.564 | 0.564 | **1.128** | 1.128 | 2.000 |
| 3 | 0.846 | 0.846 | **1.326** | 1.318 | 1.567 |
| **4** | **1.029** | **1.029** | **1.465** | 1.451 | **1.423** |
| 5 | 1.163 | 1.163 | **1.570** | 1.557 | 1.350 |
| 6 | 1.267 | 1.267 | **1.654** | 1.642 | 1.305 |
| 10 | 1.539 | 1.539 | **1.881** | 1.872 | 1.222 |
| 24 | 1.948 | 1.948 | **2.238** | 2.232 | 1.149 |
| 25 | 1.965 | 1.965 | **2.254** | 2.247 | 1.147 |
| 144 | 2.635 | 2.635 | **2.865** | 2.862 | 1.087 |
| 1770 | 3.402 | 3.402 | **3.588** | 3.587 | 1.055 |

Notes: `b_2 = 2 a_2 = 2/√π` exactly (a useful sanity check). REPORT3 quotes `b_4 = 1.47`; the exact
value is **1.4647**. The Blom-style plug-in `Φ⁻¹((1+q)/2)`, `q = (k−0.375)/(k+0.25)`, is within 1%
for k ≥ 3 and is the recommended closed form. **The error shrinks with k** — it matters most exactly
where sweeps are small.

---

## 2. Correction A — recompute `R_gap` in `r3-selection-inflation-curve` with `b_4`

`run.py` lines 243–246 compute the "max adjacent gap" statistic as `jg = argmax |gm|` over
`k_gaps = 4` adjacent α-gaps — a genuinely two-sided selection — but divide by `blom_a(4) = 1.029`.
Correct denominator is `b_4 = 1.4647`, so every published `R_gap` is inflated by **×1.423**.

| target (routing-hybrid-knob) | k_gaps | max gap | gap SE | R_gap published | verdict published | **R_gap corrected** | **verdict corrected** | changes? |
|---|---|---|---|---|---|---|---|---|
| p_waste 0.5, recovery AUC ("resistance arrives between α 0.75 and 1.0") | 4 | 0.03907 | 0.00739 | 5.038 | ROBUST | **3.609** | **ROBUST** | no |
| p_waste 0.5, retain_pruned | 4 | 0.00462 | 0.00176 | 2.500 | BORDERLINE | **1.791** | **BORDERLINE** | no |
| p_waste 0.9, recovery AUC ("nearly all resistance arrives at the α=1 endpoint") | 4 | 0.15470 | 0.02833 | 5.205 | ROBUST | **3.728** | **ROBUST** | no |
| p_waste 0.9, retain_pruned | 4 | 0.04513 | 0.00569 | 7.554 | ROBUST | **5.411** | **ROBUST** | no |

**No verdict flips**, but the margin on the two headline "resistance arrives at the endpoint" rows
drops from ~5× to ~3.6–3.7×, i.e. just above the ROBUST boundary rather than comfortably above it.

**Explicitly NOT affected** (their selection direction was pre-specified, so `a_k` is correct):
the cell-level `R` for recovery AUC (`argmin`, lower = more resistant) and retain (`argmax`), and
the `misgen-bimodality` dip statistic (`argmax`, dip is non-negative and larger = more bimodal).

---

## 3. Correction B — `misgen-connectivity-graph` modularity is family-dependent

Max-over-θ modularity `Q` of the fixed 3-way worldview partition, node-label permutation null
(1000 permutations, seed 0, same incremental computation r3 used; observed curve reproduces the
saved curve to 5.6·10⁻¹⁷).

| θ family | k (thetas) | Q_max observed | θ at max | null mean | null p95 | **family-wise p** | sig at 0.05? |
|---|---|---|---|---|---|---|---|
| all θ (r3's null) | 1770 | 0.2200 | 0.00086 | 0.134 | 0.500 | **0.2028** | no |
| **published window θ ∈ [0.0015, 0.090]** | 1595 | 0.1990 | 0.00153 | — | — | **0.0280** | **yes** |
| θ with ≥ 60 edges | 1711 | 0.0599 | 0.02279 | — | — | **0.1119** | no |

**Restatement to use:** *"Worldview modularity of the barrier graph is family-dependent. Within the
window the paper reports (θ ∈ [0.0015, 0.090]) the peak Q = 0.199 exceeds a max-over-θ label
permutation null at p = 0.028; under broader families — all 1770 thresholds, or all thresholds with
at least 60 edges — it does not (p = 0.20 and p = 0.11). The all-θ null is dominated by degenerate
near-empty graphs where Q is trivially large."* Not "no community structure at any threshold"
(REPORT.md §b/§d), and not "withdrawn".

**IMMUNE and unchanged:** the experiment's pre-stated PRIMARY, the threshold-free rank
AUC = 0.725 with node-label permutation p ≤ 1·10⁻⁴. It contains no max over θ. Every downstream
sentence that rests on the AUC (barrier heights order within-worldview pairs ahead of cross-worldview
pairs) stands as written.

---

## 4. Correction C — `routing-hybrid-knob` T3: the demotion is right, the stated reason is wrong

REPORT3 §g item 8 asks to demote T3 ("interior alphas are Pareto-useful" at p_waste = 0.5) to
unsupported **because `rho_hat = 0.00` on the quality axis**. The `rho_hat = 0` is an artifact.

| quantity (p_waste = 0.5, retain axis, 5 alphas × 6 seeds) | with the MARGINAL across-seed SE (what r3 Part C used) | with the INTERACTION SE (what r3 Part B *prescribes*) |
|---|---|---|
| s | 0.00261 | **0.00129** |
| S | 0.00228 | 0.00228 |
| **rho_hat** | **0.000** | **0.680** |
| R (winner α = 0 vs sweep mean, a₅ = 1.163) | 1.231 → SUSPECT | **2.492 → BORDERLINE** |
| seed main-effect share of total SS | 72.8% | — |

`r3-selection-inflation-curve/PREREG.md` Part B says in as many words: *"Use the **interaction**
(item × cell) SE, not the marginal one."* Part C then used the marginal one. In a fully paired design
(the same 6 seeds at every α) where 73% of the variance is a seed main effect, the marginal SE is
2× too large and drives `rho_hat` to zero mechanically.

The paired contrasts the T3 criterion actually requires (n = 6 seeds, paired 95% t-CI):

| leg | contrast | mean | 95% CI | seeds in direction | exact Wilcoxon p |
|---|---|---|---|---|---|
| quality (must beat the α=1 endpoint) | retain(α=0.25) − retain(α=1) | +0.00152 | [−0.00245, +0.00548] | 4/6 | 0.56 |
| quality | **retain(α=0.50) − retain(α=1)** | **+0.00242** | **[+0.00079, +0.00404]** | 5/6 | **0.042** |
| quality | retain(α=0.75) − retain(α=1) | +0.00192 | [−0.00464, +0.00847] | 4/6 | 0.56 |
| resistance (must beat the α=0 endpoint) | AUC(α=0) − AUC(α=0.25) | +0.01465 | [+0.00204, +0.02726] | 5/6 | — |
| resistance | AUC(α=0) − AUC(α=0.50) | +0.02485 | [+0.00997, +0.03973] | 6/6 | — |
| resistance | AUC(α=0) − AUC(α=0.75) | +0.04598 | [+0.02691, +0.06506] | 6/6 | — |

Selection-corrected for taking the best of the 3 interior alphas (one-sided, a₃ = 0.846, paired
SE 0.00063): `R_T3 = 4.52` — the α = 0.50 gap is 4.5× what a null 3-cell selection would produce.

**Verdict to publish:** T3's stated criterion is **met** at p_waste = 0.5, so "unsupported because
rho_hat = 0" must not be written. What should be demoted is the *word*, not the *fact*:
> "At p_waste = 0.5 exactly one interior setting (α = 0.5) is non-dominated on cell means; its
> quality advantage over the α = 1 endpoint is +0.24 retain points (5/6 seeds, uncorrected exact
> p = 0.042, which does **not** survive Bonferroni over the three interior alphas: p = 0.13), on a
> quality axis whose entire span is 0.6 points. The knob is statistically resolvable and practically
> negligible — take α = 1."

That is what `routing-hybrid-knob/FINDINGS.md` already says in prose ("in practice you'd take
alpha=1"); only the summary word "Pareto-useful" over-claims.

**Side effect of the same SE fix on the other three hybrid-knob cells** (POST-HOC, not in PREREG):

| cell | rho_hat marginal | rho_hat interaction | R marginal | R interaction | verdict change |
|---|---|---|---|---|---|
| p_waste 0.5, retain | 0.000 | 0.680 | 1.23 SUSPECT | 2.49 BORDERLINE | **yes (up)** |
| p_waste 0.5, AUC | 0.906 | 0.951 | 4.33 ROBUST | 6.01 ROBUST | no |
| p_waste 0.9, retain | 0.918 | 0.880 | 1.69 BORDERLINE | 1.40 SUSPECT | **yes (down)** |
| p_waste 0.9, AUC | 0.902 | 0.900 | 4.87 ROBUST | 4.82 ROBUST | no |

The fix moves verdicts in **both** directions, which is what an honest correction should do. The one
that moves down — p_waste 0.9 retain, R = 1.40 SUSPECT — is the α = 0 vs sweep-mean retain excess,
*not* the preregistered T2 endpoint test (α = 0 vs α = 1, 6/6 seeds, exact p = 0.0156), which is
k = 1 and IMMUNE.

---

## 5. The two-line selection audit for every headline number

`n/a` = not applicable (k = 1). `not recov.` = not recoverable from saved artifacts without re-running.

### Round 3 (REPORT3.md)

| # | Headline number | k | n | s | S | rho_hat | R | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | r3-multiway-switch: k=3 and k=4 routed **20/20** vs control 0/20, Fisher p = 1.45e-11 | **1** | 20 seeds/arm | n/a | n/a | n/a | n/a | **IMMUNE** — prespecified (k, w) cell, Fisher exact, no max over cells |
| 2 | r3-lm-organism: routed **11/12** vs control 0/12, p = 9.6e-6; χ²(11) = 15213 | **1** | 12 seeds | n/a | n/a | n/a | n/a | **IMMUNE** — both primaries prespecified |
| 3 | r3-entangled-localization: axis_pair **25/30** vs axis_ica 13/30, p = 0.0028; rot_oracle 26/30, mix_oracle 28/30 | **1** | 30 seeds/arm | n/a | n/a | n/a | n/a | **IMMUNE** — all 10 arms prespecified, each comparison named in PREREG; no "best arm" claim is made (if one were, it would be k = 10) |
| 4 | r3-switch-tamper: flip at 0 steps **23/23**; h4 AUC 0.834 vs switch 0.840, p = 0.40 | **1** | 23 seeds | n/a | n/a | n/a | n/a | **IMMUNE** — 6 prespecified primaries; post-hocs Holm-corrected in a separate file. (Power, not selection, is this row's problem.) |
| 5 | r3-switch-mechanism: W = **0.2894** [0.2759, 0.3068]; truth tables 44/44 | **1** | 22–30 seeds | n/a | n/a | n/a | n/a | **IMMUNE** — prespecified gate + threshold |
| 6 | r3-switch-mechanism: patchwork **16/22** seeds, determinism 0.808 | **1** | 22 seeds | n/a | n/a | n/a | n/a | **IMMUNE** to selection, but **POST-HOC** discovery — needs its own prereg (REPORT3 §g item 6 agrees) |
| 7 | r3-selection-inflation-curve: P1 100/100, P2 bias 397/400, V2 p = 0.125 | **1** | 400 cells × 2000 reps / 8 layers | n/a | n/a | n/a | n/a | **IMMUNE** — the rule validating itself; criteria fixed in advance |
| 8 | r3-selection-inflation-curve: the two-sided constant it publishes | — | — | — | — | — | — | **CORRECTED here** (§1–2): b_4 = 1.4647, not 1.029 |

### Round 2 (REPORT2.md)

| # | Headline number | k | n | s | S | rho_hat | R | Verdict |
|---|---|---|---|---|---|---|---|---|
| 9 | r2-routing-switch-learned: learned_L1 **23/30** vs oracle 22/30 vs control 0/30; localization 30/30 | **1** | 30 seeds/arm | n/a | n/a | n/a | n/a | **IMMUNE** — prespecified arms. (Its *scope* was retracted by r3-entangled-localization; that is a validity issue, not a selection issue.) |
| 10 | r2-fragility-prereg: pooled d = **−0.642** [−0.918, −0.367], p = 2.54e-5 | **1** | 32 prompts × 4 layers | n/a | n/a | n/a | n/a | **IMMUNE** — preregistered fresh replication on never-selected layers. This is the portfolio's worked example of a selection-free number. |
| 11 | r2-topic-null-package: TestB topic share **0.639**, z = 3.79 | **1** | K = 10 000 perms | n/a | n/a | n/a | n/a | **IMMUNE** — prespecified null, both primaries |
| 12 | r2-lora-tamper-protocol: arm − baseline **−0.04 to −0.14**, all CIs below 0 | **1** | 6 seeds | n/a | n/a | n/a | n/a | **IMMUNE and SELECTION-AWARE** — the authors report the *worst-case* cell per arm rather than the best, which is the correct conservative direction. Model this row. |
| 13 | r2-commitment-meter: combined ρ = **0.570** [0.337, 0.740]; partial ρ = 0.106 | **1** | 60 seeds | n/a | n/a | n/a | n/a | **IMMUNE** — the "combined" meter was prespecified, not the better of two |
| 14 | r2-legible-value-readout: ERT words **0.055** vs 0.004, MW p = 0.0079 | **1** | 5+5 seeds | n/a | n/a | n/a | n/a | **IMMUNE** — prespecified; its problem is the failed variance-match, not selection |

### Round 1 (REPORT.md) — selection-exposed rows first

| # | Headline number | k | n | s | S | rho_hat | R | Verdict |
|---|---|---|---|---|---|---|---|---|
| 15 | misalignment-noise-probe (published, tested by the portfolio): L4/8/11-vs-L6 contrast **−2.72 nats** | **24** | 6 prompts/cell | 0.841 | 0.852 | **0.026** | **1.19** | **SUSPECT** — deflates to −0.07; fresh preregistered truth +0.22 [−0.72, +1.16]. Already replaced by row 10. |
| 16 | misalignment-noise-probe: **grand mean −0.798** over all 24 layers | **1** | 24 layers × 6 | n/a | n/a | n/a | n/a | **IMMUNE and replicates** — lands inside the fresh CI [−0.918, −0.367] with no correction. The sweep contained the right answer all along. |
| 17 | routing-lora-attack: **attacker-best −0.066** [−0.090, −0.042] (max-AUC LoRA cell per arm) | **6** per arm | 6 seeds | 0.0125 (routed) / 0.0084 (baseline) | 0.222 / 0.115 | **0.997 / 0.995** | **18.1 / 7.4** | **ROBUST** — the attacker grid is wildly over-dispersed relative to seed noise; selecting the best cell is not what makes this number. *(Audited here for the first time; not one of r3's three named targets.)* |
| 18 | routing-hybrid-knob: "resistance arrives between α 0.75 and 1.0" (post-hoc endpoint jump, exact p = 0.031) | **4** gaps, two-sided | 6 seeds | 0.0074 / 0.0283 | — | — | **3.61 / 3.73** (was 5.04 / 5.20) | **ROBUST**, margin reduced — see §2 |
| 19 | routing-hybrid-knob T3: "interior alphas are Pareto-useful" at p_waste 0.5 | **5** cells (3 interior) | 6 seeds | 0.00129 (interaction) | 0.00228 | **0.680** (not 0.000) | 2.49 cell / **4.52** paired contrast | **RE-WORD, do not call it unsupported** — see §4 |
| 20 | misgen-bimodality-honest: max dip over the 25-regime grid (**0.182**, cost 0.5 / width 2) | **25** | 100 seeds | 0.0086 | 0.0518 | **0.972** | **7.63** | **ROBUST** (deflation ≈ 3%) |
| 21 | misgen parent repo: "needs width 8 specifically" (width search at equal cost) | **5** | 100 seeds | 0.0105 | — | — | **1.75** | **BORDERLINE / near-noise — drop the claim.** And the R was computed for width 2, which is the actual max at equal cost. The published *replication* cell (dip p = 0.0015 on 100 fresh seeds) is separately **IMMUNE**. |
| 22 | misgen-connectivity-graph: max-over-θ modularity Q | **1770 / 1595 / 1711** | 60 seeds | n/a (permutation null used) | — | — | — | **FAMILY-DEPENDENT** — p = 0.20 / **0.028** / 0.11, see §3 |
| 23 | misgen-bimodality-honest: **9/25** Holm rejections; stats-hygiene-harness: **10/18** BH-FDR significant | 25 / 18 | 100 seeds / varies | — | — | — | — | **p-values corrected, EFFECT SIZES NOT.** Holm and BH correct the multiplicity of *tests*; a multiplicity-clean survey can still publish an inflated *magnitude* for the winner. Print R next to any winner's effect size. |
| 24 | routing-worldview-select: post-hoc **H = 16 → 30/30** | ≥ 2 widths | 30 seeds | not recov. | — | — | — | **SUSPECT as an effect size** — width chosen after seeing the width-4 misses. The prespecified 22/30 (row 27) is the number to quote. |

| # | Headline number (k = 1, IMMUNE) | n | Why immune |
|---|---|---|---|
| 25 | misgen-knee-threshold: knee = 0.086; one component under **every** principled threshold | 60 seeds | Pre-stated knee rule; the "every threshold" claim is *conjunctive over 4*, i.e. anti-selective |
| 26 | misgen-connectivity-graph: threshold-free rank **AUC = 0.725**, node-perm p ≤ 1e-4 | 60 seeds | No max over θ anywhere in the statistic |
| 27 | routing-worldview-select: **22/30** vs 0/30, Fisher p = 8.3e-10 | 30 seeds/arm | Prespecified width-4 configuration |
| 28 | misgen-removal-dose: **f₅₀ = 0.94%** [0.73, 1.19]; 9/40 vs 40/40, McNemar p = 9.3e-10 | 40 fresh seeds | Preregistered dose–response curve, no cell selected |
| 29 | misgen-hedger-mode: determinism **0.315 vs 0.868**, p = 8e-21; 0/55 patchwork | 55 + 34 seeds | Prespecified census |
| 30 | misgen-routing-unlearn: route 0.407 vs control 0.535, **p = 0.25** | 20/arm | Prespecified null (its problem is power, not selection) |
| 31 | misgen-noise-probe: P1 null p = 0.62; flip AUROC **0.71** vs clean 0.80 | 120 seeds | Primary prespecified. **Caveat:** the AUROC comparison is quoted at one σ and the report itself calls it "dose-fragile" — if a σ was chosen after the fact, k = #doses; not recoverable from the report text |
| 32 | routing-hybrid-knob T1/T2: exact one-sided Wilcoxon **p = 0.0156** both | 6 seeds | Preregistered endpoint tests (α = 0 vs α = 1) |
| 33 | routing-lora-attack: primary diff **−0.367** [−0.42, −0.31] | 6 seeds | Prespecified cell |
| 34 | unlearn-underdetermination: **0.807 vs 0.960**, exact p = 0.0078, 7/8 same map | 8 seeds | Prespecified |
| 35 | ert-capacity-curve: C(ρ) = 2 / 0.166 / 0.003 / **0** / 1.4e-4 / 0.003 | closed form + 600k MC/ρ | Analytic; no cell selection possible |
| 36 | ert-weak-paraphraser: all arms at chance, **p = 0.27**; exact leak 0.5185 | prespecified arms | Prespecified null |
| 37 | ert-noise-probe: **0 / 14 400** reveal-capable emissions | 14 400 forwards | Prespecified (and the conditional test was vacuous, reported as such) |
| 38 | ert-learned-classes: ARI **1.000**; distill M = 0.683 (prereg FAIL) | 3 seeds | Prespecified criterion, reported as failed |
| 39 | ert-value-probe: pooled **0.558 vs 0.139**, p = 0.0079 | 15 seeds | Primary prespecified (and reversed). **Caveat:** the "class-feat within-input 0.364, best in 5/5" readout was chosen among representations post-hoc → k = #representations, not recoverable |
| 40 | vineppo-probe-fixed: ΔMSE **+0.0045** [+0.0006, +0.0096]; control r = 0.750 | 23 problems | Prespecified; nested CV |
| 41 | vineppo-within-problem: sim r = **0.62** vs its 0.69 reliability ceiling | 23 problems | Prespecified |
| 42 | vineppo-rollout-budget: λ_pooled **1.15** [0.67, 2.06], λ_within **7.4** [3.6, 25.7] | 6/12 rollouts | Curve fits over the whole budget grid, no argmax |
| 43 | noiseprobe-capability-control: covariate ρ = −0.29, p = 0.17; rerun CIs all span 0 | 432 cells | Prespecified; it *tests* a k = 24-selected claim (row 15), which is the point |
| 44 | noiseprobe-prereg-replication: **p = 0.763**, contrast +0.22 [−0.72, +1.16] | 24 fresh prompts | Preregistered replication. Post-hoc pooled d = −1.33 is k = 1 but exploratory |
| 45 | noiseprobe-behavioral-judge: Δ **+0.5/100**, p = 0.47; flips 5 vs 1 | 18 saved pairs | Statistic prespecified. **Caveat:** the 18 pairs were themselves margin-flip-*selected* — the sample is selected even though the statistic is not |
| 46 | nla-within-doc-shuffle: null **+0.1815** vs observed +0.2120; residual +0.031, p = 1e-4 | 268 features | Prespecified permutation null |
| 47 | nla-hierarchical: ρ_within **0.43** vs topic null 0.16; topic share 0.715 | 540 same-doc pairs | Prespecified decomposition |
| 48 | moe-profile-stability: ρ = **0.864**, Spearman–Brown 0.91; Jaccard 0.369 / 0.292 | 100 splits | Split-half reliability, no cell selected. It *measures* the churn of a k-selected published artifact |
| 49 | moe-ert-bridge: AMI **0.495** [0.45, 0.54]; profile acc 0.55–0.61 vs 0.125 chance | 16 cells, all reported | All cells reported, no max |
| 50 | cnr-tightness: ρ_tight **1.25–1.31** at all 8 anchors; GCG ≥ 16× | 8 anchors | Range over all anchors reported, not the best |
| 51 | cnr-noise-cert: σ* median **0.088** IQR [0.077, 0.097]; 56/56 soundness cells | 512k draws | Median over anchors, no max |
| 52 | **r4-kway-learned-plus-hygiene** (this round): localization 20/20 both k; learned 20/20 vs control 0/20 vs broken 0/20 | 20 seeds/k/arm | Prespecified cell (w = 8), thresholds inherited, Fisher exact |

---

## 6. One sentence for the paper

> Every number in the switch line — 22/30, 23/30, 11/12, 20/20 at k = 3 and k = 4, and 20/20 with the
> data-only localizer — is a per-seed success count at a **single prespecified configuration**, tested
> by Fisher exact against a same-n floor. There is no maximum over cells anywhere in the line, so
> `k = 1` and no deflation applies. The portfolio's selection-exposed numbers are all outside it
> (rows 15–24), and each is printed with its own `k, n, s, S, ρ̂, R`.

## 7. Residual audit debt (rows where k > 1 and s/S are not recoverable without a re-run)

1. **misgen-noise-probe** (row 31) — was the reported σ for the flip-AUROC comparison chosen after
   seeing the doses? If yes, k = #doses. Recoverable from that repo's per-dose per-seed AUROCs.
2. **ert-value-probe** (row 39) — the "best in 5/5 seeds" class-feature readout was chosen among
   representations; k = #representations compared.
3. **routing-worldview-select** post-hoc H = 16 (row 24) — quote 22/30, not 30/30.
4. **misgen-connectivity-graph** chain gap +0.00114 (REPORT.md) — reported at the sweep max or at a
   prespecified θ? The report text does not disambiguate; if at the max, k = 1770 and it needs the
   same family-dependent treatment as §3.

None of these four touches the switch line.
