# Corrected significance table for the six-project portfolio

**Question.** Which published headline numbers in ~/vineppo_replication, ~/misalignment_noise_probe,
~/moral_framework_misgen, ~/nla_sae_compare, ~/english_bottleneck_lm, ~/gradient_routing_experiment
survive statistically hygienic re-analysis (bootstrap CIs, exact tests, cluster-aware resampling,
multiplicity correction) computed from the saved artifacts alone? Analysis-only; no new model runs.

## Pre-registration (written BEFORE running any test; deviations are labeled post-hoc below)

Global decision rule: a table row is called "significant" iff its primary p-value survives
Benjamini-Hochberg FDR at q < 0.05 across ALL primary p-values in the table. Holm-corrected
p within each project family is also reported. Rows with no defensible null hypothesis get a
CI only and are labeled descriptive. All bootstrap CIs are 95% percentile; cluster bootstrap
resamples the stated cluster unit. RNG seed 0 for all resampling.

### Family V — vineppo_replication (cluster unit = problem, 23 clusters, 69 prefixes)
- V1 pooled Pearson r(v_sim, v_gold) (pub 0.878): cluster bootstrap CI (B=4000). Descriptive
  (circular by construction per the README); no p.
- V2 within-problem r, both variables problem-mean-centered (pub ~0.62): cluster bootstrap CI +
  permutation p (permute v_sim within each problem, 10000 perms, two-sided).
- V3 judge r(v_verb, v_gold) pooled (pub -0.23): cluster bootstrap CI + problem-block permutation
  p (shuffle which problem's v_verb block aligns with which gold block; 10000, two-sided).
- V4 probe (pub 0.06) [RE-ANALYSIS, not the published split]: grouped-5-fold (by problem)
  out-of-fold Ridge(alpha=10) predictions from the saved 1536-dim hidden states; pooled r of OOF
  pred vs gold; cluster bootstrap CI + problem-block permutation p as in V3.
- V5 branch ranking, gold gap >= 0.15 (pub 0.83, n=6): recompute wins/ties from rank.jsonl;
  exact binomial two-sided vs 0.5 on wins among non-tied pairs.

### Family N — misalignment_noise_probe
- N1 judge misaligned-rate (pub noise+SFT 0.533 vs SFT 0.167 vs base 0.0; n=60/arm, counts =
  rate*60 rounded): primary Fisher exact two-sided noise+SFT vs SFT; secondary EM vs base.
  Clopper-Pearson CI per arm.
- N2 layer differentials L4/L8/L11 (pub -2.8/-2.6/-2.3 nats, post-hoc selected from 24 layers):
  only cell means saved, so the only replicate axis is sigma in {0.1,0.2,0.4}. Primary =
  per-layer sign test across the 3 sigma replicates, Holm x 24. Pre-stated expectation: cannot
  reach significance (min possible two-sided sign p = 0.25); purpose is an honest evidence ceiling.
  Also report each selected layer's rank among the 24 differentials at sigma=0.2.
- N3 battery flips (pub 11/25 Qwen3B, 14/25 Llama3B): Clopper-Pearson CI; descriptive (null flip
  rate under "no effect" is undefined).
- N4 axes x scale flip fractions (pub deception 0.75-0.89, safety 0/12, power 5/5 at 8B):
  Clopper-Pearson CIs from n_flip/n_clean in axes_scale_results_FULL.json; descriptive.
- N5 concealer (pub noise margin swing +10.9 vs base +3.8 = max over 24 layers x 2 sigma):
  primary Wilcoxon signed-rank two-sided over the 24 paired per-layer noise-induced shifts
  (trained minus base) at sigma=0.2; effect = median paired difference. Cells are means over
  5 prompts x 3 seeds; layers are not independent units — labeled approximate. The published
  +10.9 is additionally flagged as a max-of-48-cells (winner's curse) figure.
- N6 reveal rate (pub <= 0.02): pooled reveal count at largest sigma from reveal_rate.json;
  Clopper-Pearson CI; descriptive.

### Family M — moral_framework_misgen
- M1 seed multiplicity (pub 31/120 aligned, 34/120 misaligned, 55/120 hedgers): Clopper-Pearson
  CIs. Recompute Hartigan dip on the 120 scores; label post-selection (the regime was searched
  for bimodality; no valid correction exists — reported for transparency, excluded from BH family).
- M2 barrier geometry (pub across-worldview 0.064 vs within 0.029, MW p ~ 2e-15 treating 1770
  pairs as independent): primary = node-label permutation test (permute aligned/misaligned labels
  over the group-member models among the 60, hedgers excluded; statistic = mean across-pair minus
  mean within-pair barrier; 10000 perms, two-sided). Dependence-correct replacement.
- M3 mech cue-reliance correlation (pub r=0.83, n=60 independent seeds): percentile bootstrap CI
  (B=10000) + permutation p (10000, two-sided).
- M4 prevention (pub f=0.5% -> 90% aligned, 30 seeds): Fisher exact two-sided 27/30 (f=0.005)
  vs 8/30 (f=0); Clopper-Pearson CI for 27/30 (pub CI [0.8,1.0] is narrower than exact).
- M5 persistence (pub 32/34 persist): Clopper-Pearson CI; descriptive.

### Family S — nla_sae_compare (cluster unit = document, 36 docs, 216 tokens)
Reconstruction caveat (pre-stated): saved records keep only the top-20 of ~34 active SAE features
per token (values rounded to 3 dp), so feature supports and the SAE similarity matrix are
approximate rebuilds; NLA-text similarity is recomputed exactly with the same cached MiniLM
pipeline (CPU). Reproduction of published point estimates is checked first and reported.
- S1 metric A mean lift (pub +0.189 over 0.240 baseline): doc-cluster bootstrap CI (B=1000) +
  permutation p (500 permutations of text embeddings across tokens, two-sided).
- S2 metric B Spearman rho (pub 0.213): doc-cluster bootstrap CI + Mantel permutation p (500).
- S3 frac positive features (pub 0.879 of 522): doc-cluster bootstrap CI (a naive binomial over
  522 features is invalid — features share tokens).
- S4 [POST-HOC secondary, stated in advance as exploratory]: cross-document-only lift
  (within-feature pairs restricted to tokens from different docs) to probe the topic/doc confound.

### Family E — english_bottleneck_lm
- E1 dissociation at rho=1 (pub M=0.996, S=0.247; chance M 0.042, S 0.009): eval n = 3x2000 =
  6000 fresh samples per number; counts = round(acc*6000). Primary two-sided Fisher exact M-count
  vs S-count; plus 3-seed replication range from e1_seeds.out. Expected: trivially significant.
- E2 covert channel at rho=1 (pub optimal 0.496 n=4000, learned 0.518 n=10000, both "~chance"):
  primary exact binomial two-sided vs 0.5 for the LEARNED reader (the stronger adversary);
  secondary same for optimal reader. Pre-stated: if learned p < 0.05, the published "closes the
  covert channel" claim is overstated (residual capacity > 0 for this trained pair). Caveat:
  n counts eval draws for one trained sender/receiver; training-run variability not captured.

### Family G — gradient_routing_experiment (unit = seed; tiny n by design)
- G1 deletion vs control (random partition, mislabel 0; pub forget 0.0 vs 0.93): exact Wilcoxon
  rank-sum two-sided, 3 vs 3 seeds (min possible p = 0.1).
- G2 ablation-aware retain (pub retain 0.95 vs naive routing 0.20): exact rank-sum two-sided,
  dedup fix seeds (n=3) vs naive routing seeds (n=3).
- G3 capacity at p_waste=0.9 (pub retain 0.90 random vs 0.80 structural): paired-by-seed exact
  sign-flip permutation, two-sided, n=3 pairs (min possible p = 0.25).
- G4 multicap independence (pub 0.83 vs 0.60, "4/4 seeds"): per-seed composite
  indep = 0.5*(dropA+dropB) - 0.25*(|spillA_B|+|spillA_R|+|spillB_A|+|spillB_R|); check it
  reproduces the published 0.828/0.602; paired exact sign-flip permutation, two-sided, n=4
  (min possible p = 0.125). Pre-stated: "4/4 seeds" CANNOT be significant at n=4.
- G5 frontier (2 seeds) and tamper/relearning-AUC claims: no adequate saved artifacts
  (results_frontier.json has n=2/cell; no tamper/AUC json exists) -> listed as not testable.

Results below this line were written AFTER running the pre-registered analyses.

---

## Results (written after running; full machine-readable table in results.json)

Runtime 21 s CPU. 27 table rows; 18 primary p-values; **10 survive BH FDR q < 0.05**.

### Corrected significance table (primary rows)

| id | claim (published) | corrected | p (primary test) | q (BH) | verdict |
|---|---|---|---|---|---|
| V1 | pooled sim-vs-gold r=0.878 | r=0.878, cluster-boot CI [0.78, 0.94] | — | — | descriptive; circular by construction |
| V2 | within-problem r~0.62 | r=0.621, CI [0.24, 0.85] | 0.0018 (within-problem perm) | 0.0036 | **SIG** — the repo's key number survives cluster-aware testing |
| V3 | judge r=-0.23 | r=-0.225, CI [-0.50, 0.17] | 0.29 | 0.33 | null, as the README already said |
| V4 | probe r=0.06 | OOF grouped-5-fold r=-0.045, CI [-0.26, 0.23] | 0.76 | 0.81 | null; "underpowered, not chance" stands |
| V5 | branch ranking 0.83 (n=6) | 4/4 non-tied wins (2 ties), CI [0.40, 1.0] | 0.125 (exact binomial) | 0.16 | ns — "anecdotal" confirmed |
| N1 | judge decoupling: noise+SFT 0.533 vs SFT 0.167 misaligned-rate | 32/60 vs 10/60 | 4.4e-5 (Fisher exact) | 2e-4 | **SIG** — the repo's centerpiece negative result is solid |
| N2 | L4/L8/L11 differentials | ranks 1/2/3 of 24 at sigma=0.2, but sign test across 3 sigmas Holm x 24 | 1.0 | 1.0 | ns — README's own "does not survive correction" is confirmed exactly |
| N5 | concealer swing +10.9 (max of 48 cells) | median paired per-layer shift diff +29.4 nats | 1.2e-7 (Wilcoxon over 24 layers, approximate) | 1e-6 | **SIG*** — direction robust, but units are cell means (pseudo-replication) and LoRA-fragility confound stands |
| M2 | barrier across 0.064 vs within 0.029, MW p~2e-15 | gap 0.035, node-label permutation | 1e-4 (perm floor 1/10001) | 3e-4 | **SIG** — survives the dependence correction, but honest p is ~1e-4, not 2e-15 |
| M3 | cue-reliance r=0.83 | r=0.827, CI [0.77, 0.93] | 1e-4 (perm floor) | 3e-4 | **SIG** |
| M4 | prevention 90% aligned (CI [0.8,1.0]) | 27/30 vs 8/30 at f=0; exact CI [0.735, 0.979] | 9.5e-7 (Fisher) | 6e-6 | **SIG**; published CI was anti-conservative |
| S1 | NLA-SAE lift +0.189 | +0.212 (top-20 rebuild), doc-cluster CI [0.19, 0.24] | 0.002 (perm floor 1/501) | 0.0036 | **SIG** |
| S2 | structure rho=0.213 | rho=0.193 (rebuild), doc-cluster CI [0.16, 0.29] | 0.002 (Mantel floor) | 0.0036 | **SIG** |
| E1 | M=0.996 vs S=0.247 at rho=1 | 5976/6000 vs 1485/6000; 3-seed ranges tight | <1e-300 (Fisher) | ~0 | **SIG** |
| E2 | covert channel "closed to chance" (learned 0.518) | 5177/10000, CI [0.508, 0.528] | 4.2e-4 (exact binomial vs 0.5) | 0.0011 | **SIG in the WRONG direction for the claim**: the learned reader is significantly ABOVE chance — "closes the covert channel" is overstated; residual capacity ~1.8 pp (optimal reader 1983/4000, p=0.60, is at chance) |
| G1 | forget 0.0 vs control 0.93 | 0.000 vs 0.930 | 0.10 (= exact-test floor, 3v3) | 0.15 | ns — effect huge but n=3/arm caps the exact p at 0.10 |
| G2 | fix retain 0.95 vs naive 0.20 | 0.950 vs 0.198 | 0.10 (floor) | 0.15 | ns (same floor) |
| G3 | capacity 0.90 vs 0.80 | paired diffs all positive [0.055, 0.105, 0.149] | 0.25 (floor, n=3 pairs) | 0.30 | ns |
| G4 | independence 0.83 vs 0.60, "4/4 seeds" | composite reproduces exactly (0.828/0.602), 4/4 wins | 0.125 (floor, n=4 pairs) | 0.16 | ns — "4/4 seeds" cannot reach significance at n=4 |

### Descriptive rows (no defensible null)
- N3 battery flips: 11/25 CI [0.24, 0.65]; 14/25 CI [0.35, 0.76] — CIs span ~half the unit interval.
- N4 axes x scale: deception cells are 3/4 and similar (CI [0.19, 0.99]); safety 0/12 CI [0, 0.26]; power 5/5 at 8B CI [0.48, 1.0]. All far too small-n to quote as rates.
- N6 reveal rate: 0/60, CI [0, 0.06] — clean negative.
- M1 multiplicity: aligned 31/120 CI [0.18, 0.35]; misaligned 34/120 CI [0.20, 0.37]; dip p=0.0156 reproduces but is post-selection (regime searched for bimodality) — excluded from the BH family.
- M5 persistence: 32/34 CI [0.80, 0.99] (README already reframes this as drift-dependent).
- S3 frac-positive features: 0.907 of 268 rebuilt features, doc-cluster CI [0.79, 0.88] (bootstrap is biased slightly low vs the point estimate; the naive "88% of 522" binomial is invalid since features share tokens).
- S4 [post-hoc]: cross-document-only lift is +0.050 vs +0.212 all-pairs (frac_pos 0.73) — **~3/4 of the headline NLA-SAE agreement lift is same-document (topic identity) signal**, quantifying the confound the README suspected.
- G5: frontier (n=2/cell) and ALL tamper/relearning-AUC claims have no saved per-seed artifacts — not testable.

### Headline synthesis
The portfolio's centerpiece claims mostly survive hygienic re-testing (10/18 primary tests
significant at BH q<0.05), and the projects' own self-flagged weaknesses are confirmed
quantitatively (N2 layer picks: Holm p=1.0; V5 ranking: p=0.125; V3/V4 nulls). Three genuine
corrections emerged: (1) english_bottleneck's "covert channel closed to chance" is contradicted
by its own saved numbers — the learned collusion reader is significantly above chance
(0.5177, p=4.2e-4); (2) every gradient-routing random-vs-structural headline sits at or below
the exact-test floor for its seed count (p in 0.10-0.25) — the effects are large and consistent
but the saved n cannot establish significance, and "4/4 seeds" is structurally incapable of it;
(3) the NLA-SAE agreement lift shrinks ~4x (0.212 -> 0.050) when restricted to cross-document
pairs, so the headline number is mostly topic identity. The misgen barrier geometry survives a
dependence-correct permutation test (p~1e-4) but the published pair-level p~2e-15 overstated the
evidence by ~11 orders of magnitude.

## Caveats
- This whole audit is itself post-hoc relative to the published numbers; within the audit, tests
  were pre-stated above before running, and the one exploratory row (S4) is labeled.
- Judge counts (N1) reconstructed from rounded rates x 60; exact to +-0.5 counts.
- NLA-SAE rebuild uses only the saved top-20 of ~34 active features per token (268 qualifying
  features vs published 522; lift 0.212 vs 0.189; rho 0.193 vs 0.213) — close but approximate.
- N5 units are per-layer cell means (5 prompts x 3 seeds averaged) — pseudo-replication; the
  Wilcoxon there is labeled approximate and the LoRA-fragility confound is not resolved.
- E2's n counts eval draws for ONE trained sender/receiver pair; training-run variability is
  not captured (a different training run might land at chance).
- Permutation p-values are floored at 1/(B+1): 1e-4 (10k) or 2e-3 (500).
- Some published claims (tamper resistance, relearning AUC, frontier) have no saved per-item
  artifacts at all and remain unverified.

## Files
- stats_toolkit.py — reusable toolkit (bootstrap/cluster-bootstrap CIs, exact binomial/Fisher/
  rank-sum/sign-flip tests, permutation helper, Holm, BH-FDR)
- run_all.py — loads all six projects' artifacts, runs the pre-registered tests, writes results.json
- results.json — full machine-readable corrected table (27 rows)

## Post-hoc addendum (after inspecting N5)
Decomposing the N5 shifts: at sigma=0.2 the trained (concealer) margins move +16.5 (median) from
a clean value of -15.8 while the base margins move -13.1 from +12.9 — i.e., BOTH models' margins
regress toward zero under noise, from opposite signs. The significant Wilcoxon (p=1.2e-7) is
therefore consistent with sign-symmetric margin shrinkage — exactly the text-degradation confound
the repo's own judge experiment exposed — rather than a specific concealment reveal. The median
noised trained margin lands near 0 (+0.7); only the winner's-curse max cell (+10.9) crosses well
past zero. N5's "SIG" should be read as "margins move as noise degrades margins", not as
confirmation of the concealer story. This weakens N5 from "SIG*" to effectively uninformative
about concealment.
