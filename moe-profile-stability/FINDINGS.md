# Split-half reliability of MoE expert profiles (moe_legibility)

## Question
Are the per-expert statistics behind the plain-language expert profiles in
`~/moe_legibility` (specialization score `spec[l,e]`, dominant category
`dom[l,e]`, distinctive-token lists, and the top-16 profiled-expert selection)
stable when recomputed on independent halves of the corpus? A profile that is
not stable across halves is not a real characterization.

## Pre-registered method (written BEFORE running the analysis)

Data: the saved `moe_data.npz` contains only corpus-aggregated counts, so a
split cannot be done from it alone. We re-capture routing with the exact
corpus, hook, and model from `moe_route.py` (OLMoE-1B-7B-0125-Instruct-4bit,
already on disk; deterministic forward passes, no sampling), but keep it
**per text** (64 texts, 8 per category, 1,541 tokens). Sanity gate: the
per-text capture aggregated over all texts must reproduce the published
`counts` array (we report the fraction of exactly-matching entries; if < 100%
we report the discrepancy and proceed, since both halves come from the same
capture and the split-half comparison is internally consistent either way).

Split unit: the **text** (stratified: 4 of 8 texts per category to half A,
4 to half B), because tokens within a text share context and a token-level
split would leak correlated information between halves. A token-level split is
also run as a secondary, labeled comparison.

All per-half statistics use the identical formulas from `moe_route.py`
(prop -> normalized -> entropy -> spec; dom = argmax prop; distinctive tokens
by lift with count >= 2 and non-whitespace filter, top 8) and the identical
top-16 selection rule from `moe_profile.py` (rank by spec among units with
>= 3 distinctive tokens).

Random splits: seed 0, numpy default_rng. R = 200 splits for the vectorized
statistics (spec, dom, prop); R = 50 splits for the counter-based statistics
(distinctive tokens, top-16 selection), for runtime.

### Primary endpoint (pre-stated)
Mean across R = 200 stratified text-level split-halves of the **Spearman
correlation of spec[l,e] between halves over all 1024 (layer, expert) units**,
with 95% percentile interval over splits. Units with zero selections in either
half are excluded per split (count reported). Interpretation bands stated in
advance: rho >= 0.7 good reliability, 0.5-0.7 moderate, < 0.5 poor.

### Secondary endpoints (pre-stated)
1. Pearson r of spec across halves + Spearman-Brown corrected full-length
   reliability 2r/(1+r).
2. Dominant-category agreement between halves: over all units, and over the
   16 profiled units from `moe_profiles.json` (fixed target set, not re-selected).
3. Top-16 selection stability: Jaccard overlap of the per-half top-16 candidate
   lists (moe_profile.py rule), and how many of the published 16 profiled units
   appear in each half's top-16.
4. Distinctive-token stability for the 16 profiled units: Jaccard overlap of
   the per-half top-8 distinctive-token lists.
5. Per-category peak routing propensity (paper headline "0.88-0.99"): per
   category, max over (l,e) of prop in each half, and whether the argmax unit
   agrees across halves.
6. (Labeled comparison) token-level split Spearman rho of spec, to quantify
   context-leak inflation vs the text-level split.

Everything below the RESULTS line was written after running the pre-stated
analysis. Anything post-hoc is labeled as such.

---

## RESULTS

Sanity gate: PASSED exactly. The per-text re-capture aggregated over all 64
texts reproduces the published `moe_data.npz` counts (100% of 16x64x8 entries
exact), `spec` (allclose), and `moe_tokens.json` (1024/1024 distinctive-token
lists identical). So the split-half is computed on exactly the pipeline that
produced the published profiles. n = 64 texts, 1,541 tokens, 1024 units,
seed 0; runtime ~21 s total.

### Primary endpoint
Text-level split-half **Spearman rho of spec = 0.864** [0.837, 0.885]
(mean [2.5%, 97.5%] over R = 200 splits; all 1024 units usable in every split,
none excluded). By the pre-stated bands this is **good reliability** for the
continuous specialization score. Pearson r = 0.835 [0.801, 0.861];
Spearman-Brown full-length reliability = 0.910 [0.890, 0.925].

### Secondary endpoints
1. **Dominant category, all units: 58.3%** agreement [53.6%, 61.9%] - largely
   arbitrary for the near-uniform majority of experts.
   **Dominant category, the 16 published profiled units: 86.5%** [68.8%, 100%].
2. **Top-16 profile-selection stability is poor: Jaccard 0.369** [0.231, 0.508]
   (R = 50). On average only **9.6 of the published 16** profiled units
   [8.0, 10.5] appear in a half's top-16.
3. **Distinctive-token lists are unstable: mean top-8 Jaccard 0.292**
   [0.198, 0.363] across the 16 profiled units (R = 50; 0 both-empty cases).
   The "evidence tokens" shown to the labeller substantially churn with the
   corpus half.
4. **Peak routing propensity per category: values highly stable**
   (per-half peaks 0.88-0.99 across all 8 categories, e.g. python 0.994
   [0.980, 1.000], english 0.879 [0.820, 0.935]) - the paper's "0.88-0.99"
   headline range replicates in every half. But the **identity** of the peak
   expert agrees across halves for only **42.3%** of categories
   [24.7%, 62.5%]: several experts sit near ceiling, so "the dedicated expert
   for category X" is not a stable pointer even though "a dedicated expert
   exists" is.
5. Token-level split (labeled comparison): rho = 0.891 [0.871, 0.907] - only
   +0.027 above the text-level split, so context leakage inflates the naive
   token split modestly here.

### Interpretation
The continuous statistics survive the split: spec is reliable (rho 0.86,
SB 0.91), the profiled units' dominant categories mostly agree (86%), and
"every category has a >= 0.88-propensity expert" holds in every half. The
discrete artifacts do not: which experts make the profiled top-16 (Jaccard
0.37) and which tokens serve as their evidence (Jaccard 0.29) depend heavily
on the corpus half. The published profile *list* and *evidence tokens* should
be read as one sample from a noisy selection process, not a stable catalog.

## Caveats
- Split halves have ~770 tokens each; instability at half size upper-bounds
  but does not equal full-corpus (1,541-token) stability. The SB correction
  applies only to the correlation endpoint, not to the discrete set overlaps.
- The corpus is tiny and author-constructed (8 texts/category); text-level
  splits are exchangeable only within this corpus. Nothing here tests
  generalization to new categories or natural text.
- Distinctive-token instability partly reflects the c >= 2 count threshold
  interacting with small halves; this is a property of the published pipeline
  at half size, which is exactly what split-half reliability measures.
- Jaccard/agreement CIs are percentile intervals over random splits (shared
  data), not independent-sample CIs.
- The plain-language description step (llama3.2:3b) was NOT re-run
  (analysis-only); token-list churn is a lower bound on description churn.

## Files
- capture_routing.py - per-text routing re-capture (OLMoE, on-disk model)
- routing_per_text.pkl - 64 texts x 16 layers x top-8 routing
- analyze_stability.py - pre-registered analysis (seed 0)
- results.json - all numbers
