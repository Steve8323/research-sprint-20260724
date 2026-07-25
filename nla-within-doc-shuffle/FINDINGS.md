# Within-document shuffle control for the nla-vs-sae Test-A lift

## Question

The published headline of ~/nla_sae_compare (Test A) is: tokens sharing an SAE feature get
NLA descriptions +0.189 more similar than the 0.240 random-pair baseline, with a *global*
shuffle control ~= 0. But tokens sharing an SAE feature are often topically related (same or
related source documents), and NLA descriptions of topically related tokens are similar for
topic reasons alone. The global shuffle destroys topic structure too, so it cannot separate
"the SAE grouping predicts NLA text beyond topic" from "both tools just see the topic."

**Decisive control:** permute the NLA texts only WITHIN each source document. This preserves
every feature's document-composition exactly (all doc-level topic structure, including
relatedness between different documents) while destroying the token-level pairing between
activation and description. Whatever lift survives is agreement beyond document/topic identity.

## Pre-registered analysis plan (written before running)

Data: results_20260701_162505.json only (216 tokens = 36 docs x 6 positions; per-token doc
id, NLA text, top-20 SAE features). Analysis-only; no GPU, no model calls except re-embedding
the 216 saved NLA texts with the same locally-cached MiniLM embedder + pooling as run.py.

Reconstruction: qualifying features = SAE features appearing in >= 3 tokens' saved top-20
lists. KNOWN DEVIATION (flagged up front): the original used full active sets (~34
feats/token) -> 522 qualifying features; the JSON stores only each token's top-20 by
activation value -> 268 qualifying features here. Sanity gates before interpreting anything:
(a) reconstructed baseline pair similarity must match the published 0.2397 to ~1e-3;
(b) reconstructed observed mean lift must be positive and of the same order as +0.189.

**Primary test (one, stated in advance):** observed Test-A mean lift (mean over qualifying
features of within-feature mean pair similarity minus global baseline) vs K = 10,000
within-document permutations (independently permute the 6 texts inside each of the 36 docs;
numpy default_rng seed 0). One-sided Monte Carlo p = (1 + #{null >= observed}) / (K + 1).
Report: observed lift, null mean, null SD, z, p, and topic-explained share = null_mean /
observed. Alpha = 0.01. Interpretation rule fixed in advance:

- If observed clearly exceeds the null (p < 0.01), there is token-level NLA-SAE agreement
  beyond document/topic identity; report the residual lift (observed - null mean) as the
  topic-corrected effect.
- If the null mean reproduces most of the observed lift (regardless of p), the published
  +0.189 headline is mostly a topic/document-identity effect and should be re-stated as such.
  A null result here is a fully acceptable outcome.

**Comparison null:** K = 10,000 global permutations (replicates the published shuffled
control, expected ~= 0).

**Secondary test (stated in advance):** pooled cross-document lift = mean similarity of
within-feature CROSS-doc pairs minus mean similarity of all cross-doc pairs (a second
topic-robust view; the multiset of all cross-doc pair sims is invariant under within-doc
permutation, so the same null applies exactly). Same K, same p-value construction.

Anything not listed above is post-hoc and will be labelled as such.

## Results

Sanity gates passed: reconstructed baseline pair sim 0.239652 vs published 0.239652 (diff
5e-8 -- embedding pipeline is faithfully reproduced); reconstructed observed mean lift
+0.2120 vs published +0.1888 (positive, same order; larger because the saved top-20
truncation keeps only the strongest firings -> 268 qualifying features vs 522, 45,639
within-feature pairs).

| Quantity | Value |
|---|---|
| Observed Test-A mean lift (reconstructed) | **+0.2120** over baseline 0.2397 (268 features, 90.7% positive) |
| **PRIMARY: within-doc (topic-matched) null** | mean **+0.1815**, SD 0.0028, 95% null interval [0.1760, 0.1871], K=10,000, seed 0 |
| Observed vs within-doc null | z = 10.8, one-sided Monte Carlo p = 1.0e-4 (= floor 1/(K+1)) |
| **Topic-explained share** | **85.6%** of the observed lift is reproduced by the null |
| Residual (topic-corrected) lift | **+0.0305** (= +12.7% relative to baseline, vs +88% relative for the raw reconstructed headline) |
| Comparison: global null (published control) | mean +0.00002, SD 0.0038 -- replicates the published ~0 control |
| Secondary: pooled cross-doc lift | observed +0.0056 vs null mean +0.0029 (SD 0.0006), z = 4.3, p = 1.0e-4 |

**Reading (per the pre-registered rule -- both branches fire):**

1. The observed lift significantly exceeds the topic-matched null (p = 1e-4 < 0.01): there
   IS token-level NLA-SAE agreement beyond document/topic identity. It is small: +0.031,
   about 1/7 of the headline number.
2. The null reproduces 85.6% of the observed lift: the published headline ("+0.189, +79%
   relative, shuffled control ~0") is mostly document/topic identity. The global shuffle
   control was the wrong null; it destroys topic structure and therefore could not see this.
   The honest restatement is: "sharing an SAE feature predicts NLA-description similarity
   overwhelmingly because both tools see the document topic; a ~+0.03 token-level component
   survives topic matching."
3. Even in the cross-document secondary, about half the small pooled lift (+0.0029 of
   +0.0056) comes from features over-sampling topically related document PAIRS, which the
   within-doc null preserves; the remaining half is token-level.

Post-hoc descriptives (labelled, mechanism of the confound): same-doc text pairs average
similarity 0.632 vs cross-doc 0.230; single-doc features (90/268) have observed lift +0.442
vs +0.096 for multi-doc features.

## Caveats

- **Top-20 truncation.** The saved records keep only each token's top-20 SAE features
  (original used full active sets, ~34/token), so this analysis runs on 268 of the original
  522 qualifying features -- the stronger-firing subset (its raw lift is accordingly higher,
  +0.212 vs +0.189). The topic-explained share is computed on this subset; the exact
  published statistic cannot be reconstructed from the artifacts. Directionally the
  conclusion should transfer (weak-firing features are noisier, not less topic-bound), but
  that is an inference, not a measurement.
- Small scale inherited from the original: 216 tokens, 36 docs, 6 tokens/doc; one model,
  one SAE, one NLA checkpoint, one embedder (MiniLM) whose notion of text similarity all
  metrics inherit.
- p = 1.0e-4 is the Monte Carlo floor at K=10,000 (no null draw reached the observed value).
- The within-doc null conditions on document identity only; it cannot rule out that the
  surviving +0.03 reflects shallow within-document structure (e.g., token position or token
  identity) rather than fine-grained feature semantics.
- Test B (structure Spearman) could not be re-tested against this null: the full sparse SAE
  codes are not in the saved artifacts.
- Runtime 50 s on CPU; analysis-only, no GPU, no downloads (MiniLM was already cached).
