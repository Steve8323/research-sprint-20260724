# FINDINGS — Topic-matched null as a standard control (round 2)

**Headline.** The round-1 within-group ("topic-matched") permutation null is packaged
as a reusable ~200-line numpy module (topic_null.py) with a documented API and a
self-test (positive + negative controls). Applied as a *standard control* to two more
agreement claims from the local artifacts, it behaves as a null should: it deflates one
claim by ~64% (nla Test-B) yet leaves a real residual, and it fully clears another
(moe-ert profile->routing) — the load-bearing effect survives at ~78% of its above-chance
size.

All primary tests were preregistered in PREREG.md (stamped Fri Jul 24 23:11:36 EDT
2026, before any application run). The module + self-test were written first and are not
a result claim.

---

## The reusable module (topic_null.py)

One primitive — within_group_perm(labels): a permutation of item indices that only
moves items *within* their own group. Two public entry points:
- topic_null_effect(effect_fn, labels, K, seed, two_sided) — general; pass a scalar
  effect as a function of a within-group permutation.
- topic_null_pairsim(S_a, S_b, labels, K, seed, pair_mask) — Spearman rho between two
  paired similarity matrices over the same items.

Both return a TopicNullResult: observed, null_mean, null_sd, null_q025, null_q975, z,
p_one_sided = (1+#{null>=obs})/(K+1), group_explained_share = null_mean/observed,
residual = observed - null_mean. Pure numpy, no I/O, caller-supplied seed.

**Self-test (python3 topic_null.py), synthetic controls:**
- NEGATIVE control (agreement that is purely a group function; null must reproduce it):
  observed rho 0.999, null_mean 0.999, share 1.00, p = 0.029 (non-sig). PASS
- POSITIVE control (real within-group item-level agreement on top of topic; null must
  fire): observed rho 1.000, null_mean 0.163, share 0.16, p = 5e-4. PASS
The module correctly distinguishes "all topic" from "topic + real item signal."

---

## APPLICATION (a): moe-ert-bridge profile->routing under a CLASS-MATCHED null

**Question (preregistered).** Published claim: a plain-language expert profile
(meaning-class -> argmax expert) predicts held-out-synonym routing far above a *majority*
baseline. Is the *specific* class<->expert correspondence load-bearing, or would any
class->expert map with the same marginals do as well? New control: within each layer,
permute the class->expert map (24 classes = items, layer = group; the reusable
topic_null_effect), preserving class marginals / expert set / majority structure and
destroying only the correspondence.

**Re-run (reduced budget, preregistered):** MoE-ERT, 4 seeds, 400 steps, n_prof=n_eval=
2000 (published run used 800/3000). Task-M acc retained 0.987-0.998.

**Result — PRIMARY test passes; correspondence is load-bearing.**

| seed | obs acc (true map) | class-matched null mean [2.5%,97.5%] | majority baseline | chance | class-explained share | p (K=5000) | n |
|---|---|---|---|---|---|---|---|
| 0 | 0.640 | 0.136 [0.074, 0.207] | 0.191 | 0.125 | 0.212 | 2.0e-4 | 2025 |
| 1 | 0.588 | 0.123 [0.062, 0.193] | 0.117 | 0.125 | 0.209 | 2.0e-4 | 1988 |
| 2 | 0.617 | 0.140 [0.079, 0.207] | 0.190 | 0.125 | 0.227 | 2.0e-4 | 2149 |
| 3 | 0.566 | 0.141 [0.077, 0.213] | 0.180 | 0.125 | 0.249 | 2.0e-4 | 1860 |

Cross-seed: obs acc 0.603 (t-CI [0.551, 0.655]); class-matched null mean 0.135
([0.122, 0.148]); class-explained share 0.224 ([0.195, 0.253]); residual as a
fraction of above-chance signal 0.98 ([0.95, 1.01]). p<0.01 and obs>null in ALL 4
seeds; residual > half of above-chance in ALL 4 seeds. Decision (a): the class<->expert
correspondence is load-bearing (True).

Reading: only ~22% of the profile predictor's accuracy is reproducible by a
class-marginal-matched null; ~78% is the genuine class->expert correspondence. As a
sanity check, the class-matched null lands near the published majority baseline
(0.135 vs 0.170) — scrambling class labels degrades the predictor to roughly
majority-level, exactly as it should. Unlike nla Test-B, the class null here does NOT
explain the effect away: the round-1 moe-ert legibility claim survives cleanly.

*Post-hoc (labeled):* the class-matched null mean (0.135) exceeding chance (0.125) and
tracking the majority baseline is expected — a permuted map still routes whole classes to
whichever expert it lands on, catching the majority-expert mass. Reduced-budget
sensitivity: obs acc 0.603 here vs published 0.573 at 800/3000; conclusion unchanged.

## APPLICATION (b): nla Test-B total rho = 0.213 under the FULL topic null

**Question (preregistered).** Round-1 hierarchical partialled the Test-B correlation into
within-/between-doc pieces but never put the headline TOTAL rho through a topic-matched
null. Here: topic_null_pairsim(S_text, S_sae, labels=docs) over ALL 23,220 token pairs
(within-document permutation of the SAE codes), plus the published-style GLOBAL null for
comparison.

**Sanity (must match hierarchical before interpreting):** reconstructed total rho
0.19328 vs hierarchical 0.19328; baseline pair sim 0.23965 vs 0.2397. Exact match. (Both
reconstruct 0.193 from saved top-20 codes; the published 0.213 used full codes.)

**Result — both preregistered branches fire.**

| quantity | value |
|---|---|
| observed total rho | 0.1933 (n=23,220 pairs) |
| PRIMARY: full topic (within-doc) null | mean 0.1235, SD 0.0184, 95% [0.088, 0.159], K=10,000 |
| observed vs topic null | z = 3.79, one-sided MC p = 2.0e-4 |
| topic-explained share | 0.639 (64% of the rho is document/topic identity) |
| residual (topic-corrected) rho | 0.0697 |
| COMPARISON: global null (published-style shuffle) | mean 0.0007, SD 0.029, p = 1.0e-4 |

Reading (per the preregistered rule): (1) the observed rho significantly exceeds the
topic-matched null (p=2e-4), so a token-level component of the 0.213/0.193 headline
survives topic — residual rho ~ 0.07. (2) The topic null reproduces 64% of the total
rho; the published Test-B correlation is majority document/topic identity, and the
published global shuffle (null ~ 0.0007) could not have seen this. Honest restatement:
"NLA-description similarity and SAE-code similarity are Spearman-correlated 0.19-0.21, but
~64% of that is because both track the source document; a ~0.07 token-level component
survives topic matching."

This is milder than the Test-A lift (which the same null deflated ~86% in round 1):
Test-B's rank correlation is somewhat less topic-dominated than Test-A's share-a-feature
lift, but still majority-topic.

*Post-hoc (labeled):* same-doc-only rho 0.4301 under the topic null (mean 0.1617, share
0.376, residual 0.268) reproduces the hierarchical rho_within = 0.4301 exactly, and shows
that WITHIN a document token-level agreement is the majority component (only 38% topic).
Cross-doc-only rho 0.1546 (null mean 0.0813, share 0.53) — even across documents, half
the correlation is topic, because within-doc scrambling perturbs cross-doc similarities
through topically coherent tokens.

---

## Cross-application summary — the null as a standard control

| claim | headline effect | topic/class-explained share | residual survives? |
|---|---|---|---|
| nla Test-A lift (round 1) | +0.212 | 0.856 | yes, +0.031 (p=1e-4) |
| nla Test-B total rho (b) | 0.193 | 0.639 | yes, 0.070 (p=2e-4) |
| moe-ert profile->routing (a) | 0.603 acc | 0.224 | yes, ~0.47 (p=2e-4) |

The same control, one module, three claims: it deflates the two interpretability-tool
agreement claims that were topic-confounded (86% / 64%) and clears the one toy-legibility
claim whose grouping variable (meaning class) IS the intended signal (only 22% explained
by marginal matching). A null that always fired or never fired would be useless; this one
discriminates.

## Caveats
- (a) Toy scale (d=64, 4 layers, 8 experts, 24 synthetic classes); meaning classes are
  constructed. Reduced training budget (400 vs 800 steps) — preregistered, direction
  unchanged. The class-matched null conditions on the class marginal only; it cannot rule
  out that some correspondence is surface/context-driven (round-1 already showed ~1/3 of
  word-predictable routing is not class-explained).
- (b) Inherits round-1 caveats: top-20 truncation (reconstructs 0.193 not the published
  0.213), 216 tokens / 36 docs, one model / SAE / NLA / embedder. p=2e-4 is not the MC
  floor (K=10,000 floor is 1e-4). The within-doc null conditions on document identity
  only; it cannot separate the surviving 0.07 from shallow within-document structure.
- Module. The null is EXACT only when within-group item<->measurement pairing is
  exchangeable under H0; it conditions on group identity and cannot detect finer
  within-group confounds. Documented in the API contract.
- Runtime. (b) 194 s (dominated by MiniLM re-embed), (a) 255 s (dominated by 4-seed
  training), self-test <1 s. Total < 8 min, single process, CPU, no downloads.
