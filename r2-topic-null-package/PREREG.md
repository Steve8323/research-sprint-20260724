# PREREGISTRATION — Topic-matched null as a standard control (round 2)

Stamped at start (shell `date`):
```
Fri Jul 24 23:11:36 EDT 2026
```
Everything below this line was written BEFORE any application script was run.
(The reusable module `topic_null.py` and its self-test were written first; the
self-test is a synthetic positive+negative control of the primitive, not one of
the two applications, and is not a result claim.)

## Goal
Package the round-1 topic-matched (within-group) permutation null as a small,
documented, reusable module, and demonstrate generality by applying it to TWO
further agreement claims from the local artifacts, each with a preregistered
primary test.

## Module (already written, `topic_null.py`)
Primitive: `within_group_perm(labels)` — a permutation of item indices that only
moves items within their own group. Public API:
- `topic_null_effect(effect_fn, labels, K, seed, two_sided)` — general.
- `topic_null_pairsim(S_a, S_b, labels, K, seed, pair_mask)` — Spearman convenience.
Returns a `TopicNullResult` with observed, null_mean, null_sd, 95% null interval,
z, one-sided MC p = (1+#{null>=obs})/(K+1), group_explained_share = null_mean/observed,
residual = observed - null_mean. Self-test asserts share>0.7 & non-sig for a
topic-only synthetic agreement and share<that & p<0.01 for a real item-level one.

---

## APPLICATION (a): moe-ert-bridge profile->routing under a CLASS-MATCHED null

**Background.** moe-ert-bridge P2 showed a plain-language expert profile
(class -> argmax expert, built on the PROFILE split) predicts top-1 routing of
HELD-OUT-SYNONYM eval instances at pooled acc ~0.573 (per-seed 0.55-0.61) vs a
majority-expert baseline ~0.15 (McNemar p tiny). The published control is the
majority baseline. It does NOT ask whether the *specific* class->expert
correspondence is load-bearing: could any class->expert assignment with the same
marginals score similarly, given that routing is class-structured?

**Class-matched null (the new control).** Independently WITHIN EACH LAYER, permute
the profile's class->expert assignment map (shuffle which expert each of the 24
meaning classes is mapped to; this is `within_group_perm` with the 24 classes as
items and the layer as the group). Re-score pooled held-out-synonym accuracy under
each permuted map. This preserves (i) the eval-set class distribution, (ii) the set
of experts named by the profile at each layer, and (iii) the per-layer majority
structure; it destroys only the class<->expert correspondence. If the true map
predicts far above this null, the class abstraction — not marginal matching — is
what predicts routing.

**Data / re-run.** The saved results.json lacks per-instance routing, so I re-run
moe_ert with reduced budget: 4 seeds (0-3), 400 train steps (smoke-verified
task acc 0.998, profile 0.640 vs majority 0.190 at this budget), n_prof=n_eval=2000.
Deviation from published run (800 steps / 3000 seqs) recorded here, pre-run, for the
shared-machine budget. I collect, per seed, the held-out-synonym eval instances
(word, class, observed top-1 expert) pooled over the 4 layers, plus the same
profile map (`expert_of_class[layer,class]`) the published run used.

**PRIMARY TEST (a).** For each seed: observed pooled held-out accuracy of the TRUE
profile map vs K=5000 class-matched permutation maps (numpy default_rng, seed =
7000+seed). One-sided MC p=(1+#{null_acc>=obs_acc})/(K+1). Report per seed:
observed acc, null mean, null 95% interval, z, p, and
class_explained_share = null_mean/observed. Also report the class-permuted null
mean vs the published majority baseline (the null should sit at/above majority,
since it still uses class-conditional prediction with scrambled labels).

**DECISION (a), fixed in advance.**
- The class->expert correspondence is load-bearing iff observed acc exceeds the
  class-matched null in ALL 4 seeds with p<0.01 AND the residual (obs - null_mean)
  is a substantial fraction (>50%) of (obs - chance=1/8). A null-behaving outcome
  (share ~ 1, p non-sig) would mean the profile's apparent skill is explained by
  class-marginal structure alone and the "profiles predict routing" claim would need
  restating — a fully acceptable outcome.

## APPLICATION (b): nla Test-B total rho=0.213 under the FULL topic null

**Background.** Published Test-B: total Spearman rho(S_text, S_sae) over all
C(216,2) token pairs = 0.213 (reconstructs to 0.193 from the saved top-20 codes).
Round-1 hierarchical partialled this into within-doc (rho_within=0.430 on 540
same-doc pairs) and between-doc components, but never put the TOTAL 0.213 itself
through a topic-matched null. This does that.

**Full topic null.** Apply the within-document permutation to S_sae (items = 216
tokens, groups = 36 documents) and recompute the TOTAL rho over ALL i<j pairs.
This is `topic_null_pairsim(S_text, S_sae, labels=docs, pair_mask=None)`. The
permutation holds every document's token membership fixed (so the same-doc /
cross-doc pair structure is invariant) and only scrambles which SAE code attaches
to which token inside a document.

**PRIMARY TEST (b).** Observed total reconstructed rho vs K=10000 within-document
permutations (seed 0). One-sided MC p. Report observed rho, null mean, null 95%
interval, z, p, topic_explained_share = null_mean/observed, residual.

**Comparison null (b).** K=10000 GLOBAL permutations of S_sae (the analogue of the
published shuffle control), expected ~0 — to show the global null would have missed
the topic confound here too.

**DECISION (b), fixed in advance.**
- Both branches may fire, as in round 1: (1) if observed > topic null with p<0.01,
  a token-level component of the 0.213 survives topic; report the residual.
  (2) Report topic_explained_share; if the topic null reproduces most of 0.213,
  the headline correlation is largely document/topic identity and must be restated.

## Shared rigor
- Every number reported with n, seed, and either a null 95% interval or exact MC p.
- Nulls are acceptable outcomes and will be stated plainly.
- Runtime budget: whole thing <= ~15 min CPU, single process. The moe re-run is the
  only training; the two nulls are analysis-only over saved/collected arrays.
- Reconstruction sanity for (b): reconstructed total rho and baseline must match the
  hierarchical run's 0.193 / 0.2397 before interpreting.
- READ-ONLY on all existing dirs; writes only under r2-topic-null-package/.

## Anything not listed above is POST-HOC and will be labeled as such.
Post-hoc I anticipate (labeled): per-layer breakdown of (a); (b) restricted to
same-doc pairs as a cross-check against the hierarchical rho_within; sensitivity of
(a) to the reduced training budget.
