# topic_null — a topic-matched permutation null as a standard control

A ~200-line, dependency-light (`numpy` only) module that packages the round-1
within-group ("topic-matched") permutation null so any "two measurements agree on the
same items" claim can be checked against the confound that both measurements merely
track a shared grouping variable (document / topic / meaning-class / …).

## The idea in one sentence
Hold every item's group membership fixed and shuffle only the item↔measurement pairing
*inside* each group; whatever agreement survives is agreement beyond group identity.
A *global* shuffle destroys the group too, so it cannot tell "agree beyond topic" from
"both see the topic" — which is exactly how round-1 found 86% of a published NLA-SAE
lift was topic identity.

## API
```python
from topic_null import topic_null_effect, topic_null_pairsim

# General: supply any scalar effect as a function of a within-group permutation.
res = topic_null_effect(effect_fn, labels, K=10_000, seed=0)
#   effect_fn(perm) must return the real effect when perm == np.arange(n).

# Convenience: Spearman rho between two paired similarity matrices over the same items.
res = topic_null_pairsim(S_a, S_b, labels, K=10_000, seed=0, pair_mask=None)
```
`res` is a `TopicNullResult` (`.to_dict()` for JSON) with:
`observed, null_mean, null_sd, null_q025, null_q975, z, p_one_sided,`
`group_explained_share = null_mean/observed, residual = observed − null_mean,`
`K, n_items, n_groups, n_nonsingleton_groups`.

`p_one_sided = (1 + #{null ≥ observed}) / (K + 1)` (floor 1/(K+1)); pass `two_sided=True`
in `topic_null_effect` for a two-sided test about the null mean.

## Guarantees / limits (read before using)
- The null is EXACT for statistics whose within-group item↔measurement pairing is
  exchangeable under H0. It conditions on group identity ONLY; it cannot rule out
  finer within-group structure (item position, item identity). Documented, not a bug.
- Pure numpy, no global state, no I/O; you pass the seed.

## Self-test (run `python3 topic_null.py`)
- NEGATIVE control (null must reproduce it): a synthetic agreement that is purely a
  function of the group → `group_explained_share ≈ 1.00`, p non-significant.
- POSITIVE control (null must fire): a real within-group item-level agreement on top of
  topic → `group_explained_share ≈ 0.16`, p ≈ 5e-4.

## Two demonstrations (this package)
- `app_a_moe_classnull.py` — class-matched null for the moe-ert-bridge
  profile→routing claim (items = 24 meaning classes, group = layer).
- `app_b_nla_testB.py` — full topic null for the nla Test-B total Spearman
  rho = 0.213 (items = 216 tokens, group = 36 documents).

See `PREREG.md` (stamped before the runs) and `FINDINGS.md` for results.
```
Files: topic_null.py, app_a_moe_classnull.py, app_b_nla_testB.py,
       results_app_a.json, results_app_b.json, PREREG.md, FINDINGS.md, README.md
```
