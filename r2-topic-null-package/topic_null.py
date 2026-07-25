"""
topic_null : a topic-matched (group-matched) permutation null as a reusable control.

--------------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------------
Many "two independent measurements agree on the same items" claims are inflated by
a shared *grouping* variable (document, topic, meaning-class, ...). Two tools can
look like they agree token-by-token when in fact they both merely track the group
each token belongs to. A *global* permutation null (shuffle everything) destroys the
group structure too, so it cannot separate "agreement beyond the group" from "both
tools see the group". The correct control is a **within-group permutation**: it holds
the group composition of every item fixed and only scrambles the item<->item pairing
*inside* each group. Whatever effect survives is agreement beyond group identity.

Round-1 result that motivated packaging this: on the published NLA-vs-SAE Test-A lift,
a within-document null reproduced 85.6% of the headline; only ~1/7 survived.

--------------------------------------------------------------------------------
THE ONE PRIMITIVE
--------------------------------------------------------------------------------
`within_group_perm(labels, rng)` -> a permutation of item indices that only moves
items within their own group (identity for singletons). Everything else is built on
this. Two public entry points wrap it for the two shapes agreement claims come in:

  1) topic_null_effect(effect_fn, labels, ...)     -- GENERAL: you supply a scalar
     effect statistic as a function of a permutation of one modality's item labels.
     Returns observed effect, null distribution, MC p, null mean/SD, z, and the
     "group-explained share" = null_mean / observed.

  2) topic_null_pairsim(S_a, S_b, labels, ...)     -- CONVENIENCE for the common case
     where the effect is a rank correlation (Spearman) between two paired similarity
     matrices S_a, S_b over the same items. Permutes one matrix's items within group.

Both return a `TopicNullResult` dataclass and can emit a plain dict via .to_dict().

--------------------------------------------------------------------------------
API CONTRACT (read this before use)
--------------------------------------------------------------------------------
- `labels`     : length-n integer/hashable group id per item (the topic/doc/class).
- The null is EXACT for statistics that are invariant to relabeling *within* a group
  under H0 (i.e. the item<->measurement pairing is exchangeable inside a group). It
  conditions on group identity only; it cannot rule out finer within-group structure
  (position, item identity) -- that is a documented limitation, not a bug.
- p is one-sided by default (H1: observed > null), Monte-Carlo:
  p = (1 + #{null >= observed}) / (K + 1). Floor is 1/(K+1).
- No global state; pass your own numpy Generator for reproducibility.
- Pure numpy. No torch, no sklearn, no I/O.

--------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# core primitive
# --------------------------------------------------------------------------- #
def group_index(labels: Sequence) -> list[np.ndarray]:
    """Return a list of index arrays, one per distinct group label."""
    labels = np.asarray(labels)
    return [np.where(labels == g)[0] for g in np.unique(labels)]


def within_group_perm(groups: list[np.ndarray], rng: np.random.Generator,
                      n: int) -> np.ndarray:
    """A permutation of 0..n-1 that only permutes indices inside each group.

    Singletons are fixed. This is the topic-matched shuffle: it preserves every
    group's membership exactly and only scrambles the item<->measurement pairing
    within a group.
    """
    perm = np.arange(n)
    for g in groups:
        if len(g) > 1:
            perm[g] = g[rng.permutation(len(g))]
    return perm


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class TopicNullResult:
    observed: float
    null_mean: float
    null_sd: float
    null_q025: float
    null_q975: float
    z: float
    p_one_sided: float
    group_explained_share: float   # null_mean / observed  (fraction reproduced by topic)
    residual: float                # observed - null_mean  (topic-corrected effect)
    K: int
    n_items: int
    n_groups: int
    n_nonsingleton_groups: int

    def to_dict(self) -> dict:
        return asdict(self)


def _summarize(observed: float, null: np.ndarray, K: int, n_items: int,
               n_groups: int, n_nonsingleton: int,
               two_sided: bool = False) -> TopicNullResult:
    null = np.asarray(null, dtype=np.float64)
    nm, ns = float(null.mean()), float(null.std(ddof=1))
    if two_sided:
        p = (1 + int((np.abs(null - nm) >= abs(observed - nm)).sum())) / (K + 1)
    else:
        p = (1 + int((null >= observed).sum())) / (K + 1)
    z = (observed - nm) / ns if ns > 0 else float("inf") * np.sign(observed - nm)
    share = nm / observed if observed != 0 else float("nan")
    return TopicNullResult(
        observed=float(observed), null_mean=nm, null_sd=ns,
        null_q025=float(np.quantile(null, 0.025)),
        null_q975=float(np.quantile(null, 0.975)),
        z=float(z), p_one_sided=float(p),
        group_explained_share=float(share),
        residual=float(observed - nm),
        K=K, n_items=n_items, n_groups=n_groups,
        n_nonsingleton_groups=n_nonsingleton,
    )


# --------------------------------------------------------------------------- #
# GENERAL entry point
# --------------------------------------------------------------------------- #
def topic_null_effect(effect_fn: Callable[[np.ndarray], float],
                      labels: Sequence,
                      K: int = 10_000,
                      rng: np.random.Generator | None = None,
                      seed: int = 0,
                      two_sided: bool = False,
                      return_null: bool = False):
    """Group-matched permutation null for an arbitrary scalar effect.

    Parameters
    ----------
    effect_fn : callable(perm: np.ndarray[int]) -> float
        Your statistic evaluated when one modality's items are reassigned by `perm`.
        Called with the IDENTITY permutation to get the observed value, so
        effect_fn(np.arange(n)) MUST return the real, unpermuted effect.
    labels : length-n group ids (topic / document / class).
    K : number of within-group permutations.
    rng / seed : reproducibility (rng wins if both given).
    two_sided : if True, p is two-sided about the null mean.
    return_null : if True, also return the raw null array (for plotting).

    Returns
    -------
    TopicNullResult   (and the null array if return_null).
    """
    rng = rng or np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    groups = group_index(labels)
    n_nonsingleton = sum(len(g) > 1 for g in groups)

    observed = float(effect_fn(np.arange(n)))
    null = np.empty(K)
    for k in range(K):
        null[k] = effect_fn(within_group_perm(groups, rng, n))

    res = _summarize(observed, null, K, n, len(groups), n_nonsingleton, two_sided)
    return (res, null) if return_null else res


# --------------------------------------------------------------------------- #
# CONVENIENCE : Spearman between two paired similarity matrices
# --------------------------------------------------------------------------- #
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def topic_null_pairsim(S_a: np.ndarray, S_b: np.ndarray, labels: Sequence,
                       K: int = 10_000, seed: int = 0,
                       pair_mask: np.ndarray | None = None,
                       return_null: bool = False):
    """Topic-matched null for Spearman rho between two paired similarity matrices.

    S_a, S_b : (n,n) similarity matrices over the SAME n items (e.g. text-embedding
               similarity and SAE-code similarity). rho is computed over the upper
               triangle (i<j). One matrix (S_b) is permuted within group.
    labels   : group id per item.
    pair_mask: optional boolean over the i<j pairs to restrict rho to a subset of
               pairs (e.g. only same-group pairs). Default: all i<j pairs -> the
               FULL rho gets the topic-null treatment.

    The permutation preserves group identity of every item, so the multiset of
    same-group and cross-group pairs is invariant; only the S_b value attached to
    each item is reshuffled inside its group.
    """
    S_a = np.asarray(S_a, dtype=np.float64)
    S_b = np.asarray(S_b, dtype=np.float64)
    n = S_a.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    ya = S_a[iu, ju]
    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        ya = ya[pair_mask]

    def effect_fn(perm):
        xb = S_b[perm[iu], perm[ju]]
        if pair_mask is not None:
            xb = xb[pair_mask]
        return _spearman(ya, xb)

    return topic_null_effect(effect_fn, labels, K=K, seed=seed,
                             return_null=return_null)


# --------------------------------------------------------------------------- #
# self-test (positive + negative control) when run directly
# --------------------------------------------------------------------------- #
def _self_test():
    """NEGATIVE control (null must NOT fire): agreement that is ALL group -> the
    topic null reproduces it, share ~1, p non-significant.
    POSITIVE control (null MUST fire / detect real signal): a genuine within-group
    item-level agreement on top of topic -> share small, p tiny."""
    rng = np.random.default_rng(0)
    n_groups, per = 30, 6
    labels = np.repeat(np.arange(n_groups), per)
    n = len(labels)

    # a per-item latent that is PURELY a function of the group (topic)
    group_val = rng.normal(size=n_groups)
    topic_only = group_val[labels] + 0.0
    # build two "modalities" that both equal topic_only + independent noise
    a = topic_only + 0.01 * rng.normal(size=n)
    b_topic = topic_only + 0.01 * rng.normal(size=n)
    Sa = -np.abs(a[:, None] - a[None, :])
    Sb_topic = -np.abs(b_topic[:, None] - b_topic[None, :])
    r_topic = topic_null_pairsim(Sa, Sb_topic, labels, K=2000, seed=1)

    # a modality with a REAL within-group item-level signal on top of topic
    item_val = rng.normal(size=n)
    b_item = topic_only + item_val + 0.01 * rng.normal(size=n)
    a2 = topic_only + item_val + 0.01 * rng.normal(size=n)
    Sa2 = -np.abs(a2[:, None] - a2[None, :])
    Sb_item = -np.abs(b_item[:, None] - b_item[None, :])
    r_item = topic_null_pairsim(Sa2, Sb_item, labels, K=2000, seed=1)

    print("NEGATIVE control (all-topic agreement; null SHOULD reproduce it):")
    print(f"  observed rho={r_topic.observed:.3f}  null_mean={r_topic.null_mean:.3f}"
          f"  share={r_topic.group_explained_share:.2f}  p={r_topic.p_one_sided:.3f}")
    assert r_topic.group_explained_share > 0.7, "topic-only should be mostly explained"

    print("POSITIVE control (real within-group item agreement; null SHOULD fire):")
    print(f"  observed rho={r_item.observed:.3f}  null_mean={r_item.null_mean:.3f}"
          f"  share={r_item.group_explained_share:.2f}  p={r_item.p_one_sided:.4f}")
    assert r_item.p_one_sided < 0.01, "real item-level signal should survive"
    assert r_item.group_explained_share < r_topic.group_explained_share
    print("self-test PASSED")


if __name__ == "__main__":
    _self_test()
