"""stats_toolkit.py — small reusable statistical-hygiene toolkit.

Everything is numpy + scipy only. All resampling takes an explicit seed.

Contents
--------
boot_ci             percentile bootstrap CI for any statistic of one sample
cluster_boot_ci     percentile bootstrap CI resampling whole clusters
perm_p_two_sided    generic permutation p from an observed stat + null draws
clopper_pearson     exact binomial CI
binom_test_exact    exact two-sided binomial test (scipy binomtest wrapper)
fisher_exact_2x2    Fisher exact test from two (k, n) pairs
rank_sum_exact      exact Wilcoxon rank-sum (small n, full enumeration)
signflip_exact      exact paired sign-flip permutation test (small n)
sign_test           exact sign test on paired signs
holm                Holm-Bonferroni adjusted p-values
bh_fdr              Benjamini-Hochberg q-values
pearson             plain Pearson r (nan-safe)
spearman            plain Spearman rho
"""
from itertools import combinations, product

import numpy as np
from scipy import stats as ss


# ---------------------------------------------------------------- bootstrap
def boot_ci(x, stat=np.mean, B=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for stat(x) over iid units."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.array([stat(x[rng.integers(0, n, n)]) for _ in range(B)])
    vals = vals[~np.isnan(vals)]
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def cluster_boot_ci(clusters, stat, B=4000, seed=0, alpha=0.05):
    """Percentile bootstrap CI resampling whole clusters with replacement.

    clusters : list of per-cluster data blobs (any type)
    stat     : callable(list_of_clusters) -> float
    """
    rng = np.random.default_rng(seed)
    n = len(clusters)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = stat([clusters[i] for i in idx])
        if v is not None and not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


# ---------------------------------------------------------------- permutation
def perm_p_two_sided(obs, null_draws):
    """Add-one permutation p, two-sided around the null mean."""
    null_draws = np.asarray(null_draws)
    center = null_draws.mean()
    k = int(np.sum(np.abs(null_draws - center) >= abs(obs - center)))
    return float((k + 1) / (len(null_draws) + 1))


# ---------------------------------------------------------------- exact tests
def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else ss.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else ss.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def binom_test_exact(k, n, p=0.5):
    return float(ss.binomtest(k, n, p, alternative="two-sided").pvalue)


def fisher_exact_2x2(k1, n1, k2, n2):
    tbl = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(ss.fisher_exact(tbl, alternative="two-sided")[1])


def rank_sum_exact(a, b):
    """Exact two-sided Wilcoxon rank-sum by full enumeration (small n only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, na = len(pooled), len(a)
    ranks = ss.rankdata(pooled)
    obs = ranks[:na].sum()
    stats_all = np.array([ranks[list(c)].sum()
                          for c in combinations(range(n), na)])
    center = stats_all.mean()
    p = np.mean(np.abs(stats_all - center) >= abs(obs - center) - 1e-12)
    return float(p)


def signflip_exact(diffs):
    """Exact paired sign-flip permutation test on mean(diffs), two-sided."""
    d = np.asarray(diffs, float)
    n = len(d)
    obs = d.mean()
    stats_all = np.array([(d * np.array(s)).mean()
                          for s in product([1, -1], repeat=n)])
    p = np.mean(np.abs(stats_all) >= abs(obs) - 1e-12)
    return float(p)


def sign_test(n_pos, n_neg):
    """Exact two-sided sign test, ties dropped beforehand."""
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    return binom_test_exact(n_pos, n, 0.5)


# ---------------------------------------------------------------- multiplicity
def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q.tolist()


# ---------------------------------------------------------------- correlations
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(ss.rankdata(x), ss.rankdata(y))
