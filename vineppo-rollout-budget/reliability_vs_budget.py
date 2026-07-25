#!/usr/bin/env python3
"""MC-simulator reliability vs rollout budget k in {1,2,4,8}.

Analysis-only on saved rollout outcomes from ~/vineppo_replication/results/data.jsonl.
Per prefix the run saved v_gold (12 rollouts) and v_sim (6 disjoint rollouts), generated
identically (run.py:rollout_value, same policy/temp/prompt/parser) -> 18 exchangeable
Bernoulli outcomes; the success count is a sufficient statistic, so permuting a vector of
s_i ones and 18-s_i zeros is exactly equivalent to subsampling the raw saved outcomes.

Pre-registered design (see FINDINGS.md, written before this ran):
  estimator = mean of first k of a random permutation, reference = mean of next 10 (disjoint,
  fixed m=10 for all k). r(k) = Pearson across 69 prefixes, averaged over B=1000 draws (seed 0).
  Pooled + within-problem (problem-mean-centered) versions.
  CIs: cluster bootstrap over 23 problems (1000 reps, 20 inner draws, seed 1), percentile 95%.
  Primary test: paired bootstrap CI for r_within(8) - r_within(1).
  Model: rho(k) = sqrt(k/(k+lambda)); lambda moment-based + 1-param LSQ fit.
"""
import json
import numpy as np

DATA = "/Users/steven/vineppo_replication/results/data.jsonl"
OUT = "/Users/steven/moral_framework_misgen/undefined/vineppo-rollout-budget"
N_GOLD, N_SIM = 12, 6
N = N_GOLD + N_SIM          # 18 exchangeable outcomes per prefix
M_REF = 10                  # fixed reference size
KS = [1, 2, 4, 8]
B_MAIN = 1000               # subsample draws for point estimates
B_BOOT = 1000               # cluster-bootstrap replicates
B_INNER = 20                # subsample draws inside each bootstrap replicate
TOK_PER_ROLLOUT = 242127 / 1716.0   # from full_run.log accounting
SEC_PER_ROLLOUT = TOK_PER_ROLLOUT / 51.0

# ---------- load ----------
succ, prob_idx = [], []
with open(DATA) as f:
    for j, line in enumerate(f):
        d = json.loads(line)
        for p in d["prefixes"]:
            sg, ss = round(p["v_gold"] * N_GOLD), round(p["v_sim"] * N_SIM)
            assert abs(sg - p["v_gold"] * N_GOLD) < 1e-6 and abs(ss - p["v_sim"] * N_SIM) < 1e-6
            succ.append(sg + ss)
            prob_idx.append(j)
succ = np.array(succ)                       # successes out of 18 per prefix
prob_idx = np.array(prob_idx)
n_prefix, n_prob = len(succ), prob_idx.max() + 1
assert (n_prefix, n_prob) == (69, 23)

# per-prefix outcome vectors (s ones then zeros; permuted per draw)
base = (np.arange(N)[None, :] < succ[:, None]).astype(np.float64)   # (69, 18)

rng_main = np.random.default_rng(0)


def draw_estimates(rows, ks, rng):
    """One subsample draw for the given prefix rows: dict k->est means, plus ref means."""
    out = base[rows]                                     # (n, 18)
    perm = np.argsort(rng.random(out.shape), axis=1)     # independent row permutations
    shuf = np.take_along_axis(out, perm, axis=1)
    kmax = max(ks)
    ests = {k: shuf[:, :k].mean(axis=1) for k in ks}
    ref = shuf[:, kmax:kmax + M_REF].mean(axis=1)        # disjoint from every estimator
    return ests, ref


def center(v, pidx):
    """subtract problem means"""
    means = np.bincount(pidx, weights=v) / np.bincount(pidx)
    return v - means[pidx]


def pearson(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def curves(rows, pidx, nb, rng):
    """mean r(k) pooled & within over nb subsample draws"""
    acc = {("pooled", k): [] for k in KS} | {("within", k): [] for k in KS}
    for _ in range(nb):
        ests, ref = draw_estimates(rows, KS, rng)
        ref_c = center(ref, pidx)
        for k in KS:
            acc[("pooled", k)].append(pearson(ests[k], ref))
            acc[("within", k)].append(pearson(center(ests[k], pidx), ref_c))
    return {key: float(np.nanmean(v)) for key, v in acc.items()}


# ---------- point estimates ----------
rows_all = np.arange(n_prefix)
pt = curves(rows_all, prob_idx, B_MAIN, rng_main)

# ---------- cluster bootstrap ----------
rng_boot = np.random.default_rng(1)
boot = {key: [] for key in pt}
diff_within, diff_pooled = [], []
prefix_of_prob = [np.where(prob_idx == j)[0] for j in range(n_prob)]
for _ in range(B_BOOT):
    probs = rng_boot.integers(0, n_prob, n_prob)
    rows = np.concatenate([prefix_of_prob[j] for j in probs])
    pidx = np.repeat(np.arange(n_prob), [len(prefix_of_prob[j]) for j in probs])
    c = curves(rows, pidx, B_INNER, rng_boot)
    for key in boot:
        boot[key].append(c[key])
    diff_within.append(c[("within", 8)] - c[("within", 1)])
    diff_pooled.append(c[("pooled", 8)] - c[("pooled", 1)])

ci = {key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
      for key, v in boot.items()}
ci_dw = [float(np.nanpercentile(diff_within, 2.5)), float(np.nanpercentile(diff_within, 97.5))]
ci_dp = [float(np.nanpercentile(diff_pooled, 2.5)), float(np.nanpercentile(diff_pooled, 97.5))]

# ---------- binomial-reliability model ----------
def lambda_moment(s_vec, pidx=None):
    """lambda = E[p(1-p)] / Var(p), unbiased moments from counts out of N.
    If pidx given, Var(p) is the within-problem variance (ANOVA-style)."""
    ph = s_vec / N
    pq_unb = ph * (1 - ph) * N / (N - 1)                # unbiased p(1-p) per prefix
    noise_var = np.mean(pq_unb) / N                      # E[p(1-p)]/N
    if pidx is None:
        var_p = np.var(ph, ddof=1) - noise_var
    else:
        c = center(ph, pidx)
        m = np.bincount(pidx)                            # cluster sizes (all 3 here)
        # E[var of centered noise] = noise_var * mean(1 - 1/m_i)
        dof = len(ph) - len(m)
        var_within_hat = np.sum(c ** 2) / dof
        var_p = var_within_hat - noise_var * np.mean(1 - 1 / m[pidx]) * len(ph) / dof
    return float(np.mean(pq_unb) / var_p), float(var_p), float(np.mean(pq_unb))


lam_pool, varp_pool, pq_pool = lambda_moment(succ)
lam_with, varp_with, pq_with = lambda_moment(succ, prob_idx)

# bootstrap CI for moment-based lambdas (cluster bootstrap, same replicate structure)
rng_lb = np.random.default_rng(2)
lb_pool, lb_with = [], []
for _ in range(B_BOOT):
    probs = rng_lb.integers(0, n_prob, n_prob)
    rows = np.concatenate([prefix_of_prob[j] for j in probs])
    pidx = np.repeat(np.arange(n_prob), [len(prefix_of_prob[j]) for j in probs])
    try:
        lp, vp, _ = lambda_moment(succ[rows])
        lw, vw, _ = lambda_moment(succ[rows], pidx)
        lb_pool.append(lp if vp > 0 else np.inf)
        lb_with.append(lw if vw > 0 else np.inf)
    except ZeroDivisionError:
        pass
ci_lam_pool = [float(np.percentile(lb_pool, 2.5)), float(np.percentile(lb_pool, 97.5))]
ci_lam_with = [float(np.percentile(lb_with, 2.5)), float(np.percentile(lb_with, 97.5))]


def rho(k, lam):
    return np.sqrt(k / (k + lam))


def fit_lambda(rs):
    """1-param LSQ of r(k) = rho(k,l)*rho(M_REF,l) to the 4 empirical points"""
    grid = np.linspace(0.05, 60, 12000)
    err = [sum((rs[i] - rho(k, l) * rho(M_REF, l)) ** 2 for i, k in enumerate(KS)) for l in grid]
    return float(grid[int(np.argmin(err))])


lam_fit_pool = fit_lambda([pt[("pooled", k)] for k in KS])
lam_fit_with = fit_lambda([pt[("within", k)] for k in KS])

# ---------- compute-value table ----------
table = []
for k in KS:
    r2 = rho(k, lam_fit_pool) ** 2
    r2w = rho(k, lam_fit_with) ** 2
    table.append({
        "k": k,
        "tokens": round(k * TOK_PER_ROLLOUT),
        "seconds_m4": round(k * SEC_PER_ROLLOUT, 1),
        "r_pooled": round(pt[("pooled", k)], 3),
        "ci_pooled": [round(x, 3) for x in ci[("pooled", k)]],
        "r_within": round(pt[("within", k)], 3),
        "ci_within": [round(x, 3) for x in ci[("within", k)]],
        "model_rho_vs_truth_pooled": round(float(rho(k, lam_fit_pool)), 3),
        "model_rho_vs_truth_within": round(float(rho(k, lam_fit_with)), 3),
        "model_R2_vs_truth_pooled": round(float(r2), 3),
        "model_R2_vs_truth_within": round(float(r2w), 3),
    })
# marginal value per rollout (variance-of-true-value explained per added rollout), model-based
marg = []
for k in range(1, 9):
    dr2 = rho(k, lam_fit_pool) ** 2 - rho(k - 1, lam_fit_pool) ** 2 if k > 1 else rho(1, lam_fit_pool) ** 2
    marg.append({"k": k, "delta_R2_pooled_per_rollout": round(float(dr2), 4)})

# sanity check: reproduce README's saved-aggregate correlation (6-rollout sim vs 12-rollout gold)
v_sim = np.array([s and 0 for s in range(0)])  # placeholder removed below
vg, vs = [], []
with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        for p in d["prefixes"]:
            vg.append(p["v_gold"]); vs.append(p["v_sim"])
r_readme = pearson(np.array(vs), np.array(vg))

results = {
    "design": {
        "n_problems": int(n_prob), "n_prefixes": int(n_prefix),
        "outcomes_per_prefix": N, "reference_size": M_REF, "budgets": KS,
        "B_main": B_MAIN, "B_boot": B_BOOT, "B_inner": B_INNER,
        "seeds": {"main": 0, "bootstrap": 1, "lambda_bootstrap": 2},
        "tokens_per_rollout": round(TOK_PER_ROLLOUT, 1),
        "seconds_per_rollout_m4": round(SEC_PER_ROLLOUT, 2),
    },
    "reliability_curve": table,
    "primary_test": {
        "stat": "r_within(8) - r_within(1)",
        "point": round(pt[("within", 8)] - pt[("within", 1)], 3),
        "ci95_cluster_bootstrap": [round(x, 3) for x in ci_dw],
        "pooled_diff_point": round(pt[("pooled", 8)] - pt[("pooled", 1)], 3),
        "pooled_diff_ci95": [round(x, 3) for x in ci_dp],
    },
    "lambda_noise_to_signal": {
        "definition": "rho(k)=sqrt(k/(k+lambda)) vs TRUE p; lambda = rollouts to reach R2=0.5",
        "pooled_moment": round(lam_pool, 2), "pooled_moment_ci95": [round(x, 2) for x in ci_lam_pool],
        "within_moment": round(lam_with, 2), "within_moment_ci95": [round(x, 2) for x in ci_lam_with],
        "pooled_lsq_fit": round(lam_fit_pool, 2), "within_lsq_fit": round(lam_fit_with, 2),
        "var_true_p_pooled": round(varp_pool, 4), "var_true_p_within": round(varp_with, 4),
        "mean_p_1minus_p": round(pq_pool, 4),
    },
    "marginal_value_model": marg,
    "sanity_check": {
        "saved_aggregate_r_sim6_vs_gold12": round(r_readme, 3),
        "readme_claims": 0.878,
    },
}
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------- figure ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
kk = np.linspace(0.8, 10, 200)
for i, (name, lam) in enumerate([("pooled", lam_fit_pool), ("within-problem", lam_fit_with)]):
    key = "pooled" if i == 0 else "within"
    rs = [pt[(key, k)] for k in KS]
    lo = [ci[(key, k)][0] for k in KS]
    hi = [ci[(key, k)][1] for k in KS]
    ax[i].errorbar(KS, rs, yerr=[np.array(rs) - lo, np.array(hi) - np.array(rs)],
                   fmt="o", capsize=4, color="C0", label="empirical (vs 10-rollout ref)")
    ax[i].plot(kk, rho(kk, lam) * rho(M_REF, lam), "C1--",
               label=f"binomial model, lambda={lam:.1f}")
    ax[i].plot(kk, rho(kk, lam), "C2:", label="model vs TRUE value")
    ax[i].set_xscale("log", base=2); ax[i].set_xticks(KS); ax[i].set_xticklabels(KS)
    ax[i].set_xlabel("rollout budget k"); ax[i].set_ylabel("reliability r")
    ax[i].set_title(f"{name}"); ax[i].set_ylim(0, 1); ax[i].legend(fontsize=8)
fig.suptitle("MC-simulator reliability vs rollout budget (69 prefixes, 23 problems, cluster-bootstrap 95% CI)")
fig.tight_layout()
fig.savefig(f"{OUT}/reliability_curve.png", dpi=150)
print(json.dumps(results, indent=2))
