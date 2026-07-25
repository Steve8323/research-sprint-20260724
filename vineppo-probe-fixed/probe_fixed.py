#!/usr/bin/env python3
"""Properly-trained one-pass value probe over saved VinePPO-replication hidden states.

Analysis-only. Reads /Users/steven/vineppo_replication/results/data.jsonl (read-only).
Writes results.json next to this script. All CV is problem-level (LOPO / GroupKFold).
Pre-registered in FINDINGS.md (written before this script was run).
"""
import json
import os
import time

# Single-threaded BLAS: multithreaded OpenBLAS thrashes on these tiny SVDs
# (and the machine is shared). Must be set before numpy import.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(2)

DATA = "/Users/steven/vineppo_replication/results/data.jsonl"
OUT = "/Users/steven/moral_framework_misgen/undefined/vineppo-probe-fixed/results.json"

ALPHAS = np.logspace(0, 6, 13)
BOOT_N = 10000
PERM_N = 10000
MLP_SEEDS = [0, 1, 2, 3, 4]
LC_SIZES = [4, 8, 12, 16, 20, 22]
LC_REPS = 50

t0 = time.time()

# ---------------- load ----------------
X, y, groups, v_sim, v_verb, plen = [], [], [], [], [], []
with open(DATA) as f:
    for line in f:
        rec = json.loads(line)
        for p in rec["prefixes"]:
            X.append(p["hidden"])
            y.append(p["v_gold"])
            groups.append(rec["id"])
            v_sim.append(p["v_sim"])
            v_verb.append(p["v_verb"])
            plen.append(len(p["prefix"]))
X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.float64)
groups = np.asarray(groups)
v_sim = np.asarray(v_sim)
v_verb = np.asarray(v_verb)
plen = np.asarray(plen, dtype=np.float64)
uniq = np.unique(groups)
n, d = X.shape
print(f"loaded X {X.shape}, {len(uniq)} problems")


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------- ridge via one SVD per training set (all alphas at once) ----------------
def ridge_fit_predict(Xtr, ytr, Xte, alphas):
    """Standardize on train, center y, ridge via SVD. Returns preds (len(alphas), n_te)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    ym = ytr.mean()
    yc = ytr - ym
    U, S, Vt = np.linalg.svd(Xtr_s, full_matrices=False)
    Uty = U.T @ yc
    XteV = Xte_s @ Vt.T                      # (n_te, k)
    preds = np.empty((len(alphas), Xte.shape[0]))
    for i, a in enumerate(alphas):
        w = (S / (S**2 + a)) * Uty           # coords in V basis
        preds[i] = XteV @ w + ym
    return preds


def group_kfold_splits(train_groups_unique, k, rng=None):
    """List of (inner_train_groups, inner_val_groups)."""
    g = np.array(train_groups_unique)
    if rng is not None:
        g = rng.permutation(g)
    folds = np.array_split(g, k)
    out = []
    for i in range(k):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr, val))
    return out


def ridge_cv_predict(Xtr, ytr, gtr, Xte, alphas, inner_k, rng):
    """Select alpha by inner GroupKFold MSE, return (test preds, chosen alpha)."""
    tg = np.unique(gtr)
    k = min(inner_k, len(tg))
    mse = np.zeros(len(alphas))
    for itr_g, ival_g in group_kfold_splits(tg, k, rng):
        m_tr = np.isin(gtr, itr_g)
        m_va = np.isin(gtr, ival_g)
        p = ridge_fit_predict(Xtr[m_tr], ytr[m_tr], Xtr[m_va], alphas)
        mse += ((p - ytr[m_va])**2).sum(axis=1)
    a_best = alphas[int(np.argmin(mse))]
    pred = ridge_fit_predict(Xtr, ytr, Xte, np.array([a_best]))[0]
    return pred, float(a_best)


def lopo_ridge(Xf, yf, gf, alphas, seed=100):
    rng = np.random.default_rng(seed)
    preds = np.empty(len(yf))
    chosen = []
    for pid in np.unique(gf):
        te = gf == pid
        tr = ~te
        p, a = ridge_cv_predict(Xf[tr], yf[tr], gf[tr], Xf[te], alphas, 5, rng)
        preds[te] = p
        chosen.append(a)
    return preds, chosen


# ---------------- inference helpers ----------------
def problem_bootstrap_ci(pred, gold, gr, n_boot, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(gr)
    idx_by_g = {g: np.where(gr == g)[0] for g in ug}
    rs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        rs[b] = pearson(pred[idx], gold[idx])
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]


def within_center(v, gr):
    out = v.astype(float).copy()
    for g in np.unique(gr):
        m = gr == g
        out[m] -= out[m].mean()
    return out


def within_r_and_perm_p(pred, gold, gr, n_perm, seed):
    rng = np.random.default_rng(seed)
    pc = within_center(pred, gr)
    r_obs = None
    gc = within_center(gold, gr)
    r_obs = pearson(pc, gc)
    count = 0
    gperm = gold.copy()
    for _ in range(n_perm):
        for g in np.unique(gr):
            m = np.where(gr == g)[0]
            gperm[m] = gold[m][rng.permutation(len(m))]
        r_p = pearson(pc, within_center(gperm, gr))
        if abs(r_p) >= abs(r_obs) - 1e-12:
            count += 1
    p = (1 + count) / (1 + n_perm)
    return float(r_obs), float(p)


def fisher_ci(r, n):
    if abs(r) >= 1 or n < 4:
        return [None, None]
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def summarize(pred, gold, gr, name, boot_seed, perm_seed):
    r_pool = pearson(pred, gold)
    ci = problem_bootstrap_ci(pred, gold, gr, BOOT_N, boot_seed)
    r_w, p_w = within_r_and_perm_p(pred, gold, gr, PERM_N, perm_seed)
    pm_pred = np.array([pred[gr == g].mean() for g in np.unique(gr)])
    pm_gold = np.array([gold[gr == g].mean() for g in np.unique(gr)])
    r_btw = pearson(pm_pred, pm_gold)
    res = {
        "pooled_r": r_pool, "pooled_r_boot95": ci,
        "within_problem_r": r_w, "within_perm_p": p_w,
        "between_problem_r": r_btw, "between_fisher95": fisher_ci(r_btw, len(pm_pred)),
        "n_points": int(len(gold)), "n_problems": int(len(np.unique(gr))),
    }
    print(f"[{name}] pooled r={r_pool:+.3f} CI{ci[0]:+.3f},{ci[1]:+.3f} | "
          f"within r={r_w:+.3f} p={p_w:.4f} | between r={r_btw:+.3f}")
    return res


results = {"n_points": int(n), "n_problems": int(len(uniq)), "dim": int(d),
           "alphas": list(map(float, ALPHAS)),
           "preregistered": True}

# ---------------- PRIMARY: LOPO ridge on v_gold ----------------
pred_ridge, alphas_chosen = lopo_ridge(X, y, groups, ALPHAS, seed=100)
results["ridge_lopo"] = summarize(pred_ridge, y, groups, "ridge LOPO", 0, 1)
results["ridge_lopo"]["alphas_chosen"] = alphas_chosen

# ---------------- SECONDARY 3: small MLP, LOPO, 5 seeds ----------------
def mlp_lopo(Xf, yf, gf, seed):
    torch.manual_seed(seed)
    preds = np.empty(len(yf))
    for pid in np.unique(gf):
        te = gf == pid
        tr = ~te
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0)
        sd[sd == 0] = 1.0
        Xtr = torch.tensor((Xf[tr] - mu) / sd, dtype=torch.float32)
        Xte = torch.tensor((Xf[te] - mu) / sd, dtype=torch.float32)
        ytr = torch.tensor(yf[tr], dtype=torch.float32)[:, None]
        net = torch.nn.Sequential(
            torch.nn.Linear(Xf.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = torch.nn.MSELoss()
        for _ in range(300):
            opt.zero_grad()
            loss = lossf(net(Xtr), ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds[te] = net(Xte).squeeze(-1).numpy()
    return preds

mlp_preds = []
mlp_rs = []
for s in MLP_SEEDS:
    p = mlp_lopo(X, y, groups, s)
    mlp_preds.append(p)
    mlp_rs.append(pearson(p, y))
    print(f"  mlp seed {s}: pooled r={mlp_rs[-1]:+.3f}")
mlp_avg = np.mean(mlp_preds, axis=0)
results["mlp_lopo"] = summarize(mlp_avg, y, groups, "MLP LOPO seed-avg", 3, 4)
results["mlp_lopo"]["per_seed_pooled_r"] = [float(r) for r in mlp_rs]
results["mlp_lopo"]["per_seed_mean_sd"] = [float(np.mean(mlp_rs)), float(np.std(mlp_rs))]

# ---------------- SECONDARY 4: learning curve (ridge) ----------------
rng_lc = np.random.default_rng(2)
lc = {}
for m in LC_SIZES:
    rs_pool, rs_within = [], []
    for rep in range(LC_REPS):
        tr_g = rng_lc.choice(uniq, size=m, replace=False)
        tr = np.isin(groups, tr_g)
        te = ~tr
        pred, _ = ridge_cv_predict(X[tr], y[tr], groups[tr], X[te], ALPHAS,
                                   min(5, m), rng_lc)
        rs_pool.append(pearson(pred, y[te]))
        rs_within.append(pearson(within_center(pred, groups[te]),
                                 within_center(y[te], groups[te])))
    lc[str(m)] = {
        "pooled_r_mean": float(np.mean(rs_pool)),
        "pooled_r_p2p5_97p5": [float(np.percentile(rs_pool, 2.5)),
                               float(np.percentile(rs_pool, 97.5))],
        "within_r_mean": float(np.mean(rs_within)),
        "within_r_p2p5_97p5": [float(np.percentile(rs_within, 2.5)),
                               float(np.percentile(rs_within, 97.5))],
        "n_reps": LC_REPS,
    }
    print(f"  LC m={m:2d}: pooled r mean {np.mean(rs_pool):+.3f} "
          f"[{np.percentile(rs_pool,2.5):+.3f},{np.percentile(rs_pool,97.5):+.3f}] | "
          f"within {np.mean(rs_within):+.3f}")
results["learning_curve"] = lc

# ---------------- SECONDARY 5: positive control (log prefix length) ----------------
y_ctrl = np.log(plen)
pred_ctrl, _ = lopo_ridge(X, y_ctrl, groups, ALPHAS, seed=101)
r_ctrl = pearson(pred_ctrl, y_ctrl)
ci_ctrl = problem_bootstrap_ci(pred_ctrl, y_ctrl, groups, BOOT_N, 5)
results["positive_control_logprefixlen"] = {
    "pooled_r": r_ctrl, "pooled_r_boot95": ci_ctrl, "pass_threshold": 0.5,
    "passed": bool(r_ctrl > 0.5)}
print(f"[control log-len] pooled r={r_ctrl:+.3f} CI[{ci_ctrl[0]:+.3f},{ci_ctrl[1]:+.3f}]")

# ---------------- SECONDARY 6: reference lines from saved data ----------------
results["reference_simulator"] = summarize(v_sim, y, groups, "simulator 8-roll", 6, 7)
results["reference_verbalized"] = {"pooled_r": pearson(v_verb, y),
                                   "within_problem_r": float(pearson(
                                       within_center(v_verb, groups),
                                       within_center(y, groups)))}
print(f"[verbalized] pooled r={results['reference_verbalized']['pooled_r']:+.3f}")

results["runtime_sec"] = round(time.time() - t0, 1)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"done in {results['runtime_sec']} s -> {OUT}")
