#!/usr/bin/env python3
"""POST-HOC diagnostic (labeled as such; not pre-registered).

The pre-registered primary returned pooled r = -0.59 for LOPO ridge. Hypothesis:
this is the known leave-one-group-out mean-shift artifact -- with ~no signal and
strong shrinkage, predictions ~ training-fold mean, and the training-fold mean of a
LOPO fold mechanically anticorrelates with the held-out problem's mean (the overall
mean is fixed). Diagnostics:
  D1. Pure mean-predictor baseline (predict training-fold mean of v_gold) under the
      same LOPO: its pooled r shows the artifact's size with zero feature information.
  D2. Does ridge beat the mean predictor in MSE? Problem-level paired comparison
      (per-problem MSE difference), sign test + problem bootstrap CI of the difference.
  D3. Artifact-free between-problem statistic: corr over problems of
      (problem-mean prediction - that fold's training mean) vs problem-mean v_gold.
      Subtracting the fold training mean removes the mechanical component.
"""
import json
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from itertools import combinations
from math import comb

DATA = "/Users/steven/vineppo_replication/results/data.jsonl"
DIR = "/Users/steven/moral_framework_misgen/undefined/vineppo-probe-fixed"

X, y, groups = [], [], []
with open(DATA) as f:
    for line in f:
        rec = json.loads(line)
        for p in rec["prefixes"]:
            X.append(p["hidden"]); y.append(p["v_gold"]); groups.append(rec["id"])
X = np.asarray(X, float); y = np.asarray(y, float); groups = np.asarray(groups)
uniq = np.unique(groups)


def pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# reproduce the exact LOPO ridge predictions (same code path + seed as probe_fixed.py)
def ridge_fit_predict(Xtr, ytr, Xte, alphas):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd; Xte_s = (Xte - mu) / sd
    ym = ytr.mean(); yc = ytr - ym
    U, S, Vt = np.linalg.svd(Xtr_s, full_matrices=False)
    Uty = U.T @ yc; XteV = Xte_s @ Vt.T
    preds = np.empty((len(alphas), Xte.shape[0]))
    for i, a in enumerate(alphas):
        preds[i] = XteV @ ((S / (S**2 + a)) * Uty) + ym
    return preds


def group_kfold_splits(gs, k, rng):
    g = rng.permutation(np.array(gs))
    folds = np.array_split(g, k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
            for i in range(k)]


ALPHAS = np.logspace(0, 6, 13)
rng = np.random.default_rng(100)          # same seed as probe_fixed.py primary
pred_ridge = np.empty(len(y))
pred_mean = np.empty(len(y))
fold_train_mean = {}
for pid in uniq:
    te = groups == pid; tr = ~te
    tg = np.unique(groups[tr])
    mse = np.zeros(len(ALPHAS))
    for itr_g, ival_g in group_kfold_splits(tg, 5, rng):
        m_tr = np.isin(groups, itr_g) & tr
        m_va = np.isin(groups, ival_g) & tr
        p = ridge_fit_predict(X[m_tr], y[m_tr], X[m_va], ALPHAS)
        mse += ((p - y[m_va])**2).sum(axis=1)
    a_best = ALPHAS[int(np.argmin(mse))]
    pred_ridge[te] = ridge_fit_predict(X[tr], y[tr], X[te], np.array([a_best]))[0]
    fold_train_mean[pid] = y[tr].mean()
    pred_mean[te] = y[tr].mean()

out = {}
# D1: artifact size with zero information
r_meanpred = pearson(pred_mean, y)
out["D1_mean_predictor_pooled_r"] = r_meanpred
print(f"D1 mean-predictor LOPO pooled r = {r_meanpred:+.3f}  (artifact size, zero features)")

# D2: per-problem MSE, ridge vs mean predictor
mse_r = np.array([np.mean((pred_ridge[groups == g] - y[groups == g])**2) for g in uniq])
mse_m = np.array([np.mean((pred_mean[groups == g] - y[groups == g])**2) for g in uniq])
diff = mse_r - mse_m                       # negative = ridge better
n_better = int((diff < 0).sum()); n_tot = int((diff != 0).sum())
# exact two-sided binomial sign test
p_sign = sum(comb(n_tot, k) for k in range(0, n_tot + 1)
             if comb(n_tot, k) <= comb(n_tot, min(n_better, n_tot - n_better))) * 0.5**n_tot
rngb = np.random.default_rng(11)
boots = [np.mean(diff[rngb.integers(0, len(diff), len(diff))]) for _ in range(10000)]
out["D2_ridge_vs_mean_mse"] = {
    "mean_mse_ridge": float(mse_r.mean()), "mean_mse_meanpred": float(mse_m.mean()),
    "mean_diff": float(diff.mean()),
    "diff_boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    "problems_ridge_better": n_better, "n_problems": n_tot, "sign_test_p": float(p_sign)}
print(f"D2 MSE ridge {mse_r.mean():.4f} vs mean-pred {mse_m.mean():.4f}; "
      f"diff {diff.mean():+.4f} CI[{np.percentile(boots,2.5):+.4f},{np.percentile(boots,97.5):+.4f}]; "
      f"ridge better on {n_better}/{n_tot} problems, sign p={p_sign:.3f}")

# D3: artifact-free between-problem statistic
pm_pred = np.array([pred_ridge[groups == g].mean() - fold_train_mean[g] for g in uniq])
pm_gold = np.array([y[groups == g].mean() for g in uniq])
r_d3 = pearson(pm_pred, pm_gold)
z = np.arctanh(max(min(r_d3, 0.999), -0.999)); se = 1 / np.sqrt(len(uniq) - 3)
out["D3_between_r_artifact_corrected"] = {
    "r": r_d3, "fisher95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
    "n_problems": int(len(uniq))}
print(f"D3 artifact-corrected between-problem r = {r_d3:+.3f} "
      f"CI[{np.tanh(z-1.96*se):+.3f},{np.tanh(z+1.96*se):+.3f}]")

with open(os.path.join(DIR, "posthoc_diagnostic.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved posthoc_diagnostic.json")
