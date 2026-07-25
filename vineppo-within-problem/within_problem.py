#!/usr/bin/env python3
"""Within-problem credit assignment with problem-level bootstrap CIs.

Analysis-only re-analysis of /Users/steven/vineppo_replication/results/data.jsonl.
Pre-registered in FINDINGS.md (same directory) BEFORE this script was run:
  primary  = pooled within-problem Pearson r per estimator, 95% percentile CI from
             a cluster bootstrap over the 23 problems, B=10000, seed 0.
  secondary = pairwise r differences, mean per-problem Spearman, pooled r (same resamples).
  post-hoc  = simulator binomial noise-only ceiling, probe CV-seed sweep, descriptives.
Writes results.json next to this script. Runtime ~ seconds on CPU.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

DATA = Path("/Users/steven/vineppo_replication/results/data.jsonl")
OUTDIR = Path("/Users/steven/moral_framework_misgen/undefined/vineppo-within-problem")
B = 10_000
SEED = 0

# ---------------- load ------------------------------------------------------
recs = [json.loads(l) for l in DATA.open() if l.strip()]
rows = []
for rec in recs:
    for p in rec["prefixes"]:
        rows.append({"pid": rec["id"], "k": p["k"], "v_gold": p["v_gold"],
                     "v_sim": p["v_sim"], "v_verb": p["v_verb"], "hidden": p["hidden"]})
pids = sorted({r["pid"] for r in rows})
pid_rows = {pid: [i for i, r in enumerate(rows) if r["pid"] == pid] for pid in pids}
n_rows, n_probs = len(rows), len(pids)
v_gold = np.array([r["v_gold"] for r in rows])
H = np.array([r["hidden"] for r in rows])

# ---------------- probe: out-of-fold ridge, matches analyze2.py exactly -----
def probe_oof(seed):
    rng = np.random.RandomState(seed)
    fold = {pid: i % 3 for i, pid in enumerate(rng.permutation(pids))}
    v = np.zeros(n_rows)
    for f in range(3):
        te = np.array([fold[r["pid"]] == f for r in rows])
        tr = ~te
        v[te] = Ridge(alpha=10.0).fit(H[tr], v_gold[tr]).predict(H[te])
    return np.clip(v, 0, 1)

v_probe = probe_oof(SEED)  # seed 0 = analyze2.py's split
est = {"simulator": np.array([r["v_sim"] for r in rows]),
       "verbalized": np.array([r["v_verb"] for r in rows]),
       "probe": v_probe}

# ---------------- centering -------------------------------------------------
def center(v):
    c = v.astype(float).copy()
    for pid in pids:
        idx = pid_rows[pid]
        c[idx] -= c[idx].mean()
    return c

g_c = center(v_gold)
est_c = {k: center(v) for k, v in est.items()}

def r_of(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(stats.pearsonr(x, y)[0])

# per-problem Spearman (secondary)
def chain_spearmans(v):
    out = {}
    for pid in pids:
        idx = pid_rows[pid]
        g, vv = v_gold[idx], v[idx]
        if g.std() > 1e-9 and vv.std() > 1e-9:
            out[pid] = float(stats.spearmanr(vv, g)[0])
    return out

spear = {k: chain_spearmans(v) for k, v in est.items()}

# ---------------- cluster bootstrap (shared resamples) ----------------------
rng = np.random.RandomState(SEED)
boot_pids = rng.choice(n_probs, size=(B, n_probs), replace=True)  # indices into pids

def boot_stats():
    keys = list(est.keys())
    within = {k: [] for k in keys}
    pooled = {k: [] for k in keys}
    mean_sp = {k: [] for k in keys}
    diffs = {"simulator-probe": [], "simulator-verbalized": [], "probe-verbalized": []}
    n_degen = {k: 0 for k in keys}
    for b in range(B):
        sel_pids = [pids[j] for j in boot_pids[b]]
        idx = np.concatenate([pid_rows[p] for p in sel_pids])
        g_cb, g_b = g_c[idx], v_gold[idx]
        rs = {}
        for k in keys:
            rs[k] = r_of(est_c[k][idx], g_cb)
            if np.isnan(rs[k]):
                n_degen[k] += 1
            within[k].append(rs[k])
            pooled[k].append(r_of(est[k][idx], g_b))
            sps = [spear[k][p] for p in sel_pids if p in spear[k]]
            mean_sp[k].append(float(np.mean(sps)) if sps else np.nan)
        diffs["simulator-probe"].append(rs["simulator"] - rs["probe"])
        diffs["simulator-verbalized"].append(rs["simulator"] - rs["verbalized"])
        diffs["probe-verbalized"].append(rs["probe"] - rs["verbalized"])
    return within, pooled, mean_sp, diffs, n_degen

within_b, pooled_b, spear_b, diff_b, n_degen = boot_stats()

def ci(vals):
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], int(len(vals) - len(a))

results = {"n_problems": n_probs, "n_prefixes": n_rows, "B": B, "seed": SEED,
           "primary_within_problem_r": {}, "secondary": {}, "posthoc": {}}

for k in est:
    point = r_of(est_c[k], g_c)
    c, dropped = ci(within_b[k])
    results["primary_within_problem_r"][k] = {
        "r": point, "ci95": c, "excludes_zero": bool(c[0] > 0 or c[1] < 0),
        "boot_dropped_degenerate": dropped}

results["secondary"]["pairwise_diff_within_r"] = {
    k: {"diff": float(np.nanmean(v) * 0 + (results["primary_within_problem_r"][k.split("-")[0]]["r"]
        - results["primary_within_problem_r"][k.split("-")[1]]["r"])),
        "ci95": ci(v)[0],
        "excludes_zero": bool(ci(v)[0][0] > 0 or ci(v)[0][1] < 0)}
    for k, v in diff_b.items()}

results["secondary"]["mean_chain_spearman"] = {
    k: {"mean": float(np.mean(list(spear[k].values()))), "n_chains": len(spear[k]),
        "ci95": ci(spear_b[k])[0]} for k in est}

results["secondary"]["pooled_r"] = {
    k: {"r": r_of(est[k], v_gold), "ci95": ci(pooled_b[k])[0]} for k in est}

# ---------------- post-hoc 1: simulator noise-only ceiling ------------------
p_hat = (8 * est["simulator"] + 16 * v_gold) / 24
rng2 = np.random.RandomState(SEED + 1)
noise_rs = []
for _ in range(2000):
    s = rng2.binomial(8, p_hat) / 8.0
    g = rng2.binomial(16, p_hat) / 16.0
    noise_rs.append(r_of(center(s), center(g)))
noise_rs = np.array(noise_rs, float)
results["posthoc"]["simulator_noise_only_within_r"] = {
    "mean": float(np.nanmean(noise_rs)),
    "pi95": [float(np.nanpercentile(noise_rs, 2.5)), float(np.nanpercentile(noise_rs, 97.5))],
    "n_sims": 2000,
    "note": "within-problem r predicted if the 8-rollout estimator differed from the "
            "16-rollout gold only by binomial sampling noise (shared p_hat per prefix)"}

# ---------------- post-hoc 2: probe CV-seed sweep ---------------------------
seed_rs = [r_of(center(probe_oof(s)), g_c) for s in range(20)]
results["posthoc"]["probe_cv_seed_sweep"] = {
    "n_seeds": 20, "mean": float(np.mean(seed_rs)),
    "min": float(np.min(seed_rs)), "max": float(np.max(seed_rs)),
    "values": [float(x) for x in seed_rs]}

# ---------------- post-hoc 3: descriptives ----------------------------------
zero_var = [int(p) for p in pids if v_gold[pid_rows[p]].std() < 1e-9]
results["posthoc"]["descriptives"] = {
    "problems_zero_within_gold_variance": len(zero_var),
    "zero_var_pids": zero_var,
    "median_p_yes_verbalized": float(np.median(est["verbalized"])),
    "within_gold_variance_share": float(np.var(g_c) / np.var(v_gold))}

(OUTDIR / "results.json").write_text(json.dumps(results, indent=2))

# ---------------- console summary -------------------------------------------
print(f"{n_probs} problems, {n_rows} prefixes; B={B}, seed={SEED}")
print(f"within-problem share of V* variance: {np.var(g_c)/np.var(v_gold):.3f}")
print("\n== PRIMARY: within-problem Pearson r, 95% cluster-bootstrap CI ==")
for k, d in results["primary_within_problem_r"].items():
    print(f"{k:<11} r={d['r']:+.3f}  CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]  "
          f"excl0={d['excludes_zero']}  degenerate_boot={d['boot_dropped_degenerate']}")
print("\n== secondary: pairwise diffs ==")
for k, d in results["secondary"]["pairwise_diff_within_r"].items():
    print(f"{k:<22} d={d['diff']:+.3f}  CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]  "
          f"excl0={d['excludes_zero']}")
print("\n== secondary: mean per-chain Spearman ==")
for k, d in results["secondary"]["mean_chain_spearman"].items():
    print(f"{k:<11} {d['mean']:+.3f}  CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]  "
          f"({d['n_chains']} chains)")
print("\n== secondary: pooled r (for contrast) ==")
for k, d in results["secondary"]["pooled_r"].items():
    print(f"{k:<11} r={d['r']:+.3f}  CI[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
nn = results["posthoc"]["simulator_noise_only_within_r"]
print(f"\npost-hoc noise-only sim within-r: {nn['mean']:.3f} PI[{nn['pi95'][0]:.3f},{nn['pi95'][1]:.3f}]")
sw = results["posthoc"]["probe_cv_seed_sweep"]
print(f"post-hoc probe seed sweep: mean {sw['mean']:+.3f}, range [{sw['min']:+.3f},{sw['max']:+.3f}]")
print(f"descriptives: {results['posthoc']['descriptives']}")
