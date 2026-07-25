#!/usr/bin/env python3
"""
r4 PART 2 -- portfolio hygiene, ANALYSIS-ONLY on saved artifacts. No training, no writes outside
this directory. Items fixed in PREREG.md Part 2 before any of this ran.

H1  two-sided selection constant b_k = E[max_j |N_j|]; recompute r3-selection-inflation-curve's
    R_gap (a max-of-|.| statistic that used the one-sided a_k = 1.03 at k=4).
H2  misgen-connectivity-graph max-over-theta modularity under three nested theta families.
H3  routing-hybrid-knob T3 demotion at p_waste = 0.5 (quality axis reliability + paired bounds).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json, math, time
import numpy as np
from scipy import stats, integrate

T0 = time.time()
ROOT = "/Users/steven/research_sprint_20260724"
OUT = f"{ROOT}/r4-kway-learned-plus-hygiene/hygiene_results.json"
RES = {}

# --------------------------------------------------------------- H1: two-sided constant
def a_blom(k):
    """E[max of k iid N(0,1)] -- Blom approximation (what r3 used)."""
    return float(stats.norm.ppf((k - 0.375) / (k + 0.25)))


def a_exact(k):
    """E[max of k iid N(0,1)] by quadrature: int_0^inf (1-Phi(x)^k) dx - int_{-inf}^0 Phi(x)^k dx."""
    f1 = integrate.quad(lambda x: 1 - stats.norm.cdf(x) ** k, 0, 40, limit=200)[0]
    f2 = integrate.quad(lambda x: stats.norm.cdf(x) ** k, -40, 0, limit=200)[0]
    return float(f1 - f2)


def b_exact(k):
    """E[max of k iid |N(0,1)|] by quadrature: int_0^inf (1-(2Phi(x)-1)^k) dx."""
    return float(integrate.quad(lambda x: 1 - (2 * stats.norm.cdf(x) - 1) ** k, 0, 40, limit=200)[0])


def b_blom(k):
    """Blom-style plug-in for the max of |N|: Phi^-1((1+q)/2), q = (k-0.375)/(k+0.25)."""
    q = (k - 0.375) / (k + 0.25)
    return float(stats.norm.ppf((1 + q) / 2))


rng = np.random.default_rng(20260725)
MC = 400000
tab = []
for k in [2, 3, 4, 5, 6, 8, 10, 16, 24, 25, 50, 144, 1770]:
    z = rng.standard_normal((MC, k)) if k <= 50 else rng.standard_normal((60000, k))
    tab.append(dict(k=k, a_k_blom=round(a_blom(k), 4), a_k_exact=round(a_exact(k), 4),
                    b_k_exact=round(b_exact(k), 4), b_k_blom=round(b_blom(k), 4),
                    b_k_mc=round(float(np.abs(z).max(1).mean()), 4),
                    ratio_b_over_a=round(b_exact(k) / a_exact(k), 4)))
    del z
RES["H1_constants"] = dict(
    table=tab,
    note="a_k = E[max_j N_j] is correct only when the selection DIRECTION is pre-specified. "
         "For a statistic selected as argmax_j |.| the null constant is b_k = E[max_j |N_j|].",
    mc_check_max_abs_err=round(max(abs(r["b_k_exact"] - r["b_k_mc"]) for r in tab), 4))
print(f"H1 b_4 exact = {b_exact(4):.4f} (a_4 = {a_exact(4):.4f}); "
      f"MC max err {RES['H1_constants']['mc_check_max_abs_err']}", flush=True)


def verdict(R):
    return "SUSPECT" if R < 1.5 else ("BORDERLINE" if R < 3 else "ROBUST")


infl = json.load(open(f"{ROOT}/r3-selection-inflation-curve/results.json"))
hk = infl["partC"]["routing_hybrid_knob"]
b4 = b_exact(4)
gap_rows = []
for pw, e in hk.items():
    for metric, d in e.items():
        R_old = d["R_gap"]; kg = d["k_gaps"]
        R_new = abs(d["max_gap"]) / (b4 * d["gap_SE"])
        gap_rows.append(dict(p_waste=float(pw), metric=metric, k_gaps=kg,
                             a_k_used_by_r3=round(d["a_k"] if False else a_blom(kg), 4),
                             b_k_correct=round(b4, 4),
                             max_gap=round(d["max_gap"], 5), gap_SE=round(d["gap_SE"], 5),
                             gap_pair=d["max_gap_alpha_pair"],
                             R_gap_published=round(R_old, 4), verdict_published=d["gap_verdict"],
                             R_gap_corrected=round(R_new, 4), verdict_corrected=verdict(R_new),
                             inflation_factor=round(R_old / R_new, 4),
                             verdict_changes=bool(verdict(R_new) != d["gap_verdict"])))
# cell-level R values are NOT affected: their selection direction was pre-specified one-sided
unaffected = []
for pw, e in hk.items():
    for metric, d in e.items():
        unaffected.append(dict(p_waste=float(pw), metric=metric, k=d["k"],
                               selection="argmin (lower AUC = better, pre-specified)" if metric == "recovery_auc"
                               else "argmax (higher retain = better, pre-specified)",
                               R=round(d["R"], 4), verdict=d["verdict"], affected_by_two_sided_fix=False))
RES["H1_recomputed_R_gap"] = dict(rows=gap_rows, one_sided_cell_R_unaffected=unaffected,
                                  n_verdicts_changed=sum(r["verdict_changes"] for r in gap_rows))
for r in gap_rows:
    print(f"  R_gap pw={r['p_waste']} {r['metric']}: {r['R_gap_published']} {r['verdict_published']}"
          f" -> {r['R_gap_corrected']} {r['verdict_corrected']}", flush=True)

# --------------------------------------------------------------- H2: modularity families
mf = json.load(open("/Users/steven/moral_framework_misgen/stats/results.json"))
B = np.array(mf["basins"]["barrier_matrix"])
scores = np.array(mf["multimodality"]["scores"][:B.shape[0]])
lab = np.where(scores > 0.8, 0, np.where(scores < 0.2, 1, 2))
nn = B.shape[0]
iu = np.triu_indices(nn, 1)
bvals = B[iu]; ordr = np.argsort(bvals, kind="mergesort")
eu, ev = iu[0][ordr], iu[1][ordr]
thetas = bvals[ordr]; E = len(thetas); mgrid = np.arange(1, E + 1)


def Q_curve(labels):
    lu, lv = labels[eu], labels[ev]
    q = np.zeros(E)
    for c in range(3):
        w = np.cumsum(((lu == lv) & (lu == c)).astype(float))
        dg = np.cumsum((lu == c).astype(float) + (lv == c).astype(float))
        q += w / mgrid - (dg / (2 * mgrid)) ** 2
    return q


Qobs = Q_curve(lab)
saved = np.array(json.load(open(f"{ROOT}/misgen-connectivity-graph/results.json"))["curves"]["modularity_Q"])
sanity = float(np.max(np.abs(Qobs - saved)))

FAM = {"all_theta": np.ones(E, dtype=bool),
       "published_window_0.0015_0.090": (thetas >= 0.0015) & (thetas <= 0.090),
       "m_ge_60_edges": mgrid >= 60}
NPERM = 1000
rng3 = np.random.default_rng(0)
null = {f: np.empty(NPERM) for f in FAM}
for b in range(NPERM):
    qq = Q_curve(rng3.permutation(lab))
    for f, m in FAM.items():
        null[f][b] = qq[m].max()
fam_rows = []
for f, m in FAM.items():
    qo = float(Qobs[m].max()); th = float(thetas[m][int(np.argmax(Qobs[m]))])
    p = float((1 + np.sum(null[f] >= qo)) / (1 + NPERM))
    fam_rows.append(dict(family=f, n_thetas=int(m.sum()), Q_max_obs=round(qo, 4),
                         theta_at_max=round(th, 6), null_mean=round(float(null[f].mean()), 4),
                         null_p95=round(float(np.percentile(null[f], 95)), 4),
                         p_family_wise=round(p, 4),
                         sig_at_005=bool(p < 0.05)))
RES["H2_modularity_families"] = dict(sanity_max_abs_diff_vs_saved_curve=sanity, n_perm=NPERM,
                                     seed=0, rows=fam_rows,
                                     immune_note="the pre-stated PRIMARY of misgen-connectivity-graph "
                                                 "is the threshold-free rank AUC = 0.725 (node-label "
                                                 "permutation p <= 1e-4). It involves no max over "
                                                 "theta and is IMMUNE to this correction.")
for r in fam_rows:
    print(f"H2 {r['family']:32s} k={r['n_thetas']:5d} Qmax={r['Q_max_obs']} p={r['p_family_wise']}", flush=True)

# --------------------------------------------------------------- H3: hybrid-knob T3 demotion
hy = json.load(open(f"{ROOT}/routing-hybrid-knob/results.json"))
runs = hy["runs"]; ALPH = hy["config"]["ALPHAS"]; SEEDS = hy["config"]["SEEDS"]
PW = 0.5
ret = np.array([[next(r["retain_pruned"] for r in runs if r["alpha"] == a and r["p_waste"] == PW
                      and r["seed"] == sd) for a in ALPH] for sd in SEEDS])          # 6 x 5
auc = np.array([[next(r["auc"] for r in runs if r["alpha"] == a and r["p_waste"] == PW
                      and r["seed"] == sd) for a in ALPH] for sd in SEEDS])
cell = ret.mean(0)
s_cell = float(ret.std(0, ddof=1).mean() / math.sqrt(len(SEEDS)))
S2 = float(cell.var(ddof=1))
rho_hat = max(0.0, 1.0 - s_cell ** 2 / S2)
Dbar = float(cell.mean())
R_cell = abs(cell[int(np.argmax(cell))] - Dbar) / (a_exact(5) * s_cell)

# T3 requires an interior alpha STRICTLY BETTER THAN AN ENDPOINT ON EACH AXIS.
# quality axis: the only endpoint an interior alpha can beat is alpha = 1 (0.9429).
contr = []
for i, a in enumerate(ALPH):
    if a in (0.0, 1.0): continue
    d = ret[:, i] - ret[:, ALPH.index(1.0)]                       # per-seed paired difference
    tci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if d.std(ddof=1) > 0 else (np.nan, np.nan)
    w = stats.wilcoxon(d, alternative="two-sided", method="exact") if np.any(d != 0) else None
    contr.append(dict(alpha=a, mean_diff_vs_alpha1=round(float(d.mean()), 5),
                      ci95=[round(float(tci[0]), 5), round(float(tci[1]), 5)],
                      n=len(d), n_positive=int((d > 0).sum()),
                      wilcoxon_p=round(float(w.pvalue), 4) if w is not None else None,
                      excludes_zero=bool(tci[0] > 0 or tci[1] < 0)))
# resistance-axis leg of the same T3 criterion: interior must beat the alpha=0 endpoint on AUC
contr_auc = []
for i, a in enumerate(ALPH):
    if a in (0.0, 1.0): continue
    d = auc[:, ALPH.index(0.0)] - auc[:, i]                        # positive = interior more resistant
    tci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    contr_auc.append(dict(alpha=a, mean_resistance_gain_vs_alpha0=round(float(d.mean()), 5),
                          ci95=[round(float(tci[0]), 5), round(float(tci[1]), 5)],
                          n_positive=int((d > 0).sum()), n=len(d),
                          excludes_zero=bool(tci[0] > 0 or tci[1] < 0)))

# selection over the 3 interior alphas, one-sided (best interior), so a_3 is the right constant
a3 = a_exact(3)
best = max(contr, key=lambda c: c["mean_diff_vs_alpha1"])
sd_diff = float(np.std(ret[:, ALPH.index(best["alpha"])] - ret[:, ALPH.index(1.0)], ddof=1))
se_diff = sd_diff / math.sqrt(len(SEEDS))
R_T3 = abs(best["mean_diff_vs_alpha1"]) / (a3 * se_diff)
equiv = max(abs(best["ci95"][0]), abs(best["ci95"][1]))

# ---- POST-HOC (not named in PREREG): the SE r3's own Part B prescribes is the INTERACTION SE.
# Part C used the MARGINAL across-seed SE. In a fully paired design (same 6 seeds at every alpha)
# with a large seed main effect, the marginal SE overstates the per-cell error and drives rho_hat
# to 0 artifactually. Recompute every Part-C hybrid cell both ways.
def two_way(X):
    cellm = X.mean(0); S2_ = float(cellm.var(ddof=1))
    s_marg = float(X.std(0, ddof=1).mean() / math.sqrt(X.shape[0]))
    Rres = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    dfr = (X.shape[0] - 1) * (X.shape[1] - 1)
    s_int = float(math.sqrt((Rres ** 2).sum() / dfr) / math.sqrt(X.shape[0]))
    ss_seed = float(((X.mean(1) - X.mean()) ** 2).sum() * X.shape[1])
    return dict(cell_means=[round(x, 5) for x in cellm], S=round(math.sqrt(S2_), 5),
                s_marginal=round(s_marg, 5), s_interaction=round(s_int, 5),
                rho_hat_marginal=round(max(0.0, 1 - s_marg ** 2 / S2_), 4),
                rho_hat_interaction=round(max(0.0, 1 - s_int ** 2 / S2_), 4),
                seed_main_effect_SS_share=round(ss_seed / float(((X - X.mean()) ** 2).sum()), 4))

a5 = a_exact(5)
se_rows = []
for pw in hy["config"]["P_WASTES"]:
    for name in ["retain_pruned", "auc"]:
        X = np.array([[next(r[name] for r in runs if r["alpha"] == a and r["p_waste"] == pw
                            and r["seed"] == sd) for a in ALPH] for sd in SEEDS])
        t = two_way(X); cm = np.array(t["cell_means"])
        j = int(np.argmin(cm)) if name == "auc" else int(np.argmax(cm))
        exc = abs(cm[j] - cm.mean())
        t.update(p_waste=pw, metric=name, sel_alpha=ALPH[j],
                 R_marginal=round(exc / (a5 * t["s_marginal"]), 3),
                 R_interaction=round(exc / (a5 * t["s_interaction"]), 3))
        t["verdict_marginal"] = verdict(t["R_marginal"]); t["verdict_interaction"] = verdict(t["R_interaction"])
        t["verdict_changes"] = bool(t["verdict_marginal"] != t["verdict_interaction"])
        se_rows.append(t)
RES["H4_POSTHOC_marginal_vs_interaction_SE"] = dict(
    rows=se_rows,
    note="POST-HOC (not in PREREG). r3-selection-inflation-curve PREREG.md Part B requires the "
         "INTERACTION SE ('Use the interaction (item x cell) SE, not the marginal one'), but its "
         "Part C computed s_cell = mean_j SD_seeds(X[:,j]) / sqrt(n), the MARGINAL SE. The hybrid-knob "
         "design is fully paired across alphas, so any seed main effect inflates the marginal SE.")

RES["H3_hybrid_T3"] = dict(
    p_waste=PW, alphas=ALPH, n_seeds=len(SEEDS),
    quality_axis=dict(cell_means=[round(x, 5) for x in cell], per_cell_SE_marginal=round(s_cell, 5),
                      S=round(math.sqrt(S2), 5), rho_hat_marginal=round(rho_hat, 5), Dbar=round(Dbar, 5),
                      R_winner_marginal=round(float(R_cell), 4), verdict_marginal=verdict(float(R_cell)),
                      rho_hat_interaction=se_rows[0]["rho_hat_interaction"],
                      R_winner_interaction=se_rows[0]["R_interaction"],
                      verdict_interaction=se_rows[0]["verdict_interaction"],
                      seed_main_effect_SS_share=se_rows[0]["seed_main_effect_SS_share"]),
    resistance_axis=dict(cell_means=[round(x, 5) for x in auc.mean(0)],
                         rho_hat_marginal=se_rows[1]["rho_hat_marginal"],
                         rho_hat_interaction=se_rows[1]["rho_hat_interaction"],
                         note="the resistance axis is reliable either way (rho_hat 0.91 / 0.95)"),
    T3_leg_quality_interior_vs_alpha1=contr,
    T3_leg_resistance_interior_vs_alpha0=contr_auc,
    best_interior=best["alpha"], a_3=round(a3, 4), R_T3_paired=round(float(R_T3), 4),
    largest_abs_retain_effect_excluded_by_paired_CI=round(float(equiv), 5),
    PRESTATED_RULE_FIRES=bool(rho_hat == 0.0),
    VERDICT="DEMOTION-AS-STATED NOT SUPPORTED; re-word on magnitude instead",
    reason=("My PREREG's demotion rule ('if rho_hat = 0 on the quality axis, T3 is unsupported') DOES "
            "fire on the published number -- but the published rho_hat = 0 is an artifact. It comes "
            "from the MARGINAL across-seed SE in a fully paired design where the seed main effect is "
            "72.8% of the total SS. With the INTERACTION SE that r3's own Part B prescribes, "
            "rho_hat = 0.68 and R = 2.50 (BORDERLINE, not SUSPECT), and the paired contrast the T3 "
            "criterion actually requires -- interior alpha=0.5 vs the alpha=1 endpoint on retain -- is "
            "+0.0024 [0.0008, 0.0040], 6/6 seeds, with R_T3 = 4.5 after correcting for selecting the "
            "best of 3 interior alphas. T3's statistical criterion is therefore MET at p_waste=0.5. "
            "What should be demoted is the practical reading, not the test: the interior gain over the "
            "alpha=1 endpoint is 0.24 retain points on an axis whose entire span is 0.6 points, so "
            "'Pareto-useful' should be restated as 'statistically resolvable but practically "
            "negligible; take alpha = 1', which is what routing-hybrid-knob/FINDINGS.md already says "
            "in prose."))
print(f"H3 quality-axis rho_hat marginal={rho_hat:.4f} interaction={se_rows[0]['rho_hat_interaction']}; "
      f"best interior a={best['alpha']} diff={best['mean_diff_vs_alpha1']} CI={best['ci95']} "
      f"R_T3={R_T3:.2f}", flush=True)

RES["runtime_sec"] = round(time.time() - T0, 1)
json.dump(RES, open(OUT, "w"), indent=1)
print(f"wrote {OUT} ({RES['runtime_sec']}s)")

# --------------------------------------------------------------- H5: routing-lora-attack "attacker-best"
# In scope of PREREG item 4 (selection audit for every headline number); NOT one of r3's three named
# Part-C targets, so labeled as an extension. The published headline "attacker-best -0.066" takes the
# MAX recovery-AUC cell over the 6-cell LoRA grid, separately for routed and baseline -- a one-sided
# max-over-k=6 selection (the attacker deliberately maximises), so a_6 is the right constant.
la = json.load(open(f"{ROOT}/routing-lora-attack/results.json"))
CELLS = [f"{p}_lora_r{r}_lr{lr}" for r in (1, 4, 16) for lr in (0.001, 0.01) for p in ("routed",)]
BCELLS = [c.replace("routed_", "baseline_") for c in CELLS]
h5 = {}
for arm, cl in (("routed", CELLS), ("baseline", BCELLS)):
    X = np.array([[s["attacks"][c]["auc"] for c in cl] for s in la["per_seed"]])   # 6 seeds x 6 cells
    t = two_way(X); cm = np.array(t["cell_means"]); j = int(np.argmax(cm))
    a6 = a_exact(len(cl)); exc = abs(cm[j] - cm.mean())
    t.update(arm=arm, k=len(cl), n=X.shape[0], sel_cell=cl[j], a_k=round(a6, 4),
             R_marginal=round(exc / (a6 * t["s_marginal"]), 3),
             R_interaction=round(exc / (a6 * t["s_interaction"]), 3))
    t["verdict_interaction"] = verdict(t["R_interaction"])
    h5[arm] = t
RES["H5_lora_attacker_best_selection_audit"] = dict(
    arms=h5, published_mean_diff=la["posthoc_attacker_best"]["mean_diff"],
    published_ci95=la["posthoc_attacker_best"]["ci95"],
    best_routed_cell=la["posthoc_attacker_best"]["best_routed_cell"],
    best_baseline_cell=la["posthoc_attacker_best"]["best_baseline_cell"],
    note="EXTENSION, in scope of PREREG item 4 but not named as a target in advance.")
print("H5", {a: (v["sel_cell"], v["R_interaction"], v["verdict_interaction"], v["rho_hat_interaction"]) for a, v in h5.items()}, flush=True)
json.dump(RES, open(OUT, "w"), indent=1)
