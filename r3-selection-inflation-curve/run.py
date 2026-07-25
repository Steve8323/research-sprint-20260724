#!/usr/bin/env python3
"""
r3-selection-inflation-curve — How much does a sweep inflate the winner? A correction rule.

Part A  derivation + simulation of max-of-k inflation (CONFIRMATORY, criteria P1/P2 in PREREG.md)
Part B  validation against the real ~/misalignment_noise_probe 144-cell sweep (V1..V4)
Part C  application to other selected-from-sweep portfolio numbers

Pure numpy + scipy on saved artifacts. Single process, no model loads, no downloads.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json, time, math, itertools
import numpy as np
from scipy import stats

T0 = time.time()
ROOT = "/Users/steven/research_sprint_20260724"
OUT = os.path.join(ROOT, "r3-selection-inflation-curve")
RES = {}

# ---------------------------------------------------------------- helpers
def blom_a(k):
    """E[max of k iid N(0,1)], Blom approximation."""
    return stats.norm.ppf((k - 0.375) / (k + 0.25))

def mc_a(k, R=200000, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((R, k)).max(axis=1).mean()

def rho_hat_mom(D, s2):
    """method-of-moments reliability from a sweep of cell means D and known per-cell error var s2"""
    S2 = D.var(ddof=1)
    return max(0.0, 1.0 - s2 / S2) if S2 > 0 else 0.0

# ================================================================ PART A
print("=== PART A: simulation of max-of-k inflation ===", flush=True)
KS   = [5, 10, 24, 50, 144]
NS   = [6, 12, 32, 100]
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9]
DISTS = ["normal", "t5"]
CORRS = [0.0, 0.5]
REPS = 2000
rng = np.random.default_rng(20260725)

# pools of cell-mean draws for the t5 arm (mean of n iid t5 scaled to unit variance)
POOL = {}
POOL_N = 200000
for n in NS:
    t = rng.standard_t(df=5, size=(POOL_N, n)) / math.sqrt(5 / 3.0)   # var(t5)=5/3 -> unit var
    POOL[n] = t.mean(axis=1)                                          # SD = 1/sqrt(n) exactly in expectation

partA_rows = []
for k, n, rho, dist, r in itertools.product(KS, NS, RHOS, DISTS, CORRS):
    s = 1.0 / math.sqrt(n)                       # sigma_w = 1
    tau = s * math.sqrt(rho / (1 - rho)) if rho > 0 else 0.0
    theta = rng.standard_normal((REPS, k)) * tau
    if dist == "normal":
        u = rng.standard_normal((REPS, k)) * s
    else:
        u = rng.choice(POOL[n], size=(REPS, k), replace=True)
    if r > 0:
        c = rng.standard_normal((REPS, 1)) * s
        e = math.sqrt(1 - r) * u + math.sqrt(r) * c
    else:
        e = u
    D = theta + e
    jstar = D.argmax(axis=1)
    idx = np.arange(REPS)
    D_sel, th_sel = D[idx, jstar], theta[idx, jstar]

    s_int2 = (1 - r) * s ** 2                    # interaction variance (what an analyst estimates)
    Dbar = D.mean(axis=1)
    S2 = D.var(axis=1, ddof=1)
    rh = np.maximum(0.0, 1.0 - s_int2 / S2)
    EB = Dbar + rh * (D_sel - Dbar)
    # variant that (wrongly) uses the marginal SE instead of the interaction SE
    rh_marg = np.maximum(0.0, 1.0 - (s ** 2) / S2)
    EB_marg = Dbar + rh_marg * (D_sel - Dbar)

    naive_err = D_sel - th_sel
    eb_err = EB - th_sel
    ebm_err = EB_marg - th_sel
    ak = blom_a(k)
    pred_inflation = ak * s * math.sqrt(1 - rho)
    partA_rows.append(dict(
        k=k, n=n, rho=rho, dist=dist, corr=r, s=s,
        bias_naive=float(naive_err.mean()), bias_naive_se=float(naive_err.std(ddof=1) / math.sqrt(REPS)),
        pred_inflation=float(pred_inflation),
        rel_err_closed_form=float(abs(naive_err.mean() - pred_inflation) / pred_inflation) if pred_inflation > 0 else float("nan"),
        bias_eb=float(eb_err.mean()), bias_eb_over_s=float(eb_err.mean() / s),
        rmse_naive=float(np.sqrt((naive_err ** 2).mean())), rmse_eb=float(np.sqrt((eb_err ** 2).mean())),
        rmse_eb_marginalSE=float(np.sqrt((ebm_err ** 2).mean())),
        rho_hat_mean=float(rh.mean()),
    ))

sub = [r_ for r_ in partA_rows if r_["dist"] == "normal" and r_["corr"] == 0.0]
p1_ok = sum(1 for r_ in sub if r_["rel_err_closed_form"] <= 0.15)
P1 = dict(n_cells=len(sub), n_within_15pct=p1_ok, criterion="≥90% of 100 cells within 15%",
          pass_=bool(p1_ok >= 0.9 * len(sub)),
          worst=sorted(sub, key=lambda r_: -r_["rel_err_closed_form"])[:3])
n_bias_ok = sum(1 for r_ in partA_rows if abs(r_["bias_eb_over_s"]) <= 0.25)
n_rmse_ok = sum(1 for r_ in partA_rows if r_["rmse_eb"] <= r_["rmse_naive"])
P2 = dict(n_cells=len(partA_rows), n_bias_ok=n_bias_ok, n_rmse_ok=n_rmse_ok,
          frac_bias_ok=n_bias_ok / len(partA_rows), frac_rmse_ok=n_rmse_ok / len(partA_rows),
          criterion="both ≥90% of 400 cells",
          pass_=bool(n_bias_ok >= 0.9 * len(partA_rows) and n_rmse_ok >= 0.9 * len(partA_rows)),
          worst_bias=sorted(partA_rows, key=lambda r_: -abs(r_["bias_eb_over_s"]))[:5],
          worst_rmse=sorted(partA_rows, key=lambda r_: -(r_["rmse_eb"] - r_["rmse_naive"]))[:5])
print(f"P1 pass={P1['pass_']} ({p1_ok}/{len(sub)});  P2 pass={P2['pass_']} bias {n_bias_ok}/400 rmse {n_rmse_ok}/400", flush=True)

# a_k table (Blom vs MC) + the null-winner rule of thumb
ak_table = []
for k in [2, 3, 4, 5, 8, 10, 16, 24, 25, 50, 100, 144, 500, 1770]:
    ak_table.append(dict(k=k, a_k_blom=float(blom_a(k)), a_k_mc=float(mc_a(k, R=100000, seed=k))))
RES["partA"] = dict(grid=partA_rows, P1=P1, P2=P2, a_k_table=ak_table, reps=REPS,
                    rule="theta_hat = Dbar + rho_hat*(D_sel - Dbar); rho_hat = max(0, 1 - s^2/S^2); "
                         "expected inflation = a_k * s * sqrt(1-rho), s = sigma_w/sqrt(n)")

# ================================================================ PART B
print("=== PART B: validation on the noise-probe 144-cell sweep ===", flush=True)
sweep = json.load(open("/Users/steven/misalignment_noise_probe/layer_importance_results.json"))
base02 = np.array(sweep["base"]["grid"]["0.2"]); mis02 = np.array(sweep["misaligned"]["grid"]["0.2"])
d_orig = mis02 - base02                                   # 24 layers, sigma=0.2 differential
K_SWEEP, N_SWEEP = len(d_orig), 6

r2 = json.load(open(os.path.join(ROOT, "r2-fragility-prereg/results.json")))
LAYERS8 = [2, 4, 6, 8, 11, 12, 16, 20]
def d_matrix(records, layers):
    M = np.zeros((len(records), len(layers)))
    for i, rec in enumerate(records):
        for j, L in enumerate(layers):
            nz = rec["noised"][str(L)]
            M[i, j] = (np.mean(nz["misaligned"]) - rec["clean"]["misaligned"]) - \
                      (np.mean(nz["base"]) - rec["clean"]["base"])
    return M
Mp = d_matrix(r2["per_item_primary"], [2, 12, 16, 20])
Mc = d_matrix(r2["per_item_control"], [4, 6, 8, 11])
order = {L: i for i, L in enumerate([2, 12, 16, 20])}
orderc = {L: i for i, L in enumerate([4, 6, 8, 11])}
M8 = np.column_stack([Mp[:, order[L]] if L in order else Mc[:, orderc[L]] for L in LAYERS8])  # 32 x 8
assert M8.shape == (32, 8)

# variance-only: two-way (prompt x layer) ANOVA residual SD
resid = M8 - M8.mean(axis=1, keepdims=True) - M8.mean(axis=0, keepdims=True) + M8.mean()
dof = (M8.shape[0] - 1) * (M8.shape[1] - 1)
sigma_w = float(np.sqrt((resid ** 2).sum() / dof))
s_sweep = sigma_w / math.sqrt(N_SWEEP)
Dbar = float(d_orig.mean()); S2 = float(d_orig.var(ddof=1))
rho_hat = rho_hat_mom(d_orig, s_sweep ** 2)
fresh = M8.mean(axis=0)
fresh_se = M8.std(axis=0, ddof=1) / math.sqrt(M8.shape[0])
print(f"sigma_w={sigma_w:.3f}  s={s_sweep:.3f}  Dbar={Dbar:.3f}  S={math.sqrt(S2):.3f}  rho_hat={rho_hat:.3f}", flush=True)

naive8 = np.array([d_orig[L] for L in LAYERS8])
EB8 = Dbar + rho_hat * (naive8 - Dbar)

# V1
c_sel = float(np.mean([d_orig[4], d_orig[8], d_orig[11]]) - d_orig[6])
c_pred = rho_hat * c_sel
FRESH_C_CI = (-0.72, 1.16); FRESH_C_OBS = 0.22       # r1, n=24 fresh prompts (published)
# same contrast recomputed on the r2 32-prompt fresh data (pre-stated as the same estimand)
c_items_r2 = (Mc[:, orderc[4]] + Mc[:, orderc[8]] + Mc[:, orderc[11]]) / 3 - Mc[:, orderc[6]]
c_r2 = float(c_items_r2.mean()); c_r2_ci = stats.t.interval(0.95, len(c_items_r2) - 1,
                                                            loc=c_r2, scale=stats.sem(c_items_r2))
V1 = dict(c_selected_original=c_sel, rho_hat=rho_hat, prediction=float(c_pred),
          fresh_r1_obs=FRESH_C_OBS, fresh_r1_ci=list(FRESH_C_CI),
          pred_in_CI=bool(FRESH_C_CI[0] <= c_pred <= FRESH_C_CI[1]),
          naive_in_CI=bool(FRESH_C_CI[0] <= c_sel <= FRESH_C_CI[1]),
          frac_shrinkage_explained=float((c_sel - c_pred) / (c_sel - FRESH_C_OBS)),
          c_on_r2_fresh32=c_r2, c_on_r2_fresh32_ci=[float(x) for x in c_r2_ci],
          pred_in_r2_CI=bool(c_r2_ci[0] <= c_pred <= c_r2_ci[1]))
V1["pass_"] = bool(V1["pred_in_CI"] and not V1["naive_in_CI"])

# V2 / V2b
err_eb = np.abs(EB8 - fresh); err_naive = np.abs(naive8 - fresh)
w = stats.wilcoxon(err_eb, err_naive, alternative="less", method="exact")
off_eb = (EB8 - fresh).mean(); off_nv = (naive8 - fresh).mean()
err_eb_b = np.abs(EB8 - off_eb - fresh); err_naive_b = np.abs(naive8 - off_nv - fresh)
wb = stats.wilcoxon(err_eb_b, err_naive_b, alternative="less", method="exact")
V2 = dict(layers=LAYERS8, naive=naive8.tolist(), EB=EB8.tolist(), fresh=fresh.tolist(),
          fresh_se=fresh_se.tolist(), err_naive=err_naive.tolist(), err_eb=err_eb.tolist(),
          W=float(w.statistic), p=float(w.pvalue),
          rmse_naive=float(np.sqrt((err_naive ** 2).mean())), rmse_eb=float(np.sqrt((err_eb ** 2).mean())),
          pass_=bool(w.pvalue < 0.05))
V2b = dict(offset_eb=float(off_eb), offset_naive=float(off_nv), W=float(wb.statistic), p=float(wb.pvalue),
           rmse_naive=float(np.sqrt((err_naive_b ** 2).mean())), rmse_eb=float(np.sqrt((err_eb_b ** 2).mean())))

# V3 over-shrinkage on never-selected layers
unsel = [i for i, L in enumerate(LAYERS8) if L in (2, 12, 16, 20)]
sgn_eb = (EB8 - fresh)[unsel]; sgn_nv = (naive8 - fresh)[unsel]
t3 = stats.ttest_1samp(sgn_eb, 0.0)
V3 = dict(layers=[LAYERS8[i] for i in unsel], signed_err_eb=sgn_eb.tolist(),
          signed_err_naive=sgn_nv.tolist(), mean_signed_eb=float(sgn_eb.mean()),
          mean_signed_naive=float(sgn_nv.mean()), t=float(t3.statistic), p=float(t3.pvalue))

# V4 unselected pooled quantity replicates without correction
V4 = dict(sweep_grand_mean=Dbar, fresh_pooled_mean=-0.6424701809883118,
          fresh_pooled_ci=[-0.9177323892716696, -0.3672079727049539],
          in_CI=bool(-0.9177323892716696 <= Dbar <= -0.3672079727049539),
          fresh_pooled_8layer_mean=float(fresh.mean()))

# selection-inflation ratio for the original winner
a24 = blom_a(24)
win_idx = int(np.argmin(d_orig))
R_win = abs(d_orig[win_idx] - Dbar) / (a24 * s_sweep)
RES["partB"] = dict(sweep_d_orig=d_orig.tolist(), k=K_SWEEP, n=N_SWEEP, sigma_w=sigma_w,
                    s=s_sweep, Dbar=Dbar, S=math.sqrt(S2), rho_hat=rho_hat,
                    a_k=float(a24), winner_layer=win_idx, winner_value=float(d_orig[win_idx]),
                    R_winner=float(R_win), null_winner_bound=float(a24 * s_sweep),
                    V1=V1, V2=V2, V2b=V2b, V3=V3, V4=V4)
print(f"V1 pass={V1['pass_']} pred={c_pred:.3f};  V2 pass={V2['pass_']} p={w.pvalue:.4f};  "
      f"V4 in_CI={V4['in_CI']} (Dbar={Dbar:.3f})", flush=True)

# ================================================================ PART C
print("=== PART C: apply the rule to other portfolio numbers ===", flush=True)
partC = {}

def verdict(R):
    return "SUSPECT" if R < 1.5 else ("BORDERLINE" if R < 3 else "ROBUST")

# ---- C1 routing-hybrid-knob
hy = json.load(open(os.path.join(ROOT, "routing-hybrid-knob/results.json")))
runs = hy["runs"]; ALPH = hy["config"]["ALPHAS"]; SEEDS = hy["config"]["SEEDS"]
c1 = {}
for pw in hy["config"]["P_WASTES"]:
    auc = np.array([[next(r_["auc"] for r_ in runs if r_["alpha"] == a and r_["p_waste"] == pw and r_["seed"] == sd)
                     for a in ALPH] for sd in SEEDS])          # 6 seeds x 5 alphas
    ret = np.array([[next(r_["retain_pruned"] for r_ in runs if r_["alpha"] == a and r_["p_waste"] == pw and r_["seed"] == sd)
                     for a in ALPH] for sd in SEEDS])
    entry = {}
    for name, X in (("recovery_auc", auc), ("retain_pruned", ret)):
        cell = X.mean(axis=0)
        s_cell = float(X.std(axis=0, ddof=1).mean() / math.sqrt(len(SEEDS)))   # pooled per-cell SE
        Db, S2c = float(cell.mean()), float(cell.var(ddof=1))
        rh = rho_hat_mom(cell, s_cell ** 2)
        k = len(ALPH); ak = blom_a(k)
        j = int(np.argmin(cell)) if name == "recovery_auc" else int(np.argmax(cell))
        R = abs(cell[j] - Db) / (ak * s_cell)
        # max adjacent gap (paired) — the "all the effect arrives at alpha=1" claim
        gaps = X[:, :-1] - X[:, 1:]                        # per-seed adjacent differences
        gm = gaps.mean(axis=0); gse = gaps.std(axis=0, ddof=1) / math.sqrt(len(SEEDS))
        jg = int(np.argmax(np.abs(gm))); akg = blom_a(gaps.shape[1])
        Rg = abs(gm[jg]) / (akg * float(gse.mean()))
        entry[name] = dict(cell_means=cell.tolist(), per_cell_SE=s_cell, k=k, a_k=float(ak),
                           Dbar=Db, S=math.sqrt(S2c), rho_hat=rh, sel_alpha=ALPH[j],
                           excess=float(cell[j] - Db), null_bound=float(ak * s_cell),
                           R=float(R), verdict=verdict(R),
                           deflated_excess=float(rh * (cell[j] - Db)),
                           max_gap_alpha_pair=[ALPH[jg], ALPH[jg + 1]], max_gap=float(gm[jg]),
                           gap_SE=float(gse.mean()), R_gap=float(Rg), gap_verdict=verdict(Rg),
                           k_gaps=int(gaps.shape[1]))
    c1[str(pw)] = entry
partC["routing_hybrid_knob"] = c1
for pw, e in c1.items():
    print(f"  hybrid p_waste={pw} AUC: R={e['recovery_auc']['R']:.2f} {e['recovery_auc']['verdict']}; "
          f"max-gap R={e['recovery_auc']['R_gap']:.2f} {e['recovery_auc']['gap_verdict']}", flush=True)

# ---- C2 misgen-bimodality-honest
bi = json.load(open(os.path.join(ROOT, "misgen-bimodality-honest/results.json")))
regs = bi["regimes"]
import diptest                      # same library the survey used (survey.py line 22)
def dip_stat(x):
    return float(diptest.dipstat(np.asarray(x, dtype=np.float64)))

rng2 = np.random.default_rng(4242)
c2_cells = []
for r_ in regs:
    sc = np.array(r_["scores"])
    boots = np.array([dip_stat(rng2.choice(sc, size=len(sc), replace=True)) for _ in range(200)])
    c2_cells.append(dict(cost=r_["cost"], width=r_["width"], dip=r_["dip"], p_dip=r_["p_dip"],
                         dip_recomputed=dip_stat(sc), dip_boot_se=float(boots.std(ddof=1)),
                         holm=r_["holm_reject"], underfit=r_["underfit"]))
dips = np.array([c["dip"] for c in c2_cells])
s_dip = float(np.mean([c["dip_boot_se"] for c in c2_cells]))
Db2, S22 = float(dips.mean()), float(dips.var(ddof=1))
rh2 = rho_hat_mom(dips, s_dip ** 2)
ak25 = blom_a(25); j2 = int(np.argmax(dips))
R2 = abs(dips[j2] - Db2) / (ak25 * s_dip)
# width-only selection at equal cost c=1.0 (the parent repo's search)
eq = [c for c in c2_cells if c["cost"] == 1.0]
deq = np.array([c["dip"] for c in eq]); s_eq = float(np.mean([c["dip_boot_se"] for c in eq]))
ak5 = blom_a(5); j5 = int(np.argmax(deq)); Db5 = float(deq.mean())
R5 = abs(deq[j5] - Db5) / (ak5 * s_eq)
partC["misgen_bimodality"] = dict(
    cells=c2_cells, k_full=25, a_k_full=float(ak25), s_dip=s_dip, Dbar=Db2, S=math.sqrt(S22),
    rho_hat=rh2, winner=dict(cost=c2_cells[j2]["cost"], width=c2_cells[j2]["width"], dip=float(dips[j2])),
    R_full=float(R2), verdict_full=verdict(R2),
    width_search_equal_cost=dict(k=5, a_k=float(ak5), dips=deq.tolist(), s=s_eq, Dbar=Db5,
                                 winner_width=eq[j5]["width"], R=float(R5), verdict=verdict(R5)),
    note="Holm corrects p-values across the 25 regimes; it does NOT correct the reported effect "
         "sizes (dip statistics) of the selected cells.")
print(f"  bimodality: full-grid R={R2:.2f} {verdict(R2)}; width-search R={R5:.2f} {verdict(R5)}", flush=True)

# ---- C3 misgen-connectivity-graph: max-over-theta null for modularity Q
mf = json.load(open("/Users/steven/moral_framework_misgen/stats/results.json"))
B = np.array(mf["basins"]["barrier_matrix"]); scores = np.array(mf["multimodality"]["scores"][:B.shape[0]])
lab = np.where(scores > 0.8, 0, np.where(scores < 0.2, 1, 2))       # A=0, M=1, H=2
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
saved_Q = np.array(json.load(open(os.path.join(ROOT, "misgen-connectivity-graph/results.json")))["curves"]["modularity_Q"])
sanity = float(np.max(np.abs(Qobs - saved_Q)))
NPERM = 1000
rng3 = np.random.default_rng(0)
maxQ_null = np.empty(NPERM)
for b in range(NPERM):
    maxQ_null[b] = Q_curve(rng3.permutation(lab)).max()
Qmax_obs = float(Qobs.max()); theta_argmax = float(thetas[int(Qobs.argmax())])
p_max = float((1 + np.sum(maxQ_null >= Qmax_obs)) / (1 + NPERM))
partC["misgen_connectivity"] = dict(
    sanity_max_abs_diff_vs_saved_Q=sanity, k_thetas=E,
    Q_max_obs=Qmax_obs, theta_at_Q_max=theta_argmax,
    maxQ_null_mean=float(maxQ_null.mean()), maxQ_null_p95=float(np.percentile(maxQ_null, 95)),
    maxQ_null_max=float(maxQ_null.max()), p_family_wise=p_max, n_perm=NPERM,
    verdict_maxQ="SUSPECT" if Qmax_obs <= np.percentile(maxQ_null, 95) else "SURVIVES max-over-theta null",
    a_k_1770=float(blom_a(E)),
    primary_AUC_note="the pre-stated primary AUC=0.725 is threshold-free -> IMMUNE to theta selection")
print(f"  connectivity: Q_max={Qmax_obs:.3f} vs null p95={np.percentile(maxQ_null,95):.3f}, "
      f"family-wise p={p_max:.4f} (sanity {sanity:.2e})", flush=True)

# secondary: max-over-theta null for ARI(components, labels), stride-10 subgrid
def components_at(idx_edges):
    parent = list(range(nn))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    comps = []
    ptr = 0
    for t in idx_edges:
        while ptr <= t:
            a, b_ = find(eu[ptr]), find(ev[ptr])
            if a != b_: parent[a] = b_
            ptr += 1
        comps.append(np.array([find(i) for i in range(nn)]))
    return comps
sub_idx = list(range(0, E, 10))
comps = components_at(sub_idx)
def ari(a, b_):
    ca = np.unique(a, return_inverse=True)[1]; cb = np.unique(b_, return_inverse=True)[1]
    cont = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(cont, (ca, cb), 1)
    def comb2(x): return x * (x - 1) / 2
    sij = comb2(cont).sum(); sa = comb2(cont.sum(1)).sum(); sb = comb2(cont.sum(0)).sum()
    n2 = comb2(len(a)); exp = sa * sb / n2; mx = (sa + sb) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 0.0
ari_obs = max(ari(c, lab) for c in comps)
NP2 = 200
rng4 = np.random.default_rng(1)
ari_null = np.array([max(ari(c, rng4.permutation(lab)) for c in comps) for _ in range(NP2)])
partC["misgen_connectivity"]["max_ARI_subgrid"] = dict(
    stride=10, n_thetas=len(sub_idx), ari_max_obs=float(ari_obs), n_perm=NP2,
    null_p95=float(np.percentile(ari_null, 95)), null_max=float(ari_null.max()),
    p_family_wise=float((1 + np.sum(ari_null >= ari_obs)) / (1 + NP2)))
print(f"  connectivity ARI: obs {ari_obs:.3f} vs null p95 {np.percentile(ari_null,95):.3f}", flush=True)

RES["partC"] = partC
RES["runtime_sec"] = time.time() - T0
json.dump(RES, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print(f"done in {RES['runtime_sec']:.1f}s -> results.json", flush=True)
