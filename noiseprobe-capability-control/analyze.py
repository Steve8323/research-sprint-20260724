"""
Analysis for the capability-matched control (pre-registered in FINDINGS.md).

Primary (sigma=0.2, n=24 layers):
  dS_diff(L)  from SAVED layer_importance_results.json (misaligned - base, cell means)
  dD_diff(L)  new differential neutral degradation (mean per-token dNLL, mis - base)
  1) Spearman(dS_diff, dD_diff), two-sided permutation p (20,000 perms)
  2) OLS dS_diff ~ dD_diff; residual z-scores; L4/8/11 before vs after control
Decision rule is in FINDINGS.md (written before the run).

Secondary (labelled): dCM covariate; rerun-vs-saved reproduction; per-item bootstrap CIs
on hot-layer differentials; sigma=0.1/0.4 correlations; absolute degradation at hot cells.
"""
import json
import numpy as np

rng = np.random.default_rng(0)
RAW = "/Users/steven/undefined/noiseprobe-capability-control/raw_control_measurements.json"
SAVED = "/Users/steven/misalignment_noise_probe/layer_importance_results.json"
OUT = "/Users/steven/undefined/noiseprobe-capability-control/results.json"
HOT = [4, 8, 11]
NL = 24


def spearman_perm(x, y, n_perm=20000):
    def rank(a):
        r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r
    rx, ry = rank(x), rank(y)
    def corr(a, b):
        return np.corrcoef(a, b)[0, 1]
    obs = corr(rx, ry)
    cnt = 0
    for _ in range(n_perm):
        if abs(corr(rx, rng.permutation(ry))) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def ols_resid_z(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    z = (resid - resid.mean()) / resid.std(ddof=1)
    return beta, resid, z


def zscores(v):
    return (v - v.mean()) / v.std(ddof=1)


raw = json.load(open(RAW))
saved = json.load(open(SAVED))

# ---------- saved dS_diff at sigma=0.2 ----------
dS_base_saved = np.array(saved["base"]["grid"]["0.2"])
dS_mis_saved = np.array(saved["misaligned"]["grid"]["0.2"])
dS_diff_saved = dS_mis_saved - dS_base_saved
z_uncontrolled = zscores(dS_diff_saved)

# ---------- new covariates ----------
clean = {tag: {"nll": np.array([r["nll_c"] for r in raw["clean"][tag]["neutral"]]),
               "cm": np.array([r["cm"] for r in raw["clean"][tag]["neutral"]]),
               "smarg": np.array([r["margin"] for r in raw["clean"][tag]["safety"]])}
         for tag in ["base", "misaligned"]}

# cells[(model, sigma, layer, seed)] -> per-item arrays
def collect(sigma):
    """Return per model: dNLL[L, seed, item], dCM[L, seed, item], dS[L, seed, item] (if present)."""
    out = {}
    for tag in ["base", "misaligned"]:
        dnll = np.full((NL, 3, 6), np.nan)
        dcm = np.full((NL, 3, 6), np.nan)
        ds = np.full((NL, 3, 6), np.nan)
        for c in raw["cells"]:
            if c["model"] != tag or abs(c["sigma"] - sigma) > 1e-9:
                continue
            L, s = c["layer"], c["seed"]
            for r in c["neutral"]:
                dnll[L, s, r["item"]] = r["nll_c"] - clean[tag]["nll"][r["item"]]
                dcm[L, s, r["item"]] = r["cm"] - clean[tag]["cm"][r["item"]]
            if "safety" in c:
                for r in c["safety"]:
                    ds[L, s, r["item"]] = r["margin"] - clean[tag]["smarg"][r["item"]]
        out[tag] = {"dnll": dnll, "dcm": dcm, "ds": ds}
    return out

res = {"config": {"hot_layers": HOT, "n_layers": NL,
                  "saved_source": SAVED,
                  "clean_neutral_nll_per_tok": {t: float(clean[t]["nll"].mean()) for t in clean},
                  "clean_competence_margin": {t: float(clean[t]["cm"].mean()) for t in clean},
                  "clean_safety_margin_rerun": {t: float(clean[t]["smarg"].mean()) for t in clean},
                  "clean_safety_margin_saved": {"base": saved["base"]["clean"],
                                                "misaligned": saved["misaligned"]["clean"]}}}

c02 = collect(0.2)
dD_diff = (c02["misaligned"]["dnll"] - c02["base"]["dnll"]).mean(axis=(1, 2))   # [L]
dCM_diff = (c02["misaligned"]["dcm"] - c02["base"]["dcm"]).mean(axis=(1, 2))    # [L]
dD_diff_sem = (c02["misaligned"]["dnll"] - c02["base"]["dnll"]).reshape(NL, -1).std(axis=1, ddof=1) / np.sqrt(18)

# ---------- PRIMARY ----------
rho, p_perm = spearman_perm(dS_diff_saved, dD_diff)
r_pear = pearson(dS_diff_saved, dD_diff)
beta, resid, z_ctrl = ols_resid_z(dS_diff_saved, dD_diff)
rank_ctrl = np.argsort(resid)  # most negative first
primary = {
    "dS_diff_saved_sigma0.2": dS_diff_saved.tolist(),
    "dD_diff_dNLLpertok_sigma0.2": dD_diff.tolist(),
    "dD_diff_sem": dD_diff_sem.tolist(),
    "spearman_rho": float(rho), "perm_p_two_sided": float(p_perm),
    "pearson_r": r_pear,
    "ols_intercept_slope": [float(beta[0]), float(beta[1])],
    "r_squared": float(1 - resid.var() / dS_diff_saved.var()),
    "hot_layers": {str(L): {
        "dS_diff": float(dS_diff_saved[L]),
        "dD_diff": float(dD_diff[L]),
        "z_uncontrolled": float(z_uncontrolled[L]),
        "z_after_control": float(z_ctrl[L]),
        "resid_rank_most_negative_first": int(np.where(rank_ctrl == L)[0][0]),
    } for L in HOT},
    "bottom3_resid_layers": rank_ctrl[:3].tolist(),
}
# decision rule
explained = (p_perm < 0.05 and rho < 0) and all(abs(z_ctrl[L]) < 1.5 for L in HOT)
survives = sum(z_ctrl[L] <= -1.5 for L in HOT) >= 2
primary["decision"] = ("explained_by_degradation" if explained
                       else "survives_control" if survives else "partial_mixed")
res["primary"] = primary

# bootstrap uncertainty of z_after_control from covariate sampling (items x seeds), post-hoc labelled
boot_z = {L: [] for L in HOT}
paired = (c02["misaligned"]["dnll"] - c02["base"]["dnll"]).reshape(NL, 18)
for b in range(2000):
    idx = rng.integers(0, 18, 18)
    dD_b = paired[:, idx].mean(axis=1)
    _, _, z_b = ols_resid_z(dS_diff_saved, dD_b)
    for L in HOT:
        boot_z[L].append(z_b[L])
res["primary"]["z_after_control_boot95CI_covariate_only_POSTHOC"] = {
    str(L): [float(np.percentile(boot_z[L], 2.5)), float(np.percentile(boot_z[L], 97.5))]
    for L in HOT}

# ---------- SECONDARY (labelled) ----------
sec = {}
# competence-margin covariate
rho_cm, p_cm = spearman_perm(dS_diff_saved, dCM_diff)
_, _, z_cm = ols_resid_z(dS_diff_saved, dCM_diff)
sec["cm_covariate"] = {"dCM_diff": dCM_diff.tolist(), "spearman_rho": float(rho_cm),
                       "perm_p": float(p_cm), "pearson_r": pearson(dS_diff_saved, dCM_diff),
                       "hot_z_after_control": {str(L): float(z_cm[L]) for L in HOT}}
# both covariates jointly
X2 = np.column_stack([np.ones(NL), dD_diff, dCM_diff])
b2, *_ = np.linalg.lstsq(X2, dS_diff_saved, rcond=None)
r2j = dS_diff_saved - X2 @ b2
zj = (r2j - r2j.mean()) / r2j.std(ddof=1)
sec["joint_covariates"] = {"r_squared": float(1 - r2j.var() / dS_diff_saved.var()),
                           "hot_z": {str(L): float(zj[L]) for L in HOT}}

# reproduction check: rerun safety dS vs saved (per model)
ds_rerun = {t: c02[t]["ds"].mean(axis=(1, 2)) for t in ["base", "misaligned"]}
dS_diff_rerun = ds_rerun["misaligned"] - ds_rerun["base"]
sec["reproduction"] = {
    "pearson_rerun_vs_saved_dS_base": pearson(ds_rerun["base"], dS_base_saved),
    "pearson_rerun_vs_saved_dS_mis": pearson(ds_rerun["misaligned"], dS_mis_saved),
    "pearson_rerun_vs_saved_dS_diff": pearson(dS_diff_rerun, dS_diff_saved),
    "dS_diff_rerun": dS_diff_rerun.tolist(),
    "hot_z_uncontrolled_rerun": {str(L): float(zscores(dS_diff_rerun)[L]) for L in HOT},
}
# per-item bootstrap CI on rerun hot-layer differentials (cluster-resample the 6 prompts)
ds_pair = c02["misaligned"]["ds"] - c02["base"]["ds"]  # [L, seed, item] noise seeds shared
hot_ci = {}
for L in HOT + [int(np.argsort(dS_diff_rerun)[0])]:
    vals = []
    for b in range(10000):
        items = rng.integers(0, 6, 6)
        vals.append(ds_pair[L][:, items].mean())
    hot_ci[str(L)] = {"mean": float(ds_pair[L].mean()),
                      "boot95CI_over_items": [float(np.percentile(vals, 2.5)),
                                              float(np.percentile(vals, 97.5))]}
sec["hot_layer_dS_diff_rerun_CIs"] = hot_ci

# primary test repeated on the RERUN dS_diff (labelled secondary)
rho_r, p_r = spearman_perm(dS_diff_rerun, dD_diff)
_, _, z_rr = ols_resid_z(dS_diff_rerun, dD_diff)
sec["primary_repeated_on_rerun_dS"] = {"spearman_rho": float(rho_r), "perm_p": float(p_r),
                                       "hot_z_after_control": {str(L): float(z_rr[L]) for L in HOT}}

# other sigmas (if measured)
for s in [0.1, 0.4]:
    key = str(s)
    cs = collect(s)
    if np.isnan(cs["base"]["dnll"]).all():
        sec[f"sigma_{key}"] = "not measured (budget)"
        continue
    dSd = np.array(saved["misaligned"]["grid"][key]) - np.array(saved["base"]["grid"][key])
    dDd = (cs["misaligned"]["dnll"] - cs["base"]["dnll"]).mean(axis=(1, 2))
    rho_s, p_s = spearman_perm(dSd, dDd)
    _, _, z_s = ols_resid_z(dSd, dDd)
    sec[f"sigma_{key}"] = {"spearman_rho": float(rho_s), "perm_p": float(p_s),
                           "pearson_r": pearson(dSd, dDd),
                           "hot_z_uncontrolled": {str(L): float(zscores(dSd)[L]) for L in HOT},
                           "hot_z_after_control": {str(L): float(z_s[L]) for L in HOT}}

# absolute degradation at hot cells (how broken is the text there?)
sec["absolute_dNLL_per_tok_sigma0.2"] = {
    t: {str(L): float(c02[t]["dnll"][L].mean()) for L in HOT + [6]}
    for t in ["base", "misaligned"]}
res["secondary"] = sec

json.dump(res, open(OUT, "w"), indent=1)

# ---------- console summary ----------
print(f"clean NLL/tok base={clean['base']['nll'].mean():.3f} mis={clean['misaligned']['nll'].mean():.3f}")
print(f"\nPRIMARY sigma=0.2 (saved dS_diff vs new dNLL_diff, n=24 layers)")
print(f"  Spearman rho = {rho:+.3f}, perm p = {p_perm:.4f}   Pearson r = {r_pear:+.3f}  R^2 = {res['primary']['r_squared']:.3f}")
for L in HOT:
    h = primary["hot_layers"][str(L)]
    print(f"  L{L}: dS_diff={h['dS_diff']:+.2f}  dD_diff={h['dD_diff']:+.3f}  "
          f"z {h['z_uncontrolled']:+.2f} -> {h['z_after_control']:+.2f}  (resid rank {h['resid_rank_most_negative_first']})")
print(f"  decision: {primary['decision']}")
print(f"\nreproduction: r(dS_diff rerun vs saved) = {sec['reproduction']['pearson_rerun_vs_saved_dS_diff']:+.3f}")
for L in HOT:
    print(f"  L{L} rerun dS_diff = {hot_ci[str(L)]['mean']:+.2f}  CI {hot_ci[str(L)]['boot95CI_over_items']}")
print(f"\nsaved {OUT}")
