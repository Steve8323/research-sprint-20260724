#!/usr/bin/env python3
"""POST-HOC sensitivity + figure for r3-selection-inflation-curve (not preregistered).

1) bootstrap CI on sigma_w -> rho_hat -> the V1 prediction (the rule's only free parameter)
2) break-even sigma_w at which V1 would have failed
3) design rule: n needed so that the null-winner bound falls below a target delta
4) figure: the inflation curve + the noise-probe validation
"""
import os, json, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/steven/research_sprint_20260724"
OUT = os.path.join(ROOT, "r3-selection-inflation-curve")
res = json.load(open(os.path.join(OUT, "results.json")))
B = res["partB"]

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
Mp = d_matrix(r2["per_item_primary"], [2, 12, 16, 20]); Mc = d_matrix(r2["per_item_control"], [4, 6, 8, 11])
o1 = {L: i for i, L in enumerate([2, 12, 16, 20])}; o2 = {L: i for i, L in enumerate([4, 6, 8, 11])}
M8 = np.column_stack([Mp[:, o1[L]] if L in o1 else Mc[:, o2[L]] for L in LAYERS8])
d_orig = np.array(B["sweep_d_orig"]); Dbar = float(d_orig.mean()); S2 = float(d_orig.var(ddof=1))
c_sel = B["V1"]["c_selected_original"]

def sigw(M):
    R = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    return math.sqrt((R ** 2).sum() / ((M.shape[0] - 1) * (M.shape[1] - 1)))

rng = np.random.default_rng(31337)
boot = []
for _ in range(4000):
    idx = rng.integers(0, M8.shape[0], M8.shape[0])
    sw = sigw(M8[idx]); s = sw / math.sqrt(6)
    rh = max(0.0, 1 - s ** 2 / S2)
    boot.append((sw, rh, rh * c_sel))
boot = np.array(boot)
out = dict(
    sigma_w_point=sigw(M8), sigma_w_ci=[float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
    rho_hat_point=B["rho_hat"], rho_hat_ci=[float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
    v1_pred_ci=[float(np.percentile(boot[:, 2], 2.5)), float(np.percentile(boot[:, 2], 97.5))],
    frac_boot_pred_inside_fresh_CI=float(np.mean((boot[:, 2] >= -0.72) & (boot[:, 2] <= 1.16))),
)
# break-even sigma_w for V1 (prediction must stay inside [-0.72, 1.16])
rho_break = 0.72 / abs(c_sel)
s_break = math.sqrt(S2 * (1 - rho_break))
out["rho_hat_breakeven_for_V1"] = rho_break
out["sigma_w_breakeven_for_V1"] = s_break * math.sqrt(6)
out["sigma_w_margin_pct"] = 100 * (1 - out["sigma_w_breakeven_for_V1"] / out["sigma_w_point"])
# design rule: n needed for the null-winner bound to fall below delta
ak24 = B["a_k"]; sw = out["sigma_w_point"]
out["design_rule"] = {f"delta={d}": math.ceil((ak24 * sw / d) ** 2) for d in (0.25, 0.5, 1.0, 2.0)}
json.dump(out, open(os.path.join(OUT, "sensitivity_posthoc.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ks = np.unique(np.round(np.logspace(0.3, 3.3, 60)).astype(int))
ak = stats.norm.ppf((ks - 0.375) / (ks + 0.25))
for rho, c in [(0.0, "#c0392b"), (0.5, "#e67e22"), (0.9, "#2980b9")]:
    ax[0].plot(ks, ak * math.sqrt(1 - rho), color=c, lw=2, label=f"reliability rho={rho}")
ax[0].set_xscale("log"); ax[0].set_xlabel("k = number of cells searched")
ax[0].set_ylabel("expected inflation of the winner\n(units of per-cell SE s = sigma_w/sqrt(n))")
ax[0].set_title("Inflation curve:  E[winner - truth] = a_k * s * sqrt(1-rho)")
for k, lbl in [(24, "noise-probe\nlayer sweep"), (1770, "theta grid")]:
    ax[0].axvline(k, color="k", ls=":", lw=0.8)
    ax[0].text(k * 1.1, 0.15, lbl, fontsize=7, rotation=90, va="bottom")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

V = B["V2"]; x = np.arange(8)
ax[1].errorbar(x, V["fresh"], yerr=1.96 * np.array(V["fresh_se"]), fmt="o", color="k",
               capsize=3, label="fresh 32 prompts (truth)")
ax[1].plot(x, V["naive"], "s--", color="#c0392b", label="original sweep (naive)")
ax[1].plot(x, V["EB"], "^-", color="#2980b9", label="deflated (rule)")
for i, L in enumerate(V["layers"]):
    if L in (4, 8, 11): ax[1].axvspan(i - .4, i + .4, color="#f1c40f", alpha=.18)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"L{L}" for L in V["layers"]])
ax[1].set_ylabel("misaligned - base margin change (nats)")
ax[1].set_title("Validation (yellow = the 3 originally selected layers)\n"
                f"RMSE naive {V['rmse_naive']:.2f} -> deflated {V['rmse_eb']:.2f}")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "inflation_curve.png"), dpi=140)
print("figure -> inflation_curve.png")
