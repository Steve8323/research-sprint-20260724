"""POST-HOC tests (all labeled) + pareto.png figure. Run AFTER analyze.py.

These four tests were chosen after seeing the pre-registered results, to localize
where the resistance gain arrives on the alpha axis. Their p-values are NOT
corrected for the 4 comparisons. Appends under results.json key 'analysis.posthoc'
so every number in FINDINGS.md regenerates from code on disk.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/steven/moral_framework_misgen/undefined/routing-hybrid-knob"
d = json.load(open(OUT + "/results.json"))
runs = d["runs"]
ALPHAS = d["config"]["ALPHAS"]; SEEDS = d["config"]["SEEDS"]; PW = d["config"]["P_WASTES"]


def cell(p, a, k):
    rows = sorted([r for r in runs if r["p_waste"] == p and r["alpha"] == a],
                  key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == SEEDS
    return [r[k] for r in rows]


ph = {}
# 1. where does the resistance gain arrive? endpoint jump a=1 vs a=0.75 at p=0.9
a1, a075 = cell(0.9, 1.0, "auc"), cell(0.9, 0.75, "auc")
w = stats.wilcoxon(a1, a075, alternative="less", method="exact")
ph["p09_auc_a1_lt_a075"] = dict(per_seed_a1=a1, per_seed_a075=a075,
                                W=float(w.statistic), p_one_sided=float(w.pvalue),
                                label="post-hoc")
# 2. do interior alphas gain any resistance over random at p=0.9?
a0 = cell(0.9, 0.0, "auc")
w = stats.wilcoxon(a075, a0, alternative="less", method="exact")
ph["p09_auc_a075_lt_a0"] = dict(per_seed_a0=a0, W=float(w.statistic),
                                p_one_sided=float(w.pvalue), label="post-hoc")
# 3. is the tiny retain cost at p=0.5 paired-consistent?
w = stats.wilcoxon(cell(0.5, 0.0, "retain_pruned"), cell(0.5, 1.0, "retain_pruned"),
                   alternative="two-sided", method="exact")
ph["p05_retain_a0_vs_a1_two_sided"] = dict(W=float(w.statistic), p=float(w.pvalue),
                                           label="post-hoc")
# 4. is interior retain flat vs random at p=0.9?
w = stats.wilcoxon(cell(0.9, 0.75, "retain_pruned"), cell(0.9, 0.0, "retain_pruned"),
                   alternative="two-sided", method="exact")
ph["p09_retain_a075_vs_a0_two_sided"] = dict(W=float(w.statistic), p=float(w.pvalue),
                                             label="post-hoc")

for k, v in ph.items():
    print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("per_seed")})
d["analysis"]["posthoc"] = ph
json.dump(d, open(OUT + "/results.json", "w"), indent=1)

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
cmap = plt.cm.viridis
for ax, p in zip(axes, PW):
    for a in ALPHAS:
        c = cmap(a * 0.9)
        x = cell(p, a, "auc"); y = cell(p, a, "U")
        ax.scatter(x, y, s=18, color=c, alpha=0.45)
        mx, my = np.mean(x), np.mean(y)
        ax.scatter([mx], [my], s=160, color=c, edgecolor="k", zorder=5,
                   label=f"alpha={a:g}")
        ax.annotate(f"{a:g}", (mx, my), textcoords="offset points",
                    xytext=(7, 7), fontsize=9)
    means = [(np.mean(cell(p, a, "auc")), np.mean(cell(p, a, "U"))) for a in ALPHAS]
    ax.plot([m[0] for m in means], [m[1] for m in means], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("recovery AUC  (lower = more relearning-resistant)")
    ax.set_ylabel("U = retain_pruned - forget_pruned  (higher = better unlearning)")
    ax.set_title(f"p_waste = {p}   (per-seed pale, means bold; alpha 0=random, 1=structural)")
    ax.invert_xaxis()   # so up-right = better on both axes
ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Random(0) <-> structural(1) routing knob: quality vs relearning-resistance, 6 seeds")
fig.tight_layout()
fig.savefig(OUT + "/pareto.png", dpi=130)
print("wrote pareto.png + appended analysis.posthoc")
