"""Pre-registered analysis of results.json (tests T1/T2/T3 + labeled secondaries).
Appends machine-readable summary into results.json under key 'analysis'."""
import json, itertools
import numpy as np
from scipy import stats

OUT = "/Users/steven/moral_framework_misgen/undefined/routing-hybrid-knob"
d = json.load(open(OUT + "/results.json"))
runs = d["runs"]
ALPHAS = d["config"]["ALPHAS"]; SEEDS = d["config"]["SEEDS"]; PW = d["config"]["P_WASTES"]
T975_5 = 2.5706  # t crit, df=5


def cell(p, a):
    rows = sorted([r for r in runs if r["p_waste"] == p and r["alpha"] == a],
                  key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == SEEDS
    return rows


def mci(v):
    v = np.array(v, float)
    m = v.mean(); h = T975_5 * v.std(ddof=1) / np.sqrt(len(v))
    return round(m, 4), round(m - h, 4), round(m + h, 4)


summary = {}
for p in PW:
    summary[str(p)] = {}
    for a in ALPHAS:
        rows = cell(p, a)
        s = {}
        for k in ["forget_pruned", "retain_pruned", "U", "auc"]:
            m, lo, hi = mci([r[k] for r in rows])
            s[k] = dict(mean=m, ci95=[lo, hi], per_seed=[r[k] for r in rows])
        summary[str(p)][str(a)] = s

print("=" * 78)
print("CELL MEANS (n=6 seeds, 95% t-CI)   U = retain_pruned - forget_pruned")
print("=" * 78)
for p in PW:
    print(f"\np_waste = {p}")
    print(f"  {'alpha':>6s} {'forget_pruned':>16s} {'retain_pruned':>16s} "
          f"{'U':>16s} {'recovery AUC':>18s}")
    for a in ALPHAS:
        s = summary[str(p)][str(a)]
        f = lambda k: f"{s[k]['mean']:.3f} [{s[k]['ci95'][0]:.3f},{s[k]['ci95'][1]:.3f}]"
        print(f"  {a:>6.2f} {f('forget_pruned'):>16s} {f('retain_pruned'):>16s} "
              f"{f('U'):>16s} {f('auc'):>18s}")

# ---------------- T1: AUC(alpha=1) < AUC(alpha=0) at p_waste=0.5 ----------------
a0 = [r["auc"] for r in cell(0.5, 0.0)]
a1 = [r["auc"] for r in cell(0.5, 1.0)]
w1 = stats.wilcoxon(a1, a0, alternative="less", method="exact")
print("\n" + "=" * 78)
print("T1 (pre-registered): structural (a=1) more relearning-resistant than random (a=0)")
print("     at p_waste=0.5.  Exact one-sided Wilcoxon, n=6.")
print(f"  per-seed AUC a=1: {a1}")
print(f"  per-seed AUC a=0: {a0}")
print(f"  seeds with AUC(a=1)<AUC(a=0): {sum(x < y for x, y in zip(a1, a0))}/6")
print(f"  W={w1.statistic}, p={w1.pvalue:.4f}  -> {'SIGNIFICANT' if w1.pvalue < 0.05 else 'NOT significant'}")

# ---------------- T2: retain(a=0) > retain(a=1) at p_waste=0.9 ----------------
r0 = [r["retain_pruned"] for r in cell(0.9, 0.0)]
r1 = [r["retain_pruned"] for r in cell(0.9, 1.0)]
w2 = stats.wilcoxon(r0, r1, alternative="greater", method="exact")
print("\nT2 (pre-registered): random (a=0) better retain than structural (a=1)")
print("     at p_waste=0.9.  Exact one-sided Wilcoxon, n=6.")
print(f"  per-seed retain a=0: {r0}")
print(f"  per-seed retain a=1: {r1}")
print(f"  seeds with retain(a=0)>retain(a=1): {sum(x > y for x, y in zip(r0, r1))}/6")
print(f"  W={w2.statistic}, p={w2.pvalue:.4f}  -> {'SIGNIFICANT' if w2.pvalue < 0.05 else 'NOT significant'}")

# ---------------- T3: Pareto non-dominated set on (mean U, mean -AUC) ----------------
print("\n" + "=" * 78)
print("T3 (pre-registered): non-dominated alphas in (mean U up, mean -AUC up)")
print("  validity gate: mean forget_pruned <= 0.05 (all cells checked)")
pareto = {}
for p in PW:
    pts = {}
    for a in ALPHAS:
        s = summary[str(p)][str(a)]
        assert s["forget_pruned"]["mean"] <= 0.05, f"validity gate failed p={p} a={a}"
        pts[a] = (s["U"]["mean"], -s["auc"]["mean"])
    nd = []
    for a, (u, r) in pts.items():
        dominated = any((u2 >= u and r2 >= r and (u2 > u or r2 > r))
                        for a2, (u2, r2) in pts.items() if a2 != a)
        if not dominated:
            nd.append(a)
    pareto[str(p)] = sorted(nd)
    interior = [a for a in nd if 0 < a < 1]
    # pre-stated usefulness: interior nd point strictly better than one endpoint per axis
    useful = []
    for a in interior:
        u, r = pts[a]
        u_ends = [pts[0.0][0], pts[1.0][0]]; r_ends = [pts[0.0][1], pts[1.0][1]]
        if (u > min(u_ends) and r > min(r_ends)) and (u > max(u_ends) or r > max(r_ends) or
                (u > min(u_ends) and r > min(r_ends))):
            useful.append(a)
    # strict pre-stated criterion: better than ONE endpoint on EACH axis
    strict_useful = [a for a in interior
                     if pts[a][0] > min(pts[0.0][0], pts[1.0][0])
                     and pts[a][1] > min(pts[0.0][1], pts[1.0][1])]
    print(f"  p_waste={p}: non-dominated alphas = {sorted(nd)}; "
          f"interior non-dominated = {interior}; "
          f"Pareto-useful (pre-stated criterion) = {strict_useful}")
    pareto[str(p) + "_interior_useful"] = strict_useful

# ---------------- secondary (labeled): Spearman trend alpha vs AUC ----------------
print("\nSECONDARY (labeled): Spearman trend of alpha vs per-seed AUC")
sec = {}
for p in PW:
    xs, ys = [], []
    for a in ALPHAS:
        for r in cell(p, a):
            xs.append(a); ys.append(r["auc"])
    rho, pv = stats.spearmanr(xs, ys)
    sec[str(p)] = dict(spearman_rho=round(rho, 4), p=round(pv, 6))
    print(f"  p_waste={p}: rho={rho:.3f}, p={pv:.4g}  (n=30 obs, seeds pooled -- "
          f"treat as descriptive)")

# per-seed dominance count for the modal useful alpha (secondary)
print("\nSECONDARY (labeled): per-seed check of interior candidates vs endpoints")
per_seed_checks = {}
for p in PW:
    for a in [x for x in ALPHAS if 0 < x < 1]:
        wins = 0
        det = []
        for i, s in enumerate(SEEDS):
            u = cell(p, a)[i]["U"]; auc_ = cell(p, a)[i]["auc"]
            u0, auc0 = cell(p, 0.0)[i]["U"], cell(p, 0.0)[i]["auc"]
            u1, auc1 = cell(p, 1.0)[i]["U"], cell(p, 1.0)[i]["auc"]
            ok = (u > min(u0, u1)) and (-auc_ > min(-auc0, -auc1))
            wins += ok
        per_seed_checks[f"p{p}_a{a}"] = wins
        print(f"  p_waste={p} alpha={a}: strictly-inside-both-axes in {wins}/6 seeds")

d["analysis"] = dict(cell_summary=summary,
                     T1=dict(auc_a1=a1, auc_a0=a0, W=float(w1.statistic), p=float(w1.pvalue)),
                     T2=dict(retain_a0=r0, retain_a1=r1, W=float(w2.statistic), p=float(w2.pvalue)),
                     T3_pareto=pareto,
                     secondary_spearman=sec,
                     secondary_per_seed_interior=per_seed_checks)
json.dump(d, open(OUT + "/results.json", "w"), indent=1)
print("\nappended analysis to results.json")
