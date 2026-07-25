"""
Connectivity graph vs worldview: theta-sweep of the saved 60x60 permutation-aligned
barrier census from ~/moral_framework_misgen/stats/results.json (READ-ONLY input).

Pre-registered in FINDINGS.md (same directory) BEFORE this script was run.
PRIMARY: threshold-free AUC = P(within-group barrier < cross-group barrier),
         node-label permutation test (10,000 perms, seed=0), one-sided, alpha=0.01.
SECONDARY: curves vs theta (direct / path-connected fractions, chain hops, modularity
         with permutation null band, ARI vs components, NMI vs greedy communities),
         landmark thetas, hedger-mediation test.

Analysis-only; no training. Runtime well under the budget.
"""
import json
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RES = "/Users/steven/moral_framework_misgen/stats/results.json"
OUT = "/Users/steven/misgen-connectivity-graph/results.json"
FIG = "/Users/steven/misgen-connectivity-graph/curves.png"

# ---------------- load fixed data ----------------
d = json.load(open(RES))
Bm = np.array(d["basins"]["barrier_matrix"])           # 60x60 symmetric, zero diag
scores = np.array(d["multimodality"]["scores"][:60])   # same seeds/order as census
n = 60
lab = np.where(scores > 0.8, "A", np.where(scores < 0.2, "M", "H"))
lab_int = np.array([{"A": 0, "M": 1, "H": 2}[x] for x in lab])
nA, nM, nH = int((lab == "A").sum()), int((lab == "M").sum()), int((lab == "H").sum())
assert (nA, nM, nH) == (12, 17, 31), "label counts drifted from saved census"

iu = np.triu_indices(n, 1)
b_all = Bm[iu]
ii, jj = iu

def pair_masks(labels):
    li, lj = labels[ii], labels[jj]
    cross = ((li == "A") & (lj == "M")) | ((li == "M") & (lj == "A"))
    within = ((li == "A") & (lj == "A")) | ((li == "M") & (lj == "M"))
    return within, cross

within_mask, cross_mask = pair_masks(lab)
w_b, c_b = b_all[within_mask], b_all[cross_mask]
assert len(w_b) == 202 and len(c_b) == 204

# ---------------- PRIMARY: rank AUC + node-label permutation ----------------
def auc_within_lt_cross(wb, cb):
    # P(within < cross) + 0.5 P(tie), computed via ranks (O(N log N))
    comb = np.concatenate([wb, cb])
    order = comb.argsort(kind="mergesort")
    ranks = np.empty(len(comb))
    # average ranks for ties
    sorted_c = comb[order]
    ranks_sorted = np.arange(1, len(comb) + 1, dtype=float)
    i = 0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and sorted_c[j + 1] == sorted_c[i]:
            j += 1
        ranks_sorted[i:j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    ranks[order] = ranks_sorted
    Rc = ranks[len(wb):].sum()          # rank-sum of cross
    nw, nc = len(wb), len(cb)
    U_cross_greater = Rc - nc * (nc + 1) / 2.0
    return U_cross_greater / (nw * nc)  # = P(within < cross) + .5 ties

auc_obs = auc_within_lt_cross(w_b, c_b)

NPERM = 10_000
rng = np.random.default_rng(0)
perm_aucs = np.empty(NPERM)
for k in range(NPERM):
    pl = rng.permutation(lab)
    wm, cm = pair_masks(pl)
    perm_aucs[k] = auc_within_lt_cross(b_all[wm], b_all[cm])
p_primary = (1 + int((perm_aucs >= auc_obs).sum())) / (1 + NPERM)

print(f"PRIMARY  AUC = P(within < cross) = {auc_obs:.4f}")
print(f"         node-label permutation p (one-sided, {NPERM} perms) = {p_primary:.5f}")
print(f"         perm null: mean {perm_aucs.mean():.3f}, 97.5pct {np.percentile(perm_aucs,97.5):.3f}, max {perm_aucs.max():.3f}")

# ---------------- SECONDARY: theta sweep ----------------
thetas = np.unique(b_all)                     # every point where the graph changes
A_idx = np.where(lab == "A")[0]
M_idx = np.where(lab == "M")[0]
H_idx = np.where(lab == "H")[0]
keep = np.array([x for x in range(n) if lab[x] != "H"])  # hedger-deleted subgraph

deg_norm = None
curves = dict(theta=[], frac_cross_direct=[], frac_within_direct=[],
              frac_cross_path=[], frac_cross_path_noH=[],
              med_hops_cross=[], max_hops_cross=[],
              n_components=[], modularity_Q=[], ari_components=[],
              nmi_greedy=[], n_greedy_comms=[])

def label_partition_modularity(adj, labels_int):
    # Newman modularity of the fixed partition for an unweighted simple graph
    deg = adj.sum(1).astype(float)
    two_m = deg.sum()
    if two_m == 0:
        return np.nan
    Q = 0.0
    for g in np.unique(labels_int):
        idx = labels_int == g
        e_in = adj[np.ix_(idx, idx)].sum()          # 2 * internal edges
        Q += e_in / two_m - (deg[idx].sum() / two_m) ** 2
    return Q

# permutation null band for modularity (same 1000 label perms reused across theta)
NPERM_Q = 1000
rngQ = np.random.default_rng(1)
perm_labels_Q = [rngQ.permutation(lab_int) for _ in range(NPERM_Q)]
Q_null_lo, Q_null_hi = [], []

for th in thetas:
    adj = (Bm <= th).astype(int)
    np.fill_diagonal(adj, 0)
    # direct fractions
    dvals = adj[iu]
    curves["theta"].append(float(th))
    curves["frac_cross_direct"].append(float(dvals[cross_mask].mean()))
    curves["frac_within_direct"].append(float(dvals[within_mask].mean()))
    # components / path connectivity
    ncomp, comp = connected_components(adj, directed=False)
    curves["n_components"].append(int(ncomp))
    same_comp = comp[ii] == comp[jj]
    curves["frac_cross_path"].append(float(same_comp[cross_mask].mean()))
    curves["ari_components"].append(float(adjusted_rand_score(lab_int, comp)))
    # hop lengths among path-connected A-M pairs
    sp = shortest_path(adj, method="D", unweighted=True)
    hops = sp[np.ix_(A_idx, M_idx)].ravel()
    finite = hops[np.isfinite(hops)]
    curves["med_hops_cross"].append(float(np.median(finite)) if len(finite) else np.nan)
    curves["max_hops_cross"].append(float(finite.max()) if len(finite) else np.nan)
    # hedger-deleted subgraph path connectivity (A-M pairs only)
    sub = adj[np.ix_(keep, keep)]
    ncomp2, comp2 = connected_components(sub, directed=False)
    lab_keep = lab[keep]
    Ak = np.where(lab_keep == "A")[0]; Mk = np.where(lab_keep == "M")[0]
    same2 = comp2[Ak][:, None] == comp2[Mk][None, :]
    curves["frac_cross_path_noH"].append(float(same2.mean()))
    # modularity of fixed worldview partition + null band
    Q = label_partition_modularity(adj, lab_int)
    curves["modularity_Q"].append(float(Q) if np.isfinite(Q) else None)
    if adj.sum() > 0:
        qn = np.array([label_partition_modularity(adj, pl) for pl in perm_labels_Q])
        Q_null_lo.append(float(np.percentile(qn, 2.5)))
        Q_null_hi.append(float(np.percentile(qn, 97.5)))
    else:
        Q_null_lo.append(None); Q_null_hi.append(None)
    # greedy-modularity communities vs labels
    if adj.sum() > 0:
        G = nx.from_numpy_array(adj)
        comms = nx.algorithms.community.greedy_modularity_communities(G)
        cl = np.empty(n, dtype=int)
        for cidx, cset in enumerate(comms):
            for node in cset:
                cl[node] = cidx
        curves["nmi_greedy"].append(float(normalized_mutual_info_score(lab_int, cl)))
        curves["n_greedy_comms"].append(int(len(comms)))
    else:
        curves["nmi_greedy"].append(None)
        curves["n_greedy_comms"].append(None)

th_arr = np.array(curves["theta"])
fcd = np.array(curves["frac_cross_direct"])
fwd = np.array(curves["frac_within_direct"])
fcp = np.array(curves["frac_cross_path"])
fcp_noH = np.array(curves["frac_cross_path_noH"])
Qarr = np.array([q if q is not None else np.nan for q in curves["modularity_Q"]])
Qhi = np.array([q if q is not None else np.nan for q in Q_null_hi])

def first_theta(cond):
    w = np.where(cond)[0]
    return float(th_arr[w[0]]) if len(w) else None

landmarks = dict(
    theta_first_cross_direct=first_theta(fcd > 0),
    theta_cross_path_50=first_theta(fcp >= 0.5),
    theta_cross_path_100=first_theta(fcp >= 1.0),
    theta_cross_direct_50=first_theta(fcd >= 0.5),
    theta_cross_direct_100=first_theta(fcd >= 1.0),
    theta_within_direct_50=first_theta(fwd >= 0.5),
    theta_within_direct_100=first_theta(fwd >= 1.0),
    theta_cross_path_noH_100=first_theta(fcp_noH >= 1.0),
)
# theta at which A (resp. M) alone becomes internally path-connected
for name, idxg in [("theta_A_internal_connected", A_idx), ("theta_M_internal_connected", M_idx)]:
    val = None
    for th in thetas:
        adj = (Bm[np.ix_(idxg, idxg)] <= th).astype(int)
        np.fill_diagonal(adj, 0)
        ncomp, _ = connected_components(adj, directed=False)
        if ncomp == 1:
            val = float(th); break
    landmarks[name] = val

# window where the label-partition modularity exceeds its 97.5pct permutation null
sig = np.isfinite(Qarr) & np.isfinite(Qhi) & (Qarr > Qhi)
landmarks["Q_sig_theta_lo"] = first_theta(sig)
w = np.where(sig)[0]
landmarks["Q_sig_theta_hi"] = float(th_arr[w[-1]]) if len(w) else None
landmarks["Q_sig_frac_of_grid"] = float(sig.mean())
landmarks["Q_max"] = float(np.nanmax(Qarr))
landmarks["Q_argmax_theta"] = float(th_arr[int(np.nanargmax(Qarr))])

# legacy threshold 0.05 details
adj05 = (Bm <= 0.05).astype(int); np.fill_diagonal(adj05, 0)
sp05 = shortest_path(adj05, method="D", unweighted=True)
hops05 = sp05[np.ix_(A_idx, M_idx)].ravel()
hops05_f = hops05[np.isfinite(hops05)]
hist05 = {str(int(k)): int((hops05_f == k).sum()) for k in np.unique(hops05_f)}
ncomp05, _ = connected_components(adj05, directed=False)
sub05 = adj05[np.ix_(keep, keep)]
_, comp05s = connected_components(sub05, directed=False)
lab_keep = lab[keep]
Ak = np.where(lab_keep == "A")[0]; Mk = np.where(lab_keep == "M")[0]
frac_noH_05 = float((comp05s[Ak][:, None] == comp05s[Mk][None, :]).mean())
legacy = dict(theta=0.05,
              frac_cross_direct=float((c_b <= 0.05).mean()),
              frac_cross_path=float(np.isfinite(hops05).mean()),
              n_components=int(ncomp05),
              hop_histogram_AM=hist05,
              median_hops_AM=float(np.median(hops05_f)),
              max_hops_AM=float(hops05_f.max()),
              frac_cross_path_hedgers_deleted=frac_noH_05)
print("\nLegacy theta=0.05:", json.dumps(legacy, indent=2))
print("\nLandmark thetas:", json.dumps(landmarks, indent=2))

# ---------------- save ----------------
out = dict(
    input=RES,
    n_nodes=n, labels=dict(A=nA, M=nM, H=nH),
    label_rule="A: score>0.8, M: score<0.2, H: otherwise (repo convention)",
    sanity=dict(cross_mean=float(c_b.mean()), within_mean=float(w_b.mean()),
                frac_cross_not_direct_at_005=float((c_b > 0.05).mean()),
                n_cross=int(len(c_b)), n_within=int(len(w_b))),
    primary=dict(statistic="AUC = P(within-group barrier < cross-group barrier)",
                 auc=float(auc_obs), n_perm=NPERM, perm_seed=0,
                 p_one_sided=float(p_primary),
                 null_mean=float(perm_aucs.mean()),
                 null_p975=float(np.percentile(perm_aucs, 97.5)),
                 null_max=float(perm_aucs.max()),
                 prestated_alpha=0.01,
                 verdict=("landscape sees worldview (threshold-free)"
                          if p_primary < 0.01 else "not established")),
    curves=curves,
    modularity_null=dict(n_perm=NPERM_Q, perm_seed=1, lo=Q_null_lo, hi=Q_null_hi),
    landmarks=landmarks,
    legacy_theta_005=legacy,
)
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nsaved {OUT}")

# ---------------- figure ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(th_arr, fwd, label="within (A-A, M-M) direct", color="tab:blue")
ax.plot(th_arr, fcd, label="cross (A-M) direct", color="tab:red")
ax.plot(th_arr, fcp, label="cross path-connected (full graph)", color="tab:red", ls="--")
ax.plot(th_arr, fcp_noH, label="cross path-connected (hedgers deleted)", color="tab:orange", ls=":")
ax.axvline(0.05, color="gray", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("theta (barrier threshold)"); ax.set_ylabel("fraction of pairs")
ax.set_title(f"Connectivity vs theta (AUC={auc_obs:.3f}, perm p={p_primary:.4f})")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(th_arr, Qarr, color="tab:green", label="Q of worldview partition")
ax.fill_between(th_arr, [q if q is not None else np.nan for q in Q_null_lo],
                [q if q is not None else np.nan for q in Q_null_hi],
                color="gray", alpha=0.3, label="perm null 95% band (1000 perms)")
ax.axvline(0.05, color="gray", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("theta"); ax.set_ylabel("modularity Q")
ax.set_title("Label-partition modularity vs theta")
ax.legend(fontsize=7)

ax = axes[2]
ax.plot(th_arr, curves["med_hops_cross"], label="median A-M hops", color="tab:purple")
ax.plot(th_arr, curves["max_hops_cross"], label="max A-M hops", color="tab:purple", ls="--")
axb = ax.twinx()
axb.plot(th_arr, curves["n_components"], label="# components", color="tab:gray", lw=0.8)
axb.set_ylabel("# components", color="tab:gray")
ax.axvline(0.05, color="gray", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("theta"); ax.set_ylabel("shortest-path hops (A-M)")
ax.set_title("Chain length vs theta")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print(f"saved {FIG}")
