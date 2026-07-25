#!/usr/bin/env python3
"""Connectivity graph vs worldview: sweep theta over the saved 60x60 barrier census.

Analysis-only. Data: stats/results.json (basins/barrier_matrix, multimodality/scores).
Pre-registered plan: see FINDINGS.md (written before this script was run).

PRIMARY : threshold-free chained-barrier (minimax path) test, seed-level permutation.
S1      : dependence-aware seed-level permutation redo of the direct-barrier gap.
S2      : worldview-partition modularity Q(theta) with exact max-over-theta perm test.
D1-D4   : descriptive curves (direct-connection ECDFs, components, chain hops, chain gain).
EXPL    : greedy-modularity community detection ARI/NMI vs labels (post-hoc, no test).
"""
import json
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path

t0 = time.time()
RNG_PRIMARY, RNG_S1, RNG_S2, RNG_BOOT = 0, 1, 2, 3
NPERM = 20_000
OUT = "/Users/steven/moral_framework_misgen/undefined/misgen-connectivity-graph"

# ---------------------------------------------------------------- load + verify
d = json.load(open("/Users/steven/moral_framework_misgen/stats/results.json"))
B = np.array(d["basins"]["barrier_matrix"])            # 60x60
scores = np.array(d["multimodality"]["scores"][:60])   # deployment alignment
assert B.shape == (60, 60)
assert np.allclose(B, B.T) and np.allclose(np.diag(B), 0)
iu = np.triu_indices(60, 1)
allb = B[iu]
assert np.isclose(allb.min(), d["basins"]["barrier_min"])
assert np.isclose(np.median(allb), d["basins"]["barrier_median"])
assert np.isclose(allb.max(), d["basins"]["barrier_max"])
A_idx = np.where(scores > 0.8)[0]
M_idx = np.where(scores < 0.2)[0]
H_idx = np.where((scores >= 0.2) & (scores <= 0.8))[0]
assert len(A_idx) == 12 and len(M_idx) == 17 and len(H_idx) == 31, (len(A_idx), len(M_idx), len(H_idx))
# cross-check vs saved cluster summary
assert np.isclose(scores.mean(), d["basins"]["cluster_align"]["0"]["mean_align"], atol=1e-6)
print(f"verified: 60x60 symmetric, |A|={len(A_idx)} |M|={len(M_idx)} |H|={len(H_idx)}, "
      f"barriers min/med/max = {allb.min():.5f}/{np.median(allb):.5f}/{allb.max():.5f}")

committed = np.concatenate([A_idx, M_idx])             # 29 seeds
lab29 = np.array([1] * len(A_idx) + [0] * len(M_idx))  # 1 = aligned
B29 = B[np.ix_(committed, committed)]

def pair_masks(lab):
    """boolean masks over the upper triangle of a 29x29 matrix for cross / within."""
    same = lab[:, None] == lab[None, :]
    iu29 = np.triu_indices(len(lab), 1)
    return ~same[iu29], same[iu29]                     # cross_mask, within_mask

iu29 = np.triu_indices(29, 1)
cross_m, within_m = pair_masks(lab29)
assert cross_m.sum() == 204 and within_m.sum() == 202

# ---------------------------------------------------------------- minimax (chain) barriers
# minimax path cost between all 60 nodes = max edge on the MST path (exact).
mst = minimum_spanning_tree(csr_matrix(B)).toarray()
mst = np.maximum(mst, mst.T)                           # symmetrize
adj_mst = [[] for _ in range(60)]
for i in range(60):
    for j in range(60):
        if mst[i, j] > 0:
            adj_mst[i].append((j, mst[i, j]))
C = np.zeros((60, 60))                                 # chain (minimax) barrier matrix
for s in range(60):
    seen = np.zeros(60, bool); seen[s] = True
    stack = [(s, 0.0)]
    while stack:
        u, mx = stack.pop()
        for v, w in adj_mst[u]:
            if not seen[v]:
                seen[v] = True
                C[s, v] = max(mx, w)
                stack.append((v, C[s, v]))
    assert seen.all()
assert np.allclose(C, C.T)
C29 = C[np.ix_(committed, committed)]
c_pairs = C29[iu29]
b_pairs = B29[iu29]

# ---------------------------------------------------------------- PRIMARY test
def gap_stat(pairvals, lab):
    cm, wm = pair_masks(lab)
    return pairvals[cm].mean() - pairvals[wm].mean()

T_obs = gap_stat(c_pairs, lab29)
rng = np.random.default_rng(RNG_PRIMARY)
perm_T = np.empty(NPERM)
for p in range(NPERM):
    perm_T[p] = gap_stat(c_pairs, rng.permutation(lab29))
p_primary = float((np.sum(perm_T >= T_obs) + 1) / (NPERM + 1))

# bootstrap CIs on the two means (resampling SEEDS within each group, not pairs)
def boot_mean_ci(mat, rows, cols, nboot=2000, seed=RNG_BOOT, same_group=False):
    """CI for mean of mat[rows x cols] pairs, resampling the seed lists."""
    r = np.random.default_rng(seed); vals = []
    for _ in range(nboot):
        rr = r.choice(rows, len(rows), replace=True)
        cc = rr if same_group else r.choice(cols, len(cols), replace=True)
        sub = mat[np.ix_(rr, cc)]
        if same_group:
            m = rr[:, None] != rr[None, :]              # exclude self-pairs incl. resampled dups
            if m.sum() == 0:
                continue
            vals.append(sub[m].mean())
        else:
            vals.append(sub.mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

chain_cross_mean = float(C[np.ix_(A_idx, M_idx)].mean())
wA = C[np.ix_(A_idx, A_idx)][np.triu_indices(12, 1)]
wM = C[np.ix_(M_idx, M_idx)][np.triu_indices(17, 1)]
chain_within_mean = float(np.concatenate([wA, wM]).mean())
ci_cross = boot_mean_ci(C, A_idx, M_idx)

def boot_within_ci(mat, nboot=2000, seed=RNG_BOOT + 1):
    """CI for pooled within-group pair mean, resampling seeds within each group."""
    r = np.random.default_rng(seed); vals = []
    for _ in range(nboot):
        parts = []
        for grp in (A_idx, M_idx):
            rr = r.choice(grp, len(grp), replace=True)
            sub = mat[np.ix_(rr, rr)]
            m = rr[:, None] != rr[None, :]
            if m.any():
                parts.append(sub[m])
        vals.append(np.concatenate(parts).mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

ci_within = boot_within_ci(C)
print(f"\nPRIMARY chain-barrier gap: cross mean {chain_cross_mean:.5f} "
      f"(CI {ci_cross}) vs within mean {chain_within_mean:.5f}; "
      f"T = {T_obs:+.5f}, one-sided perm p = {p_primary:.4g} (n_perm={NPERM})")

# ---------------------------------------------------------------- S1: direct-barrier gap, seed-level
T1_obs = gap_stat(b_pairs, lab29)
rng = np.random.default_rng(RNG_S1)
perm_T1 = np.empty(NPERM)
for p in range(NPERM):
    perm_T1[p] = gap_stat(b_pairs, rng.permutation(lab29))
p_s1 = float((np.sum(perm_T1 >= T1_obs) + 1) / (NPERM + 1))
direct_cross_mean = float(B[np.ix_(A_idx, M_idx)].mean())
dwA = B[np.ix_(A_idx, A_idx)][np.triu_indices(12, 1)]
dwM = B[np.ix_(M_idx, M_idx)][np.triu_indices(17, 1)]
direct_within_mean = float(np.concatenate([dwA, dwM]).mean())
print(f"S1 direct-barrier gap (seed-level perm): cross {direct_cross_mean:.5f} vs "
      f"within {direct_within_mean:.5f}; T1 = {T1_obs:+.5f}, one-sided p = {p_s1:.4g}")

# ---------------------------------------------------------------- theta grid
qs = np.linspace(0.005, 0.995, 100)
thetas = np.quantile(allb, qs)

# ---------------------------------------------------------------- S2: modularity curve + max test
# vectorized fixed-partition modularity over label permutations
rng = np.random.default_rng(RNG_S2)
Z = np.stack([rng.permutation(lab29) for _ in range(NPERM)]).astype(float)  # NPERM x 29
z_obs = lab29.astype(float)[None, :]

def Q_curve(Zmat):
    """modularity of the 2-group partition for each label row, each theta."""
    out = np.zeros((Zmat.shape[0], len(thetas)))
    for t, th in enumerate(thetas):
        Aadj = (B29 <= th).astype(float); np.fill_diagonal(Aadj, 0)
        m = Aadj.sum() / 2
        if m == 0:
            continue
        v = Zmat @ Aadj                                  # P x 29
        dA = v.sum(1)                                    # sum of degrees in group 1
        eA2 = (v * Zmat).sum(1)                          # 2 * edges within group 1
        e_cross = dA - eA2
        e_within = m - e_cross
        out[:, t] = e_within / m - (dA**2 + (2 * m - dA)**2) / (4 * m**2)
    return out

Q_obs = Q_curve(z_obs)[0]
Q_null = Q_curve(Z)
T2_obs = float(Q_obs.max())
theta_at_maxQ = float(thetas[int(Q_obs.argmax())])
null_max = Q_null.max(1)
p_s2 = float((np.sum(null_max >= T2_obs) + 1) / (NPERM + 1))
q_null_lo = np.percentile(Q_null, 2.5, axis=0)
q_null_hi = np.percentile(Q_null, 97.5, axis=0)
print(f"S2 modularity: max_theta Q = {T2_obs:.4f} at theta = {theta_at_maxQ:.4f} "
      f"(pair-quantile {qs[int(Q_obs.argmax())]:.3f}); exact max-over-theta perm p = {p_s2:.4g}")
n_above_band = int((Q_obs > q_null_hi).sum())
# POST-HOC (not pre-registered; chosen after seeing Q_obs sits above the pointwise band
# at most thetas while the pre-registered max test is dominated by the noisy sparse
# regime): mean-over-theta summary with the same permutation null.
T2b_obs = float(Q_obs.mean())
p_s2_posthoc_mean = float((np.sum(Q_null.mean(1) >= T2b_obs) + 1) / (NPERM + 1))
print(f"   Q_obs above pointwise 95% null band at {n_above_band}/100 thetas; "
      f"POST-HOC mean-over-theta Q = {T2b_obs:.4f}, perm p = {p_s2_posthoc_mean:.4g}")

# ---------------------------------------------------------------- D1-D3 + exploratory sweep
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

lab60 = np.zeros(60, int)          # 0 = hedger
lab60[A_idx] = 1; lab60[M_idx] = 2

cross_full = (np.ix_(A_idx, M_idx))
f_cross, f_wA, f_wM, f_within = [], [], [], []
ncomp, giant = [], []
hops_med_cross, hops_max_cross, frac_reach_cross, hops_med_within = [], [], [], []
ari_curve, nmi_curve, ncomm_curve = [], [], []
for th in thetas:
    Adj60 = (B <= th).astype(float); np.fill_diagonal(Adj60, 0)
    # D1 direct-connection fractions (ECDFs by pair type)
    f_cross.append(float((B[cross_full] <= th).mean()))
    f_wA.append(float((dwA <= th).mean()))
    f_wM.append(float((dwM <= th).mean()))
    f_within.append(float((np.concatenate([dwA, dwM]) <= th).mean()))
    # D2 components
    nc, labels_cc = connected_components(csr_matrix(Adj60), directed=False)
    ncomp.append(int(nc)); giant.append(int(np.bincount(labels_cc).max()))
    # D3 chain lengths (hops) through any of the 60 seeds
    D = shortest_path(csr_matrix(Adj60), method="D", unweighted=True)
    dc = D[cross_full].ravel()
    reach = np.isfinite(dc)
    frac_reach_cross.append(float(reach.mean()))
    hops_med_cross.append(float(np.median(dc[reach])) if reach.any() else None)
    hops_max_cross.append(float(dc[reach].max()) if reach.any() else None)
    dwithin = np.concatenate([D[np.ix_(A_idx, A_idx)][np.triu_indices(12, 1)],
                              D[np.ix_(M_idx, M_idx)][np.triu_indices(17, 1)]])
    rw = np.isfinite(dwithin)
    hops_med_within.append(float(np.median(dwithin[rw])) if rw.any() else None)
    # exploratory community detection
    G = nx.from_numpy_array(Adj60)
    if G.number_of_edges() > 0:
        comms = greedy_modularity_communities(G)
        cl = np.zeros(60, int)
        for k, com in enumerate(comms):
            for node in com:
                cl[node] = k
        ncomm_curve.append(len(comms))
        ari_curve.append(float(adjusted_rand_score(lab60, cl)))
        nmi_curve.append(float(normalized_mutual_info_score(lab60, cl)))
    else:
        ncomm_curve.append(60); ari_curve.append(0.0); nmi_curve.append(0.0)

f_cross = np.array(f_cross); f_within = np.array(f_within)
gap_curve = f_within - f_cross
i_gapmax = int(gap_curve.argmax())
# smallest theta where median cross chain is direct (1 hop)
theta_direct = next((float(thetas[i]) for i, h in enumerate(hops_med_cross)
                     if h is not None and h <= 1.0), None)
# smallest theta with all cross pairs reachable
theta_all_reach = next((float(thetas[i]) for i, f in enumerate(frac_reach_cross) if f == 1.0), None)
print(f"\nD1 gap curve peaks at theta = {thetas[i_gapmax]:.4f}: "
      f"f_within = {f_within[i_gapmax]:.3f} vs f_cross = {f_cross[i_gapmax]:.3f} "
      f"(gap {gap_curve[i_gapmax]:.3f})")
print(f"D3 all cross pairs reachable from theta = {theta_all_reach}, "
      f"median cross chain reaches 1 hop at theta = {theta_direct}")
i_best_ari = int(np.argmax(ari_curve))
print(f"EXPL best ARI (post-hoc) = {ari_curve[i_best_ari]:.3f} at theta = {thetas[i_best_ari]:.4f} "
      f"(NMI {nmi_curve[i_best_ari]:.3f}, {ncomm_curve[i_best_ari]} communities)")

# D4 chain gain
chain_gain_cross = B[cross_full] - C[cross_full]
chain_gain_within = np.concatenate([dwA, dwM]) - np.concatenate([wA, wM])
print(f"D4 chain gain (direct - chain barrier): cross median "
      f"{np.median(chain_gain_cross):.4f}, within median {np.median(chain_gain_within):.4f}")

# ---------------------------------------------------------------- save
res = dict(
    data=dict(n_seeds=60, n_aligned=12, n_misaligned=17, n_hedger=31,
              n_cross_pairs=204, n_within_pairs=202,
              barrier_min=float(allb.min()), barrier_median=float(np.median(allb)),
              barrier_max=float(allb.max())),
    primary_chain_barrier=dict(
        stat="mean chain (minimax path) barrier, cross - within, committed seeds",
        cross_mean=chain_cross_mean, cross_mean_ci95_bootseeds=ci_cross,
        within_mean=chain_within_mean, within_mean_ci95_bootseeds=ci_within,
        T_obs=float(T_obs),
        p_one_sided_perm=p_primary, n_perm=NPERM, rng_seed=RNG_PRIMARY,
        chain_cross_median=float(np.median(C[cross_full])),
        chain_within_median=float(np.median(np.concatenate([wA, wM])))),
    s1_direct_barrier_seed_level=dict(
        cross_mean=direct_cross_mean, within_mean=direct_within_mean,
        T1_obs=float(T1_obs), p_one_sided_perm=p_s1, n_perm=NPERM, rng_seed=RNG_S1,
        note="dependence-aware redo of README Mann-Whitney p=2e-15"),
    s2_modularity=dict(
        Q_obs_curve=[float(x) for x in Q_obs],
        Q_null_lo=[float(x) for x in q_null_lo], Q_null_hi=[float(x) for x in q_null_hi],
        maxQ=T2_obs, theta_at_maxQ=theta_at_maxQ,
        p_max_over_theta=p_s2, n_perm=NPERM, rng_seed=RNG_S2,
        n_thetas_above_pointwise_band=n_above_band,
        posthoc_meanQ=T2b_obs, posthoc_meanQ_perm_p=p_s2_posthoc_mean,
        posthoc_note="mean-over-theta summary chosen AFTER seeing the curve; "
                     "pre-registered inference is p_max_over_theta only"),
    theta_grid=dict(quantiles=[float(x) for x in qs], thetas=[float(x) for x in thetas]),
    d1_direct_connection=dict(
        f_cross=[float(x) for x in f_cross], f_within=[float(x) for x in f_within],
        f_withinA=f_wA, f_withinM=f_wM,
        gap_peak_theta=float(thetas[i_gapmax]), gap_peak=float(gap_curve[i_gapmax]),
        f_cross_at_peak=float(f_cross[i_gapmax]), f_within_at_peak=float(f_within[i_gapmax])),
    d2_components=dict(n_components=ncomp, giant_size=giant),
    d3_chains=dict(hops_median_cross=hops_med_cross, hops_max_cross=hops_max_cross,
                   frac_cross_reachable=frac_reach_cross, hops_median_within=hops_med_within,
                   theta_all_cross_reachable=theta_all_reach,
                   theta_median_cross_direct=theta_direct),
    d4_chain_gain=dict(cross_median=float(np.median(chain_gain_cross)),
                       within_median=float(np.median(chain_gain_within)),
                       cross_mean=float(np.mean(chain_gain_cross)),
                       within_mean=float(np.mean(chain_gain_within))),
    exploratory_communities=dict(ari=ari_curve, nmi=nmi_curve, n_communities=ncomm_curve,
                                 best_ari=float(ari_curve[i_best_ari]),
                                 best_ari_theta=float(thetas[i_best_ari]),
                                 note="POST-HOC, no test"),
    runtime_sec=float(time.time() - t0),
)
with open(f"{OUT}/results.json", "w") as f:
    json.dump(res, f, indent=1)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax = axes[0, 0]
ax.plot(thetas, f_within, label="within-worldview pairs", color="tab:blue")
ax.plot(thetas, f_cross, label="cross-worldview pairs", color="tab:red")
ax.fill_between(thetas, f_cross, f_within, alpha=0.15, color="gray")
ax.axvline(0.05, ls=":", c="k", lw=1, label="paper thr 0.05")
ax.set_xscale("log"); ax.set_xlabel("theta (barrier threshold)")
ax.set_ylabel("fraction directly connected")
ax.set_title("D1: direct connection ECDF by pair type"); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(thetas, Q_obs, color="tab:purple", label="Q(worldview partition)")
ax.fill_between(thetas, q_null_lo, q_null_hi, alpha=0.2, color="gray",
                label="perm null 95% band (pointwise)")
ax.axvline(theta_at_maxQ, ls="--", c="tab:purple", lw=1)
ax.set_xscale("log"); ax.set_xlabel("theta"); ax.set_ylabel("modularity Q")
ax.set_title(f"S2: worldview modularity, max-test p={p_s2:.3g}"); ax.legend(fontsize=8)

ax = axes[1, 0]
hm = [h if h is not None else np.nan for h in hops_med_cross]
ax.plot(thetas, hm, color="tab:red", label="median hops, cross pairs")
hw = [h if h is not None else np.nan for h in hops_med_within]
ax.plot(thetas, hw, color="tab:blue", label="median hops, within pairs")
ax2 = ax.twinx()
ax2.plot(thetas, frac_reach_cross, color="tab:green", ls="--", label="frac cross reachable")
ax2.set_ylabel("fraction reachable", color="tab:green")
ax.set_xscale("log"); ax.set_xlabel("theta"); ax.set_ylabel("median chain length (hops)")
ax.set_title("D3: chain lengths"); ax.legend(fontsize=8, loc="upper right")

ax = axes[1, 1]
bins = np.logspace(np.log10(2e-4), np.log10(0.4), 30)
ax.hist(np.concatenate([wA, wM]), bins=bins, alpha=0.6, label="chain barrier, within", color="tab:blue")
ax.hist(C[cross_full].ravel(), bins=bins, alpha=0.6, label="chain barrier, cross", color="tab:red")
ax.hist(b_pairs, bins=bins, histtype="step", color="k", lw=1, label="direct (all committed pairs)")
ax.set_xscale("log"); ax.set_xlabel("barrier"); ax.set_ylabel("pairs")
ax.set_title(f"PRIMARY: chain barriers, perm p={p_primary:.3g}"); ax.legend(fontsize=8)
fig.suptitle("Connectivity graph vs worldview (60-seed barrier census)", y=0.995)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_connectivity_curves.png", dpi=150)
print(f"\nsaved results.json + fig  [{time.time()-t0:.1f}s]")
