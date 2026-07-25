"""Data-driven knee threshold for the 60x60 barrier census (analysis-only).

Loads the saved permutation-aligned barrier matrix from stats/results.json,
computes the pre-registered Kneedle elbow threshold on the sorted pairwise
barriers, re-runs the connectivity census at it, and runs the pre-stated
secondary analyses (Otsu, GMM crossing, jackknife, percolation, hedger-chain).
Writes results.json and two figures next to this script. No training, no GPU.
"""
import json
import os

import numpy as np
from scipy import stats as sstats
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = "/Users/steven/moral_framework_misgen/stats/results.json"

# ---------------------------------------------------------------- load + verify
d = json.load(open(STATS))
Bm = np.array(d["basins"]["barrier_matrix"])
scores = np.array(d["multimodality"]["scores"])[:60]
assert Bm.shape == (60, 60) and np.allclose(Bm, Bm.T) and np.abs(np.diag(Bm)).max() == 0
ca = d["basins"]["cluster_align"]["0"]
assert abs(scores.mean() - ca["mean_align"]) < 1e-9
assert int((scores > 0.8).sum()) == ca["n_hi"] == 12
assert int((scores < 0.2).sum()) == ca["n_lo"] == 17

iu = np.triu_indices(60, 1)
allb = Bm[iu]
M = len(allb)
hi = scores > 0.8          # aligned
lo = scores < 0.2          # misaligned
mid = ~hi & ~lo            # hedgers


def knee(vals):
    """Pre-registered Kneedle elbow: max vertical distance below the diagonal
    of the normalized sorted-value curve."""
    b = np.sort(np.asarray(vals, float))
    n = len(b)
    x = np.arange(n) / (n - 1)
    y = (b - b[0]) / (b[-1] - b[0])
    i = int(np.argmax(x - y))
    return float(b[i]), i, x, y


def census(B, thr):
    adj = (B <= thr).astype(int)
    np.fill_diagonal(adj, 1)
    nc, labels = connected_components(adj, directed=False)
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    return int(nc), labels, sizes


def verdict(labels, hi_mask, lo_mask):
    hs = set(labels[hi_mask].tolist())
    ls = set(labels[lo_mask].tolist())
    shared = hs & ls
    return dict(aligned_labels=sorted(hs), misaligned_labels=sorted(ls),
                n_shared=len(shared), shared=bool(shared))


def clopper_pearson(k, n, alpha=0.05):
    lob = sstats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hib = sstats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lob), float(hib)


out = {"data": dict(n_seeds=60, n_pairs=M, n_aligned=12, n_misaligned=17, n_hedger=31,
                    barrier_min=float(allb.min()), barrier_median=float(np.median(allb)),
                    barrier_max=float(allb.max()))}

# ---------------------------------------------------------------- PRIMARY: knee
thr_knee, i_knee, xs, ys = knee(allb)
nc, labels, sizes = census(Bm, thr_knee)
v = verdict(labels, hi, lo)
cross = Bm[np.ix_(np.where(hi)[0], np.where(lo)[0])].ravel()   # 12*17 = 204
k_direct = int((cross <= thr_knee).sum())
ci = clopper_pearson(k_direct, len(cross))
out["primary"] = dict(
    method="Kneedle max-distance-below-diagonal on sorted pairwise barriers",
    thr_knee=thr_knee, knee_index=i_knee, knee_quantile=i_knee / (M - 1),
    n_components=nc, component_sizes=sizes, worldview=v,
    cross_pairs_direct=k_direct, cross_pairs_total=len(cross),
    frac_cross_direct=k_direct / len(cross), frac_cross_direct_ci95=ci,
    frac_all_pairs_leq_thr=float((allb <= thr_knee).mean()),
)
print(f"PRIMARY  thr_knee={thr_knee:.5f} (pair-quantile {i_knee/(M-1):.3f})")
print(f"  census: {nc} components, sizes {sizes}")
print(f"  aligned+misaligned share a component: {v['shared']}")
print(f"  direct aligned-misaligned edges: {k_direct}/{len(cross)} "
      f"= {k_direct/len(cross):.3f}  CI95 [{ci[0]:.3f},{ci[1]:.3f}]")

# component composition at the knee
comp = []
for lab in range(nc):
    idx = np.where(labels == lab)[0]
    comp.append(dict(size=int(len(idx)), n_aligned=int(hi[idx].sum()),
                     n_misaligned=int(lo[idx].sum()), n_hedger=int(mid[idx].sum()),
                     mean_align=float(scores[idx].mean()),
                     seeds=idx.tolist()))
out["primary"]["components"] = comp
for c in comp:
    print(f"    comp size={c['size']:2d}  A={c['n_aligned']:2d} M={c['n_misaligned']:2d} "
          f"H={c['n_hedger']:2d}  mean_align={c['mean_align']:.2f}")

# ------------------------------------------------- S1: Otsu + GMM alternatives
def otsu(vals, nbins=512):
    h, edges = np.histogram(vals, bins=nbins)
    cbin = 0.5 * (edges[:-1] + edges[1:])
    w = h.astype(float)
    p = w / w.sum()
    best_t, best_var = None, -1.0
    csum = np.cumsum(p)
    cmean = np.cumsum(p * cbin)
    mu = cmean[-1]
    for t in range(1, nbins):
        w0, w1 = csum[t - 1], 1 - csum[t - 1]
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = cmean[t - 1] / w0
        m1 = (mu - cmean[t - 1]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, 0.5 * (edges[t] + edges[t - 1])
    return float(best_t)


thr_otsu = otsu(allb)
lg = np.log10(allb).reshape(-1, 1)
gm = GaussianMixture(n_components=2, n_init=10, random_state=0, reg_covar=1e-6).fit(lg)
grid = np.linspace(lg.min(), lg.max(), 20000).reshape(-1, 1)
post = gm.predict_proba(grid)
lo_comp = int(np.argmin(gm.means_.ravel()))
sw = np.where(np.diff((post[:, lo_comp] > 0.5).astype(int)) != 0)[0]
thr_gmm = float(10 ** grid[sw[0], 0]) if len(sw) else float("nan")

out["S1_alternatives"] = {}
for name, thr in [("otsu", thr_otsu), ("gmm_log_crossing", thr_gmm)]:
    ncx, labx, sizx = census(Bm, thr)
    vx = verdict(labx, hi, lo)
    kx = int((cross <= thr).sum())
    out["S1_alternatives"][name] = dict(thr=thr, n_components=ncx, sizes=sizx,
                                        shared=vx["shared"],
                                        cross_direct=f"{kx}/{len(cross)}")
    print(f"S1 {name:16s} thr={thr:.5f} -> {ncx} comps {sizx}, shared={vx['shared']}, "
          f"direct cross {kx}/{len(cross)}")

# ------------------------------------------------- S2: leave-one-seed-out jackknife
jk_thr, jk_nc, jk_shared = [], [], []
for s in range(60):
    keep = np.ones(60, bool)
    keep[s] = False
    B2 = Bm[np.ix_(keep, keep)]
    a2 = B2[np.triu_indices(59, 1)]
    t2, _, _, _ = knee(a2)
    nc2, lab2, _ = census(B2, t2)
    v2 = verdict(lab2, hi[keep], lo[keep])
    jk_thr.append(t2)
    jk_nc.append(nc2)
    jk_shared.append(v2["shared"])
jk_thr = np.array(jk_thr)
out["S2_jackknife"] = dict(
    thr_min=float(jk_thr.min()), thr_max=float(jk_thr.max()),
    thr_mean=float(jk_thr.mean()),
    n_components_counts={str(k): int(vv) for k, vv in
                         zip(*np.unique(jk_nc, return_counts=True))},
    frac_shared=float(np.mean(jk_shared)))
print(f"S2 jackknife: thr in [{jk_thr.min():.5f},{jk_thr.max():.5f}], "
      f"n_comp counts {out['S2_jackknife']['n_components_counts']}, "
      f"shared in {np.mean(jk_shared)*100:.0f}% of 60 drops")

# ------------------------------------------------- S3: census curve + percolation
grid_thr = np.linspace(0, allb.max(), 800)
curve = [census(Bm, t)[0] for t in grid_thr]
mst = minimum_spanning_tree(Bm + 1e-12).toarray()   # weights > 0 required
thr_perc = float(mst.max()) - 1e-12                  # bottleneck barrier
ncp, labp, _ = census(Bm, thr_perc * 0.999)          # just below percolation
last_seeds = [int(i) for i in np.where(labp != np.bincount(labp).argmax())[0]]
out["S3_percolation"] = dict(thr_full_connect=thr_perc,
                             n_comp_just_below=int(ncp),
                             last_merging_seeds=last_seeds,
                             last_merging_scores=[float(scores[i]) for i in last_seeds],
                             curve_thr=grid_thr.tolist(), curve_ncomp=curve)
print(f"S3 percolation: graph fully connects at thr={thr_perc:.5f} "
      f"({thr_knee/thr_perc:.1f}x below the knee); last to merge: seeds {last_seeds} "
      f"scores {[round(scores[i],2) for i in last_seeds]}")

# ------------------------------------------------- S4: hedger-chain test at knee
committed = hi | lo
Bc = Bm[np.ix_(np.where(committed)[0], np.where(committed)[0])]
ncc, labc, sizc = census(Bc, thr_knee)
vc = verdict(labc, hi[committed], lo[committed])
out["S4_no_hedgers"] = dict(n_committed=int(committed.sum()), thr=thr_knee,
                            n_components=ncc, sizes=sizc, shared=vc["shared"])
print(f"S4 hedgers removed (n=29) at thr_knee: {ncc} comps {sizc}, "
      f"aligned-misaligned still connected: {vc['shared']}")

# ------------------------------------------------- POST-HOC: same checks at the
# paper's ad hoc thr = 0.05 (labeled post-hoc; motivated by the README's
# "only via chains through hedger seeds" claim, decided after seeing the data)
k05 = int((cross <= 0.05).sum())
nc05, lab05, siz05 = census(Bm, 0.05)
Bc05 = Bm[np.ix_(np.where(committed)[0], np.where(committed)[0])]
ncc05, labc05, sizc05 = census(Bc05, 0.05)
vc05 = verdict(labc05, hi[committed], lo[committed])
out["posthoc_at_paper_thr"] = dict(
    posthoc=True, thr=0.05,
    cross_direct=f"{k05}/{len(cross)}", frac_cross_direct=k05 / len(cross),
    frac_cross_direct_ci95=clopper_pearson(k05, len(cross)),
    n_components=nc05, sizes=siz05,
    no_hedgers_n_components=ncc05, no_hedgers_sizes=sizc05,
    no_hedgers_shared=vc05["shared"])
print(f"POST-HOC at paper thr=0.05: direct cross {k05}/{len(cross)} = {k05/len(cross):.3f}; "
      f"hedgers removed -> {ncc05} comps {sizc05}, shared={vc05['shared']}")

# ------------------------------------------------- descriptive cross-check (README)
within_grp = np.concatenate([
    Bm[np.ix_(np.where(hi)[0], np.where(hi)[0])][np.triu_indices(12, 1)],
    Bm[np.ix_(np.where(lo)[0], np.where(lo)[0])][np.triu_indices(17, 1)]])
mw_u, mw_p = sstats.mannwhitneyu(cross, within_grp, alternative="greater")
out["descriptive_check"] = dict(cross_mean=float(cross.mean()),
                                within_group_mean=float(within_grp.mean()),
                                n_cross=len(cross), n_within=len(within_grp),
                                mannwhitney_p=float(mw_p))
print(f"check: cross mean {cross.mean():.3f} (n={len(cross)}) vs within-group "
      f"{within_grp.mean():.3f} (n={len(within_grp)}), MW p={mw_p:.2g} "
      f"(README: 0.064 vs 0.029, p~2e-15)")

# ------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
b_sorted = np.sort(allb)
ax[0].plot(xs, ys, lw=1.5, label="sorted barriers (normalized)")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chord")
ax[0].axvline(i_knee / (M - 1), color="r", lw=1,
              label=f"knee: thr={thr_knee:.4f} (q={i_knee/(M-1):.2f})")
ax[0].set_xlabel("pair quantile")
ax[0].set_ylabel("normalized barrier")
ax[0].legend(fontsize=8)
ax[0].set_title("Pre-registered Kneedle elbow, 1770 pairs")
ax[1].step(grid_thr, curve, where="post", lw=1.5)
for t, lab, c in [(thr_knee, f"knee {thr_knee:.3f}", "r"),
                  (0.05, "paper 0.05", "gray"),
                  (thr_otsu, f"Otsu {thr_otsu:.3f}", "g"),
                  (thr_perc, f"percolation {thr_perc:.3f}", "b")]:
    ax[1].axvline(t, color=c, lw=1, ls=":", label=lab)
ax[1].set_xlabel("connectivity threshold")
ax[1].set_ylabel("n connected components")
ax[1].set_yscale("log")
ax[1].legend(fontsize=8)
ax[1].set_title("Census curve n_components(thr)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_knee_census.png"), dpi=150)
print("figure saved")

json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
print("results.json saved")
