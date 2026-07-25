"""
Hierarchical decomposition of NLA-SAE agreement (analysis-only, reuses saved results).

===================== PRE-REGISTRATION (written before running) =====================
The nla_sae_compare project reports two independently-trained interpretability tools
(kitft NLA -> English description; andyrdt SAE -> sparse code) agree on the same
Qwen2.5-7B activations: Metric A lift +0.189, Metric B Spearman rho ~= 0.213. Its own
limitation flags the confound: agreement may reflect shallow TOPIC identity, not
fine-grained feature semantics. Same-document tokens share a topic, so they can be
similar in BOTH modalities for topic reasons alone.

We decompose agreement into a topic-level (between-document) and a token-level
(within-document) component. The token-level component is the honest feature-agreement
estimate.

PRIMARY TEST (pre-stated):
  Estimand: rho_within = Spearman corr(S_text, S_sae) over same-document token pairs
            only (topic held constant; 36 docs x C(6,2)=15 pairs).
  H0: within-document agreement is no greater than a topic-PRESERVING null that scrambles
      the SAE<->NLA token correspondence inside each document.
  Test: within-document label-permutation, 2000 perms, seed 0, one-sided.
        p = (1 + #{rho_perm >= rho_obs}) / (1 + 2000).
  CI: 95% CI for rho_within via delete-one-document jackknife (36 leave-one-out fits).
  Decision: token-level agreement supported iff rho_within>0 AND perm p<0.05 AND
            jackknife CI excludes 0.

SECONDARY (labeled): Mundlak within/between pair-regression (clusters = unordered
  document-pairs) -> b_within (token) vs b_between (topic), jackknife CIs; fraction of
  total agreement attributable to topic vs token; topic-level rho over doc-pair means.

Validity check first: reconstructed total rho(S_text,S_sae) and baseline pair similarity
must match published 0.213 / 0.2397 before trusting the decomposition.
=====================================================================================
"""
import json, os, itertools
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESULTS = "/Users/steven/nla_sae_compare/results_20260701_162505.json"
OUT = "/Users/steven/nla-hierarchical/results.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 0
N_PERM = 2000

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- load records
d = json.load(open(RESULTS))
recs = d["records"]
N = len(recs)
docs = np.array([r["doc"] for r in recs])
uniq_docs = sorted(set(docs.tolist()))
print(f"N tokens = {N}, N docs = {len(uniq_docs)}")

# ---------------------------------------------------------------- S_text (re-embed)
import torch
from transformers import AutoModel, AutoTokenizer

dev = "mps" if torch.backends.mps.is_available() else "cpu"
etok = AutoTokenizer.from_pretrained(EMBED_MODEL)
emodel = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

texts = [r["nla_text"] for r in recs]

def embed(text_list):
    out = []
    for j in range(0, len(text_list), 32):
        batch = text_list[j:j + 32]
        enc = etok(batch, padding=True, truncation=True, max_length=256,
                   return_tensors="pt").to(dev)
        with torch.no_grad():
            m = emodel(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (m * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(out).numpy()

E = embed(texts).astype(np.float64)
S_text = E @ E.T
np.fill_diagonal(S_text, 1.0)

# ---------------------------------------------------------------- S_sae (top-20 codes)
feat_cols = {}
for r in recs:
    for tf in r["top_features"]:
        if tf["feat"] not in feat_cols:
            feat_cols[tf["feat"]] = len(feat_cols)
K = len(feat_cols)
M = np.zeros((N, K), dtype=np.float64)
for i, r in enumerate(recs):
    for tf in r["top_features"]:
        M[i, feat_cols[tf["feat"]]] = tf["val"]
Mn = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
S_sae = Mn @ Mn.T
np.fill_diagonal(S_sae, 1.0)
print(f"SAE feature-space dim (distinct top-20 feats) = {K}")

# top-20 feature SETS (for reproducing Metric A)
feat_sets = [set(tf["feat"] for tf in r["top_features"]) for r in recs]

# ---------------------------------------------------------------- helpers
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = (np.sqrt((ra*ra).sum()) * np.sqrt((rb*rb).sum()))
    if denom == 0:
        return 0.0
    return float((ra*rb).sum() / denom)

def pair_indices(idx):
    """all i<j pairs among a set of token indices -> (I, J) arrays"""
    idx = np.asarray(idx)
    I, J = np.triu_indices(len(idx), k=1)
    return idx[I], idx[J]

# global upper-triangle pairs
IU, JU = np.triu_indices(N, k=1)
xt_all = S_sae[IU, JU]
yt_all = S_text[IU, JU]
same_doc_all = (docs[IU] == docs[JU])

# ---------------------------------------------------------------- VALIDITY CHECKS
total_rho = spearman(yt_all, xt_all)           # compare to published 0.213
baseline_pair_sim = float(yt_all.mean())        # compare to published 0.2397

# reproduce Metric A (share-a-feature lift) using saved top-20 sets
support = {}
for i, fs in enumerate(feat_sets):
    for f in fs:
        support.setdefault(f, []).append(i)
qualifying = {f: ts for f, ts in support.items() if len(ts) >= 3}
lifts = []
for f, ts in qualifying.items():
    ii, jj = pair_indices(ts)
    lifts.append(float(S_text[ii, jj].mean()) - baseline_pair_sim)
lifts = np.array(lifts)
metricA_lift = float(lifts.mean())
metricA_fracpos = float((lifts > 0).mean())
metricA_nfeat = len(qualifying)

print("\n--- VALIDITY (reconstruction vs published) ---")
print(f"total Spearman rho(S_text,S_sae) = {total_rho:.4f}   (published 0.213)")
print(f"baseline pair sim               = {baseline_pair_sim:.4f}   (published 0.2397)")
print(f"Metric A lift                   = {metricA_lift:.4f}   (published 0.189)")
print(f"Metric A frac positive          = {metricA_fracpos:.4f}   (published 0.879)")
print(f"Metric A n_features (>=3)        = {metricA_nfeat}   (published 522)")

# ---------------------------------------------------------------- PRIMARY: token-level rho_within
def within_pairs_mask():
    return same_doc_all
sd_mask = within_pairs_mask()
x_within = xt_all[sd_mask]
y_within = yt_all[sd_mask]
n_within_pairs = int(sd_mask.sum())
rho_within_obs = spearman(y_within, x_within)

# between (cross-doc) pairs, for comparison
x_between = xt_all[~sd_mask]
y_between = yt_all[~sd_mask]
rho_between_pairs = spearman(y_between, x_between)
n_between_pairs = int((~sd_mask).sum())

# ---- within-document permutation null (topic preserving) ----
# scramble SAE<->NLA correspondence inside each doc: permute S_sae rows/cols within
# each document independently, recompute rho over same-doc pairs.
doc_to_idx = {dc: np.where(docs == dc)[0] for dc in uniq_docs}
# precompute same-doc pair list as (i,j) with i<j
sd_I, sd_J = IU[sd_mask], JU[sd_mask]
y_sd = S_text[sd_I, sd_J]  # NLA side fixed

def perm_rho_within(rng_local):
    # build a permutation of token indices that only shuffles within each doc
    perm = np.arange(N)
    for dc, idx in doc_to_idx.items():
        p = rng_local.permutation(len(idx))
        perm[idx] = idx[p]
    # permuted SAE similarity for same-doc pairs
    xi = S_sae[perm[sd_I], perm[sd_J]]
    return spearman(y_sd, xi)

perm_rng = np.random.default_rng(SEED)
null = np.array([perm_rho_within(perm_rng) for _ in range(N_PERM)])
p_perm = (1 + int((null >= rho_within_obs).sum())) / (1 + N_PERM)
null_mean = float(null.mean()); null_sd = float(null.std(ddof=1))

# ---------------------------------------------------------------- Mundlak within/between regression
# clusters = unordered document-pair for each token pair
cl_a = np.minimum(docs[IU], JU*0 + docs[IU])  # placeholder; recompute properly below
def cluster_ids(di, dj):
    lo = np.minimum(di, dj); hi = np.maximum(di, dj)
    return lo.astype(np.int64) * 1000 + hi.astype(np.int64)
cl = cluster_ids(docs[IU], docs[JU])

def within_between_slopes(x, y, cl):
    # group means
    order = np.argsort(cl, kind="stable")
    cls = cl[order]; xs = x[order]; ys = y[order]
    uniq, start = np.unique(cls, return_index=True)
    ng = np.diff(np.append(start, len(cls)))
    # per-group means via reduceat
    sx = np.add.reduceat(xs, start); sy = np.add.reduceat(ys, start)
    mx = sx / ng; my = sy / ng
    # map back
    inv = np.repeat(np.arange(len(uniq)), ng)
    gmx = mx[inv]; gmy = my[inv]
    # within slope (fixed-effects): regress centered y on centered x
    xc = xs - gmx; yc = ys - gmy
    denw = (xc*xc).sum()
    b_within = float((xc*yc).sum()/denw) if denw > 0 else np.nan
    # between slope: regress group-mean y on group-mean x, weighted by group size
    w = ng.astype(np.float64)
    wmx = (w*mx).sum()/w.sum(); wmy = (w*my).sum()/w.sum()
    dbx = mx - wmx
    denb = (w*dbx*dbx).sum()
    b_between = float((w*dbx*(my-wmy)).sum()/denb) if denb > 0 else np.nan
    # total slope
    xcm = x - x.mean(); ycm = y - y.mean()
    b_total = float((xcm*ycm).sum()/(xcm*xcm).sum())
    return b_within, b_between, b_total, len(uniq)

b_within, b_between, b_total, n_clusters = within_between_slopes(xt_all, yt_all, cl)

# topic-level rho: doc-pair mean S_text vs mean S_sae across clusters
def cluster_mean_rho(x, y, cl):
    order = np.argsort(cl, kind="stable")
    cls = cl[order]; xs = x[order]; ys = y[order]
    uniq, start = np.unique(cls, return_index=True)
    ng = np.diff(np.append(start, len(cls)))
    mx = np.add.reduceat(xs, start)/ng
    my = np.add.reduceat(ys, start)/ng
    return spearman(my, mx), len(uniq)
topic_rho, n_docpairs = cluster_mean_rho(xt_all, yt_all, cl)

# ---------------------------------------------------------------- delete-one-document JACKKNIFE
def stats_on_subset(keep_docs):
    keep = np.isin(docs, list(keep_docs))
    kidx = np.where(keep)[0]
    dsub = docs[kidx]
    Ik, Jk = np.triu_indices(len(kidx), k=1)
    gi = kidx[Ik]; gj = kidx[Jk]
    x = S_sae[gi, gj]; y = S_text[gi, gj]
    sd = (dsub[Ik] == dsub[Jk])
    rho_w = spearman(y[sd], x[sd])
    clk = cluster_ids(dsub[Ik], dsub[Jk])
    bw, bb, bt, _ = within_between_slopes(x, y, clk)
    trho, _ = cluster_mean_rho(x, y, clk)
    total = spearman(y, x)
    return dict(rho_within=rho_w, b_within=bw, b_between=bb, b_total=bt,
                topic_rho=trho, total_rho=total)

jk = {k: [] for k in ["rho_within", "b_within", "b_between", "b_total",
                       "topic_rho", "total_rho"]}
for d_out in uniq_docs:
    keep = [dc for dc in uniq_docs if dc != d_out]
    s = stats_on_subset(keep)
    for k in jk:
        jk[k].append(s[k])

def jack_ci(full_value, replicates):
    reps = np.array(replicates, dtype=np.float64)
    n = len(reps)
    mean = reps.mean()
    se = np.sqrt((n - 1) / n * ((reps - mean) ** 2).sum())
    # bias-corrected jackknife estimate
    theta_jack = n * full_value - (n - 1) * mean
    lo, hi = full_value - 1.96 * se, full_value + 1.96 * se
    return dict(estimate=float(full_value), jack_mean=float(mean), se=float(se),
                ci95=[float(lo), float(hi)], jack_bias_corrected=float(theta_jack))

ci_rho_within = jack_ci(rho_within_obs, jk["rho_within"])
ci_b_within = jack_ci(b_within, jk["b_within"])
ci_b_between = jack_ci(b_between, jk["b_between"])
ci_b_total = jack_ci(b_total, jk["b_total"])
ci_topic_rho = jack_ci(topic_rho, jk["topic_rho"])
ci_total_rho = jack_ci(total_rho, jk["total_rho"])

# fraction of total slope that is topic-level vs token-level (descriptive)
# additive Mundlak-style: contribution of between vs within to total covariance
def cov_decomp(x, y, cl):
    order = np.argsort(cl, kind="stable")
    cls = cl[order]; xs = x[order]; ys = y[order]
    uniq, start = np.unique(cls, return_index=True)
    ng = np.diff(np.append(start, len(cls)))
    inv = np.repeat(np.arange(len(uniq)), ng)
    mx = (np.add.reduceat(xs, start)/ng)[inv]
    my = (np.add.reduceat(ys, start)/ng)[inv]
    gx = xs.mean(); gy = ys.mean()
    cov_total = ((xs-gx)*(ys-gy)).mean()
    cov_between = ((mx-gx)*(my-gy)).mean()
    cov_within = ((xs-mx)*(ys-my)).mean()
    return cov_total, cov_between, cov_within
cov_total, cov_between, cov_within = cov_decomp(xt_all, yt_all, cl)
frac_topic = float(cov_between / cov_total) if cov_total != 0 else np.nan
frac_token = float(cov_within / cov_total) if cov_total != 0 else np.nan

# ---------------------------------------------------------------- report
print("\n--- PRIMARY: token-level within-document agreement ---")
print(f"rho_within (same-doc pairs, n={n_within_pairs}) = {rho_within_obs:.4f}")
print(f"  perm null mean={null_mean:.4f} sd={null_sd:.4f}  one-sided p={p_perm:.4g}")
print(f"  jackknife 95% CI = [{ci_rho_within['ci95'][0]:.4f}, {ci_rho_within['ci95'][1]:.4f}]")
print(f"rho_between (cross-doc pairs, n={n_between_pairs}) = {rho_between_pairs:.4f}")
print("\n--- SECONDARY: Mundlak within/between & topic decomposition ---")
print(f"b_within (token) = {b_within:.4f}  CI {ci_b_within['ci95']}")
print(f"b_between (topic)= {b_between:.4f}  CI {ci_b_between['ci95']}")
print(f"b_total          = {b_total:.4f}  CI {ci_b_total['ci95']}")
print(f"topic_rho (doc-pair means) = {topic_rho:.4f}  CI {ci_topic_rho['ci95']}")
print(f"cov decomposition: total={cov_total:.5f} between={cov_between:.5f} within={cov_within:.5f}")
print(f"  frac topic-level = {frac_topic:.3f}   frac token-level = {frac_token:.3f}")

out = {
    "meta": {
        "n_tokens": N, "n_docs": len(uniq_docs), "tokens_per_doc": 6,
        "seed": SEED, "n_perm": N_PERM, "embed_model": EMBED_MODEL,
        "sae_feature_space_dim": K,
        "note_top20_truncation": "SAE codes reconstructed from saved top-20 features "
            "per token (mean 34 active); dominant components retained.",
    },
    "validity_reconstruction": {
        "total_spearman_rho": total_rho, "published_metricB_rho": 0.213,
        "baseline_pair_sim": baseline_pair_sim, "published_baseline": 0.2397,
        "metricA_lift": metricA_lift, "published_metricA_lift": 0.189,
        "metricA_frac_positive": metricA_fracpos, "published_metricA_fracpos": 0.879,
        "metricA_n_features_ge3": metricA_nfeat, "published_n_features": 522,
    },
    "primary_token_level": {
        "rho_within": rho_within_obs,
        "n_same_doc_pairs": n_within_pairs,
        "perm_null_mean": null_mean, "perm_null_sd": null_sd,
        "perm_p_one_sided": p_perm,
        "jackknife_ci95": ci_rho_within["ci95"],
        "jackknife_se": ci_rho_within["se"],
        "decision_supported": bool(rho_within_obs > 0 and p_perm < 0.05
                                   and ci_rho_within["ci95"][0] > 0),
    },
    "between_doc_pairs": {
        "rho_between": rho_between_pairs, "n_cross_doc_pairs": n_between_pairs,
    },
    "secondary_mundlak": {
        "b_within_token": {"value": b_within, **ci_b_within},
        "b_between_topic": {"value": b_between, **ci_b_between},
        "b_total": {"value": b_total, **ci_b_total},
        "n_clusters_docpairs": n_clusters,
        "cov_total": cov_total, "cov_between_topic": cov_between,
        "cov_within_token": cov_within,
        "frac_topic_level": frac_topic, "frac_token_level": frac_token,
    },
    "secondary_topic_rho": {
        "topic_rho_docpair_means": topic_rho, "n_docpairs": n_docpairs,
        "jackknife_ci95": ci_topic_rho["ci95"],
    },
    "total_rho_jackknife": ci_total_rho,
    "jackknife_replicates": {k: [float(v) for v in vals] for k, vals in jk.items()},
}
json.dump(out, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
