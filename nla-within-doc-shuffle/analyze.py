"""
Within-document (topic-matched) permutation null for the nla-vs-sae Test-A lift.

Analysis-only. Reads the saved artifacts of ~/nla_sae_compare (READ-ONLY), re-embeds the
216 saved NLA texts with the same MiniLM embedder + pooling as run.py (locally cached,
CPU), reconstructs the Test-A statistic from the saved top-20 SAE features, and compares
the observed mean lift against:
  (1) PRIMARY  : K=10,000 WITHIN-document permutations of the texts (topic-matched null)
  (2) COMPARISON: K=10,000 global permutations (replicates the published shuffled control)
plus the pre-registered secondary pooled cross-document lift under null (1).

Pre-registration: see FINDINGS.md (written before this script was run).
Everything under "post_hoc" in the output JSON is post-hoc descriptive.
"""

import json
import time

import numpy as np
import torch

RESULTS = "/Users/steven/nla_sae_compare/results_20260701_162505.json"
OUT = "/Users/steven/undefined/nla-within-doc-shuffle/results.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 10_000
SEED = 0
MIN_SUPPORT = 3

t_start = time.time()
d = json.load(open(RESULTS))
recs = d["records"]
pub = d["metrics"]["agreement_A_feature_conditioned"]
N = len(recs)
docs = np.array([r["doc"] for r in recs])
texts = [r["nla_text"] for r in recs]
n_docs = len(set(docs.tolist()))
assert N == 216 and n_docs == 36

# ---- embed exactly as run.py (MiniLM, mean-pool over attention mask, L2 norm), CPU ----
from transformers import AutoModel, AutoTokenizer

emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
emb_model = AutoModel.from_pretrained(EMBED_MODEL).eval()
out = []
with torch.no_grad():
    for j in range(0, len(texts), 32):
        enc = emb_tok(texts[j:j + 32], padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
        m = emb_model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (m * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1))
E = torch.cat(out).numpy().astype(np.float64)
S = E @ E.T

iu = np.triu_indices(N, k=1)
baseline = float(S[iu].mean())  # invariant under any permutation of tokens

# ---- reconstruct Test-A qualifying features from saved top-20 lists ----
support = {}
for i, r in enumerate(recs):
    for tf in r["top_features"]:
        support.setdefault(tf["feat"], []).append(i)
qual = {f: ts for f, ts in support.items() if len(ts) >= MIN_SUPPORT}
feat_ids = sorted(qual)
F = len(feat_ids)

# flat pair arrays: (A, B) token indices, seg = feature index per pair
A_idx, B_idx, seg = [], [], []
for fi, f in enumerate(feat_ids):
    ts = qual[f]
    for k, a in enumerate(ts):
        for b in ts[k + 1:]:
            A_idx.append(a); B_idx.append(b); seg.append(fi)
A_idx = np.array(A_idx); B_idx = np.array(B_idx); seg = np.array(seg)
pair_counts = np.bincount(seg, minlength=F).astype(np.float64)
S_flat = S.ravel()

def mean_lift(perm):
    """Test-A mean lift with texts reassigned by perm (identity = observed)."""
    sims = S_flat[perm[A_idx] * N + perm[B_idx]]
    per_feat = np.bincount(seg, weights=sims, minlength=F) / pair_counts
    return per_feat, float(per_feat.mean() - baseline)

identity = np.arange(N)
per_feat_obs, obs_lift = mean_lift(identity)
frac_pos = float((per_feat_obs - baseline > 0).mean())

# same-doc / cross-doc masks over within-feature pairs (invariant under within-doc perm)
cross_mask = docs[A_idx] != docs[B_idx]
cross_all_mask = docs[iu[0]] != docs[iu[1]]
cross_baseline = float(S[iu][cross_all_mask].mean())  # invariant under within-doc perm

def cross_lift(perm):
    sims = S_flat[perm[A_idx] * N + perm[B_idx]]
    return float(sims[cross_mask].mean() - cross_baseline)

obs_cross_lift = cross_lift(identity)

# ---- permutation nulls ----
rng = np.random.default_rng(SEED)
doc_groups = [np.where(docs == g)[0] for g in range(n_docs)]

null_within = np.empty(K)
null_within_cross = np.empty(K)
null_global = np.empty(K)
for k in range(K):
    # within-document permutation: shuffle assignment inside each doc independently
    perm = identity.copy()
    for g in doc_groups:
        perm[g] = g[rng.permutation(len(g))]
    _, null_within[k] = mean_lift(perm)
    null_within_cross[k] = cross_lift(perm)
    # global permutation (published control, for comparison)
    _, null_global[k] = mean_lift(rng.permutation(N))

def one_sided_p(obs, null):
    return float((1 + (null >= obs).sum()) / (len(null) + 1))

p_within = one_sided_p(obs_lift, null_within)
p_cross = one_sided_p(obs_cross_lift, null_within_cross)
p_global = one_sided_p(obs_lift, null_global)

wm, ws = float(null_within.mean()), float(null_within.std(ddof=1))
gm, gs = float(null_global.mean()), float(null_global.std(ddof=1))
cm, cs = float(null_within_cross.mean()), float(null_within_cross.std(ddof=1))

# ---- post-hoc descriptives (labelled) ----
single_doc = np.array([len(set(docs[t] for t in qual[f])) == 1 for f in feat_ids])
post_hoc = {
    "note": "descriptive only, not pre-registered",
    "n_single_doc_features": int(single_doc.sum()),
    "obs_mean_lift_single_doc_features": float(per_feat_obs[single_doc].mean() - baseline),
    "obs_mean_lift_multi_doc_features": float(per_feat_obs[~single_doc].mean() - baseline),
    "within_feature_pairs_same_doc": int((~cross_mask).sum()),
    "within_feature_pairs_cross_doc": int(cross_mask.sum()),
    "mean_sim_same_doc_pairs_all": float(S[iu][~cross_all_mask].mean()),
    "mean_sim_cross_doc_pairs_all": cross_baseline,
}

results = {
    "config": {"K": K, "seed": SEED, "min_support": MIN_SUPPORT, "n_tokens": N,
               "n_docs": n_docs, "embed_model": EMBED_MODEL,
               "source": RESULTS},
    "reconstruction_sanity": {
        "published_baseline_pair_sim": pub["baseline_pair_sim"],
        "reconstructed_baseline_pair_sim": baseline,
        "published_n_features_ge3": pub["n_features_ge3"],
        "reconstructed_n_features_ge3": F,
        "published_mean_lift": pub["mean_lift"],
        "reconstructed_obs_mean_lift": obs_lift,
        "published_frac_positive": pub["frac_positive"],
        "reconstructed_frac_positive": frac_pos,
        "n_within_feature_pairs": int(len(A_idx)),
        "deviation": "records store only top-20 features/token (orig used full ~34 "
                     "active/token), so the qualifying-feature set is a subset",
    },
    "primary_within_doc_null": {
        "observed_mean_lift": obs_lift,
        "null_mean": wm, "null_sd": ws,
        "null_q025": float(np.quantile(null_within, 0.025)),
        "null_q975": float(np.quantile(null_within, 0.975)),
        "z": (obs_lift - wm) / ws,
        "p_one_sided_MC": p_within,
        "topic_explained_share": wm / obs_lift,
        "residual_lift_obs_minus_null_mean": obs_lift - wm,
    },
    "comparison_global_null": {
        "observed_mean_lift": obs_lift,
        "null_mean": gm, "null_sd": gs,
        "z": (obs_lift - gm) / gs,
        "p_one_sided_MC": p_global,
    },
    "secondary_cross_doc_pooled": {
        "observed_cross_doc_lift": obs_cross_lift,
        "cross_doc_baseline": cross_baseline,
        "null_mean": cm, "null_sd": cs,
        "z": (obs_cross_lift - cm) / cs if cs > 0 else None,
        "p_one_sided_MC": p_cross,
    },
    "post_hoc": post_hoc,
    "runtime_sec": round(time.time() - t_start, 1),
}

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2)
print(json.dumps(results, indent=2))
