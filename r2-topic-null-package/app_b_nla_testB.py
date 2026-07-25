"""
APPLICATION (b): full topic-matched null for the nla Test-B total rho = 0.213.

Analysis-only. READ-ONLY reuse of /Users/steven/nla_sae_compare artifacts. Re-embeds
the 216 saved NLA texts with the same MiniLM pooling as the published run, reconstructs
S_text and S_sae (from saved top-20 codes), and puts the TOTAL Spearman rho through the
reusable topic_null module (within-document permutation of S_sae).

Pre-registration: PREREG.md, application (b). Post-hoc items are labeled.
"""
import json
import time

import numpy as np
import torch

from topic_null import topic_null_pairsim, _spearman, within_group_perm, group_index

RESULTS = "/Users/steven/nla_sae_compare/results_20260701_162505.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 10_000
SEED = 0

t0 = time.time()
d = json.load(open(RESULTS))
recs = d["records"]
N = len(recs)
docs = np.array([r["doc"] for r in recs])
texts = [r["nla_text"] for r in recs]
assert N == 216 and len(set(docs.tolist())) == 36

# ---- S_text: re-embed exactly as run.py (MiniLM, mean-pool, L2 norm), CPU ----
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
mdl = AutoModel.from_pretrained(EMBED_MODEL).eval()
out = []
with torch.no_grad():
    for j in range(0, N, 32):
        enc = tok(texts[j:j + 32], padding=True, truncation=True, max_length=256,
                  return_tensors="pt")
        m = mdl(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (m * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1))
E = torch.cat(out).numpy().astype(np.float64)
S_text = E @ E.T
np.fill_diagonal(S_text, 1.0)

# ---- S_sae: cosine over saved top-20 codes (as hierarchical did) ----
feat_cols = {}
for r in recs:
    for tf in r["top_features"]:
        feat_cols.setdefault(tf["feat"], len(feat_cols))
M = np.zeros((N, len(feat_cols)))
for i, r in enumerate(recs):
    for tf in r["top_features"]:
        M[i, feat_cols[tf["feat"]]] = tf["val"]
Mn = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
S_sae = Mn @ Mn.T
np.fill_diagonal(S_sae, 1.0)

# ---- sanity: total rho + baseline must match hierarchical (0.193 / 0.2397) ----
iu, ju = np.triu_indices(N, k=1)
total_rho = _spearman(S_text[iu, ju], S_sae[iu, ju])
baseline = float(S_text[iu, ju].mean())
print(f"[sanity] reconstructed total rho = {total_rho:.4f} (hierarchical 0.1933, pub 0.213)")
print(f"[sanity] baseline pair sim       = {baseline:.4f} (hierarchical/pub 0.2397)")

# ================= PRIMARY (b): full topic null over ALL pairs =================
res_full = topic_null_pairsim(S_text, S_sae, labels=docs, K=K, seed=SEED)
print(f"\n[PRIMARY b] total rho={res_full.observed:.4f}  topic-null mean={res_full.null_mean:.4f}"
      f"  share={res_full.group_explained_share:.3f}  residual={res_full.residual:.4f}"
      f"  z={res_full.z:.1f}  p={res_full.p_one_sided:.2g}")

# ================= COMPARISON (b): GLOBAL null over ALL pairs ==================
rng = np.random.default_rng(SEED)
ya = S_text[iu, ju]
null_global = np.empty(K)
for k in range(K):
    perm = rng.permutation(N)
    null_global[k] = _spearman(ya, S_sae[perm[iu], perm[ju]])
gm, gs = float(null_global.mean()), float(null_global.std(ddof=1))
p_global = (1 + int((null_global >= res_full.observed).sum())) / (K + 1)
print(f"[COMPARE b] global-null mean={gm:.4f} sd={gs:.4f} p={p_global:.2g}  (published-style shuffle)")

# ================= POST-HOC (labeled): same-doc-only cross-check ==============
same_doc = docs[iu] == docs[ju]
res_sd = topic_null_pairsim(S_text, S_sae, labels=docs, K=K, seed=SEED,
                            pair_mask=same_doc)
# cross-doc-only, too
res_cd = topic_null_pairsim(S_text, S_sae, labels=docs, K=K, seed=SEED,
                            pair_mask=~same_doc)

results = {
    "application": "b_nla_testB_full_topic_null",
    "config": {"K": K, "seed": SEED, "n_tokens": N, "n_docs": 36,
               "embed_model": EMBED_MODEL, "source": RESULTS,
               "sae_similarity": "cosine over saved top-20 codes"},
    "sanity": {
        "reconstructed_total_rho": total_rho,
        "hierarchical_total_rho": 0.19328276736719305,
        "published_rho": 0.213,
        "reconstructed_baseline_pair_sim": baseline,
        "published_baseline": 0.2397,
        "n_all_pairs": int(len(iu)),
        "n_same_doc_pairs": int(same_doc.sum()),
        "n_cross_doc_pairs": int((~same_doc).sum()),
    },
    "primary_full_topic_null": res_full.to_dict(),
    "comparison_global_null": {
        "observed": res_full.observed, "null_mean": gm, "null_sd": gs,
        "z": (res_full.observed - gm) / gs, "p_one_sided": p_global,
    },
    "post_hoc_same_doc_only": {**res_sd.to_dict(),
        "note": "topic null on rho over same-doc pairs only; cross-check vs "
                "hierarchical rho_within=0.430 (which used a within-doc scramble too)"},
    "post_hoc_cross_doc_only": {**res_cd.to_dict(),
        "note": "topic null on rho over cross-doc pairs only"},
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(results, open("results_app_b.json", "w"), indent=2)
print(f"\nwrote results_app_b.json  ({results['runtime_sec']}s)")
