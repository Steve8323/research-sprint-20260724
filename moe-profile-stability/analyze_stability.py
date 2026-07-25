"""
Split-half reliability of moe_legibility expert profiles.
Pre-registered in FINDINGS.md before this run. Reads routing_per_text.pkl,
verifies it reproduces the published moe_data.npz / moe_tokens.json, then runs
R=200 stratified text-level splits (vectorized stats) and R=50 splits
(counter-based stats), plus a token-level split comparison. Writes results.json.
"""
import json
import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr

RNG = np.random.default_rng(0)
R_VEC = 200          # splits for vectorized stats (spec/dom/prop)
R_TOK = 50           # splits for counter-based stats (distinctive tokens, top-16)
OUTDIR = "/Users/steven/undefined/moe-profile-stability"
REPO = "/Users/steven/moe_legibility"

D = pickle.load(open(f"{OUTDIR}/routing_per_text.pkl", "rb"))
meta, texts = D["meta"], D["texts"]
NL, NE, TOPK, cats = meta["NL"], meta["NE"], meta["TOPK"], meta["cats"]
NC = len(cats)
NT = len(texts)

pub = np.load(f"{REPO}/moe_data.npz", allow_pickle=True)
pub_counts, pub_spec, pub_dom = pub["counts"], pub["spec"], pub["dom"]
pub_tokens = json.load(open(f"{REPO}/moe_tokens.json"))
profiles = json.load(open(f"{REPO}/moe_profiles.json"))
prof_units = [(p["layer"], p["expert"]) for p in profiles]  # fixed published set (16)

# ---------- per-text tensors ---------- #
ci_of = np.array([t["ci"] for t in texts])
ntok_of = np.array([len(t["tokens"]) for t in texts])
counts_text = np.zeros((NT, NL, NE), dtype=np.int32)      # selections per text
vocab = {}
tri = []                                                   # (t, l, e, tok_id, c) sparse
text_tokvec = []                                           # (t, tok_id, c)
for ti, t in enumerate(texts):
    toks = t["tokens"]
    for s in toks:
        vocab.setdefault(s, len(vocab))
    tv = {}
    for s in toks:
        tv[vocab[s]] = tv.get(vocab[s], 0) + 1
    for k, c in tv.items():
        text_tokvec.append((ti, k, c))
    le = {}
    for lid, inds in t["routes"].items():
        for p in range(inds.shape[0]):
            tid = vocab[toks[p]]
            for e in inds[p]:
                counts_text[ti, lid, int(e)] += 1
                key = (lid, int(e), tid)
                le[key] = le.get(key, 0) + 1
    for (l, e, k), c in le.items():
        tri.append((ti, l, e, k, c))
V = len(vocab)
tri = np.array(tri, dtype=np.int32)
text_tokvec = np.array(text_tokvec, dtype=np.int32)
inv_vocab = [None] * V
for s, k in vocab.items():
    inv_vocab[k] = s
strip_ok = np.array([bool(s.strip()) for s in inv_vocab])
print(f"texts={NT} vocab={V} sparse-unit-token entries={len(tri)}")

# ---------- formulas identical to moe_route.py ---------- #
def stats_from_counts(counts, tokc):
    prop = counts / np.maximum(tokc[None, None, :], 1)
    qn = prop / np.maximum(prop.sum(axis=2, keepdims=True), 1e-9)
    ent = -(qn * np.log(np.maximum(qn, 1e-12))).sum(axis=2)
    spec = 1 - ent / np.log(NC)
    dom = prop.argmax(axis=2)
    return prop, spec, dom

def counts_of(text_idx):
    counts = np.zeros((NL, NE, NC), dtype=np.int64)
    for c in range(NC):
        sel = text_idx[ci_of[text_idx] == c]
        counts[:, :, c] = counts_text[sel].sum(axis=0)
    tokc = np.array([ntok_of[text_idx[ci_of[text_idx] == c]].sum() for c in range(NC)])
    return counts, tokc

def unit_token_arrays(text_idx):
    """dense arr[NL,NE,V] of per-unit token-selection counts + global token counts."""
    m = np.isin(tri[:, 0], text_idx)
    sub = tri[m]
    arr = np.zeros((NL, NE, V), dtype=np.int32)
    np.add.at(arr, (sub[:, 1], sub[:, 2], sub[:, 3]), sub[:, 4])
    mg = np.isin(text_tokvec[:, 0], text_idx)
    g = np.zeros(V, dtype=np.int64)
    np.add.at(g, text_tokvec[mg, 1], text_tokvec[mg, 2])
    return arr, g

def distinctive_from(arr_le, g, G, l, e, topn=8):
    """identical to moe_route.distinctive: lift, c>=2, t.strip(), sort (lift,c,t) desc."""
    row = arr_le[l, e]
    tot = row.sum() or 1
    idx = np.nonzero((row >= 2) & strip_ok)[0]
    scored = [((row[k] / tot) / (g[k] / G), int(row[k]), inv_vocab[k]) for k in idx]
    scored.sort(reverse=True)
    return [t for _, _, t in scored[:topn]]

def top16(spec, arr_le, g, G):
    cands = []
    for l in range(NL):
        for e in range(NE):
            toks = distinctive_from(arr_le, g, G, l, e)
            if len(toks) >= 3:
                cands.append((float(spec[l, e]), l, e, toks))
    cands.sort(reverse=True)
    return cands[:16]

# ---------- sanity gate: reproduce published artifacts ---------- #
all_idx = np.arange(NT)
counts_all, tokc_all = counts_of(all_idx)
match_counts = float((counts_all == pub_counts).mean())
prop_all, spec_all, dom_all = stats_from_counts(counts_all, tokc_all)
match_spec = bool(np.allclose(spec_all, pub_spec))
arr_all, g_all = unit_token_arrays(all_idx)
G_all = int(g_all.sum())
tok_match = np.mean([distinctive_from(arr_all, g_all, G_all, l, e) == pub_tokens[f"{l}_{e}"]
                     for l in range(NL) for e in range(NE)])
sanity = {"counts_exact_match_frac": match_counts, "spec_allclose": match_spec,
          "tokens_exact_match_frac": float(tok_match), "total_tokens": int(tokc_all.sum())}
print("SANITY:", sanity)

# ---------- the 200 text-level splits (vectorized stats) ---------- #
splits = []
for r in range(R_VEC):
    A, B = [], []
    for c in range(NC):
        idx = np.nonzero(ci_of == c)[0]
        perm = RNG.permutation(idx)
        A += list(perm[: len(idx) // 2]); B += list(perm[len(idx) // 2:])
    splits.append((np.array(sorted(A)), np.array(sorted(B))))

prof_l = np.array([u[0] for u in prof_units]); prof_e = np.array([u[1] for u in prof_units])
rows = []
for A, B in splits:
    cA, tA = counts_of(A); cB, tB = counts_of(B)
    pA, sA, dA = stats_from_counts(cA, tA)
    pB, sB, dB = stats_from_counts(cB, tB)
    used = (cA.sum(2) > 0) & (cB.sum(2) > 0)
    rho = spearmanr(sA[used], sB[used]).statistic
    r_p = pearsonr(sA[used], sB[used]).statistic
    peakA = pA.reshape(-1, NC).max(0); peakB = pB.reshape(-1, NC).max(0)
    argA = pA.reshape(-1, NC).argmax(0); argB = pB.reshape(-1, NC).argmax(0)
    rows.append(dict(
        rho=rho, r=r_p, sb=2 * r_p / (1 + r_p), n_used=int(used.sum()),
        dom_agree=float((dA[used] == dB[used]).mean()),
        dom_agree_prof=float((dA[prof_l, prof_e] == dB[prof_l, prof_e]).mean()),
        peak_argmax_agree=float((argA == argB).mean()),
        peakA=peakA, peakB=peakB))

def ci(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), lo=float(np.percentile(x, 2.5)),
                hi=float(np.percentile(x, 97.5)))

vec = {k: ci([r[k] for r in rows]) for k in
       ["rho", "r", "sb", "n_used", "dom_agree", "dom_agree_prof", "peak_argmax_agree"]}
peak_all = np.concatenate([[r["peakA"], r["peakB"]] for r in rows])   # (2R, NC)
vec["peak_prop_per_cat"] = {cats[c]: ci(peak_all[:, c]) for c in range(NC)}
print("PRIMARY  spearman rho:", vec["rho"])
print("         pearson r:", vec["r"], " SB:", vec["sb"])
print("         dom agree:", vec["dom_agree"], " dom agree (16 profiled):", vec["dom_agree_prof"])
print("         peak-argmax agree:", vec["peak_argmax_agree"])

# ---------- counter-based stats on the first R_TOK splits ---------- #
tok_rows = []
for A, B in splits[:R_TOK]:
    cA, tA = counts_of(A); cB, tB = counts_of(B)
    _, sA, _ = stats_from_counts(cA, tA); _, sB, _ = stats_from_counts(cB, tB)
    arrA, gA = unit_token_arrays(A); arrB, gB = unit_token_arrays(B)
    GA, GB = int(gA.sum()), int(gB.sum())
    tA16 = top16(sA, arrA, gA, GA); tB16 = top16(sB, arrB, gB, GB)
    setA = {(l, e) for _, l, e, _ in tA16}; setB = {(l, e) for _, l, e, _ in tB16}
    jac16 = len(setA & setB) / len(setA | setB)
    pubset = set(prof_units)
    in_half = (len(pubset & setA) + len(pubset & setB)) / 2
    jacs, empty = [], 0
    for l, e in prof_units:
        ta = set(distinctive_from(arrA, gA, GA, l, e))
        tb = set(distinctive_from(arrB, gB, GB, l, e))
        if not ta and not tb:
            empty += 1; jacs.append(1.0)
        else:
            jacs.append(len(ta & tb) / len(ta | tb))
    tok_rows.append(dict(jac16=jac16, pub_in_half_top16=in_half,
                         tok_jac_prof=float(np.mean(jacs)), n_both_empty=empty))

tokres = {k: ci([r[k] for r in tok_rows]) for k in
          ["jac16", "pub_in_half_top16", "tok_jac_prof"]}
tokres["n_both_empty_total"] = int(sum(r["n_both_empty"] for r in tok_rows))
print("TOP16 jaccard:", tokres["jac16"])
print("published-16 present per half:", tokres["pub_in_half_top16"])
print("token jaccard (16 profiled):", tokres["tok_jac_prof"])

# ---------- secondary: token-level split (context leak comparison) ---------- #
# per-position selection tensor
pos_ci, pos_counts = [], []
for t in texts:
    n = len(t["tokens"])
    arr = np.zeros((n, NL, NE), dtype=np.int8)
    for lid, inds in t["routes"].items():
        for p in range(n):
            arr[p, lid, inds[p].astype(int)] = 1
    pos_counts.append(arr)
    pos_ci += [t["ci"]] * n
pos_counts = np.concatenate(pos_counts)       # (1541, NL, NE)
pos_ci = np.array(pos_ci)

tl_rho = []
for r in range(R_VEC):
    selA = np.zeros(len(pos_ci), dtype=bool)
    for c in range(NC):
        idx = np.nonzero(pos_ci == c)[0]
        perm = RNG.permutation(idx)
        selA[perm[: len(idx) // 2]] = True
    def half_stats(mask):
        counts = np.zeros((NL, NE, NC), dtype=np.int64)
        for c in range(NC):
            counts[:, :, c] = pos_counts[mask & (pos_ci == c)].sum(0)
        tokc = np.array([(mask & (pos_ci == c)).sum() for c in range(NC)])
        return counts, stats_from_counts(counts, tokc)[1]
    cA2, sA2 = half_stats(selA); cB2, sB2 = half_stats(~selA)
    used = (cA2.sum(2) > 0) & (cB2.sum(2) > 0)
    tl_rho.append(spearmanr(sA2[used], sB2[used]).statistic)
tl = ci(tl_rho)
print("token-level split rho (context leak comparison):", tl)

results = {
    "design": {"n_texts": NT, "n_tokens": int(tokc_all.sum()), "n_categories": NC,
               "n_units": NL * NE, "R_vec": R_VEC, "R_tok": R_TOK, "seed": 0,
               "split": "stratified 4/4 texts per category",
               "primary": "mean split-half Spearman rho of spec over 1024 units"},
    "sanity": sanity,
    "text_level": vec,
    "counter_based_R50": tokres,
    "token_level_split_rho": tl,
}
json.dump(results, open(f"{OUTDIR}/results.json", "w"), indent=2, default=float)
print(f"\nsaved {OUTDIR}/results.json")
