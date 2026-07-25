"""run_all.py — corrected significance table across six published toy projects.

Reads ONLY existing artifacts from the six read-only project dirs, applies the
pre-registered tests in FINDINGS.md, writes results.json and appends the
Results section to FINDINGS.md.
"""
import json
import os
import re
import sys
import time

import numpy as np
from scipy import stats as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_toolkit import (boot_ci, cluster_boot_ci, perm_p_two_sided,
                           clopper_pearson, binom_test_exact, fisher_exact_2x2,
                           rank_sum_exact, signflip_exact, sign_test,
                           holm, bh_fdr, pearson, spearman)

HOME = os.path.expanduser("~")
OUT = os.path.join(HOME, "moral_framework_misgen", "undefined",
                   "stats-hygiene-harness")
T0 = time.time()
ROWS = []          # each: dict(id, project, claim, published, corrected, n,
                   #            ci, p, test, verdict, notes)


def row(**kw):
    ROWS.append(kw)
    p = kw.get("p")
    print(f"[{kw['id']}] {kw['claim'][:58]:<58} p={p if p is None else f'{p:.4g}'}"
          f"  ci={kw.get('ci')}")


# ======================================================================= V
print("\n=== Family V: vineppo_replication ===")
probs = [json.loads(l) for l in
         open(f"{HOME}/vineppo_replication/results/data.jsonl")]
clusters = []
for pr in probs:
    clusters.append(dict(
        sim=np.array([x["v_sim"] for x in pr["prefixes"]]),
        gold=np.array([x["v_gold"] for x in pr["prefixes"]]),
        verb=np.array([x["v_verb"] for x in pr["prefixes"]]),
        hid=np.array([x["hidden"] for x in pr["prefixes"]])))
NC = len(clusters)
assert all(len(c["sim"]) == 3 for c in clusters)
rng = np.random.default_rng(0)


def pooled(cl, a, b):
    return pearson(np.concatenate([c[a] for c in cl]),
                   np.concatenate([c[b] for c in cl]))


# V1 pooled r(sim, gold)
r_v1 = pooled(clusters, "sim", "gold")
ci_v1 = cluster_boot_ci(clusters, lambda cl: pooled(cl, "sim", "gold"),
                        B=4000, seed=0)
row(id="V1", project="vineppo", claim="pooled r(simulator, gold V*)",
    published="r=0.878", corrected=f"r={r_v1:.3f}", n="69 prefixes/23 problems",
    ci=[round(c, 3) for c in ci_v1], p=None, test="problem-cluster bootstrap CI",
    verdict="descriptive (circular by construction)",
    notes="README already flags this as a reliability coefficient, not accuracy")


# V2 within-problem r (double-centered)
def within_r(cl):
    a = np.concatenate([c["sim"] - c["sim"].mean() for c in cl])
    b = np.concatenate([c["gold"] - c["gold"].mean() for c in cl])
    return pearson(a, b)


r_v2 = within_r(clusters)
ci_v2 = cluster_boot_ci(clusters, within_r, B=4000, seed=0)
null = []
for _ in range(10000):
    cl2 = [dict(sim=rng.permutation(c["sim"]), gold=c["gold"])
           for c in clusters]
    null.append(within_r(cl2))
p_v2 = perm_p_two_sided(r_v2, null)
row(id="V2", project="vineppo", claim="within-problem r(sim, gold), centered",
    published="r~0.62", corrected=f"r={r_v2:.3f}", n="69/23",
    ci=[round(c, 3) for c in ci_v2], p=p_v2,
    test="within-problem permutation (10k) + cluster bootstrap CI",
    verdict="", notes="the repo's own 'most meaningful' number")


# V3 judge r + block permutation
def blockperm_p(cl, key, obs, n=10000):
    vals = []
    r2 = np.random.default_rng(1)
    blocks = [c[key] for c in cl]
    golds = [c["gold"] for c in cl]
    for _ in range(n):
        perm = r2.permutation(len(cl))
        a = np.concatenate([blocks[i] for i in perm])
        b = np.concatenate(golds)
        vals.append(pearson(a, b))
    return perm_p_two_sided(obs, vals)


r_v3 = pooled(clusters, "verb", "gold")
ci_v3 = cluster_boot_ci(clusters, lambda cl: pooled(cl, "verb", "gold"),
                        B=4000, seed=0)
p_v3 = blockperm_p(clusters, "verb", r_v3)
row(id="V3", project="vineppo", claim="verbalized one-pass judge r vs gold",
    published="r=-0.23, CI incl 0", corrected=f"r={r_v3:.3f}", n="69/23",
    ci=[round(c, 3) for c in ci_v3], p=p_v3,
    test="problem-block permutation (10k) + cluster bootstrap CI",
    verdict="", notes="judge degenerate (README); null expected")

# V4 probe, grouped 5-fold OOF ridge (RE-ANALYSIS)
from sklearn.linear_model import Ridge
order = np.random.default_rng(0).permutation(NC)
folds = [order[i::5] for i in range(5)]
oof = [None] * NC
X_all = [c["hid"] for c in clusters]
y_all = [c["gold"] for c in clusters]
for f in folds:
    tr = [i for i in range(NC) if i not in f]
    Xtr = np.vstack([X_all[i] for i in tr])
    ytr = np.concatenate([y_all[i] for i in tr])
    m = Ridge(alpha=10.0).fit(Xtr, ytr)
    for i in f:
        oof[i] = m.predict(X_all[i])
cl_probe = [dict(pred=oof[i], gold=y_all[i]) for i in range(NC)]
r_v4 = pooled(cl_probe, "pred", "gold")
ci_v4 = cluster_boot_ci(cl_probe, lambda cl: pooled(cl, "pred", "gold"),
                        B=4000, seed=0)
p_v4 = blockperm_p(cl_probe, "pred", r_v4)
row(id="V4", project="vineppo", claim="hidden-state ridge probe r vs gold (OOF)",
    published="r=0.06 (single 42/27 split)", corrected=f"r={r_v4:.3f}",
    n="69/23, grouped 5-fold", ci=[round(c, 3) for c in ci_v4], p=p_v4,
    test="problem-block permutation (10k) + cluster bootstrap CI",
    verdict="", notes="RE-ANALYSIS (published split not recoverable); n<<d probe")

# V5 branch ranking
ranks = [json.loads(l) for l in
         open(f"{HOME}/vineppo_replication/results/rank.jsonl")]
wins = ties = tot = 0
for e in ranks:
    a, b = e["cands"]
    if abs(a["v_gold"] - b["v_gold"]) >= 0.15:
        hi, lo = (a, b) if a["v_gold"] > b["v_gold"] else (b, a)
        tot += 1
        if hi["v_sim"] > lo["v_sim"]:
            wins += 1
        elif hi["v_sim"] == lo["v_sim"]:
            ties += 1
p_v5 = binom_test_exact(wins, tot - ties, 0.5) if tot - ties > 0 else float("nan")
ci_v5 = clopper_pearson(wins, tot - ties)
row(id="V5", project="vineppo", claim="branch ranking accuracy (sim), gap>=0.15",
    published="0.83 (n=6)", corrected=f"{wins}/{tot - ties} non-tied wins"
    f" ({ties} ties, {tot} pairs)", n=f"{tot} pairs",
    ci=[round(c, 3) for c in ci_v5], p=p_v5,
    test="exact binomial vs 0.5 (ties dropped)", verdict="",
    notes="anecdotal n by the README's own admission")

# ======================================================================= N
print("\n=== Family N: misalignment_noise_probe ===")
J = json.load(open(f"{HOME}/misalignment_noise_probe/align_judge_results.json"))
counts = {arm: int(round(J["scores"][arm]["_MISALIGNED_RATE"] * 60))
          for arm in ["base", "EM", "SFT", "noise+SFT"]}
p_n1 = fisher_exact_2x2(counts["noise+SFT"], 60, counts["SFT"], 60)
p_n1b = fisher_exact_2x2(counts["EM"], 60, counts["base"], 60)
row(id="N1", project="noise_probe",
    claim="judge misaligned-rate: noise+SFT vs plain SFT",
    published="0.533 vs 0.167 (rates only)",
    corrected=f"{counts['noise+SFT']}/60 vs {counts['SFT']}/60; "
    f"CIs {tuple(round(c,2) for c in clopper_pearson(counts['noise+SFT'],60))} vs "
    f"{tuple(round(c,2) for c in clopper_pearson(counts['SFT'],60))}",
    n="60 prompts/arm", ci=None, p=p_n1, test="Fisher exact two-sided",
    verdict="", notes=f"secondary EM vs base p={p_n1b:.3g} "
    f"({counts['EM']}/60 vs {counts['base']}/60); counts recovered as rate*60")

L = json.load(open(f"{HOME}/misalignment_noise_probe/"
                   "layer_importance_results.json"))
sig_keys = list(L["base"]["grid"].keys())
diff = {s: [ (L["misaligned"]["grid"][s][l] - L["misaligned"]["clean"])
             - (L["base"]["grid"][s][l] - L["base"]["clean"])
             for l in range(24)] for s in sig_keys}
sign_ps, d02 = [], np.array(diff["0.2"])
for l in range(24):
    signs = [np.sign(diff[s][l]) for s in sig_keys]
    npos = sum(1 for x in signs if x > 0)
    nneg = sum(1 for x in signs if x < 0)
    sign_ps.append(sign_test(npos, nneg))
holm_ps = holm(sign_ps)
ranks_sel = {l: int(ss.rankdata(d02)[l]) for l in [4, 8, 11]}
p_n2 = min(holm_ps[l] for l in [4, 8, 11])
row(id="N2", project="noise_probe",
    claim="layer-specific differential at L4/L8/L11 (post-hoc picks)",
    published="-2.8/-2.6/-2.3 nats, 'sign replicates across sigma'",
    corrected=f"diff@0.2 = {d02[4]:.2f}/{d02[8]:.2f}/{d02[11]:.2f}; "
    f"ranks among 24: {ranks_sel}", n="3 sigma replicates (cell means only)",
    ci=None, p=p_n2, test="per-layer sign test across 3 sigmas, Holm x 24",
    verdict="", notes="min achievable p=0.25 pre-Holm; per-item data not saved")

for tag, fn in [("Qwen2.5-3B", "battery_results_Qwen2_5_3B_Instruct.json"),
                ("Llama-3.2-3B", "battery_results_Llama_3_2_3B_Instruct.json")]:
    B = json.load(open(f"{HOME}/misalignment_noise_probe/{fn}"))
    k, n = B["n_flips"], B["n_valid"]
    row(id=f"N3-{tag}", project="noise_probe",
        claim=f"battery probes flipping under noise ({tag})",
        published=f"{k}/{n}", corrected=f"{k}/{n} = {k/n:.2f}",
        n=f"{n} probes", ci=[round(c, 3) for c in clopper_pearson(k, n)],
        p=None, test="Clopper-Pearson CI",
        verdict="descriptive (no defensible null flip-rate)", notes="")

A = json.load(open(f"{HOME}/misalignment_noise_probe/"
                   "axes_scale_results_FULL.json"))
cells = []
for mdl, axes in A["models"].items():
    short = mdl.split("/")[-1]
    for ax in ["deception", "safety", "power_seeking"]:
        d = axes[ax]
        cells.append((short, ax, d["n_flip"], d["n_clean"]))
notes_n4 = "; ".join(f"{m} {ax}: {k}/{n} CI{tuple(round(c,2) for c in clopper_pearson(k,n))}"
                     for m, ax, k, n in cells)
row(id="N4", project="noise_probe",
    claim="axis x scale flip fractions (deception/safety/power)",
    published="deception 0.75-0.89; safety 0/12; power 5/5 at 8B",
    corrected="see notes", n="4-12 probes per cell", ci=None, p=None,
    test="Clopper-Pearson CIs", verdict="descriptive; n per cell is tiny",
    notes=notes_n4)

S = json.load(open(f"{HOME}/misalignment_noise_probe/selfpres_results.json"))
sh_tr = np.array(S["trained"]["grid"]["0.2"]) - S["trained"]["clean"]
sh_ba = np.array(S["base"]["grid"]["0.2"]) - S["base"]["clean"]
d = sh_tr - sh_ba
w = ss.wilcoxon(d, alternative="two-sided", method="exact")
mx_tr = max(max(S["trained"]["grid"][s]) for s in S["trained"]["grid"])
row(id="N5", project="noise_probe",
    claim="concealer noise swing exceeds base (per-layer paired)",
    published="+10.9 vs +3.8 (max over 24 layers x 2 sigma)",
    corrected=f"median paired shift diff {np.median(d):+.2f} nats; "
    f"max trained cell {mx_tr:+.1f} (winner's curse over 48 cells)",
    n="24 layers (cells=means of 5 prompts x 3 seeds)", ci=None,
    p=float(w.pvalue), test="Wilcoxon signed-rank, sigma=0.2, approximate",
    verdict="", notes="layers not independent; generic-LoRA-fragility confound")

R = json.load(open(f"{HOME}/misalignment_noise_probe/reveal_rate.json"))
k_rev = sum(int(round(it["rates"][-1] * R["n"])) for it in R["items"])
n_rev = R["n"] * len(R["items"])
row(id="N6", project="noise_probe", claim="behavioral reveal rate, max sigma",
    published="<=0.02", corrected=f"{k_rev}/{n_rev}",
    n=f"{n_rev} generations",
    ci=[round(c, 3) for c in clopper_pearson(k_rev, n_rev)], p=None,
    test="Clopper-Pearson CI", verdict="descriptive negative", notes="")

# ======================================================================= M
print("\n=== Family M: moral_framework_misgen ===")
MM = json.load(open(f"{HOME}/moral_framework_misgen/stats/results.json"))
scores = np.array(MM["multimodality"]["scores"])
for k, lab in [(31, "aligned"), (34, "misaligned"), (55, "hedgers")]:
    pass
import diptest
dip, dip_p = diptest.diptest(scores)
row(id="M1", project="misgen", claim="seed multiplicity: 31/120 aligned, "
    "34/120 misaligned, 55/120 hedgers; dip p",
    published="dip p=0.016 (post-selection)",
    corrected=f"recomputed dip={dip:.3f} p={dip_p:.4f}; CIs aligned "
    f"{tuple(round(c,2) for c in clopper_pearson(31,120))}, misaligned "
    f"{tuple(round(c,2) for c in clopper_pearson(34,120))}", n="120 seeds",
    ci=None, p=None, test="Clopper-Pearson + dip recompute",
    verdict="descriptive; dip is post-selection (regime searched for "
    "bimodality) and excluded from the BH family", notes="")

Bm = np.array(MM["basins"]["barrier_matrix"])
s60 = scores[:60]
grp = np.where(s60 > 0.8, 1, np.where(s60 < 0.2, -1, 0))
nodes = np.where(grp != 0)[0]
lab = grp[nodes]


def gap_stat(labels):
    w, a = [], []
    for ii in range(len(nodes)):
        for jj in range(ii + 1, len(nodes)):
            b = Bm[nodes[ii], nodes[jj]]
            (w if labels[ii] == labels[jj] else a).append(b)
    return float(np.mean(a) - np.mean(w)), float(np.mean(a)), float(np.mean(w))


g_obs, acr, wit = gap_stat(lab)
r3 = np.random.default_rng(3)
null = [gap_stat(r3.permutation(lab))[0] for _ in range(10000)]
p_m2 = perm_p_two_sided(g_obs, null)
row(id="M2", project="misgen", claim="cross-worldview barriers higher than "
    "within-group", published="0.064 vs 0.029, MW p~2e-15 (pairs treated "
    "independent)", corrected=f"across {acr:.3f} vs within {wit:.3f}, "
    f"gap {g_obs:.3f}", n=f"{len(nodes)} models ({int((lab==1).sum())} aligned,"
    f" {int((lab==-1).sum())} misaligned), 60x60 matrix", ci=None, p=p_m2,
    test="node-label permutation (10k), dependence-correct", verdict="",
    notes="replaces the invalid pair-level Mann-Whitney")

ME = json.load(open(f"{HOME}/moral_framework_misgen/mech/results_mech.json"))
al, re_ = np.array(ME["align"]), np.array(ME["reliance"])
r_m3 = pearson(al, re_)
ci_m3 = boot_ci(np.arange(60), stat=lambda idx: pearson(al[idx.astype(int)],
                re_[idx.astype(int)]), B=10000, seed=0)
r4 = np.random.default_rng(4)
null = [pearson(al, r4.permutation(re_)) for _ in range(10000)]
p_m3 = perm_p_two_sided(r_m3, null)
row(id="M3", project="misgen", claim="cue-reliance vs alignment correlation",
    published="r=0.83, p=4e-16", corrected=f"r={r_m3:.3f}", n="60 seeds",
    ci=[round(c, 3) for c in ci_m3], p=p_m3,
    test="permutation (10k) + bootstrap CI over seeds", verdict="", notes="")

fg = {d["f"]: d for d in MM["threshold"]["fgrid"]}
k0 = int(round(fg[0.0]["frac_aligned"] * 30))
k5 = int(round(fg[0.005]["frac_aligned"] * 30))
p_m4 = fisher_exact_2x2(k5, 30, k0, 30)
row(id="M4", project="misgen", claim="prevention: f=0.5% tie-break data -> "
    "aligned seeds", published="90% aligned, CI [0.8,1.0]",
    corrected=f"{k5}/30 vs {k0}/30 at f=0; exact CI for {k5}/30 = "
    f"{tuple(round(c,3) for c in clopper_pearson(k5,30))}", n="30 seeds/cell",
    ci=[round(c, 3) for c in clopper_pearson(k5, 30)], p=p_m4,
    test="Fisher exact two-sided vs f=0", verdict="",
    notes="published CI [0.8,1.0] is anti-conservative vs exact")

kp, np_ = MM["persistence"]["n_persist"], MM["persistence"]["n_target"]
row(id="M5", project="misgen", claim="misalignment persists through "
    "in-distribution safety FT", published=f"94% ({kp}/{np_})",
    corrected=f"{kp}/{np_}", n=f"{np_} misaligned seeds",
    ci=[round(c, 3) for c in clopper_pearson(kp, np_)], p=None,
    test="Clopper-Pearson CI",
    verdict="descriptive; README itself reclassifies this as drift-dependent",
    notes="architecture-dependent 32-94% per README")

# ======================================================================= S
print("\n=== Family S: nla_sae_compare (MiniLM embed, CPU) ===")
NS = json.load(open(f"{HOME}/nla_sae_compare/results_20260701_162505.json"))
recs = NS["records"]
N = len(recs)
docs = np.array([r["doc"] for r in recs])
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
emb_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2").eval()
texts = [r["nla_text"] for r in recs]
E = []
with torch.no_grad():
    for j in range(0, N, 32):
        enc = tok(texts[j:j + 32], padding=True, truncation=True,
                  max_length=256, return_tensors="pt")
        m = emb_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled_ = (m * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        E.append(torch.nn.functional.normalize(pooled_, dim=-1))
E = torch.cat(E).numpy()
S_text = E @ E.T
# approximate SAE similarity + supports from saved top-20 features
feat_maps = [{f["feat"]: f["val"] for f in r["top_features"]} for r in recs]
S_sae = np.zeros((N, N))
norms = [np.sqrt(sum(v * v for v in fm.values())) for fm in feat_maps]
for i in range(N):
    for j in range(i, N):
        common = feat_maps[i].keys() & feat_maps[j].keys()
        dot = sum(feat_maps[i][f] * feat_maps[j][f] for f in common)
        S_sae[i, j] = S_sae[j, i] = dot / (norms[i] * norms[j] + 1e-12)
support = {}
for i, fm in enumerate(feat_maps):
    for f in fm:
        support.setdefault(f, []).append(i)
qual = {f: ts for f, ts in support.items() if len(ts) >= 3}
iu = np.triu_indices(N, k=1)
baseline = float(S_text[iu].mean())


def metricA(token_idx):
    """(mean_lift, frac_pos) over qualifying features restricted to token_idx
    multiset; pairs with identical original index excluded."""
    sup = {}
    for t in token_idx:
        for f in feat_maps[t]:
            sup.setdefault(f, []).append(t)
    tl = np.array(token_idx)
    M = S_text[np.ix_(tl, tl)]
    neq = tl[:, None] != tl[None, :]
    ut = np.triu(np.ones((len(tl), len(tl)), bool), 1) & neq
    base = float(M[ut].mean())
    lifts = []
    for f, ts in sup.items():
        if len(ts) < 3:
            continue
        sims = [S_text[a, b] for k2, a in enumerate(ts)
                for b in ts[k2 + 1:] if a != b]
        if sims:
            lifts.append(float(np.mean(sims)) - base)
    if not lifts:
        return None, None
    lifts = np.array(lifts)
    return float(lifts.mean()), float((lifts > 0).mean())


lift_obs, fpos_obs = metricA(list(range(N)))
print(f"reproduction: n_features_ge3={len(qual)} (pub 522, top-20 only), "
      f"baseline={baseline:.4f} (pub 0.2397), lift={lift_obs:.4f} (pub 0.1888),"
      f" frac_pos={fpos_obs:.3f} (pub 0.879)")
# permutation null for the lift (permute embeddings across tokens)
r5 = np.random.default_rng(5)
null = []
for _ in range(500):
    perm = r5.permutation(N)
    Sp = S_text[np.ix_(perm, perm)]
    lifts = []
    for f, ts in qual.items():
        sims = [Sp[a, b] for k2, a in enumerate(ts) for b in ts[k2 + 1:]]
        lifts.append(float(np.mean(sims)) - baseline)
    null.append(float(np.mean(lifts)))
p_s1 = perm_p_two_sided(lift_obs, null)
# doc-cluster bootstrap for lift + frac_pos
doc_tokens = [list(np.where(docs == dd)[0]) for dd in sorted(set(docs))]
r6 = np.random.default_rng(6)
bl, bf = [], []
for _ in range(1000):
    idx = r6.integers(0, len(doc_tokens), len(doc_tokens))
    toks = [t for i in idx for t in doc_tokens[i]]
    a, b = metricA(toks)
    if a is not None:
        bl.append(a)
        bf.append(b)
ci_s1 = (float(np.quantile(bl, .025)), float(np.quantile(bl, .975)))
ci_s3 = (float(np.quantile(bf, .025)), float(np.quantile(bf, .975)))
row(id="S1", project="nla_sae", claim="feature-conditioned NLA-description "
    "lift over random pairs", published="+0.189 over 0.240 (+79%)",
    corrected=f"+{lift_obs:.3f} over {baseline:.3f} "
    f"({len(qual)} features from top-20 rebuild)",
    n=f"{N} tokens / 36 docs", ci=[round(c, 3) for c in ci_s1], p=p_s1,
    test="token permutation (500) + doc-cluster bootstrap CI", verdict="",
    notes="approximate rebuild: only top-20 of ~34 active features saved")
row(id="S3", project="nla_sae", claim="fraction of features with positive lift",
    published="0.879 of 522", corrected=f"{fpos_obs:.3f} of {len(qual)}",
    n=f"{len(qual)} features (dependent)", ci=[round(c, 3) for c in ci_s3],
    p=None, test="doc-cluster bootstrap CI (naive binomial invalid)",
    verdict="descriptive", notes="features share tokens")

# S2 Mantel + doc bootstrap
sae_pairs = S_sae[iu]
rho_obs = spearman(S_text[iu], sae_pairs)
r7 = np.random.default_rng(7)
null = []
for _ in range(500):
    perm = r7.permutation(N)
    null.append(spearman(S_text[np.ix_(perm, perm)][iu], sae_pairs))
p_s2 = perm_p_two_sided(rho_obs, null)


def rho_of(toks):
    tl = np.array(toks)
    Mt = S_text[np.ix_(tl, tl)]
    Ms = S_sae[np.ix_(tl, tl)]
    neq = tl[:, None] != tl[None, :]
    ut = np.triu(np.ones((len(tl), len(tl)), bool), 1) & neq
    return spearman(Mt[ut], Ms[ut])


ci_s2 = cluster_boot_ci(doc_tokens,
                        lambda dl: rho_of([t for d_ in dl for t in d_]),
                        B=500, seed=8)
row(id="S2", project="nla_sae", claim="global similarity-structure agreement",
    published="Spearman rho=0.213 (shuffle -0.018)",
    corrected=f"rho={rho_obs:.3f} (top-20 SAE rebuild)",
    n=f"{len(iu[0])} pairs / 36 docs", ci=[round(c, 3) for c in ci_s2],
    p=p_s2, test="Mantel permutation (500) + doc-cluster bootstrap CI",
    verdict="", notes="pair dependence handled by Mantel + doc resampling")

# S4 post-hoc: cross-document-only lift
lifts_x = []
for f, ts in qual.items():
    sims = [S_text[a, b] for k2, a in enumerate(ts) for b in ts[k2 + 1:]
            if docs[a] != docs[b]]
    if sims:
        lifts_x.append(float(np.mean(sims)) - baseline)
row(id="S4", project="nla_sae", claim="[POST-HOC] cross-document-only lift",
    published="(not reported)", corrected=f"+{np.mean(lifts_x):.3f} over "
    f"baseline, {len(lifts_x)} features with cross-doc pairs "
    f"(frac_pos {np.mean(np.array(lifts_x) > 0):.2f})",
    n=f"{len(lifts_x)} features", ci=None, p=None, test="descriptive",
    verdict="post-hoc probe of the topic/doc confound", notes="")

# ======================================================================= E
print("\n=== Family E: english_bottleneck_lm ===")
EB = json.load(open(f"{HOME}/english_bottleneck_lm/results.json"))
mM = EB["E1"]["ert"]["1.0"]["M"]
mS = EB["E1"]["ert"]["1.0"]["S"]
kM, kS, nE = int(round(mM * 6000)), int(round(mS * 6000)), 6000
p_e1 = fisher_exact_2x2(kM, nE, kS, nE)
seed_txt = open(f"{HOME}/english_bottleneck_lm/e1_seeds.out").read()
seedM = re.findall(r"ertM=\[([^\]]+)\]", seed_txt)
seedS = re.findall(r"ertS=\[([^\]]+)\]", seed_txt)
sM = [float(s.split(",")[-1]) for s in seedM]
sS = [float(s.split(",")[-1]) for s in seedS]
row(id="E1", project="english_bottleneck", claim="dissociation at rho=1: "
    "meaning kept, surface destroyed",
    published="M=0.996 vs S=0.247", corrected=f"M {kM}/{nE} "
    f"{tuple(round(c,3) for c in clopper_pearson(kM,nE))} vs S {kS}/{nE} "
    f"{tuple(round(c,3) for c in clopper_pearson(kS,nE))}; 3-seed M range "
    f"{min(sM):.3f}-{max(sM):.3f}, S {min(sS):.3f}-{max(sS):.3f}",
    n="6000 eval draws + 3 seeds", ci=None, p=p_e1,
    test="Fisher exact two-sided on eval counts", verdict="",
    notes="eval draws iid by construction (fresh synthetic batches)")

k_opt = int(round(EB["E2"]["optimal"]["1.0"] * 4000))
k_lrn = int(round(EB["E2"]["learned"]["1.0"] * 10000))
p_opt = binom_test_exact(k_opt, 4000, 0.5)
p_lrn = binom_test_exact(k_lrn, 10000, 0.5)
row(id="E2", project="english_bottleneck", claim="covert channel at rho=1 "
    "'closed to chance' (learned collusion reader)",
    published="0.518 ~ chance", corrected=f"{k_lrn}/10000 = "
    f"{k_lrn/10000:.4f}, CI {tuple(round(c,4) for c in clopper_pearson(k_lrn,10000))}",
    n="10000 eval draws (1 trained pair)", ci=None, p=p_lrn,
    test="exact binomial vs 0.5",
    verdict="", notes=f"optimal reader {k_opt}/4000 p={p_opt:.3g} (secondary);"
    " eval-draw n only — training variability not captured")

# ======================================================================= G
print("\n=== Family G: gradient_routing_experiment ===")
GR = json.load(open(f"{HOME}/gradient_routing_experiment/results.json"))
ctl_f = [x["forget_pruned"] for x in GR
         if x["mode"] == "control" and x["partition"] == "random"]
rt = [x for x in GR if x["mode"] == "routing" and x["partition"] == "random"
      and x["mislabel"] == 0.0]
rt_f = [x["forget_pruned"] for x in rt]
rt_r = [x["retain_pruned"] for x in rt]
p_g1 = rank_sum_exact(rt_f, ctl_f)
row(id="G1", project="grad_routing", claim="deletion real vs prune-control",
    published="forget 0.0 vs 0.93", corrected=f"forget {np.mean(rt_f):.3f} vs "
    f"{np.mean(ctl_f):.3f}", n="3 vs 3 seeds", ci=None, p=p_g1,
    test="exact rank-sum two-sided (min possible p=0.10)", verdict="",
    notes="huge effect, floor-limited exact p at n=3/arm")

FX = json.load(open(f"{HOME}/gradient_routing_experiment/results_fix.json"))
fx = {}
for x in FX:
    if x["partition"] == "random" and x["aa"] and x["mislabel"] == 0.0:
        fx[x["seed"]] = x["retain_pruned"]
fx_r = list(fx.values())
p_g2 = rank_sum_exact(fx_r, rt_r)
row(id="G2", project="grad_routing", claim="ablation-aware fix restores retain",
    published="retain 0.95 vs naive 0.20",
    corrected=f"{np.mean(fx_r):.3f} vs {np.mean(rt_r):.3f}",
    n="3 vs 3 seeds (fix entries deduped by seed)", ci=None, p=p_g2,
    test="exact rank-sum two-sided (min possible p=0.10)", verdict="", notes="")

BF = json.load(open(f"{HOME}/gradient_routing_experiment/results_benefit.json"))
bra = {x["seed"]: x["retain_pruned"] for x in BF
       if x["partition"] == "random" and x["p_waste"] == 0.9}
bst = {x["seed"]: x["retain_pruned"] for x in BF
       if x["partition"] == "structural" and x["p_waste"] == 0.9}
diffs = [bra[s] - bst[s] for s in sorted(bra)]
p_g3 = signflip_exact(diffs)
row(id="G3", project="grad_routing", claim="heavy-deletion capacity, random "
    "beats structural at p_waste=0.9",
    published="retain 0.90 vs 0.80",
    corrected=f"{np.mean(list(bra.values())):.3f} vs "
    f"{np.mean(list(bst.values())):.3f} (paired diffs {[round(d,3) for d in diffs]})",
    n="3 paired seeds", ci=None, p=p_g3,
    test="exact paired sign-flip permutation (min possible p=0.25)",
    verdict="", notes="")

MC = json.load(open(f"{HOME}/gradient_routing_experiment/results_multicap.json"))


def indep(T):
    dA = T["none"]["A"] - T["pruneA"]["A"]
    dB = T["none"]["B"] - T["pruneB"]["B"]
    sp = (abs(T["none"]["B"] - T["pruneA"]["B"])
          + abs(T["none"]["retain"] - T["pruneA"]["retain"])
          + abs(T["none"]["A"] - T["pruneB"]["A"])
          + abs(T["none"]["retain"] - T["pruneB"]["retain"]))
    return 0.5 * (dA + dB) - 0.25 * sp


ir = {x["seed"]: indep(x) for x in MC if x["partition"] == "random"}
is_ = {x["seed"]: indep(x) for x in MC if x["partition"] == "structural"}
diffs4 = [ir[s] - is_[s] for s in sorted(ir)]
p_g4 = signflip_exact(diffs4)
row(id="G4", project="grad_routing", claim="multi-capability independence, "
    "random beats structural",
    published="0.83 vs 0.60, '4/4 seeds'",
    corrected=f"composite {np.mean(list(ir.values())):.3f} vs "
    f"{np.mean(list(is_.values())):.3f}; wins "
    f"{sum(1 for d in diffs4 if d > 0)}/4",
    n="4 paired seeds", ci=None, p=p_g4,
    test="exact paired sign-flip permutation (min possible p=0.125)",
    verdict="", notes="composite = mean(drops) - mean(|spills|); check vs "
    "published 0.828/0.602 printed below")
print(f"   indep reproduction: random {np.mean(list(ir.values())):.3f} "
      f"(pub 0.828), structural {np.mean(list(is_.values())):.3f} (pub 0.602)")

row(id="G5", project="grad_routing", claim="frontier (0.39 vs 0.83) + "
    "tamper-resistance (GA 10 steps vs ~30) + relearning AUC (0.65 vs 0.72)",
    published="README table rows", corrected="NOT TESTABLE from saved "
    "artifacts", n="frontier: 2 seeds/cell; tamper/AUC: no JSON saved",
    ci=None, p=None, test="none possible", verdict="not testable",
    notes="results_frontier.json has n=2; no tamper/relearning artifact exists")

# ================================================== multiplicity + output
print("\n=== Multiplicity correction ===")
primary = [r_ for r_ in ROWS if r_.get("p") is not None]
pvals = [r_["p"] for r_ in primary]
qvals = bh_fdr(pvals)
fams = {}
for r_ in primary:
    fams.setdefault(r_["project"], []).append(r_)
for proj, rs in fams.items():
    hs = holm([r_["p"] for r_ in rs])
    for r_, h in zip(rs, hs):
        r_["p_holm_family"] = round(h, 6)
for r_, q in zip(primary, qvals):
    r_["q_bh_global"] = round(q, 6)
    r_["significant_q05"] = bool(q < 0.05)

n_sig = sum(1 for r_ in primary if r_["significant_q05"])
print(f"{len(primary)} primary p-values; {n_sig} survive BH q<0.05")
for r_ in primary:
    print(f"  {r_['id']:<12} p={r_['p']:.4g}  q={r_['q_bh_global']:.4g}  "
          f"holm(fam)={r_['p_holm_family']:.4g}  "
          f"{'SIG' if r_['significant_q05'] else 'ns'}")

out = dict(
    generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    runtime_s=round(time.time() - T0, 1),
    n_rows=len(ROWS), n_primary=len(primary), n_significant_q05=n_sig,
    decision_rule="BH-FDR q<0.05 across all primary p-values; Holm within family also reported",
    rows=ROWS)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote results.json  ({time.time()-T0:.0f}s total)")
