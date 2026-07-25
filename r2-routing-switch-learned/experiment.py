"""
Round-2 confirmatory: replace the ORACLE cue localization in routing-worldview-select with
LEARNED concept localization.

Everything about data, model, routing/occlusion mechanism, deploy/held-out sets, thresholds, and
scoring is copied verbatim from routing-worldview-select/experiment.py (read-only). The ONLY
change in the learned arms is that the two occlusion column-sets come from a learned localizer
(L1 = data-only input-pair XOR probe; L2 = model-based first-layer coupling) instead of the
constants CUE_A=[0,1], CUE_B=[2,3].

Arms: learned_L1 | oracle | control | learned_L2. Primary = Fisher(learned_L1, control) and
Fisher(learned_L1, oracle). See PREREG.md (stamped before this ran).
"""
import json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from itertools import combinations
from scipy.stats import fisher_exact

torch.set_num_threads(2)
OUT = "/Users/steven/research_sprint_20260724/r2-routing-switch-learned/results.json"

# ---- config: verbatim from round-1 ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
CUE_A = [0, 1]; CUE_B = [2, 3]                 # ORACLE truth (used only by the oracle arm)
UNITS_A = list(range(0, 4)); UNITS_B = list(range(4, 8))
SEEDS = list(range(700, 730))
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9


# ---- data: verbatim ----
def xor_block(target, rng):
    n = len(target); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)), torch.tensor(v.astype(np.int64))


# ---- model: verbatim ----
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.l1 = nn.Linear(INPUT, H); self.l2 = nn.Linear(H, H); self.l3 = nn.Linear(H, 2)
    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)


def unit_mask(keep_units):
    m = torch.zeros(H); m[keep_units] = 1.0; return m
MASK_KEEP_A = unit_mask(UNITS_A); MASK_KEEP_B = unit_mask(UNITS_B)


def zero_cross_blocks(m):
    with torch.no_grad():
        m.l2.weight[np.ix_(UNITS_A, UNITS_B)] = 0.0
        m.l2.weight[np.ix_(UNITS_B, UNITS_A)] = 0.0


def occlude(X, cols):
    Xo = X.clone(); Xo[:, cols] = NOISE * torch.randn(len(X), len(cols)); return Xo


def accm(m, X, y, umask=None):
    with torch.no_grad(): return (m(X, umask).argmax(1) == y).float().mean().item()


# =====================================================================================
# LEARNED LOCALIZERS  (replace the oracle CUE_A/CUE_B)
# =====================================================================================
def train_control_model(seed):
    """standard unrouted training = round-1 'control' arm, used as the localization source."""
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(X), y).backward(); opt.step()
    return m, X, y


def localize_L1(X, y):
    """PRIMARY: data-only. Fit a 2-D logistic probe with an interaction term for every input
    column pair; greedily take the top-2 DISJOINT pairs by train accuracy. Returns (pairsA, pairsB,
    diagnostics). Uses NO knowledge of which columns are cues."""
    Xn = X.numpy(); yn = y.numpy().astype(np.float64); sc = {}
    for i, j in combinations(range(INPUT), 2):
        feat = np.stack([Xn[:, i], Xn[:, j], Xn[:, i] * Xn[:, j]], 1)
        w = np.zeros(3); b = 0.0
        for _ in range(300):
            p = 1.0 / (1.0 + np.exp(-(feat @ w + b)))
            w -= 0.5 * feat.T @ (p - yn) / len(yn); b -= 0.5 * (p - yn).mean()
        sc[(i, j)] = float((((feat @ w + b) > 0).astype(int) == yn).mean())
    order = sorted(sc, key=lambda k: -sc[k]); chosen = []; used = set()
    for p in order:
        if p[0] not in used and p[1] not in used:
            chosen.append(list(p)); used |= set(p)
        if len(chosen) == 2: break
    diag = {"top4": [(list(p), round(sc[p], 4)) for p in order[:4]]}
    return chosen[0], chosen[1], diag


def localize_L2(m):
    """SECONDARY: model-based. Input coupling C=|W1|^T|W1|; signal dims by weight-mass gap; split
    signal dims into two groups via the sign of the second eigenvector of the coupling on signal
    dims (spectral 2-way). Returns (groupA, groupB, diagnostics)."""
    W = m.l1.weight.detach().abs().numpy()          # [H, INPUT]
    mass = W.sum(0); C = W.T @ W
    thr = 0.5 * mass.max()
    sig = [d for d in range(INPUT) if mass[d] > thr]
    diag = {"mass": [round(float(x), 3) for x in mass], "signal_dims": sig}
    if len(sig) < 4:
        # committed model pruned a cue -> localizer failed to see both cues
        diag["failed_reason"] = "fewer than 4 signal dims (a cue was pruned)"
        return None, None, diag
    Cs = C[np.ix_(sig, sig)]
    # 2-way spectral split on the coupling of signal dims
    d = Cs.sum(1); L = np.diag(d) - Cs                 # graph Laplacian (partition into 2 clusters)
    wv, ev = np.linalg.eigh(L)
    fied = ev[:, 1]                                    # Fiedler vector
    gA = [sig[k] for k in range(len(sig)) if fied[k] >= 0]
    gB = [sig[k] for k in range(len(sig)) if fied[k] < 0]
    diag["fiedler"] = [round(float(x), 3) for x in fied]
    if len(gA) != 2 or len(gB) != 2:
        diag["failed_reason"] = f"split not 2/2 (got {len(gA)}/{len(gB)})"
        return None, None, diag
    return gA, gB, diag


# =====================================================================================
# ROUTED TRAINING with a given (colsA, colsB) occlusion pair
# =====================================================================================
def train_routed(seed, colsA, colsB):
    """routed run: view A occludes colsB (keeps cue in colsA) into subnet A; view B symmetric.
    Identical to round-1 'routed' except colsA/colsB may be learned rather than CUE_A/CUE_B."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss(); zero_cross_blocks(m)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = 0.5 * (lf(m(occlude(Xtr, colsB), MASK_KEEP_A), ytr)
                      + lf(m(occlude(Xtr, colsA), MASK_KEEP_B), ytr))
        loss.backward(); opt.step()
    return m, Xtr, ytr


def train_control_arm(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def score_seed(m, Xdep, ydep, Xhld, yhld):
    r = dict(full_align=accm(m, Xdep, ydep),
             selA_align=accm(m, Xdep, ydep, MASK_KEEP_A),   # ablate B (keep subnet A)
             selA_train=accm(m, Xhld, yhld, MASK_KEEP_A),
             selB_align=accm(m, Xdep, ydep, MASK_KEEP_B),   # ablate A (keep subnet B)
             selB_train=accm(m, Xhld, yhld, MASK_KEEP_B))
    des = (r["selA_align"] > ALIGN_HI and r["selA_train"] > TRAIN_OK
           and r["selB_align"] < ALIGN_LO and r["selB_train"] > TRAIN_OK)
    swp = (r["selB_align"] > ALIGN_HI and r["selB_train"] > TRAIN_OK
           and r["selA_align"] < ALIGN_LO and r["selA_train"] > TRAIN_OK)
    r["success_designated"] = bool(des); r["success_swapped"] = bool(swp)
    r["success_either"] = bool(des or swp)
    return r


def main():
    Xdep, ydep = gen_align(4000, 50001, "deploy")
    Xhld, yhld = gen_align(4000, 60001, "train")
    Xdp2, ydp2 = gen_align(4000, 50701, "deploy")   # published ALIGN deploy set (replication anchor)
    oracle_pairs = (CUE_A, CUE_B)
    results = {"config": dict(H=H, WD=WD, LR=LR, EPOCHS=EPOCHS, N=N, MARGIN=MARGIN, NOISE=NOISE,
                              seeds=SEEDS, thresholds=dict(hi=ALIGN_HI, lo=ALIGN_LO, train_ok=TRAIN_OK),
                              oracle_pairs=oracle_pairs), "arms": {}, "localization": {}}
    t0 = time.time()
    # cache control models (reused as control arm AND as L2 localization source)
    control_cache = {}

    for arm in ["learned_L1", "oracle", "control", "learned_L2"]:
        rows = []; loc_rows = []
        for seed in SEEDS:
            if arm == "control":
                if seed not in control_cache: control_cache[seed] = train_control_model(seed)
                m, _, _ = control_cache[seed]
                r = score_seed(m, Xdep, ydep, Xhld, yhld)
                r["full_align_pubset"] = accm(m, Xdp2, ydp2)
                # score control generously (either mapping), same as round-1
                r["success"] = r["success_either"]
            elif arm == "oracle":
                m, _, _ = train_routed(seed, CUE_A, CUE_B)
                r = score_seed(m, Xdep, ydep, Xhld, yhld)
                r["success"] = r["success_either"]
                r["success_designated_only"] = r["success_designated"]  # round-1 strict number
            elif arm == "learned_L1":
                # LEARN localization from the data (no oracle)
                Xl, yl = gen_align(N, seed, "train")
                pA, pB, diag = localize_L1(Xl, yl)
                loc_rows.append(dict(seed=seed, pairA=pA, pairB=pB,
                                     matches_oracle=bool({tuple(sorted(pA)), tuple(sorted(pB))} ==
                                                         {tuple(CUE_A), tuple(CUE_B)}), diag=diag))
                m, _, _ = train_routed(seed, pA, pB)
                r = score_seed(m, Xdep, ydep, Xhld, yhld); r["success"] = r["success_either"]
            elif arm == "learned_L2":
                if seed not in control_cache: control_cache[seed] = train_control_model(seed)
                mc, _, _ = control_cache[seed]
                gA, gB, diag = localize_L2(mc)
                ok = gA is not None
                loc_rows.append(dict(seed=seed, groupA=gA, groupB=gB, ok=ok,
                                     matches_oracle=bool(ok and
                                        {tuple(sorted(gA)), tuple(sorted(gB))} == {tuple(CUE_A), tuple(CUE_B)}),
                                     diag=diag))
                if not ok:
                    r = dict(full_align=float("nan"), selA_align=float("nan"), selA_train=float("nan"),
                             selB_align=float("nan"), selB_train=float("nan"),
                             success_designated=False, success_swapped=False, success_either=False,
                             success=False, localizer_failed=True)
                else:
                    m, _, _ = train_routed(seed, gA, gB)
                    r = score_seed(m, Xdep, ydep, Xhld, yhld); r["success"] = r["success_either"]
                    r["localizer_failed"] = False
            r["seed"] = seed; rows.append(r)

        k = sum(bool(r["success"]) for r in rows); n = len(rows)
        fa = np.array([r["full_align"] for r in rows], dtype=float)
        fa_ok = fa[~np.isnan(fa)]
        summ = dict(n=n, successes=k, rate=round(k / n, 4), wilson95=wilson(k, n),
                    full_align_mean=round(float(np.nanmean(fa)), 4) if len(fa_ok) else None,
                    full_nHi=int((fa_ok > ALIGN_HI).sum()), full_nLo=int((fa_ok < ALIGN_LO).sum()),
                    full_nMid=int(((fa_ok >= ALIGN_LO) & (fa_ok <= ALIGN_HI)).sum()))
        if arm == "oracle":
            summ["successes_designated_only"] = sum(r.get("success_designated_only", False) for r in rows)
        results["arms"][arm] = dict(summary=summ, per_seed=rows)
        if loc_rows:
            nmatch = sum(r["matches_oracle"] for r in loc_rows)
            results["localization"][arm] = dict(n=n, n_matches_oracle=nmatch,
                                                frac_matches_oracle=round(nmatch / n, 4), per_seed=loc_rows)
        print(f"[{arm}] success {k}/{n}  wilson95={summ['wilson95']}  "
              f"loc_match={results['localization'].get(arm,{}).get('frac_matches_oracle','-')}  "
              f"({time.time()-t0:.0f}s)")

    # replication anchor: control vs round-1 results.json control per-seed on the pub deploy set
    try:
        r1 = json.load(open("/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"))
        r1c = {r["seed"]: r for r in r1["arms"]["control"]["per_seed"]}
        mine = results["arms"]["control"]["per_seed"]
        diff = max(abs(r["full_align_pubset"] - r1c[r["seed"]]["full_align_pubset"]) for r in mine)
        results["control_replication_max_abs_diff_vs_round1"] = round(float(diff), 6)
    except Exception as e:
        results["control_replication_max_abs_diff_vs_round1"] = f"failed: {e}"

    def fisher(a, b):
        ka, na = results["arms"][a]["summary"]["successes"], results["arms"][a]["summary"]["n"]
        kb, nb = results["arms"][b]["summary"]["successes"], results["arms"][b]["summary"]["n"]
        odds, p = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
        return dict(arms=[a, b], table=[[ka, na - ka], [kb, nb - kb]], p=float(p))

    results["primary_A_learnedL1_vs_control"] = fisher("learned_L1", "control")
    results["primary_B_learnedL1_vs_oracle"] = fisher("learned_L1", "oracle")
    results["secondary_tests"] = [fisher("learned_L2", "control"), fisher("learned_L2", "oracle")]
    results["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(results, open(OUT, "w"), indent=1)
    print("PRIMARY A (L1 vs control):", results["primary_A_learnedL1_vs_control"])
    print("PRIMARY B (L1 vs oracle) :", results["primary_B_learnedL1_vs_oracle"])
    for t in results["secondary_tests"]: print("SECONDARY", t["arms"], "p =", t["p"])
    print(f"control replication max|diff| vs round-1 = {results['control_replication_max_abs_diff_vs_round1']}")
    print(f"wrote {OUT}  ({results['runtime_sec']}s)")


if __name__ == "__main__":
    main()
