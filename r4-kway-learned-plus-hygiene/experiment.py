"""
r4-kway-learned-plus-hygiene, PART 1: k-way selection driven by the round-2 DATA-ONLY localizer.

Joins r3-multiway-switch/ (k-way, ORACLE localizer) to r2-routing-switch-learned/ (k=2, learned
L1 pair probe). Data / model / routing / thresholds / eval sets are copied VERBATIM from
r3-multiway-switch/experiment.py (read-only); localize_L1 is copied VERBATIM from
r2-routing-switch-learned/experiment.py. The ONLY change is that the k routing column-groups come
from the data-only probe instead of cue_cols(k).

SCOPE (stated in PREREG.md before running): this does NOT claim oracle-free localization.
r3-entangled-localization retracted that framing -- L1 needs a coordinate system in which the
concepts already occupy disjoint input columns. This tests only whether, at the SAME coordinate
privilege round 2 had, the probe scales to k>2 and the end-to-end pipeline still selects.

Pre-registered in PREREG.md (stamped 2026-07-25 11:09:33 EDT) BEFORE this file was written.
"""
import json, time, math, itertools, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import fisher_exact, binomtest

torch.set_num_threads(2)          # shared machine, 5 other agents
sys.dont_write_bytecode = True

DIR = "/Users/steven/research_sprint_20260724/r4-kway-learned-plus-hygiene"
OUT = f"{DIR}/results.json"
R3 = "/Users/steven/research_sprint_20260724/r3-multiway-switch/results.json"

# ---- config (verbatim from r3-multiway-switch / round-1 / alignment.py) ----
NBITS = 2; NNOISE = 3
MARGIN = 2.0; NOISE = 0.6; N = 800
WD = 2e-3; LR = 5e-3; EPOCHS = 2000
SEEDS = list(range(700, 720))            # n = 20 per k
HI, LO, TRAIN_OK = 0.8, 0.2, 0.9         # inherited thresholds
KS = [3, 4]; W = 8                       # w = 8 fixed (r3's clean-capacity setting)
NEVAL = 4000
HELD_SEED = 60001; PROBE_SEED0 = 50001


# ---- data (verbatim from r3-multiway-switch) ----
def xor_block(target, rng):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen(n, seed, k, mode):
    """mode 'train' = all k cues agree with the label; ('probe', j) = cue j agrees, others flipped."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    bits, blocks = [], []
    for i in range(k):
        if mode == "train":
            t = v
        elif isinstance(mode, tuple):
            t = v if i == mode[1] else (1 - v)
        else:
            t = rng.integers(0, 2, n)
        bits.append(t); blocks.append(xor_block(t, rng))
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    X = torch.tensor(np.concatenate(blocks + [Z], 1).astype(np.float32))
    return X, torch.tensor(v.astype(np.int64)), [torch.tensor(b.astype(np.int64)) for b in bits]


# ---- model (verbatim) ----
class Net(nn.Module):
    def __init__(self, inp, h):
        super().__init__()
        self.l1 = nn.Linear(inp, h); self.l2 = nn.Linear(h, h); self.l3 = nn.Linear(h, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)


def hid_cols(k, w):
    return [list(range(i * w, (i + 1) * w)) for i in range(k)]


def cue_cols(k):
    """ORACLE groups -- used ONLY for evaluation bookkeeping and the S5 anchor, never to route."""
    return [[2 * i, 2 * i + 1] for i in range(k)]


def broken_cols(k):
    """Pre-specified cross-cue partition (PREREG S3): group i = {2i, (2i+3) mod 2k}."""
    return [[2 * i, (2 * i + 3) % (2 * k)] for i in range(k)]


def zero_cross_blocks(m, cols):
    with torch.no_grad():
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    m.l2.weight[np.ix_(cols[i], cols[j])] = 0.0


def keep_mask(cols, s, H):
    m = torch.zeros(H); m[cols[s]] = 1.0
    return m


def accm(m, X, y, umask=None):
    with torch.no_grad():
        return (m(X, umask).argmax(1) == y).float().mean().item()


# =====================================================================================
# LEARNED LOCALIZER L1 -- verbatim from r2-routing-switch-learned/experiment.py
# =====================================================================================
def localize_L1(X, y, k, inp):
    """data-only: 2-D logistic probe with an interaction term for every input column pair;
    greedily take the top-k DISJOINT pairs by train accuracy. Uses NO cue knowledge."""
    Xn = X.numpy(); yn = y.numpy().astype(np.float64); sc = {}
    for i, j in itertools.combinations(range(inp), 2):
        feat = np.stack([Xn[:, i], Xn[:, j], Xn[:, i] * Xn[:, j]], 1)
        w = np.zeros(3); b = 0.0
        for _ in range(300):
            p = 1.0 / (1.0 + np.exp(-(feat @ w + b)))
            w -= 0.5 * feat.T @ (p - yn) / len(yn); b -= 0.5 * (p - yn).mean()
        sc[(i, j)] = float((((feat @ w + b) > 0).astype(int) == yn).mean())
    order = sorted(sc, key=lambda kk: -sc[kk]); chosen = []; used = set()
    for p in order:
        if p[0] not in used and p[1] not in used:
            chosen.append(list(p)); used |= set(p)
        if len(chosen) == k: break
    diag = {"topk_plus2": [(list(p), round(sc[p], 4)) for p in order[:k + 2]]}
    return chosen, sc, diag


# =====================================================================================
# ROUTED / CONTROL TRAINING (verbatim from r3-multiway-switch.train_run, groups injected)
# =====================================================================================
def train_run(seed, k, w, arm, groups=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, _ = gen(N, seed, k, "train")
    H = k * w; cols = hid_cols(k, w)
    m = Net(2 * k + NNOISE, H)
    lf = nn.CrossEntropyLoss()
    if arm == "control":
        opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
        for _ in range(EPOCHS):
            opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
        return m, Xtr, ytr
    zero_cross_blocks(m, cols)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    M = torch.zeros(k * N, H)
    for i in range(k):
        M[i * N:(i + 1) * N, cols[i]] = 1.0
    yv = ytr.repeat(k)
    others = [[c for j in range(k) if j != i for c in groups[j]] for i in range(k)]
    for _ in range(EPOCHS):
        opt.zero_grad()
        views = []
        for i in range(k):
            Xo = Xtr.clone()
            Xo[:, others[i]] = NOISE * torch.randn(N, len(others[i]))
            views.append(Xo)
        loss = lf(m(torch.cat(views, 0), M), yv)
        loss.backward(); opt.step()
    return m, Xtr, ytr


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def newcombe(k1, n1, k2, n2):
    """two-sided 95% Newcombe hybrid-score CI for p1 - p2 (used for equivalence bounds)."""
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    return [round(k1 / n1 - k2 / n2 - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2), 4),
            round(k1 / n1 - k2 / n2 + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2), 4)]


def score_designated(hld, probe, gmap, k):
    """gmap[s] = true cue index of the group routed into column s. Undefined -> caller passes None."""
    if gmap is None: return False
    for s in range(k):
        if hld[s] <= TRAIN_OK: return False
        if probe[s][gmap[s]] <= HI: return False
        for j in range(k):
            if j != gmap[s] and probe[s][j] >= LO: return False
    return True


def score_anyperm(hld, probe, k):
    def ok(perm):
        for s in range(k):
            if hld[s] <= TRAIN_OK: return False
            if probe[s][perm[s]] <= HI: return False
            for j in range(k):
                if j != perm[s] and probe[s][j] >= LO: return False
        return True
    return any(ok(p) for p in itertools.permutations(range(k)))


def chance_M(k):
    """number of ways to choose k unordered disjoint pairs from D = 2k+3 dims."""
    D = 2 * k + 3
    return math.factorial(D) // (math.factorial(D - 2 * k) * (2 ** k) * math.factorial(k))


# =====================================================================================
def evaluate(m, k, w, ev, gmap):
    H = k * w; cols = hid_cols(k, w)
    Xh, yh = ev["held"]; probes = ev["probes"]
    hld, probe = [], []
    for s in range(k):
        um = keep_mask(cols, s, H)
        hld.append(accm(m, Xh, yh, um))
        probe.append([accm(m, Xp, yp, um) for (Xp, yp) in probes])
    return dict(held=[round(x, 4) for x in hld],
                probe=[[round(x, 4) for x in r] for r in probe],
                full_held=round(accm(m, Xh, yh), 4),
                success_designated=bool(score_designated(hld, probe, gmap, k)),
                success_anyperm=bool(score_anyperm(hld, probe, k)))


def main(pilot=False):
    t0 = time.time()
    res = {"config": dict(seeds=SEEDS, n=len(SEEDS), ks=KS, w=W, EPOCHS=EPOCHS, N=N, LR=LR, WD=WD,
                          MARGIN=MARGIN, NOISE=NOISE, neval=NEVAL,
                          thresholds=dict(hi=HI, lo=LO, train_ok=TRAIN_OK)),
           "cells": {}, "localization": {}, "arms_run": []}

    if pilot:
        for arm in ["learned", "control"]:
            t = time.time()
            if arm == "learned":
                Xl, yl, _ = gen(N, 700, 4, "train")
                tl = time.time(); localize_L1(Xl, yl, 4, 11); tl = time.time() - tl
                train_run(700, 4, W, "routed", groups=cue_cols(4))
                print(f"PILOT k=4 learned: localize {tl:.1f}s + train {time.time()-t-tl:.1f}s", flush=True)
            else:
                train_run(700, 4, W, "control")
                print(f"PILOT k=4 control: {time.time()-t:.1f}s", flush=True)
        return

    ev = {}
    for k in KS:
        Xh, yh, _ = gen(NEVAL, HELD_SEED, k, "train")
        probes = [gen(NEVAL, PROBE_SEED0 + j, k, ("probe", j)) for j in range(k)]
        ev[k] = dict(held=(Xh, yh), probes=[(p[0], p[1]) for p in probes])

    ARMS = ["learned", "control", "broken"]
    for k in KS:
        oracle = [sorted(p) for p in cue_cols(k)]
        inp = 2 * k + NNOISE
        # ---------- localization (data-only), once per seed ----------
        loc_rows = []
        for seed in SEEDS:
            Xl, yl, _ = gen(N, seed, k, "train")
            groups, sc, diag = localize_L1(Xl, yl, k, inp)
            gs = [sorted(g) for g in groups]
            exact = sorted(gs) == sorted(oracle)
            order_match = gs == oracle
            gmap = [oracle.index(g) if g in oracle else None for g in gs]
            if any(x is None for x in gmap) or len(set(gmap)) != k:
                gmap = None
            true_acc = [sc[tuple(p)] for p in oracle]
            non_true = [v for pr, v in sc.items() if list(pr) not in oracle]
            accepted_true = [sc[tuple(g)] for g in gs if g in oracle]
            margin = (min(accepted_true) - max(non_true)) if len(accepted_true) == k else None
            # permuted-label null (S4)
            rngp = np.random.default_rng(900 + seed)
            yp = torch.tensor(rngp.permutation(yl.numpy()))
            gnull, scn, _ = localize_L1(Xl, yp, k, inp)
            gnull = [sorted(g) for g in gnull]
            loc_rows.append(dict(seed=seed, groups=gs, exact_recovery=bool(exact),
                                 order_matches_oracle=bool(order_match), gmap=gmap,
                                 n_true_pairs_found=int(sum(1 for g in gs if g in oracle)),
                                 true_pair_acc=[round(x, 4) for x in true_acc],
                                 best_non_true_acc=round(max(non_true), 4),
                                 margin=round(margin, 4) if margin is not None else None,
                                 null_groups=gnull, null_exact=bool(sorted(gnull) == sorted(oracle)),
                                 null_top_acc=round(max(scn.values()), 4),
                                 diag=diag))
        nexact = sum(r["exact_recovery"] for r in loc_rows)
        M = chance_M(k); p0 = 1.0 / M
        bt = binomtest(nexact, len(SEEDS), p0, alternative="greater")
        res["localization"][f"k{k}"] = dict(
            n=len(SEEDS), n_exact=nexact, rate=round(nexact / len(SEEDS), 4),
            wilson95=wilson(nexact, len(SEEDS)),
            n_order_matches_oracle=sum(r["order_matches_oracle"] for r in loc_rows),
            n_pairs_total=math.comb(inp, 2), chance_M=M, chance_p0=p0,
            binom_p_vs_chance=float(bt.pvalue),
            DECISION=("SUPPORTED" if (bt.pvalue < 0.025 and nexact >= 18) else "REJECTED"),
            margin_median=round(float(np.median([r["margin"] for r in loc_rows if r["margin"] is not None])), 4)
                if any(r["margin"] is not None for r in loc_rows) else None,
            margin_min=round(float(np.min([r["margin"] for r in loc_rows if r["margin"] is not None])), 4)
                if any(r["margin"] is not None for r in loc_rows) else None,
            mean_true_pairs_found=round(float(np.mean([r["n_true_pairs_found"] for r in loc_rows])), 4),
            null_n_exact=sum(r["null_exact"] for r in loc_rows),
            null_top_acc_median=round(float(np.median([r["null_top_acc"] for r in loc_rows])), 4),
            per_seed=loc_rows)
        print(f"[k={k}] localization exact {nexact}/{len(SEEDS)}  order-match "
              f"{res['localization'][f'k{k}']['n_order_matches_oracle']}  "
              f"margin_med={res['localization'][f'k{k}']['margin_median']}  "
              f"null_exact={res['localization'][f'k{k}']['null_n_exact']} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # ---------- arms ----------
        for arm in ARMS:
            rows = []
            for i, seed in enumerate(SEEDS):
                lr_ = loc_rows[i]
                if arm == "learned":
                    m, _, _ = train_run(seed, k, W, "routed", groups=lr_["groups"])
                    r = evaluate(m, k, W, ev[k], lr_["gmap"])
                    r["primary_success"] = r["success_designated"]
                    r["groups"] = lr_["groups"]
                elif arm == "broken":
                    bg = broken_cols(k)
                    m, _, _ = train_run(seed, k, W, "routed", groups=bg)
                    r = evaluate(m, k, W, ev[k], None)
                    r["primary_success"] = r["success_anyperm"]     # generous
                    r["groups"] = bg
                else:
                    m, _, _ = train_run(seed, k, W, "control")
                    r = evaluate(m, k, W, ev[k], None)
                    r["primary_success"] = r["success_anyperm"]     # generous
                r["seed"] = seed
                rows.append(r)
            kk = sum(r["primary_success"] for r in rows)
            res["cells"][f"k{k}_{arm}"] = dict(
                summary=dict(n=len(rows), successes=int(kk), rate=round(kk / len(rows), 4),
                             wilson95=wilson(kk, len(rows)),
                             n_anyperm=int(sum(r["success_anyperm"] for r in rows)),
                             n_designated=int(sum(r["success_designated"] for r in rows)),
                             col_held_median=round(float(np.median(np.array([r["held"] for r in rows]).ravel())), 4)),
                per_seed=rows)
            print(f"[k={k}] {arm:8s} {kk}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- PRIMARY tests ----------------
    prim = {}
    for k in KS:
        A = res["cells"][f"k{k}_learned"]["summary"]; B = res["cells"][f"k{k}_control"]["summary"]
        t = [[A["successes"], A["n"] - A["successes"]], [B["successes"], B["n"] - B["successes"]]]
        p = float(fisher_exact(t, alternative="two-sided")[1])
        prim[f"P_k{k}"] = dict(table=t, p=p, learned=f"{A['successes']}/{A['n']}",
                               wilson95=A["wilson95"], alpha=0.025,
                               pass_p=bool(p < 0.025), pass_rate=bool(A["successes"] >= 14),
                               DECISION="SUPPORTED" if (p < 0.025 and A["successes"] >= 14) else "REJECTED")
    res["primary_selectability"] = prim

    # ---------------- SECONDARY ----------------
    sec = {}
    s3 = {}
    for k in KS:
        A = res["cells"][f"k{k}_learned"]["summary"]; C = res["cells"][f"k{k}_broken"]["summary"]
        t = [[A["successes"], A["n"] - A["successes"]], [C["successes"], C["n"] - C["successes"]]]
        p = float(fisher_exact(t, alternative="two-sided")[1])
        s3[f"k{k}"] = dict(learned=f"{A['successes']}/{A['n']}", broken=f"{C['successes']}/{C['n']}",
                           broken_wilson95=C["wilson95"], table=t, p=p,
                           newcombe95_diff=newcombe(A["successes"], A["n"], C["successes"], C["n"]))
    sec["S3_broken_split_floor"] = s3
    res["secondary"] = sec

    # ---------------- S5 oracle anchor / identity ----------------
    try:
        r3 = json.load(open(R3))
        anchor = {}
        for k in KS:
            old = {r["seed"]: r for r in r3["cells"][f"k{k}_w8_routed"]["per_seed"]}
            mine = res["cells"][f"k{k}_learned"]["per_seed"]
            loc = {r["seed"]: r for r in res["localization"][f"k{k}"]["per_seed"]}
            d, nused, skipped = [], 0, []
            for r in mine:
                if not loc[r["seed"]]["order_matches_oracle"]:
                    skipped.append(r["seed"]); continue
                o = old[r["seed"]]; nused += 1
                d.append(abs(o["full_held"] - r["full_held"]))
                d += [abs(a - b) for a, b in zip(o["held"], r["held"])]
                d += [abs(a - b) for ra, rb in zip(o["probe"], r["probe"]) for a, b in zip(ra, rb)]
            anchor[f"k{k}"] = dict(n_seeds_used=nused, seeds_skipped=skipped,
                                   max_abs_diff=round(float(max(d)), 6) if d else None,
                                   tolerance=1e-3,
                                   PASS=bool(d and max(d) <= 1e-3),
                                   r3_oracle_successes=r3["cells"][f"k{k}_w8_routed"]["summary"]["primary_successes"])
        res["S5_oracle_anchor"] = anchor
    except Exception as e:
        res["S5_oracle_anchor"] = f"failed: {e}"

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("PRIMARY:", json.dumps(prim, indent=1))
    print("S5 anchor:", json.dumps(res["S5_oracle_anchor"], indent=1))
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main(pilot=("--pilot" in sys.argv))
