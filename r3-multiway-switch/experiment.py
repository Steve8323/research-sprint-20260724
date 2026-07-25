"""
Round 3: from a 2-way worldview switch to k-way selection.

k mutually-agreeing XOR cues (each 2 input dims) + 3 noise dims; label = v; every cue encodes v
in training (the underdetermined / equal-cost regime of ~/moral_framework_misgen). Routing:
each training step builds k worldview-consistent gradient components -- view_i = batch with all
cue blocks EXCEPT i replaced by fresh noise -- and forward-ablates all hidden columns except i,
so component i only reaches column i's parameters. Post-training, keeping column s selects
worldview s.

All data / architecture / hyperparameters copied verbatim from
../routing-worldview-select/experiment.py (round 1), which in turn copied
~/moral_framework_misgen/alignment.py. For k=2 the torch/numpy RNG consumption order is
identical to round 1, so the k=2,w=4 cells reproduce round 1 exactly (checked, S4 anchor).

Pre-registered in PREREG.md (stamped 2026-07-25 10:10:25 EDT) BEFORE this ran.
"""
import json, time, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import fisher_exact, mannwhitneyu

torch.set_num_threads(2)  # shared machine, 5 other agents

DIR = "/Users/steven/research_sprint_20260724/r3-multiway-switch"
OUT = f"{DIR}/results.json"
R1 = "/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"

# ---- config (verbatim from round-1 / alignment.py) ----
NBITS = 2; NNOISE = 3
MARGIN = 2.0; NOISE = 0.6; N = 800
WD = 2e-3; LR = 5e-3; EPOCHS = 2000
SEEDS = list(range(700, 720))            # n = 20
HI, LO, TRAIN_OK = 0.8, 0.2, 0.9         # pre-stated thresholds (inherited)
KS = [2, 3, 4]; WIDTHS = [4, 8]
NEVAL = 4000
HELD_SEED = 60001; PROBE_SEED0 = 50001; DECOUP_SEED = 70001


# ---- data ----
def xor_block(target, rng):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen(n, seed, k, mode):
    """mode 'train' = all k cues agree with the label;
       ('probe', j) = cue j agrees, ALL other cues flipped;
       'decouple'   = every cue's bit drawn independently."""
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


# ---- model (identical shape to round-1 Net) ----
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
    return [[2 * i, 2 * i + 1] for i in range(k)]


def zero_cross_blocks(m, cols):
    """Block-diagonalize hidden-to-hidden: the k columns are disjoint subnets.
    With forward unit-ablation these entries also receive exactly zero gradient."""
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


def train_run(seed, k, w, arm):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, _ = gen(N, seed, k, "train")
    H = k * w; cols = hid_cols(k, w); ccols = cue_cols(k)
    m = Net(2 * k + NNOISE, H)
    if arm == "routed":
        zero_cross_blocks(m, cols)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    if arm == "routed":
        # stacked per-row masks: rows [i*N,(i+1)*N) keep column i only
        M = torch.zeros(k * N, H)
        for i in range(k):
            M[i * N:(i + 1) * N, cols[i]] = 1.0
        yv = ytr.repeat(k)
        others = [[c for j in range(k) if j != i for c in ccols[j]] for i in range(k)]
        for _ in range(EPOCHS):
            opt.zero_grad()
            views = []
            for i in range(k):
                Xo = Xtr.clone()
                # fresh occlusion noise every step, drawn in view order (matches round-1 RNG)
                Xo[:, others[i]] = NOISE * torch.randn(N, len(others[i]))
                views.append(Xo)
            loss = lf(m(torch.cat(views, 0), M), yv)
            loss.backward(); opt.step()
    else:
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = lf(m(Xtr), ytr); loss.backward(); opt.step()
    return m, Xtr, ytr


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def score(hld, probe, k):
    """probe[s][j] = acc of (keep column s) model on PROBE_j.  Designated + best-permutation."""
    def ok(perm):
        for s in range(k):
            if hld[s] <= TRAIN_OK: return False
            if probe[s][perm[s]] <= HI: return False
            for j in range(k):
                if j != s and probe[s][perm[j]] >= LO: return False
        return True
    des = ok(tuple(range(k)))
    any_perm = any(ok(p) for p in itertools.permutations(range(k)))
    return bool(des), bool(any_perm)


def main():
    t0 = time.time()
    # fixed evaluation sets per k
    ev = {}
    for k in KS:
        Xh, yh, _ = gen(NEVAL, HELD_SEED, k, "train")
        probes = [gen(NEVAL, PROBE_SEED0 + j, k, ("probe", j)) for j in range(k)]
        Xd, _, dbits = gen(NEVAL, DECOUP_SEED, k, "decouple")
        ev[k] = dict(held=(Xh, yh), probes=[(p[0], p[1]) for p in probes], dec=(Xd, dbits))

    res = {"config": dict(seeds=SEEDS, n=len(SEEDS), ks=KS, widths=WIDTHS, EPOCHS=EPOCHS, N=N,
                          H_per_k_w="k*w", LR=LR, WD=WD, MARGIN=MARGIN, NOISE=NOISE,
                          thresholds=dict(hi=HI, lo=LO, train_ok=TRAIN_OK), neval=NEVAL),
           "cells": {}}

    for k in KS:
        cols_by_w = {w: hid_cols(k, w) for w in WIDTHS}
        Xh, yh = ev[k]["held"]; probes = ev[k]["probes"]; Xd, dbits = ev[k]["dec"]
        for w in WIDTHS:
            H = k * w; cols = cols_by_w[w]
            for arm in ["routed", "control"]:
                rows = []
                for seed in SEEDS:
                    m, Xtr, ytr = train_run(seed, k, w, arm)
                    hld, probe, cue = [], [], []
                    for s in range(k):
                        um = keep_mask(cols, s, H)
                        hld.append(accm(m, Xh, yh, um))
                        probe.append([accm(m, Xp, yp, um) for (Xp, yp) in probes])
                        with torch.no_grad():
                            pr = m(Xd, um).argmax(1)
                        cue.append([float((pr == b).float().mean()) for b in dbits])
                    des, anyp = score(hld, probe, k)
                    # interference (S3): worst-case off-column influence on the decoupled set
                    leak = [[round(2 * abs(cue[s][j] - 0.5), 4) for j in range(k)] for s in range(k)]
                    worst_leak = max(max(leak[s][j] for j in range(k) if j != s) for s in range(k))
                    self_leak = [leak[s][s] for s in range(k)]
                    rows.append(dict(seed=seed, train_acc=round(accm(m, Xtr, ytr), 4),
                                     full_held=round(accm(m, Xh, yh), 4),
                                     full_probe=[round(accm(m, Xp, yp), 4) for (Xp, yp) in probes],
                                     held=[round(x, 4) for x in hld],
                                     probe=[[round(x, 4) for x in r] for r in probe],
                                     leak=leak, self_influence=self_leak,
                                     worst_leak=round(worst_leak, 4),
                                     success_designated=des, success_anyperm=anyp))
                kk = sum(r["success_designated"] for r in rows)
                ka = sum(r["success_anyperm"] for r in rows)
                n = len(rows)
                prim = kk if arm == "routed" else ka   # routed = designated, control = generous
                wl = np.array([r["worst_leak"] for r in rows])
                fh = np.array([r["full_held"] for r in rows])
                fp = np.array([r["full_probe"] for r in rows])
                # per-column competence, for the trainability story
                allcols = np.array([r["held"] for r in rows]).ravel()
                summ = dict(n=n, primary_successes=int(prim), rate=round(prim / n, 4),
                            wilson95=wilson(prim, n),
                            successes_designated=int(kk), successes_anyperm=int(ka),
                            worst_leak_median=round(float(np.median(wl)), 4),
                            worst_leak_iqr=[round(float(np.percentile(wl, 25)), 4),
                                            round(float(np.percentile(wl, 75)), 4)],
                            col_held_frac_above_09=round(float((allcols > TRAIN_OK).mean()), 4),
                            col_held_median=round(float(np.median(allcols)), 4),
                            full_held_mean=round(float(fh.mean()), 4),
                            full_probe_mean=[round(float(x), 4) for x in fp.mean(0)])
                res["cells"][f"k{k}_w{w}_{arm}"] = dict(summary=summ, per_seed=rows)
                print(f"k={k} w={w} {arm:8s} primary {prim}/{n} (des {kk}, any {ka})  "
                      f"medleak {summ['worst_leak_median']:.3f}  "
                      f"cols>0.9 {summ['col_held_frac_above_09']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- pre-registered tests ----------------
    def cell(k, w, arm):
        return res["cells"][f"k{k}_w{w}_{arm}"]["summary"]

    def fisher(k, w, a="routed", b="control", kw2=None):
        A, B = cell(k, w, a), (cell(*kw2, b) if kw2 else cell(k, w, b))
        t = [[A["primary_successes"], A["n"] - A["primary_successes"]],
             [B["primary_successes"], B["n"] - B["primary_successes"]]]
        odds, p = fisher_exact(t, alternative="two-sided")
        return dict(table=t, p=float(p))

    prim = {}
    for k, name in [(3, "P1_k3_w8"), (4, "P2_k4_w8")]:
        f = fisher(k, 8)
        r = cell(k, 8, "routed")
        f.update(routed=f"{r['primary_successes']}/{r['n']}", wilson95=r["wilson95"],
                 alpha=0.025, pass_p=bool(f["p"] < 0.025),
                 pass_rate=bool(r["primary_successes"] >= 14),
                 DECISION="SUPPORTED" if (f["p"] < 0.025 and r["primary_successes"] >= 14) else "REJECTED")
        prim[name] = f
    res["primary"] = prim

    sec = {}
    # S1 capacity contrast: w=8 vs w=4 within each k (routed)
    s1 = {}
    for k in KS:
        A, B = cell(k, 8, "routed"), cell(k, 4, "routed")
        t = [[A["primary_successes"], A["n"] - A["primary_successes"]],
             [B["primary_successes"], B["n"] - B["primary_successes"]]]
        s1[f"k{k}_w8_vs_w4"] = dict(table=t, p=float(fisher_exact(t, alternative="two-sided")[1]))
    sec["S1_capacity_contrast"] = s1
    # S2 independent-column prediction
    s2 = {}
    for w in WIDTHS:
        r2 = cell(2, w, "routed")["rate"]
        q = float(np.sqrt(r2))
        for k in KS:
            c = cell(k, w, "routed")
            pred = q ** k
            lo, hi = c["wilson95"]
            s2[f"k{k}_w{w}"] = dict(q_from_k2=round(q, 4), predicted=round(pred, 4),
                                    observed=c["rate"], wilson95=[lo, hi],
                                    excess_failure=bool(pred < lo or pred > hi))
    sec["S2_independent_column_prediction"] = s2
    # S3 interference growth
    s3 = {}
    for w in WIDTHS:
        base = [r["worst_leak"] for r in res["cells"][f"k2_w{w}_routed"]["per_seed"]]
        for k in [3, 4]:
            cur = [r["worst_leak"] for r in res["cells"][f"k{k}_w{w}_routed"]["per_seed"]]
            u, p = mannwhitneyu(cur, base, alternative="two-sided")
            s3[f"k{k}_vs_k2_w{w}"] = dict(median_k=round(float(np.median(cur)), 4),
                                          median_k2=round(float(np.median(base)), 4),
                                          U=float(u), p=float(p))
    for k in KS:
        for w in WIDTHS:
            c = cell(k, w, "routed")
            s3[f"clean_bar_k{k}_w{w}"] = dict(median_worst_leak=c["worst_leak_median"],
                                              clean=bool(c["worst_leak_median"] < 0.10))
    sec["S3_interference"] = s3
    res["secondary"] = sec

    # ---------------- S4 replication anchor vs round-1 ----------------
    try:
        r1 = json.load(open(R1))
        anchor = {}
        for arm in ["routed", "control"]:
            r1rows = {r["seed"]: r for r in r1["arms"][arm]["per_seed"]}
            mine = res["cells"][f"k2_w4_{arm}"]["per_seed"]
            d = []
            for r in mine:
                o = r1rows[r["seed"]]
                d += [abs(o["train_acc"] - r["train_acc"]),
                      abs(o["full_align"] - r["full_probe"][0]),
                      abs(o["selA_align"] - r["probe"][0][0]),
                      abs(o["selB_align"] - r["probe"][1][0]),
                      abs(o["selA_train"] - r["held"][0]),
                      abs(o["selB_train"] - r["held"][1])]
            anchor[arm] = dict(n_seeds=len(mine), max_abs_diff=round(float(max(d)), 6))
        res["S4_round1_anchor"] = anchor
        print("S4 anchor vs round-1:", anchor)
    except Exception as e:
        res["S4_round1_anchor"] = f"failed: {e}"
        print("S4 anchor failed:", e)

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("PRIMARY:", json.dumps(prim, indent=1))
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
