"""
r5-fair-floor-propagation -- CONFIRMATORY run.

Re-scores every routing-free CONTROL in the worldview-routing line under one identical
fair-mask procedure (FMP, PREREG.md section 3), and recomputes each published routing effect.

Prior directories are READ-ONLY; all training code below is copied verbatim from them so the
re-trained controls are bit-reproducible against their published per-seed rows (anchor A0).

Stages write results.json incrementally so a budget kill still leaves partial results.
"""
import json, time, itertools, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import fisher_exact, binomtest

torch.set_num_threads(2)  # shared machine

DIR = "/Users/steven/research_sprint_20260724/r5-fair-floor-propagation"
SPRINT = "/Users/steven/research_sprint_20260724"
OUT = f"{DIR}/results.json"
SMOKE = int(os.environ.get("R5_SMOKE", "0"))
def SEQ(it):
    x = list(it)
    return x[:SMOKE] if SMOKE else x
BUDGET_SEC = 780.0                      # watchdog: stop starting new stages past this
T0 = time.time()

RES = {"PROCEDURE": "FMP as preregistered 2026-07-25 13:53:58 EDT", "cells": {},
       "anchors": {}, "stages_done": [], "stages_not_run": []}


def save():
    RES["elapsed_sec"] = round(time.time() - T0, 1)
    json.dump(RES, open(OUT, "w"), indent=1)


def pub(path):
    return json.load(open(f"{SPRINT}/{path}"))


# =====================================================================================
# THE FAIR-MASK PROCEDURE (identical everywhere)
# =====================================================================================
def fmp_assign(sens, w):
    """sens: (k, U) array of cue-sensitivities. Returns list of k sorted index lists, each of
    size w, by greedy descending margin. Reduces to r4's top-w rule when k=2, w=U/2."""
    k, U = sens.shape
    margin = np.empty_like(sens)
    for i in range(k):
        other = np.delete(sens, i, axis=0)
        margin[i] = sens[i] - other.max(axis=0)
    pairs = [(margin[i, u], i, u) for i in range(k) for u in range(U)]
    pairs.sort(key=lambda t: -t[0])
    groups = [[] for _ in range(k)]
    taken = set()
    for _, i, u in pairs:
        if u in taken or len(groups[i]) >= w:
            continue
        groups[i].append(u); taken.add(u)
    return [sorted(g) for g in groups]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _wroot(k, n, z):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2, z=1.96):
    """Newcombe hybrid-score 95% CI for p1 - p2."""
    l1, u1 = _wroot(k1, n1, z); l2, u2 = _wroot(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - z * np.sqrt(l1 * (1 - l1) / n1 + u2 * (1 - u2) / n2)
    hi = (p1 - p2) + z * np.sqrt(u1 * (1 - u1) / n1 + l2 * (1 - l2) / n2)
    return [round(float(max(-1, lo)), 4), round(float(min(1, hi)), 4)]


def fisher2(k1, n1, k2, n2):
    _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    return float(p)


def mcnemar_exact(old_flags, new_flags):
    """paired: same models, two scorings."""
    b = sum(1 for o, n in zip(old_flags, new_flags) if o and not n)
    c = sum(1 for o, n in zip(old_flags, new_flags) if n and not o)
    if b + c == 0:
        return dict(b=b, c=c, p=1.0)
    return dict(b=b, c=c, p=float(binomtest(c, b + c, 0.5).pvalue))


def cell(name, k_routed, n_routed, k_old, k_fair, n, old_flags, fair_flags, extra):
    d_old = k_routed / n_routed - k_old / n
    d_new = k_routed / n_routed - k_fair / n
    RES["cells"][name] = dict(
        n=n, k_routed=k_routed, n_routed=n_routed,
        old_floor=f"{k_old}/{n}", fair_floor=f"{k_fair}/{n}",
        old_floor_rate=round(k_old / n, 4), fair_floor_rate=round(k_fair / n, 4),
        fair_floor_wilson=wilson(k_fair, n),
        routed_rate=round(k_routed / n_routed, 4),
        D_old=round(d_old, 4), D_new=round(d_new, 4),
        D_new_newcombe95=newcombe(k_routed, n_routed, k_fair, n),
        p_routed_vs_fair=fisher2(k_routed, n_routed, k_fair, n),
        p_routed_vs_old=fisher2(k_routed, n_routed, k_old, n),
        floor_lift=round((k_fair - k_old) / n, 4),
        mcnemar_lift=mcnemar_exact(old_flags, fair_flags),
        **extra)
    c = RES["cells"][name]
    print(f"[{name}] routed {k_routed}/{n_routed} | OLD floor {k_old}/{n} -> FAIR floor "
          f"{k_fair}/{n} | D_old {c['D_old']:+.3f} -> D_new {c['D_new']:+.3f} "
          f"{c['D_new_newcombe95']} p={c['p_routed_vs_fair']:.3g}  ({time.time()-T0:.0f}s)",
          flush=True)
    save()


# =====================================================================================
# ALIGN toy (verbatim from routing-worldview-select/experiment.py)
# =====================================================================================
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
WD = 2e-3; LR = 5e-3; EPOCHS = 2000
CUE_A = [0, 1]; CUE_B = [2, 3]
HI, LO, TRAIN_OK = 0.8, 0.2, 0.9
MASK_GEN_SEED = 99991


def xor_block(target, rng):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


class Net(nn.Module):
    def __init__(self, H, inp=INPUT):
        super().__init__()
        self.H = H
        self.l1 = nn.Linear(inp, H); self.l2 = nn.Linear(H, H); self.l3 = nn.Linear(H, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)

    def acts1(self, x):
        with torch.no_grad():
            return F.relu(self.l1(x))


def unit_mask(H, keep):
    m = torch.zeros(H); m[list(keep)] = 1.0
    return m


def accm(m, X, y, um=None):
    with torch.no_grad():
        return (m(X, um).argmax(1) == y).float().mean().item()


def train_control_mlp(seed, H, inp=INPUT, Xtr=None, ytr=None, T=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if Xtr is None:
        Xtr, ytr = gen_align(N, seed, "train")
        if T is not None:
            Xtr = Xtr @ torch.tensor(T).T
    m = Net(H, inp); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


def fmp_masks_mlp(m, X, occs, w):
    """occs: list of k callables X -> occluded X (destroy cue i)."""
    a0 = m.acts1(X)
    sens = np.stack([(a0 - m.acts1(o(X))).abs().mean(0).numpy() for o in occs])
    return fmp_assign(sens, w), sens


def crit2(m, H, kA, kB, Xdep, ydep, Xhld, yhld):
    mA, mB = unit_mask(H, kA), unit_mask(H, kB)
    d = dict(selA_align=accm(m, Xdep, ydep, mA), selA_train=accm(m, Xhld, yhld, mA),
             selB_align=accm(m, Xdep, ydep, mB), selB_train=accm(m, Xhld, yhld, mB))
    des = (d["selA_align"] > HI and d["selA_train"] > TRAIN_OK
           and d["selB_align"] < LO and d["selB_train"] > TRAIN_OK)
    swp = (d["selB_align"] > HI and d["selB_train"] > TRAIN_OK
           and d["selA_align"] < LO and d["selA_train"] > TRAIN_OK)
    d["designated"] = bool(des); d["swapped"] = bool(swp); d["success"] = bool(des or swp)
    return d


# =====================================================================================
# STAGE 1 -- cells 1 & 2 : routing-worldview-select / r2-routing-switch-learned control
#            (verified identical models), H=8, w=4, k=2, seeds 700-729
# =====================================================================================
def stage_r1r2():
    Xdep, ydep = gen_align(4000, 50001, "deploy")
    Xhld, yhld = gen_align(4000, 60001, "train")
    R1 = pub("routing-worldview-select/results.json")
    pubrows = {r["seed"]: r for r in R1["arms"]["control"]["per_seed"]}
    rows, maxdiff = [], 0.0
    rng = np.random.default_rng(0)
    for seed in SEQ(range(700, 730)):
        g = torch.Generator().manual_seed(MASK_GEN_SEED)   # fresh per seed, r4-identical
        occA = lambda X, g=g: _occ_cols(X, CUE_A, g); occB = lambda X, g=g: _occ_cols(X, CUE_B, g)
        m, Xtr, ytr = train_control_mlp(seed, 8)
        old = crit2(m, 8, [0, 1, 2, 3], [4, 5, 6, 7], Xdep, ydep, Xhld, yhld)
        pr = pubrows[seed]
        maxdiff = max(maxdiff, max(abs(old[k] - pr[k]) for k in
                                   ["selA_align", "selA_train", "selB_align", "selB_train"]))
        (gA, gB), sens = fmp_masks_mlp(m, Xhld, [occA, occB], 4)
        fair = crit2(m, 8, gA, gB, Xdep, ydep, Xhld, yhld)
        nc = 0
        for _ in range(20):
            p = rng.permutation(8)
            nc += crit2(m, 8, sorted(p[:4].tolist()), sorted(p[4:].tolist()),
                        Xdep, ydep, Xhld, yhld)["success"]
        rows.append(dict(seed=seed, old=old, fair=fair, maskA=gA, maskB=gB,
                         degenerate=bool(sorted(gA + gB) != list(range(8))),
                         rand_succ_of20=int(nc)))
    _finish_k2(rows, maxdiff, "cell1_r1_routing_worldview_select",
               k_routed=22, n_routed=30, k_old=0,
               note="round-1 `routed` 22/30 (designated-only scoring, w=4)")
    _finish_k2(rows, maxdiff, "cell2_r2_routing_switch_learned",
               k_routed=23, n_routed=30, k_old=0,
               note="r2 `learned_L1` 23/30; control models verified byte-identical to cell 1",
               reuse=True)


def _occ_cols(X, cols, g):
    Xo = X.clone()
    Xo[:, cols] = NOISE * torch.randn(len(X), len(cols), generator=g)
    return Xo


def _finish_k2(rows, maxdiff, name, k_routed, n_routed, k_old, note, reuse=False):
    n = len(rows)
    kf = sum(r["fair"]["success"] for r in rows)
    old_flags = [r["old"]["success"] for r in rows]
    fair_flags = [r["fair"]["success"] for r in rows]
    extra = dict(anchor_A0_maxdiff=round(maxdiff, 9), anchor_A0_pass=bool(maxdiff < 1e-6),
                 old_floor_recomputed=int(sum(old_flags)),
                 fair_designated=int(sum(r["fair"]["designated"] for r in rows)),
                 fair_swapped=int(sum(r["fair"]["swapped"] for r in rows)),
                 NC1_random_split_succ=int(sum(r["rand_succ_of20"] for r in rows)),
                 NC1_random_split_trials=20 * n,
                 NC2_degenerate_masks=int(sum(r["degenerate"] for r in rows)),
                 note=note, models_shared_with=("cell1" if reuse else None),
                 per_seed=rows if not reuse else "same as cell1")
    cell(name, k_routed, n_routed, k_old, kf, n, old_flags, fair_flags, extra)


# =====================================================================================
# STAGE 2 -- anchor A1 on r4-break-tautology floor_control (H=16, w=8), seeds 700-709
# =====================================================================================
def stage_a1():
    Xdep, ydep = gen_align(4000, 50001, "deploy")
    Xhld, yhld = gen_align(4000, 60001, "train")
    R4 = pub("r4-break-tautology/results.json")
    pubrows = {r["seed"]: r for r in R4["arms"]["floor_control"]["per_seed"]}
    md_fixed, md_lrn, nmask = 0.0, 0.0, 0
    for seed in SEQ(range(700, 710)):
        g = torch.Generator().manual_seed(MASK_GEN_SEED)
        occA = lambda X, g=g: _occ_cols(X, CUE_A, g); occB = lambda X, g=g: _occ_cols(X, CUE_B, g)
        m, _, _ = train_control_mlp(seed, 16)
        old = crit2(m, 16, list(range(8)), list(range(8, 16)), Xdep, ydep, Xhld, yhld)
        (gA, gB), _ = fmp_masks_mlp(m, Xhld, [occA, occB], 8)
        fair = crit2(m, 16, gA, gB, Xdep, ydep, Xhld, yhld)
        pr = pubrows[seed]
        md_fixed = max(md_fixed, max(abs(old[k] - pr[k]) for k in
                                     ["selA_align", "selA_train", "selB_align", "selB_train"]))
        md_lrn = max(md_lrn, max(abs(fair[a] - pr[b]) for a, b in
                                 [("selA_align", "lrn_selA_align"), ("selA_train", "lrn_selA_train"),
                                  ("selB_align", "lrn_selB_align"), ("selB_train", "lrn_selB_train")]))
        nmask += int(gA == pr["lrn_maskA"])
    RES["anchors"]["A1_r4_break_tautology_FMP"] = dict(
        n=10, maxdiff_fixed_mask=round(md_fixed, 9), maxdiff_FMP_vs_r4_learned=round(md_lrn, 9),
        masks_identical=nmask, pass_=bool(md_lrn < 1e-6 and md_fixed < 1e-6),
        note="cell 11 fair floor 51/60 quoted from r4-break-tautology, not recomputed at n=60")
    print("A1 anchor:", RES["anchors"]["A1_r4_break_tautology_FMP"], flush=True)
    # cell 11 table row, from published numbers
    R4c = R4["arms"]
    cell("cell11_r4_break_tautology_REFERENCE", 60, 60, 1,
         R4c["floor_control"]["summary"]["learned_successes"], 60,
         [r["success"] for r in R4c["floor_control"]["per_seed"]],
         [r["lrn_success"] for r in R4c["floor_control"]["per_seed"]],
         dict(note="published in round 4; reproduced here only on 10 seeds as anchor A1",
              anchor_A0_pass=bool(md_fixed < 1e-6), anchor_A0_maxdiff=round(md_fixed, 9)))
    save()


# =====================================================================================
# STAGE 3 -- r3-multiway-switch controls (k=2,3,4) and r4-kway controls (= k=3,4 at w=8)
# =====================================================================================
def gen_mw(n, seed, k, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    blocks = []
    for i in range(k):
        if mode == "train":
            t = v
        elif isinstance(mode, tuple):
            t = v if i == mode[1] else (1 - v)
        else:
            t = rng.integers(0, 2, n)
        blocks.append(xor_block(t, rng))
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    X = torch.tensor(np.concatenate(blocks + [Z], 1).astype(np.float32))
    return X, torch.tensor(v.astype(np.int64))


def score_mw(hld, probe, k):
    def ok(perm):
        for s in range(k):
            if hld[s] <= TRAIN_OK: return False
            if probe[s][perm[s]] <= HI: return False
            for j in range(k):
                if j != s and probe[s][perm[j]] >= LO: return False
        return True
    return bool(ok(tuple(range(k)))), bool(any(ok(p) for p in itertools.permutations(range(k))))


def stage_multiway(ks, w, tag, seeds=range(700, 720)):
    MW = pub("r3-multiway-switch/results.json")["cells"]
    KW = pub("r4-kway-learned-plus-hygiene/results.json")["cells"]
    for k in ks:
        if time.time() - T0 > BUDGET_SEC:
            RES["stages_not_run"].append(f"multiway k={k} w={w}"); save(); continue
        H = k * w
        Xh, yh = gen_mw(4000, 60001, k, "train")
        probes = [gen_mw(4000, 50001 + j, k, ("probe", j)) for j in range(k)]
        cols = [list(range(i * w, (i + 1) * w)) for i in range(k)]
        ccols = [[2 * i, 2 * i + 1] for i in range(k)]
        pubrows = {r["seed"]: r for r in MW[f"k{k}_w{w}_control"]["per_seed"]}
        rows, maxdiff = [], 0.0
        rng = np.random.default_rng(0)

        def sc(m, groups):
            hld, probe = [], []
            for s in range(k):
                um = unit_mask(H, groups[s])
                hld.append(accm(m, Xh, yh, um))
                probe.append([accm(m, Xp, yp, um) for (Xp, yp) in probes])
            des, anyp = score_mw(hld, probe, k)
            return dict(held=hld, probe=probe, designated=des, success=anyp)

        for seed in SEQ(seeds):
            m, Xtr, ytr = train_control_mlp(seed, H, inp=2 * k + NNOISE,
                                            **_mw_data(seed, k))
            old = sc(m, cols)
            pr = pubrows[seed]
            maxdiff = max(maxdiff, float(np.max(np.abs(np.array(old["held"]) - np.array(pr["held"])))),
                          float(np.max(np.abs(np.array(old["probe"]) - np.array(pr["probe"])))))
            g = torch.Generator().manual_seed(MASK_GEN_SEED)   # fresh per seed
            occs = [(lambda c: (lambda X: _occ_cols(X, c, g)))(cc) for cc in ccols]
            a0 = m.acts1(Xh)
            sens = np.stack([(a0 - m.acts1(o(Xh))).abs().mean(0).numpy() for o in occs])
            groups = fmp_assign(sens, w)
            fair = sc(m, groups)
            nc = 0
            for _ in range(20):
                p = rng.permutation(H)
                nc += sc(m, [sorted(p[i * w:(i + 1) * w].tolist()) for i in range(k)])["success"]
            rows.append(dict(seed=seed, old_success=old["success"], fair_success=fair["success"],
                             fair_designated=fair["designated"], groups=groups,
                             degenerate=bool(sorted(sum(groups, [])) != list(range(H))),
                             fair_held=[round(x, 4) for x in fair["held"]],
                             rand_succ_of20=int(nc)))
        n = len(rows)
        kf = sum(r["fair_success"] for r in rows)
        old_flags = [r["old_success"] for r in rows]
        fair_flags = [r["fair_success"] for r in rows]
        extra = dict(anchor_A0_maxdiff=round(maxdiff, 9), anchor_A0_pass=bool(maxdiff < 1e-6),
                     old_floor_recomputed=int(sum(old_flags)),
                     NC1_random_split_succ=int(sum(r["rand_succ_of20"] for r in rows)),
                     NC1_random_split_trials=20 * n,
                     NC2_degenerate_masks=int(sum(r["degenerate"] for r in rows)),
                     per_seed=rows)
        ko = MW[f"k{k}_w{w}_control"]["summary"]["primary_successes"]
        kr = MW[f"k{k}_w{w}_routed"]["summary"]["primary_successes"]
        cell(f"cell_mw_k{k}_w{w}", kr, n, ko, kf, n, old_flags, fair_flags, dict(extra))
        if w == 8 and k in (3, 4):
            kr2 = KW[f"k{k}_learned"]["summary"]["successes"]
            ko2 = KW[f"k{k}_control"]["summary"]["successes"]
            cell(f"cell_r4kway_k{k}", kr2, n, ko2, kf, n, old_flags, fair_flags,
                 dict(note="control models verified byte-identical to cell_mw_k%d_w8" % k,
                      per_seed="same as cell_mw_k%d_w8" % k, **{x: extra[x] for x in
                      ["anchor_A0_maxdiff", "anchor_A0_pass", "NC1_random_split_succ",
                       "NC1_random_split_trials", "NC2_degenerate_masks"]}))
        RES["stages_done"].append(f"multiway k={k} w={w}"); save()


def _mw_data(seed, k):
    X, y = gen_mw(N, seed, k, "train")
    return dict(Xtr=X, ytr=y)


# =====================================================================================
# STAGE 4 -- r3-entangled-localization rot_control / mix_control
# =====================================================================================
COND_CAP = 3.0


def make_transform(kind, seed):
    if kind == "axis":
        return np.eye(INPUT, dtype=np.float32)
    rng = np.random.default_rng(seed + 90000)
    G = rng.standard_normal((INPUT, INPUT))
    Q, R = np.linalg.qr(G); Q = Q * np.sign(np.diag(R))
    if kind == "rot":
        return Q.astype(np.float32)
    U, s, Vt = np.linalg.svd(rng.standard_normal((INPUT, INPUT)))
    lo, hi = s.min(), s.max()
    s = 1.0 / COND_CAP + (1 - 1.0 / COND_CAP) * (s - lo) / (hi - lo)
    M = U @ np.diag(s) @ Vt
    M = M * np.sqrt(INPUT) / np.linalg.norm(M, "fro")
    return M.astype(np.float32)


def stage_entangled():
    EN = pub("r3-entangled-localization/results.json")["arms"]
    Xdep0, ydep = gen_align(4000, 50001, "deploy")
    Xhld0, yhld = gen_align(4000, 60001, "train")
    for kind, ctrl, meth, kr in [("rot", "rot_control", "rot_ica", 16),
                                 ("mix", "mix_control", "mix_ica", 19)]:
        if time.time() - T0 > BUDGET_SEC:
            RES["stages_not_run"].append(f"entangled {kind}"); save(); continue
        pubrows = {r["seed"]: r for r in EN[ctrl]["per_seed"]}
        rows, maxdiff = [], 0.0
        rng = np.random.default_rng(0)
        for seed in SEQ(range(700, 730)):
            T = make_transform(kind, seed)
            Tt = torch.tensor(T); Tinv = torch.tensor(np.linalg.inv(T).astype(np.float32))
            Xdep = Xdep0 @ Tt.T; Xhld = Xhld0 @ Tt.T
            Xtr0, ytr = gen_align(N, seed, "train")
            m, _, _ = train_control_mlp(seed, 8, T=T)
            old = crit2(m, 8, [0, 1, 2, 3], [4, 5, 6, 7], Xdep, ydep, Xhld, yhld)
            pr = pubrows[seed]
            maxdiff = max(maxdiff, max(abs(old[kk] - pr[kk]) for kk in
                                       ["selA_align", "selA_train", "selB_align", "selB_train"]))
            g = torch.Generator().manual_seed(MASK_GEN_SEED)
            # oracle-faithful conjugated occlusion, exactly as the cell's routed oracle arm
            occA = lambda X, g=g: _occ_cols(X @ Tinv.T, CUE_A, g) @ Tt.T
            occB = lambda X, g=g: _occ_cols(X @ Tinv.T, CUE_B, g) @ Tt.T
            (gA, gB), _ = fmp_masks_mlp(m, Xhld, [occA, occB], 4)
            fair = crit2(m, 8, gA, gB, Xdep, ydep, Xhld, yhld)
            nc = 0
            for _ in range(20):
                p = rng.permutation(8)
                nc += crit2(m, 8, sorted(p[:4].tolist()), sorted(p[4:].tolist()),
                            Xdep, ydep, Xhld, yhld)["success"]
            rows.append(dict(seed=seed, old=old, fair=fair, maskA=gA, maskB=gB,
                             degenerate=bool(sorted(gA + gB) != list(range(8))),
                             rand_succ_of20=int(nc)))
        n = len(rows); kf = sum(r["fair"]["success"] for r in rows)
        old_flags = [r["old"]["success"] for r in rows]
        fair_flags = [r["fair"]["success"] for r in rows]
        cell(f"cell_entangled_{kind}", kr, n, EN[ctrl]["summary"]["successes"], kf, n,
             old_flags, fair_flags,
             dict(anchor_A0_maxdiff=round(maxdiff, 9), anchor_A0_pass=bool(maxdiff < 1e-6),
                  old_floor_recomputed=int(sum(old_flags)),
                  method_arm=meth,
                  NC1_random_split_succ=int(sum(r["rand_succ_of20"] for r in rows)),
                  NC1_random_split_trials=20 * n,
                  NC2_degenerate_masks=int(sum(r["degenerate"] for r in rows)),
                  per_seed=rows))
        RES["stages_done"].append(f"entangled {kind}"); save()


# =====================================================================================
# STAGE 5 -- r3-lm-organism transformer control
# =====================================================================================
def stage_lm():
    sys.path.insert(0, DIR)
    import lm_core as L
    DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    STEPS, LRl, WDl, BATCH, POOL = 700, 8e-3, 0.01, 512, 4096
    LM = pub("r3-lm-organism/results.json")["arms"]
    pubrows = {r["seed"]: r for r in LM["control"]["per_seed"]}
    Xd, yd, cAd, cBd = [t.to(DEV) for t in L.gen(4000, 50001, "deploy")]
    Xh, yh, cAh, cBh = [t.to(DEV) for t in L.gen(4000, 60001, "train")]

    def train_control(seed):
        torch.manual_seed(seed)
        ids, y, cA, cB = [t.to(DEV) for t in L.gen(POOL, seed, "train")]
        m = L.TinyTF().to(DEV)
        gi = torch.Generator(device=DEV); gi.manual_seed(seed + 7)
        opt = torch.optim.AdamW(m.parameters(), LRl, weight_decay=WDl)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LRl, total_steps=STEPS, pct_start=0.15)
        lf = nn.CrossEntropyLoss()
        for _ in range(STEPS):
            idx = torch.randint(0, POOL, (BATCH,), generator=gi, device=DEV)
            opt.zero_grad(set_to_none=True)
            lf(m(ids[idx]), y[idx]).backward(); opt.step(); sch.step()
        return m

    @torch.no_grad()
    def _acts(m, ids):
        """per-layer pre-mask activations: heads o (B,H,T,dh) and MLP units u (B,T,DFF)."""
        x = m.emb(ids) + m.pos.unsqueeze(0)
        Os, Us = [], []
        for b in m.blocks:
            B, Lq, _ = x.shape
            h = b.ln1(x)
            q, kk, v = b.qkv(h).chunk(3, dim=-1)
            q = q.view(B, Lq, L.NHEAD, b.dh).transpose(1, 2)
            kk = kk.view(B, Lq, L.NHEAD, b.dh).transpose(1, 2)
            v = v.view(B, Lq, L.NHEAD, b.dh).transpose(1, 2)
            att = torch.softmax(q @ kk.transpose(-2, -1) / b.dh ** 0.5, dim=-1)
            o = att @ v
            Os.append(o)
            x = x + b.proj(o.transpose(1, 2).reshape(B, Lq, L.D))
            u = F.gelu(b.fc1(b.ln2(x)))
            Us.append(u)
            x = x + b.fc2(u)
        return Os, Us

    @torch.no_grad()
    def _dsens(m, ids0, ids1, chunk=1000):
        m.eval()
        sh = torch.zeros(L.NHEAD, device=DEV); su = torch.zeros(L.DFF, device=DEV); nb = 0
        for i in range(0, len(ids0), chunk):
            O0, U0 = _acts(m, ids0[i:i + chunk]); O1, U1 = _acts(m, ids1[i:i + chunk])
            for l in range(L.NLAYER):
                sh += (O0[l] - O1[l]).abs().mean(dim=(0, 2, 3))
                su += (U0[l] - U1[l]).abs().mean(dim=(0, 1))
            nb += 1
        m.train()
        return (sh / (nb * L.NLAYER)).cpu().numpy(), (su / (nb * L.NLAYER)).cpu().numpy()

    def sens_lm(m, ids, cA, cB, g):
        idsA = L.occlude(ids, cA, L.CUE_A_FIELD, g)   # destroy cue A
        idsB = L.occlude(ids, cB, L.CUE_B_FIELD, g)   # destroy cue B
        shA, suA = _dsens(m, ids, idsA)
        shB, suB = _dsens(m, ids, idsB)
        return np.stack([shA, shB]), np.stack([suA, suB])

    def mk(heads, units):
        hm = torch.zeros(L.NHEAD, device=DEV); hm[list(heads)] = 1.0
        um = torch.zeros(L.DFF, device=DEV); um[list(units)] = 1.0
        return (hm, um)

    def crit(m, mA, mB):
        d = dict(selA_deploy=L.acc(m, Xd, yd, mA), selA_held=L.acc(m, Xh, yh, mA),
                 selB_deploy=L.acc(m, Xd, yd, mB), selB_held=L.acc(m, Xh, yh, mB))
        des = (d["selA_deploy"] > HI and d["selA_held"] > TRAIN_OK
               and d["selB_deploy"] < LO and d["selB_held"] > TRAIN_OK)
        swp = (d["selB_deploy"] > HI and d["selB_held"] > TRAIN_OK
               and d["selA_deploy"] < LO and d["selA_held"] > TRAIN_OK)
        d["success"] = bool(des or swp); d["designated"] = bool(des)
        return d

    rows, maxdiff = [], 0.0
    rng = np.random.default_rng(0)
    for seed in SEQ(range(700, 712)):
        m = train_control(seed)
        old = crit(m, mk(L.HEADS_A, L.UNITS_A), mk(L.HEADS_B, L.UNITS_B))
        pr = pubrows[seed]
        maxdiff = max(maxdiff, max(abs(old[k] - pr[k]) for k in
                                   ["selA_deploy", "selA_held", "selB_deploy", "selB_held"]))
        g = torch.Generator(device=DEV); g.manual_seed(MASK_GEN_SEED)
        sh, su = sens_lm(m, Xh, cAh, cBh, g)
        gh = fmp_assign(sh, L.NHEAD // 2)
        gu = fmp_assign(su, L.DFF // 2)
        fair = crit(m, mk(gh[0], gu[0]), mk(gh[1], gu[1]))
        nc = 0
        for _ in range(5):
            ph = rng.permutation(L.NHEAD); pu = rng.permutation(L.DFF)
            nc += crit(m, mk(ph[:2].tolist(), pu[:64].tolist()),
                       mk(ph[2:].tolist(), pu[64:].tolist()))["success"]
        rows.append(dict(seed=seed, old=old, fair=fair, headsA=gh[0], headsB=gh[1],
                         degenerate=bool(sorted(gh[0] + gh[1]) != list(range(L.NHEAD))
                                         or sorted(gu[0] + gu[1]) != list(range(L.DFF))),
                         rand_succ_of5=int(nc)))
        print(f"  lm s{seed} old={old['success']} fair={fair['success']} headsA={gh[0]} "
              f"selA(dep/held)={fair['selA_deploy']:.3f}/{fair['selA_held']:.3f} "
              f"selB={fair['selB_deploy']:.3f}/{fair['selB_held']:.3f} ({time.time()-T0:.0f}s)",
              flush=True)
    n = len(rows); kf = sum(r["fair"]["success"] for r in rows)
    old_flags = [r["old"]["success"] for r in rows]
    fair_flags = [r["fair"]["success"] for r in rows]
    cell("cell_lm_organism", 12, 12, LM["control"]["summary"]["successes"], kf, n,
         old_flags, fair_flags,
         dict(anchor_A0_maxdiff=round(maxdiff, 9), anchor_A0_pass=bool(maxdiff < 1e-6),
              old_floor_recomputed=int(sum(old_flags)),
              NC1_random_split_succ=int(sum(r["rand_succ_of5"] for r in rows)),
              NC1_random_split_trials=5 * n,
              NC2_degenerate_masks=int(sum(r["degenerate"] for r in rows)),
              per_seed=rows))
    RES["stages_done"].append("lm"); save()


# =====================================================================================
def holm_and_verdicts():
    confirmatory = [k for k in RES["cells"] if "REFERENCE" not in k]
    ps = sorted([(RES["cells"][c]["p_routed_vs_fair"], c) for c in confirmatory])
    mfam = len(ps); out = {}
    prev = 0.0
    for i, (p, c) in enumerate(ps):
        thr = 0.05 / (mfam - i)
        adj = max(prev, min(1.0, p * (mfam - i))); prev = adj
        cc = RES["cells"][c]
        passed = adj < 0.05
        if not passed:
            v = "DIES"
        elif cc["D_new"] < 0.5 * cc["D_old"]:
            v = "SHRINKS"
        else:
            v = "SURVIVES"
        cc["holm_p_adj"] = round(adj, 6); cc["holm_threshold"] = round(thr, 6)
        cc["VERDICT"] = v
        out[c] = dict(p=p, holm_adj=round(adj, 6), verdict=v,
                      D_old=cc["D_old"], D_new=cc["D_new"], ci=cc["D_new_newcombe95"])
    RES["holm_family"] = dict(m=mfam, alpha=0.05, results=out)
    save()
    print("\nVERDICTS:", json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    stages = [("r1r2", stage_r1r2), ("a1", stage_a1),
              ("mw_w8", lambda: stage_multiway([2, 3, 4], 8, "w8")),
              ("entangled", stage_entangled),
              ("lm", stage_lm),
              ("mw_w4", lambda: stage_multiway([2, 3, 4], 4, "w4"))]
    for nm, fn in stages:
        if time.time() - T0 > BUDGET_SEC:
            RES["stages_not_run"].append(nm); save()
            print("BUDGET: skipping", nm, flush=True); continue
        try:
            fn(); RES["stages_done"].append(nm)
        except Exception as e:
            import traceback; traceback.print_exc()
            RES.setdefault("errors", {})[nm] = repr(e)
        save()
    holm_and_verdicts()
    RES["runtime_sec"] = round(time.time() - T0, 1)
    save()
    print("wrote", OUT, RES["runtime_sec"], "s")
