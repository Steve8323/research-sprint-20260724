"""r5-optionality-case -- shared harness.

Imports round-4's core READ-ONLY (sys.dont_write_bytecode so no __pycache__ is written into it).
Everything here is forward-only plumbing plus stats; the models come from core.train_arm.
"""
import sys, copy, math
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r4-break-tautology")

import numpy as np
import torch, torch.nn as nn
from scipy import stats as sps

import core as C  # read-only parent harness (round 4)

torch.set_num_threads(2)  # shared machine

SEEDS = list(range(700, 760))          # n = 60, prereg
DEP_SEED, HLD_SEED, SEL_SEED = 50001, 60001, 50777
NSEL = 32                               # selection pool size (first m used)
N_RAND = 200                            # blind random-mask family (100 subsets + complements)

# cost ledger (example-forwards)
COST_ORD_TRAIN = C.EPOCHS * C.N          # 1.6e6
COST_ROUT_TRAIN = C.EPOCHS * 2 * C.N     # 3.2e6


# ---------------------------------------------------------------- data / interventions
def eval_sets():
    Xd, yd = C.gen_align(4000, DEP_SEED, "deploy")
    Xh, yh = C.gen_align(4000, HLD_SEED, "train")
    Xs, ys = C.gen_align(NSEL, SEL_SEED, "deploy")
    return (Xd, yd), (Xh, yh), (Xs, ys)


def occluded_views(X, seed):
    """Deterministic per-seed occluded copies: cue-A noised, cue-B noised."""
    torch.manual_seed(seed * 1000 + 7)
    return C.occlude(X, C.CUE_A), C.occlude(X, C.CUE_B)


def sens_score(model, sp, X, gen_seed=99991):
    """Byte-identical to core.learned_masks' scoring (anchor A0 checks this)."""
    g = torch.Generator().manual_seed(gen_seed)
    XA = X.clone(); XA[:, C.CUE_A] = C.NOISE * torch.randn(len(X), len(C.CUE_A), generator=g)
    XB = X.clone(); XB[:, C.CUE_B] = C.NOISE * torch.randn(len(X), len(C.CUE_B), generator=g)
    L = sp["score_layer"]
    a0, aA, aB = model.acts(X, L), model.acts(XA, L), model.acts(XB, L)
    return ((a0 - aA).abs().mean(0) - (a0 - aB).abs().mean(0)).numpy()


def topk_mask(score, H, w, sign):
    idx = np.argsort(-sign * score)[:w]
    return C.unit_mask(H, sorted(idx.tolist()))


class Cand:
    """A no-gradient deployment intervention."""
    __slots__ = ("name", "umask", "view", "flip")

    def __init__(self, name, umask=None, view="raw", flip=False):
        self.name, self.umask, self.view, self.flip = name, umask, view, flip


def cand_preds(model, views, c):
    X = views[c.view]
    with torch.no_grad():
        lo = model(X, c.umask)
        if c.flip:
            lo = -lo
    return lo.argmax(1)


def cand_metrics(model, dviews, hviews, sviews, yd, yh, ys, c):
    """(deploy alignment, heldout acc, acc-on-selection-pool toward TRUE)."""
    ad = (cand_preds(model, dviews, c) == yd).float().mean().item()
    ah = (cand_preds(model, hviews, c) == yh).float().mean().item()
    asel = (cand_preds(model, sviews, c) == ys).float().numpy()   # per-example correctness
    return ad, ah, asel


def success(align, hold, target):
    if target == "TRUE":
        return bool(align > C.ALIGN_HI and hold > C.TRAIN_OK)
    return bool(align < C.ALIGN_LO and hold > C.TRAIN_OK)


def select(cands, sel_correct, m, target):
    """Prereg selection rule: argmax accuracy-toward-target on the first m labelled deployment
    examples; ties -> first in the fixed candidate order.  m = 0 is not allowed here."""
    best, bi = -1.0, 0
    for i, c in enumerate(cands):
        v = sel_correct[i][:m]
        acc = v.mean() if target == "TRUE" else (1.0 - v).mean()
        if acc > best + 1e-12:
            best, bi = acc, i
    return bi, best


# ---------------------------------------------------------------- fine-tuning
def finetune(model, X, y, k, lr=None):
    m2 = copy.deepcopy(model)
    if k == 0:
        return m2
    opt = torch.optim.Adam(m2.parameters(), lr or C.LR, weight_decay=C.WD)
    lf = nn.CrossEntropyLoss()
    for _ in range(k):
        opt.zero_grad(); lf(m2(X), y).backward(); opt.step()
    return m2


def entropy_min(model, X, k):
    m2 = copy.deepcopy(model)
    opt = torch.optim.Adam(m2.parameters(), C.LR, weight_decay=C.WD)
    for _ in range(k):
        opt.zero_grad()
        p = torch.softmax(m2(X), 1)
        loss = -(p * torch.log(p + 1e-9)).sum(1).mean()
        loss.backward(); opt.step()
    return m2


# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar_exact(a, b):
    """a, b boolean arrays (paired). Returns dict with discordant counts + two-sided exact p."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int((a & ~b).sum())   # a wins
    n10 = int((~a & b).sum())   # b wins
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, 2 * sps.binom.cdf(k, n, 0.5))
    return dict(a_only=n01, b_only=n10, p_exact=float(p))


def boot_paired(a, b, reps=10000, seed=12345, level=95):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed); n = len(a)
    idx = rng.integers(0, n, (reps, n))
    d = (a[idx] - b[idx]).mean(1)
    lo, hi = np.percentile(d, [(100 - level) / 2, 100 - (100 - level) / 2])
    return dict(diff=round(float(a.mean() - b.mean()), 4),
                ci=[round(float(lo), 4), round(float(hi), 4)], level=level)


def tost_paired(a, b, delta=0.15, reps=10000, seed=12345):
    """Equivalence via the 90% bootstrap CI of the paired difference (TOST-equivalent)."""
    r = boot_paired(a, b, reps, seed, level=90)
    r["delta"] = delta
    r["equivalent"] = bool(r["ci"][0] > -delta and r["ci"][1] < delta)
    return r


def binom_vs_half(k, n):
    return float(sps.binomtest(k, n, 0.5).pvalue) if n else 1.0


def fisher(k1, n1, k2, n2):
    return float(sps.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
