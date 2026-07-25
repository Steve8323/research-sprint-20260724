"""
r5-latent-switch-standard -- MLP substrate.

Reimplements the round-4 `floor_control` arm (READ-ONLY parent:
/Users/steven/research_sprint_20260724/r4-break-tautology/core.py) with three generalizations
that the round-5 prereg needs:

  (a) hidden width H is a free parameter (round 4 fixed H = 16),
  (b) the training data can DISAGREE (fraction f of examples where the proxy cue is wrong) --
      the negative control, since then the worldview is determined and there should be
      nothing latent to select,
  (c) k cues instead of 2 (multi-way latent selection).

For k = 2, f = 0, H = 16 the RNG consumption order is identical to round 4's `floor_control`,
so those cells reproduce round-4's stored per-seed numbers exactly (anchor A0).

NOTHING here is routed. Every model in this file is STANDARD-trained on the agreeing data.
"""
import itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)  # shared machine, other agents concurrent

# ---- config: verbatim from round-4 / round-1 / alignment.py ----
NBITS = 2; NNOISE = 3
MARGIN = 2.0; NOISE = 0.6; N = 800
WD = 2e-3; LR = 5e-3; EPOCHS = 2000
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9   # pre-stated thresholds, inherited unchanged
NEVAL = 4000
DEPLOY_SEED, HELD_SEED = 50001, 60001
MASK_GEN_SEED = 99991                          # round-4's learned_masks generator seed


def inp_dim(k):
    return 2 * k + NNOISE


def cue_cols(k):
    return [[2 * i, 2 * i + 1] for i in range(k)]


def noise_cols(k):
    return list(range(2 * k, 2 * k + NNOISE))


# ---- data ----
def xor_block(target, rng):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, f=0.0):
    """k=2 ALIGN toy, verbatim round-1/round-4 for f=0 (identical RNG consumption).
    f>0 mixes in anti-proxy examples (the proxy cue disagrees with the label) -- exactly the
    `f` of ~/moral_framework_misgen/alignment.py. With f>0 the worldview is DETERMINED."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    if mode == "deploy":
        a = 1 - v
    else:
        a = v.copy()
        if f > 0:
            msk = rng.random(n) < f
            a[msk] = 1 - v[msk]
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


def gen_k(n, seed, k, mode):
    """k mutually-agreeing cues (round-3 multiway toy).
    mode 'train'      : every cue == label.
    mode ('probe', j) : cue j == label, ALL other cues flipped -> isolates worldview j."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    blocks = []
    for i in range(k):
        t = v if (mode == "train" or (isinstance(mode, tuple) and i == mode[1])) else (1 - v)
        blocks.append(xor_block(t, rng))
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate(blocks + [Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


# ---- model: identical shape to round 1 / round 4 ----
class Net(nn.Module):
    def __init__(self, H, inp):
        super().__init__()
        self.H = H
        self.l1 = nn.Linear(inp, H); self.l2 = nn.Linear(H, H); self.l3 = nn.Linear(H, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)

    def acts(self, x, layer=1):
        with torch.no_grad():
            h1 = F.relu(self.l1(x))
            return h1 if layer == 1 else F.relu(self.l2(h1))


def unit_mask(H, keep):
    m = torch.zeros(H); m[list(keep)] = 1.0
    return m


def accm(m, X, y, umask=None):
    with torch.no_grad():
        return (m(X, umask).argmax(1) == y).float().mean().item()


def train_standard(seed, H, inp, Xtr, ytr):
    """STANDARD training. No routing, no occlusion, no mask. Model init consumes torch RNG in
    exactly round-4's order (l1, l2, l3)."""
    m = Net(H, inp)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        lf(m(Xtr), ytr).backward()
        opt.step()
    return m


# ---- the data-chosen mask (round-4's S4 procedure, unchanged) ----
def sensitivity(m, X, cols_list, layer=1, gen_seed=MASK_GEN_SEED):
    """sens[i][u] = mean |change in unit u's pre-mask activation| when the columns in
    cols_list[i] are resampled as noise. Computed on HELD-OUT AGREEING data only."""
    g = torch.Generator().manual_seed(gen_seed)
    a0 = m.acts(X, layer)
    out = []
    for cols in cols_list:
        Xc = X.clone()
        Xc[:, cols] = NOISE * torch.randn(len(X), len(cols), generator=g)
        out.append((a0 - m.acts(Xc, layer)).abs().mean(0))
    return out


def learned_masks_2(m, X, k=2, layer=1, cols_list=None, gen_seed=MASK_GEN_SEED):
    """Round-4 S4 verbatim: score = sens_A - sens_B; top-H/2 selects A, bottom-H/2 selects B.
    Returns (keepA, keepB, score)."""
    if cols_list is None:
        cc = cue_cols(k); cols_list = [cc[0], cc[1]]
    sA, sB = sensitivity(m, X, cols_list, layer, gen_seed)
    score = (sA - sB).numpy()
    w = m.H // 2
    return sorted(np.argsort(-score)[:w].tolist()), sorted(np.argsort(score)[:w].tolist()), score.tolist()


def learned_partition_k(m, X, k, layer=1, gen_seed=MASK_GEN_SEED):
    """k-way data-chosen partition: assign each unit to argmax_s sens_s(u). A genuine partition
    (groups may be unbalanced; sizes are reported and the random control is size-matched)."""
    sens = sensitivity(m, X, cue_cols(k), layer, gen_seed)
    S = torch.stack(sens, 0).numpy()          # (k, H)
    assign = S.argmax(0)
    return [sorted(np.where(assign == s)[0].tolist()) for s in range(k)], S.tolist()


def crit2(m, H, kA, kB, Xd, yd, Xh, yh):
    """Success criterion verbatim from rounds 1/3/4."""
    mA, mB = unit_mask(H, kA), unit_mask(H, kB)
    d = dict(aA=accm(m, Xd, yd, mA), tA=accm(m, Xh, yh, mA),
             aB=accm(m, Xd, yd, mB), tB=accm(m, Xh, yh, mB))
    d["designated"] = bool(d["aA"] > ALIGN_HI and d["tA"] > TRAIN_OK
                           and d["aB"] < ALIGN_LO and d["tB"] > TRAIN_OK)
    d["swapped"] = bool(d["aB"] > ALIGN_HI and d["tB"] > TRAIN_OK
                        and d["aA"] < ALIGN_LO and d["tA"] > TRAIN_OK)
    d["success"] = bool(d["designated"] or d["swapped"])
    return d


def critk(m, H, groups, Xh, yh, probes):
    """k-way criterion, verbatim from r3-multiway-switch.score()."""
    k = len(groups)
    if any(len(g) == 0 for g in groups):
        return dict(designated=False, any_perm=False, success=False, empty_group=True,
                    held=[], probe=[])
    hld, probe = [], []
    for s in range(k):
        um = unit_mask(H, groups[s])
        hld.append(accm(m, Xh, yh, um))
        probe.append([accm(m, Xp, yp, um) for (Xp, yp) in probes])

    def ok(perm):
        for s in range(k):
            if hld[s] <= TRAIN_OK: return False
            if probe[s][perm[s]] <= ALIGN_HI: return False
            for j in range(k):
                if j != s and probe[s][perm[j]] >= ALIGN_LO: return False
        return True

    des = ok(tuple(range(k)))
    anyp = any(ok(p) for p in itertools.permutations(range(k)))
    return dict(designated=bool(des), any_perm=bool(anyp), success=bool(anyp),
                empty_group=False, held=[round(x, 4) for x in hld],
                probe=[[round(x, 4) for x in r] for r in probe])
