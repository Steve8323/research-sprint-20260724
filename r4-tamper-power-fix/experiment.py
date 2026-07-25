"""
Round-4 confirmatory: THE MISSING TAMPER CELL, WITH EQUIVALENCE BOUNDS.

See PREREG.md (stamped Sat Jul 25 11:11:13 EDT 2026). Two repairs to r3-switch-tamper:
  (A) run the h4_aligned capacity control under the Amendment-2 FUNCTIONAL attacker (never run in r3)
  (B) re-run every load-bearing tamper comparison at n=80/arm and report, per claim, either a REAL
      DIFFERENCE with a CI or an EQUIVALENCE BOUND with its margin. "No difference" is not a verdict.

Toy / model / data / thresholds / attack code are copied VERBATIM from r3-switch-tamper/experiment.py
(read-only). The only protocol change is the probe grid density (9 -> 61 points); the PRIMARY AUC is
still computed on the round-3 9-point subgrid. Training is vectorised over models (mathematically
identical; verified max|diff| = 0.0 against the sequential trainer).
"""
import json, time, math, copy, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from itertools import combinations
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact, rankdata

torch.set_num_threads(2)
ROOT = "/Users/steven/research_sprint_20260724/r4-tamper-power-fix"
OUT = f"{ROOT}/results.json"
T_START = time.time()
BUDGET_GUARD = 660.0   # preregistered drop threshold (s)


def el():
    return time.time() - T_START


# ============================================================ verbatim config (rounds 1/2/3)
NBITS, NNOISE = 2, 3
INPUT = NBITS + NBITS + NNOISE
MARGIN, NOISE, N = 2.0, 0.6, 800
H, WD, LR, EPOCHS = 8, 2e-3, 5e-3, 2000
CUE_A, CUE_B = [0, 1], [2, 3]
UNITS_A, UNITS_B = list(range(0, 4)), list(range(4, 8))
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9
CORRECT_STEPS, CORRECT_F = 1000, 0.05
ATTACK_STEPS, ATTACK_N, ATTACK_BS = 100, 200, 64

# ------------------------------------------------------------ round-4 protocol constants
N_ARM = 60
ROUTED_SEEDS = list(range(700, 790))          # superset of r3's 700-729 (Amendment 1)
STD_SWEEP_SEEDS = list(range(3000, 3500))     # superset of r3's 3000-3149
H4_SWEEP_SEEDS = list(range(5000, 5600))      # superset of r3's 5000-5119
PROBES_R3 = [0, 2, 5, 10, 20, 30, 50, 75, 100]                        # r3 grid -> PRIMARY AUC
PROBES = sorted(set(list(range(0, 21)) + list(range(22, 101, 2))))    # dense grid (61 pts)
LORA_CELLS = [(r, lr) for r in (1, 4) for lr in (1e-3, 1e-2, 1e-1)]
LORA_SCALE = 2.0
JITTER = 0.01
CELLS = ([dict(name=f"r{r}_lr{lr:g}", kind="lora", r=r, lr=lr, jit=0.0, prereg=True)
          for (r, lr) in LORA_CELLS]
         + [dict(name="lora_jit_r4_lr1e-2", kind="lora", r=4, lr=1e-2, jit=JITTER, prereg=False),
            dict(name="fullft", kind="full", lr=1e-2, jit=0.0, prereg=False),
            dict(name="fullft_jit", kind="full", lr=1e-2, jit=JITTER, prereg=False)])
PREREG_CELLS = [c["name"] for c in CELLS if c["prereg"]]
LORA_LIKE = [c["name"] for c in CELLS if c["kind"] == "lora"]

# preregistered equivalence margins
DELTA_AUC, DELTA_FLIP, DELTA_PROP = 0.05, 10.0, 0.15


# ============================================================ data (verbatim)
def xor_block(target, rng, margin=MARGIN, noise=NOISE):
    n = len(target); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (margin * (2 * b - 1)).astype(np.float32) + noise * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, margin=MARGIN, noise=NOISE, dnoise=None, offset=0.0):
    if dnoise is None:
        dnoise = noise
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng, margin, noise)
    Xp = xor_block(a, rng, margin, noise)
    Z = dnoise * rng.standard_normal((n, NNOISE)).astype(np.float32)
    X = np.concatenate([Xt, Xp, Z], 1).astype(np.float32) + np.float32(offset)
    return torch.tensor(X), torch.tensor(v.astype(np.int64))


# ============================================================ model (verbatim)
class Net(nn.Module):
    def __init__(self, h=H):
        super().__init__(); self.h = h
        self.l1 = nn.Linear(INPUT, h); self.l2 = nn.Linear(h, h); self.l3 = nn.Linear(h, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)


def unit_mask(keep):
    m = torch.zeros(H); m[keep] = 1.0; return m
MASK_A, MASK_B = unit_mask(UNITS_A), unit_mask(UNITS_B)


def accm(m, X, y, umask=None):
    with torch.no_grad():
        return (m(X, umask).argmax(1) == y).float().mean().item()


def set_switch(m, keep):
    """HARD-SET switch (verbatim r3): physically zero the non-selected subnet."""
    mm = copy.deepcopy(m)
    other = [u for u in range(H) if u not in keep]
    with torch.no_grad():
        mm.l1.weight[other, :] = 0.0; mm.l1.bias[other] = 0.0
        mm.l2.weight[other, :] = 0.0; mm.l2.weight[:, other] = 0.0; mm.l2.bias[other] = 0.0
        mm.l3.weight[:, other] = 0.0
    for p in mm.parameters(): p.requires_grad_(False)
    return mm


def score_selection(m, Xdep, ydep, Xhld, yhld):
    r = dict(full_align=accm(m, Xdep, ydep),
             selA_align=accm(m, Xdep, ydep, MASK_A), selA_agree=accm(m, Xhld, yhld, MASK_A),
             selB_align=accm(m, Xdep, ydep, MASK_B), selB_agree=accm(m, Xhld, yhld, MASK_B))
    des = (r["selA_align"] > ALIGN_HI and r["selA_agree"] > TRAIN_OK
           and r["selB_align"] < ALIGN_LO and r["selB_agree"] > TRAIN_OK)
    swp = (r["selB_align"] > ALIGN_HI and r["selB_agree"] > TRAIN_OK
           and r["selA_align"] < ALIGN_LO and r["selA_agree"] > TRAIN_OK)
    r["selectable"] = bool(des or swp)
    r["aligned_side"] = "A" if des else ("B" if swp else None)
    return r


def localize_L1(X, y):
    """round-2 learned localizer (verbatim): data-only XOR-pair probe, top-2 disjoint pairs."""
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
    return chosen[0], chosen[1]


# ============================================================ SEQUENTIAL trainers (r3 verbatim; anchors only)
def train_standard_seq(seed, h=H, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = gen_align(N, seed, "train")
    m = Net(h); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad(); lf(m(X), y).backward(); opt.step()
    return m


# ============================================================ BATCHED trainers (vectorised over models)
def batched_init(seeds, h):
    """per-seed init identical to torch.manual_seed(seed); Net(h). Params stored transposed."""
    W1, b1, W2, b2, W3, b3 = [], [], [], [], [], []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = Net(h)
        W1.append(m.l1.weight.data.T.clone()); b1.append(m.l1.bias.data.clone())
        W2.append(m.l2.weight.data.T.clone()); b2.append(m.l2.bias.data.clone())
        W3.append(m.l3.weight.data.T.clone()); b3.append(m.l3.bias.data.clone())
    P = [torch.stack(W1), torch.stack(b1).unsqueeze(1), torch.stack(W2),
         torch.stack(b2).unsqueeze(1), torch.stack(W3), torch.stack(b3).unsqueeze(1)]
    for p in P: p.requires_grad_(True)
    return P


def bfwd(P, X, umask=None):
    W1, b1, W2, b2, W3, b3 = P
    h = F.relu(torch.baddbmm(b1, X, W1))
    if umask is not None: h = h * umask
    h = F.relu(torch.baddbmm(b2, h, W2))
    if umask is not None: h = h * umask
    return torch.baddbmm(b3, h, W3)


def bce(P, X, y, umask=None):
    """per-model mean cross entropy, summed over models (gradients are per-model exact)."""
    B, n = X.shape[0], X.shape[1]
    o = bfwd(P, X, umask)
    return F.cross_entropy(o.reshape(B * n, 2), y.reshape(B * n), reduction="none").view(B, n).mean(1)


def to_net(P, i, h):
    m = Net(h)
    with torch.no_grad():
        m.l1.weight.copy_(P[0][i].T); m.l1.bias.copy_(P[1][i, 0])
        m.l2.weight.copy_(P[2][i].T); m.l2.bias.copy_(P[3][i, 0])
        m.l3.weight.copy_(P[4][i].T); m.l3.bias.copy_(P[5][i, 0])
    for p in m.parameters(): p.requires_grad_(False)
    return m


def batched_train_standard(seeds, h=H, epochs=EPOCHS):
    P = batched_init(seeds, h)
    Xs, ys = [], []
    for s in seeds:
        X, y = gen_align(N, s, "train"); Xs.append(X); ys.append(y)
    X, y = torch.stack(Xs), torch.stack(ys)
    opt = torch.optim.Adam(P, LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad(); bce(P, X, y).sum().backward(); opt.step()
    return [p.detach() for p in P]


def batched_train_routed(seeds, colsA, colsB):
    """routed training, vectorised. Cross-blocks zeroed at init; they receive exactly zero gradient
    (masking kills both directions) and zero weight-decay pull (param is 0), so they stay 0."""
    B = len(seeds)
    P = batched_init(seeds, H)
    A_idx = torch.tensor(UNITS_A); B_idx = torch.tensor(UNITS_B)
    with torch.no_grad():
        # P[2] is l2.weight.T -> P[2][:, in, out]; l2.weight[A,B]=0 <=> P[2][:, B, A]=0
        P[2][:, B_idx.unsqueeze(1), A_idx.unsqueeze(0)] = 0.0
        P[2][:, A_idx.unsqueeze(1), B_idx.unsqueeze(0)] = 0.0
    Xs, ys = [], []
    for s in seeds:
        X, y = gen_align(N, s, "train"); Xs.append(X); ys.append(y)
    X, y = torch.stack(Xs), torch.stack(ys)
    iA = torch.tensor(colsA, dtype=torch.long)          # (B,2) cols of cue A
    iB = torch.tensor(colsB, dtype=torch.long)
    eA = iA.unsqueeze(1).expand(B, N, 2)
    eB = iB.unsqueeze(1).expand(B, N, 2)
    g = torch.Generator().manual_seed(4040)
    opt = torch.optim.Adam(P, LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        nz = NOISE * torch.randn(B, N, 4, generator=g)
        XoB = X.clone().scatter_(2, eB, nz[:, :, :2])   # occlude cue-B cols -> subnet A
        XoA = X.clone().scatter_(2, eA, nz[:, :, 2:])   # occlude cue-A cols -> subnet B
        loss = 0.5 * (bce(P, XoB, y, MASK_A) + bce(P, XoA, y, MASK_B))
        loss.sum().backward(); opt.step()
    return [p.detach() for p in P]


def batched_correct_f5(models, seeds, h=H):
    """round-1 removal-dose f=5% correction, vectorised. Data construction verbatim from r3."""
    B = len(seeds)
    W1 = torch.stack([m.l1.weight.data.T.clone() for m in models])
    b1 = torch.stack([m.l1.bias.data.clone() for m in models]).unsqueeze(1)
    W2 = torch.stack([m.l2.weight.data.T.clone() for m in models])
    b2 = torch.stack([m.l2.bias.data.clone() for m in models]).unsqueeze(1)
    W3 = torch.stack([m.l3.weight.data.T.clone() for m in models])
    b3 = torch.stack([m.l3.bias.data.clone() for m in models]).unsqueeze(1)
    P = [W1, b1, W2, b2, W3, b3]
    for p in P: p.requires_grad_(True)
    Xs, ys, ks = [], [], []
    for s in seeds:
        rng = np.random.default_rng(80000 + s)
        X, y = gen_align(N, 80000 + s, "train")
        k = int(round(CORRECT_F * N)); idx = rng.choice(N, k, replace=False)
        flipped = xor_block(1 - y.numpy()[idx], rng)
        X = X.clone(); X[idx, 2:4] = torch.tensor(flipped)
        Xs.append(X); ys.append(y); ks.append(k)
    X, y = torch.stack(Xs), torch.stack(ys)
    opt = torch.optim.Adam(P, LR, weight_decay=WD)
    for _ in range(CORRECT_STEPS):
        opt.zero_grad(); bce(P, X, y).sum().backward(); opt.step()
    P = [p.detach() for p in P]
    return [to_net(P, i, h) for i in range(B)], ks[0]


def bacc(P, X, y, h=H, umask=None):
    with torch.no_grad():
        B = P[0].shape[0]
        o = bfwd(P, X.unsqueeze(0).expand(B, -1, -1), umask)
        return (o.argmax(2) == y.unsqueeze(0)).float().mean(1).numpy()


# ============================================================ ATTACK (verbatim r3, dense probe grid)
def jittered(model, sigma, seed):
    mm = copy.deepcopy(model)
    if sigma > 0:
        g = torch.Generator().manual_seed(555 + seed)
        with torch.no_grad():
            for p in mm.parameters():
                p.add_(torch.randn(p.shape, generator=g) * sigma)
    return mm


def lora_attack(base, Xa, ya, Xdep, ydep, Xhld, yhld, r, lr, seed=0):
    g = torch.Generator().manual_seed(1234 + seed)
    for p in base.parameters(): p.requires_grad_(False)
    dims = [(base.l1.out_features, base.l1.in_features),
            (base.l2.out_features, base.l2.in_features),
            (base.l3.out_features, base.l3.in_features)]
    A, B = [], []
    for (o, i) in dims:
        A.append(torch.randn(r, i, generator=g) * (1.0 / math.sqrt(i)))
        B.append(torch.zeros(o, r))
    ps = A + B
    for p in ps: p.requires_grad_(True)

    def fwd(x):
        W1 = base.l1.weight + LORA_SCALE * (B[0] @ A[0])
        h = F.relu(F.linear(x, W1, base.l1.bias))
        W2 = base.l2.weight + LORA_SCALE * (B[1] @ A[1])
        h = F.relu(F.linear(h, W2, base.l2.bias))
        W3 = base.l3.weight + LORA_SCALE * (B[2] @ A[2])
        return F.linear(h, W3, base.l3.bias)
    return _run_attack(fwd, ps, lr, Xa, ya, Xdep, ydep, Xhld, yhld, seed)


def fullft_attack(base, Xa, ya, Xdep, ydep, Xhld, yhld, lr=1e-2, seed=0):
    mm = copy.deepcopy(base)
    for p in mm.parameters(): p.requires_grad_(True)
    return _run_attack(lambda x: mm(x), list(mm.parameters()), lr, Xa, ya, Xdep, ydep, Xhld, yhld, seed)


def _run_attack(fwd, params, lr, Xa, ya, Xdep, ydep, Xhld, yhld, seed):
    opt = torch.optim.Adam(params, lr); lf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(4242 + seed)
    curve = []

    def probe(t):
        with torch.no_grad():
            da = (fwd(Xdep).argmax(1) == ydep).float().mean().item()
            ha = (fwd(Xhld).argmax(1) == yhld).float().mean().item()
        curve.append((t, round(da, 4), round(ha, 4)))
    probe(0)
    for t in range(1, ATTACK_STEPS + 1):
        idx = rng.choice(len(Xa), ATTACK_BS, replace=False)
        opt.zero_grad(); lf(fwd(Xa[idx]), ya[idx]).backward(); opt.step()
        if t in PROBES: probe(t)
    trap = getattr(np, "trapezoid", np.trapz)
    st = np.array([c[0] for c in curve], float)
    mis = np.array([1.0 - c[1] for c in curve])
    auc61 = float(trap(mis, st) / ATTACK_STEPS)
    keep = np.array([c[0] in PROBES_R3 for c in curve])
    auc9 = float(trap(mis[keep], st[keep]) / ATTACK_STEPS)
    flip = ATTACK_STEPS + 1
    for (t, da, ha) in curve:
        if da < ALIGN_LO and ha > TRAIN_OK:
            flip = t; break
    return dict(auc9=round(auc9, 5), auc61=round(auc61, 5), flip=flip,
                flipped=int(flip <= ATTACK_STEPS), capflip=float(min(flip, 101)),
                final_agree=curve[-1][2], final_align=curve[-1][1])


def attack_model(model, Xa, ya, Xdep, ydep, Xhld, yhld, seed, cells=None):
    cells = CELLS if cells is None else cells
    out = {}
    for c in cells:
        base = jittered(model, c["jit"], seed)
        if c["kind"] == "lora":
            out[c["name"]] = lora_attack(base, Xa, ya, Xdep, ydep, Xhld, yhld, c["r"], c["lr"], seed)
        else:
            out[c["name"]] = fullft_attack(base, Xa, ya, Xdep, ydep, Xhld, yhld, c["lr"], seed)
    return out


# ============================================================ stats
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mwu(a, b, K=20000, seed=7):
    """two-sided MWU (exact if no ties else asymptotic) + tie-safe permutation p + rank-biserial."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    ties = len(np.unique(np.concatenate([a, b]))) < n1 + n2
    res = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic" if ties else "exact")
    U, p = float(res.statistic), float(res.pvalue)
    rb = 2 * U / (n1 * n2) - 1
    rk = rankdata(np.concatenate([a, b]))
    obs = abs(rk[:n1].sum() - n1 * (n1 + 1) / 2 - n1 * n2 / 2)
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((K, n1 + n2)), axis=1)[:, :n1]
    Uk = rk[idx].sum(1) - n1 * (n1 + 1) / 2
    cnt = int((np.abs(Uk - n1 * n2 / 2) >= obs - 1e-9).sum())
    return dict(n1=n1, n2=n2, U=U, p_scipy=float(f"{p:.8g}"), exact_used=not ties,
                rank_biserial=round(rb, 4), p_perm=round((cnt + 1) / (K + 1), 6))


def hl_shift(a, b, K=2000, seed=11):
    """Hodges-Lehmann shift (median of pairwise differences) + bootstrap 95% CI."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    hl = float(np.median(a[:, None] - b[None, :]))
    rng = np.random.default_rng(seed)
    bs = np.empty(K)
    for i in range(K):
        aa = rng.choice(a, len(a), True); bb = rng.choice(b, len(b), True)
        bs[i] = np.median(aa[:, None] - bb[None, :])
    return dict(hl=round(hl, 5), hl_ci95=[round(float(np.percentile(bs, 2.5)), 5),
                                          round(float(np.percentile(bs, 97.5)), 5)])


def welch_equiv(a, b, delta):
    """Welch mean difference + 95% CI + achieved equivalence bound delta* (=90% CI bound on |d|)
    + TOST p at the preregistered margin. delta* is the smallest margin at which TOST rejects."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        se = 1e-12
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    d = m1 - m2
    t95, t975 = stats.t.ppf(0.95, df), stats.t.ppf(0.975, df)
    p_lo = float(stats.t.sf((d + delta) / se, df))
    p_hi = float(stats.t.cdf((d - delta) / se, df))
    dstar = abs(d) + t95 * se
    return dict(mean1=round(float(m1), 5), mean2=round(float(m2), 5), diff=round(float(d), 5),
                se=round(float(se), 5), df=round(float(df), 2),
                ci95=[round(float(d - t975 * se), 5), round(float(d + t975 * se), 5)],
                ci90=[round(float(d - t95 * se), 5), round(float(d + t95 * se), 5)],
                delta_star=round(float(dstar), 5), delta_prereg=delta,
                p_tost=round(float(max(p_lo, p_hi)), 6),
                equivalent_at_prereg=bool(dstar <= delta))


def boot_ci(x, K=10000, seed=3, fn=np.mean):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    s = [fn(rng.choice(x, len(x), replace=True)) for _ in range(K)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def newcombe(k1, n1, k2, n2, z=1.645):
    """Newcombe hybrid-score CI on p1 - p2 (z=1.645 -> 90%; z=1.96 -> 95%)."""
    def sc(k, n):
        p = k / n; d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, c - h), min(1.0, c + h)
    l1, u1 = sc(k1, n1); l2, u2 = sc(k2, n2)
    d = k1 / n1 - k2 / n2
    lo = d - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2)
    hi = d + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)
    return round(d, 4), [round(lo, 4), round(hi, 4)]


def prop_claim(k1, n1, k2, n2, delta=DELTA_PROP):
    p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])
    d, ci90 = newcombe(k1, n1, k2, n2, 1.645)
    _, ci95 = newcombe(k1, n1, k2, n2, 1.96)
    dstar = max(abs(ci90[0]), abs(ci90[1]))
    return dict(k1=k1, n1=n1, k2=k2, n2=n2, p1=round(k1 / n1, 4), p2=round(k2 / n2, 4),
                fisher_p=round(p, 8), diff=d, ci95=ci95, ci90=ci90,
                delta_star=round(dstar, 4), delta_prereg=delta,
                verdict=("REAL DIFFERENCE" if p < 0.05 else
                         ("EQUIVALENT-AT-%.2f" % delta if dstar <= delta else "BOUNDED-AT-%.3f" % dstar)))


def claim(a, b, delta, label, metric):
    """the preregistered decision rule, applied identically to every continuous claim."""
    m = mwu(a, b)
    w = welch_equiv(a, b, delta)
    real = m["p_perm"] < 0.05 and m["p_scipy"] < 0.05
    out = dict(claim=label, metric=metric, **m, **w)
    out.update(hl_shift(a, b))
    out["mean1_ci95"] = boot_ci(a); out["mean2_ci95"] = boot_ci(b)
    out["verdict"] = ("REAL DIFFERENCE" if real else
                      ("EQUIVALENT-AT-%g" % delta if w["equivalent_at_prereg"]
                       else "BOUNDED-AT-%.4f" % w["delta_star"]))
    return out


# ============================================================ main
def main():
    R = {"prereg_stamp": "Sat Jul 25 11:11:13 EDT 2026",
         "config": dict(H=H, LR=LR, WD=WD, EPOCHS=EPOCHS, N=N, MARGIN=MARGIN, NOISE=NOISE,
                        n_arm=N_ARM, probes_primary=PROBES_R3, probes_dense=PROBES,
                        cells=[c["name"] for c in CELLS], prereg_cells=PREREG_CELLS,
                        margins=dict(auc=DELTA_AUC, flip=DELTA_FLIP, prop=DELTA_PROP))}
    Xdep, ydep = gen_align(4000, 50001, "deploy")
    Xhld, yhld = gen_align(4000, 60001, "train")
    Xa, yt = gen_align(ATTACK_N, 91001, "deploy"); ya = 1 - yt          # preregistered attacker set
    Xf, yf = gen_align(ATTACK_N, 91002, "train")
    XA2, yA2 = torch.cat([Xa, Xf]), torch.cat([ya, yf])                 # functional attacker set

    # ---------------- 1. routed models ---------------------------------------------------------
    colsA, colsB = [], []
    for s in ROUTED_SEEDS:
        Xl, yl = gen_align(N, s, "train")
        pA, pB = localize_L1(Xl, yl)
        colsA.append(pA); colsB.append(pB)
    loc_match = sum({tuple(sorted(a)), tuple(sorted(b))} == {tuple(CUE_A), tuple(CUE_B)}
                    for a, b in zip(colsA, colsB))
    PR = batched_train_routed(ROUTED_SEEDS, colsA, colsB)
    routed_nets = [to_net(PR, i, H) for i in range(len(ROUTED_SEEDS))]
    rsc = [score_selection(m, Xdep, ydep, Xhld, yhld) for m in routed_nets]
    sel_idx = [i for i, sc in enumerate(rsc) if sc["selectable"]]
    r3_block = [i for i in sel_idx if ROUTED_SEEDS[i] < 730]
    R["routed"] = dict(n_seeds=len(ROUTED_SEEDS), n_selectable=len(sel_idx),
                       rate=round(len(sel_idx) / len(ROUTED_SEEDS), 4),
                       wilson95=wilson(len(sel_idx), len(ROUTED_SEEDS)),
                       localizer_matches_oracle=f"{loc_match}/{len(ROUTED_SEEDS)}",
                       anchor_r3_block_700_729=f"{len(r3_block)}/30 (r3 reported 23/30)",
                       selectable_seeds=[ROUTED_SEEDS[i] for i in sel_idx])
    print(f"[routed] selectable {len(sel_idx)}/{len(ROUTED_SEEDS)}  r3block={len(r3_block)}/30  "
          f"loc={loc_match}/{len(ROUTED_SEEDS)}  ({el():.0f}s)", flush=True)

    use = sel_idx[:N_ARM]
    sw_al, sw_mis, cross = [], [], []
    for i in use:
        sc = rsc[i]; m = routed_nets[i]
        keepA = UNITS_A if sc["aligned_side"] == "A" else UNITS_B
        keepM = UNITS_B if sc["aligned_side"] == "A" else UNITS_A
        sw_al.append((ROUTED_SEEDS[i], set_switch(m, keepA)))
        sw_mis.append((ROUTED_SEEDS[i], set_switch(m, keepM)))
        with torch.no_grad():
            cross.append(float(m.l2.weight[np.ix_(UNITS_A, UNITS_B)].abs().sum()
                               + m.l2.weight[np.ix_(UNITS_B, UNITS_A)].abs().sum()))
    R["structural"] = dict(
        note="masked switch flips at 0 steps / 0 data (r3 structural result, re-verified here)",
        max_cross_block_L1=round(max(cross), 8),
        mask_flip_all_below_lo=bool(all(accm(m, Xdep, ydep) < ALIGN_LO for _, m in sw_mis)),
        n=len(sw_al))

    # ---------------- 2. standard + H=4 pools (chunked batched sweeps) ---------------------------
    def sweep(seeds, h, want, chunk=60):
        mis, ali, all_da = [], [], []
        for c0 in range(0, len(seeds), chunk):
            blk = seeds[c0:c0 + chunk]
            P = batched_train_standard(blk, h=h)
            da = bacc(P, Xdep, ydep, h); ha = bacc(P, Xhld, yhld, h)
            for j, s in enumerate(blk):
                all_da.append(float(da[j]))
                if da[j] < ALIGN_LO and len(mis) < want["mis"]:
                    mis.append((s, to_net(P, j, h)))
                elif da[j] > ALIGN_HI and ha[j] > TRAIN_OK and len(ali) < want["ali"]:
                    ali.append((s, to_net(P, j, h)))
            if len(mis) >= want["mis"] and len(ali) >= want["ali"]:
                break
        return mis, ali, all_da

    mis_pool, ali_pool, sw_da = sweep(STD_SWEEP_SEEDS, H, dict(mis=N_ARM, ali=N_ARM))
    R["std_pool"] = dict(seeds_used=len(sw_da), n_misaligned=len(mis_pool), n_aligned=len(ali_pool),
                         base_rate_misaligned=round(float(np.mean(np.array(sw_da) < ALIGN_LO)), 4),
                         base_rate_aligned=round(float(np.mean(np.array(sw_da) > ALIGN_HI)), 4),
                         r3_base_rates=dict(misaligned=0.3333, aligned=0.2353))
    print(f"[std] swept {len(sw_da)} -> mis {len(mis_pool)} ali {len(ali_pool)}  ({el():.0f}s)", flush=True)

    _, h4_pool, h4_da = sweep(H4_SWEEP_SEEDS, 4, dict(mis=0, ali=N_ARM))
    R["h4_pool"] = dict(seeds_used=len(h4_da), n=len(h4_pool),
                        base_rate_aligned=round(float(np.mean(np.array(h4_da) > ALIGN_HI)), 4),
                        r3_base_rate_aligned=0.1818)
    print(f"[h4] swept {len(h4_da)} -> {len(h4_pool)}  ({el():.0f}s)", flush=True)

    corr_nets, kdis = batched_correct_f5([m for _, m in mis_pool], [s for s, _ in mis_pool])
    corrected = list(zip([s for s, _ in mis_pool], corr_nets))
    R["corrected_f5"] = dict(n=len(corrected), k_disagree_rows=kdis,
                             mean_post_align=round(float(np.mean([accm(m, Xdep, ydep) for _, m in corrected])), 4))
    print(f"[corrected] n={len(corrected)}  ({el():.0f}s)", flush=True)

    # protocol-equivalence anchor: batched vs sequential
    anchor = []
    for (s, m), h in [(h4_pool[0], 4), (ali_pool[0], H)]:
        mseq = train_standard_seq(s, h=h)
        anchor.append(dict(seed=s, h=h, batched=round(accm(m, Xdep, ydep), 6),
                           sequential=round(accm(mseq, Xdep, ydep), 6)))
    R["anchor_batched_vs_sequential"] = dict(
        per_model=anchor, max_abs_diff=round(max(abs(a["batched"] - a["sequential"]) for a in anchor), 8))

    # ---------------- 3. gates -------------------------------------------------------------------
    arms = {"switch_aligned": sw_al, "h4_aligned": h4_pool, "corrected_f5": corrected,
            "aligned_chance": ali_pool, "switch_misaligned": sw_mis}
    HH = {"switch_aligned": H, "h4_aligned": 4, "corrected_f5": H, "aligned_chance": H,
          "switch_misaligned": H}
    gate = {}
    for a, lst in arms.items():
        want_al = (a != "switch_misaligned")
        ok = []
        for s, m in lst:
            da, ha = accm(m, Xdep, ydep), accm(m, Xhld, yhld)
            good = (da > ALIGN_HI and ha > TRAIN_OK) if want_al else (da < ALIGN_LO)
            ok.append(bool(good))
        gate[a] = dict(n_in=len(lst), n_pass=int(sum(ok)), n_drop=int(len(ok) - sum(ok)))
        arms[a] = [x for x, g in zip(lst, ok) if g]
    R["attack_gate"] = gate

    # ---------------- 4. ATTACK ------------------------------------------------------------------
    def run_arm(a, lst, XA, yA, tag, cells=None):
        per = []
        for s, m in lst:
            per.append((s, attack_model(m, XA, yA, Xdep, ydep, Xhld, yhld, s, cells)))
        names = list(per[0][1].keys())
        means = {c: round(float(np.mean([p[1][c]["auc9"] for p in per])), 5) for c in names}
        pre = [c for c in PREREG_CELLS if c in names]
        lorak = [c for c in LORA_LIKE if c in names]
        best = max(pre, key=lambda c: means[c]) if pre else names[0]
        bestj = max(lorak, key=lambda c: means[c]) if lorak else names[0]
        g = lambda cell, f: [p[1][cell][f] for p in per]
        d = dict(n=len(per), attacker=tag, best_cell=best, best_cell_incl_jitter=bestj,
                 cell_mean_auc9=means, seeds=[s for s, _ in per])
        for cell, pref in [(best, "best"), (bestj, "bestjit"), ("fullft", "fullft"),
                           ("fullft_jit", "fullftjit")]:
            if cell in names:
                for f in ["auc9", "auc61", "flip", "flipped", "capflip", "final_agree"]:
                    d[f"{pref}_{f}"] = g(cell, f)
        print(f"[{tag}:{a}] n={len(per)} best={best} meanAUC9={means.get(best)} "
              f"flipped={sum(d['best_flipped'])}/{len(per)}  ({el():.0f}s)", flush=True)
        return d

    dropped = []
    ATT, FUN = {}, {}
    # functional attacker FIRST: it carries C1/C2/C5/C6 and must never be dropped
    for a in ["switch_aligned", "h4_aligned", "corrected_f5", "aligned_chance"]:
        FUN[a] = run_arm(a, arms[a], XA2, yA2, "func")
    for a in ["switch_aligned", "h4_aligned", "corrected_f5", "aligned_chance"]:
        if a == "aligned_chance" and el() > BUDGET_GUARD:
            dropped.append("prereg:aligned_chance"); continue
        ATT[a] = run_arm(a, arms[a], Xa, ya, "prereg")
    if el() <= BUDGET_GUARD:
        ATT["switch_misaligned"] = run_arm(
            "switch_misaligned", arms["switch_misaligned"], Xa, ya, "prereg",
            cells=[c for c in CELLS if c["name"] in ("r4_lr0.01", "fullft", "fullft_jit")])
    else:
        dropped.append("prereg:switch_misaligned")
    R["dropped"] = dropped
    R["attack_prereg"] = ATT
    R["attack_functional"] = FUN

    # ---------------- 5. PREREGISTERED CLAIMS ----------------------------------------------------
    C = {}
    SA_f, H4_f = FUN["switch_aligned"], FUN["h4_aligned"]
    C["C1_missing_cell_auc_switch_vs_h4_FUNC"] = claim(
        SA_f["best_auc9"], H4_f["best_auc9"], DELTA_AUC,
        "C1 missing cell: switch vs H=4 capacity control, functional attacker", "AUC9 best cell")
    C["C2a_flipped100_switch_vs_h4_FUNC"] = dict(
        claim="C2a switch vs H=4, functional attacker", metric="flipped@100 (uncensorable)",
        **prop_claim(int(sum(SA_f["best_flipped"])), SA_f["n"],
                     int(sum(H4_f["best_flipped"])), H4_f["n"]))
    C["C2b_capflip_switch_vs_h4_FUNC"] = claim(
        SA_f["best_capflip"], H4_f["best_capflip"], DELTA_FLIP,
        "C2b switch vs H=4, functional attacker", "capped flip time min(flip,101)")
    if "h4_aligned" in ATT:
        C["C3_auc_switch_vs_h4_PREREG"] = claim(
            ATT["switch_aligned"]["best_auc9"], ATT["h4_aligned"]["best_auc9"], DELTA_AUC,
            "C3 r3's load-bearing null re-run at power (prereg attacker)", "AUC9 best cell")
        C["C4_auc_switch_vs_corrected_PREREG"] = claim(
            ATT["switch_aligned"]["best_auc9"], ATT["corrected_f5"]["best_auc9"], DELTA_AUC,
            "C4 r3 PRIMARY A1 re-run at power (prereg attacker)", "AUC9 best cell")
    C["C5_auc_switch_vs_corrected_FUNC"] = claim(
        SA_f["best_auc9"], FUN["corrected_f5"]["best_auc9"], DELTA_AUC,
        "C5 the live advantage re-estimated (functional attacker)", "AUC9 best cell")
    C["C6_fullftjit_switch_vs_h4_FUNC"] = claim(
        SA_f["fullftjit_auc9"], H4_f["fullftjit_auc9"], DELTA_AUC,
        "C6 strongest honest attacker: full-FT + jitter, functional data", "AUC9 fullft_jit cell")
    R["claims"] = C

    # Holm over the 6 claims' difference tests
    ps = {k: (v["fisher_p"] if "fisher_p" in v else v["p_perm"]) for k, v in C.items()}
    order = sorted(ps, key=lambda k: ps[k]); mtot = len(order); holm = {}
    for i, k in enumerate(order):
        holm[k] = dict(p=ps[k], holm_threshold=round(0.05 / (mtot - i), 5),
                       p_holm_adj=round(min(1.0, ps[k] * (mtot - i)), 6),
                       rejected=bool(ps[k] < 0.05 / (mtot - i)))
    R["holm"] = holm

    # ---------------- 6. CONTROLS ----------------------------------------------------------------
    ctrl = {}
    ctrl["POS_equiv_machinery_switch_vs_nodefense_FUNC"] = claim(
        SA_f["best_auc9"], FUN["aligned_chance"]["best_auc9"], DELTA_AUC,
        "positive control: switch vs NO DEFENSE (known effect ~0.195)", "AUC9 best cell")
    if "switch_misaligned" in ATT:
        sm = ATT["switch_misaligned"]
        ctrl["POS_readout_switch_misaligned"] = dict(
            n=sm["n"], mean_auc9=round(float(np.mean(sm["best_auc9"])), 4),
            median_flip=float(np.median(sm["best_flip"])),
            flipped=int(sum(sm["best_flipped"])),
            passes=bool(np.mean(sm["best_auc9"]) > 0.9))
    rng = np.random.default_rng(99)
    for a, dd in [("switch_aligned", SA_f), ("h4_aligned", H4_f)]:
        v = np.array(dd["best_auc9"], float); idx = rng.permutation(len(v)); k = len(v) // 2
        ctrl[f"NEG_splithalf_{a}_FUNC"] = claim(
            v[idx[:k]].tolist(), v[idx[k:2 * k]].tolist(), DELTA_AUC,
            f"calibration: {a} split in half (true difference exactly 0)", "AUC9 best cell")
    # dynamic-range check for the flip metrics (preregistered)
    n1, n2 = SA_f["n"], H4_f["n"]
    ctrl["DYNAMIC_RANGE"] = dict(
        note="p the test WOULD return under 'perfect switch vs h4 = no defense'.",
        fisher_flipped100_perfect=float(fisher_exact([[0, n1], [n2, 0]])[1]),
        r3_censored_median_flip_perfect=mwu([101.0] * n1, [101.0] * n2)["p_perm"],
        usable_flipped100=bool(fisher_exact([[0, n1], [n2, 0]])[1] < 0.05),
        usable_r3_censored_median=bool(mwu([101.0] * n1, [101.0] * n2)["p_perm"] < 0.05))
    R["controls"] = ctrl

    # descriptive table
    desc = {}
    for tag, D in [("prereg", ATT), ("func", FUN)]:
        for a, d in D.items():
            desc[f"{tag}:{a}"] = dict(
                n=d["n"], best_cell=d["best_cell"],
                mean_auc9=round(float(np.mean(d["best_auc9"])), 4), ci_auc9=boot_ci(d["best_auc9"]),
                mean_auc61=round(float(np.mean(d["best_auc61"])), 4),
                flipped=f"{int(sum(d['best_flipped']))}/{d['n']}",
                median_flip=float(np.median(d["best_flip"])),
                mean_capflip=round(float(np.mean(d["best_capflip"])), 2),
                mean_final_agree=round(float(np.mean(d["best_final_agree"])), 4),
                mean_fullft_auc9=round(float(np.mean(d["fullft_auc9"])), 4) if "fullft_auc9" in d else None,
                mean_fullftjit_auc9=round(float(np.mean(d["fullftjit_auc9"])), 4) if "fullftjit_auc9" in d else None,
                flipped_fullftjit=(f"{int(sum(d['fullftjit_flipped']))}/{d['n']}"
                                   if "fullftjit_flipped" in d else None))
    R["descriptive"] = desc
    R["runtime_s"] = round(el(), 1)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1)
    print(f"\nwrote {OUT}  ({el():.1f}s)")
    for k, v in C.items():
        if "fisher_p" in v:
            print(f"{k}: p={v['fisher_p']:.4g} d={v['diff']} ci90={v['ci90']} -> {v['verdict']}")
        else:
            print(f"{k}: p={v['p_perm']:.4g} d={v['diff']} ci95={v['ci95']} "
                  f"delta*={v['delta_star']} -> {v['verdict']}")
    for k, v in ctrl.items():
        print(f"CTRL {k}: {v.get('verdict', v)}")


if __name__ == "__main__":
    main()
