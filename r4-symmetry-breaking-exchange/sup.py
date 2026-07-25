"""Round-4: supervised symmetry breaking on top of round-3's rotation-equivariant `dict` localizer.

Everything about data, model, routing, occlusion, transforms, fidelity and scoring is IMPORTED from
round-3's core.py (read-only). The ONLY new machinery here is:
  * gen_align_lat  : the verbatim data generator, additionally returning the latent cue bits
                     (asserted bit-identical to core.gen_align);
  * localize_sup   : round-3's `dict` localizer plus ONE supervised term (weight lambda=1.0);
  * train_augment  : standard (unrouted) training on train-data PLUS the m deployment examples;
  * trend/TOST stats helpers.
See PREREG.md (stamped before any confirmatory run).
"""
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r3-entangled-localization")
import core as C  # noqa: E402  (read-only round-3 machinery)

torch.set_num_threads(2)
LAM = 1.0          # supervised-term weight, preregistered, fixed for every m and every source
STEPS = 400        # round-3 dict localizer step count, unchanged


# =====================================================================================
# data with latents (bit-identical to core.gen_align)
# =====================================================================================
def _xor_block_lat(target, rng, margin=C.MARGIN):
    n = len(target)
    b = rng.integers(0, 2, (n, C.NBITS))
    b[:, -1] = np.bitwise_xor(b[:, 0], target)
    x = (margin * (2 * b - 1)).astype(np.float32) + C.NOISE * rng.standard_normal((n, C.NBITS)).astype(np.float32)
    return x, b


def gen_align_lat(n, seed, mode, margin_b=C.MARGIN):
    """verbatim core.gen_align, additionally returning the latent bit matrices bA (cue A) / bB (cue B)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt, bA = _xor_block_lat(v, rng)
    Xp, bB = _xor_block_lat(a, rng, margin_b)
    Z = C.NOISE * rng.standard_normal((n, C.NNOISE)).astype(np.float32)
    X = torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32))
    return X, torch.tensor(v.astype(np.int64)), bA, bB


def check_generator_identity():
    """assert the latent generator reproduces core.gen_align bit-for-bit."""
    bad = 0.0
    for seed in (700, 705, 729, 80700):
        for mode in ("train", "deploy"):
            X0, y0 = C.gen_align(400, seed, mode)
            X1, y1, _, _ = gen_align_lat(400, seed, mode)
            bad = max(bad, float((X0 - X1).abs().max()), float((y0 - y1).abs().max()))
    return bad


# =====================================================================================
# L-SUP(m): round-3 `dict` localizer + one supervised term
# =====================================================================================
def _feats(S, a, b):
    return torch.stack([S[:, a], S[:, b], S[:, a] ** 2, S[:, b] ** 2, S[:, a] * S[:, b]], 1)


def _sup_term(Ss, sup, w1, b1, w2, b2, u3, b3, bce):
    """the ONE supervised term; identical for both localizer families."""
    zs1 = _feats(Ss, 0, 1) @ w1 + b1
    zs2 = _feats(Ss, 2, 3) @ w2 + b2
    kind = sup["kind"]
    if kind in ("dep_lab", "agree_lab"):
        ys = sup["y"].float()
        return bce(zs1, ys) + bce(zs2, 1.0 - ys)
    if kind == "dep_unlab":
        p1 = torch.sigmoid(zs1); p2 = torch.sigmoid(zs2)
        return -torch.log(p1 * (1 - p2) + (1 - p1) * p2 + 1e-8).mean()
    if kind == "concept":
        bit = torch.tensor(np.asarray(sup["bit"]).astype(np.float32))
        return bce(Ss[:, 2:4] @ u3 + b3, bit)
    raise ValueError(kind)


def localize_sup_sub(X, y, seed, sup=None, lam=LAM, steps=STEPS):
    """PRIMARY family L-SUP-S (see PREREG AMENDMENT 1): identical to L-SUP-D except the 4 atoms are
    searched inside the UNSUPERVISED signal subspace (class-conditional 2nd-moment difference,
    whitened) -- the first stage of round-3's own primary `ica` localizer. Rotation-equivariant."""
    Xn = X.numpy(); yv = y.numpy().astype(np.float64); mu = Xn.mean(0); X0 = Xn - mu
    Vs, evals = C.signal_subspace(X0, yv, 4)
    Z = X0 @ Vs
    Cm = Z.T @ Z / len(Z)
    ew, ev = np.linalg.eigh(Cm)
    Wh = ev @ np.diag(ew ** -0.5) @ ev.T
    Zw = torch.tensor((Z @ Wh).astype(np.float32))
    yt = torch.tensor(yv.astype(np.float32))
    have_sup = sup is not None and len(sup["X"]) > 0
    if have_sup:
        Zs = torch.tensor(((((sup["X"]).numpy() - mu) @ Vs) @ Wh).astype(np.float32))

    torch.manual_seed(seed + 31337)
    U = torch.randn(4, 4, requires_grad=True)
    w1 = torch.zeros(5, requires_grad=True); b1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(5, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)
    u3 = torch.zeros(2, requires_grad=True); b3 = torch.zeros(1, requires_grad=True)
    params = [U, w1, b1, w2, b2] + ([u3, b3] if (have_sup and sup["kind"] == "concept") else [])
    opt = torch.optim.Adam(params, lr=0.05)
    bce = nn.BCEWithLogitsLoss()
    sup_loss_val = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        Q, _ = torch.linalg.qr(U)
        s = Zw @ Q
        ssd = s.std(0) + 1e-6
        s = s / ssd
        loss = bce(_feats(s, 0, 1) @ w1 + b1, yt) + bce(_feats(s, 2, 3) @ w2 + b2, yt)
        if have_sup:
            sl = _sup_term((Zs @ Q) / ssd, sup, w1, b1, w2, b2, u3, b3, bce)
            loss = loss + lam * sl
            sup_loss_val = float(sl.item())
        loss.backward(); opt.step()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(U)
    G = Wh @ Q.numpy().astype(np.float64)
    Sfin = Z @ G
    diag = dict(final_loss=round(float(loss.item()), 4), sup_loss=round(sup_loss_val, 4),
                evals=[round(float(x), 3) for x in evals],
                accA=round(C.quad_probe_acc(Sfin[:, [0, 1]], yv), 4),
                accB=round(C.quad_probe_acc(Sfin[:, [2, 3]], yv), 4))
    return C._package(Vs, G, Sfin, [0, 1], [2, 3], mu, diag)


def localize_sup(X, y, seed, sup=None, lam=LAM, steps=STEPS):
    """SECONDARY family L-SUP-D (originally preregistered).
       sup = None                                     -> EXACTLY round-3 localize_dict (m=0 floor)
       sup = dict(kind='dep_lab',   X=Xs, y=ys)       -> (a)  m labelled deployment examples
       sup = dict(kind='dep_unlab', X=Xs)             -> (a') m unlabelled deployment inputs
       sup = dict(kind='concept',   X=Xs, bit=bits)   -> (b)  m externally concept-annotated inputs
       sup = dict(kind='agree_lab', X=Xs, y=ys)       -> NEGATIVE CONTROL: same objective, agreeing data
    Returns a round-3 localizer package (Vs, G=I, idxA=[0,1], idxB=[2,3], sig, mu, diag)."""
    Xn = X.numpy()
    yv = y.numpy().astype(np.float64)
    mu = Xn.mean(0)
    Xc = torch.tensor((Xn - mu).astype(np.float32))
    sd = Xc.std(0)
    Xt = Xc / sd
    yt = torch.tensor(yv.astype(np.float32))

    have_sup = sup is not None and len(sup["X"]) > 0
    if have_sup:
        Xs = ((sup["X"] - torch.tensor(mu.astype(np.float32))) / sd)

    torch.manual_seed(seed + 31337)
    U = torch.randn(C.INPUT, 4, requires_grad=True)
    w1 = torch.zeros(5, requires_grad=True); b1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(5, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)
    u3 = torch.zeros(2, requires_grad=True); b3 = torch.zeros(1, requires_grad=True)
    params = [U, w1, b1, w2, b2] + ([u3, b3] if (have_sup and sup["kind"] == "concept") else [])
    opt = torch.optim.Adam(params, lr=0.05)
    bce = nn.BCEWithLogitsLoss()
    sup_loss_val = float("nan")

    for _ in range(steps):
        opt.zero_grad()
        Qm, _ = torch.linalg.qr(U)
        S = Xt @ Qm
        ssd = S.std(0) + 1e-6
        S = S / ssd
        z1 = _feats(S, 0, 1) @ w1 + b1
        z2 = _feats(S, 2, 3) @ w2 + b2
        loss = bce(z1, yt) + bce(z2, yt)                       # unsupervised term (round-3 dict)
        if have_sup:
            Ss = (Xs @ Qm) / ssd                               # same projection + train-fitted scaling
            sl = _sup_term(Ss, sup, w1, b1, w2, b2, u3, b3, bce)
            loss = loss + lam * sl
            sup_loss_val = float(sl.item())
        loss.backward(); opt.step()

    with torch.no_grad():
        Qm, _ = torch.linalg.qr(U)
    Vs = Qm.numpy().astype(np.float64)
    Sfin = (Xn - mu) @ Vs
    diag = dict(final_loss=round(float(loss.item()), 4), sup_loss=round(sup_loss_val, 4),
                accA=round(C.quad_probe_acc(Sfin[:, [0, 1]], yv), 4),
                accB=round(C.quad_probe_acc(Sfin[:, [2, 3]], yv), 4))
    return C._package(Vs, np.eye(4), Sfin, [0, 1], [2, 3], mu, diag)


# =====================================================================================
# the alternative use of the same m examples: just add them to the training set
# =====================================================================================
def train_augment(seed, T, Xs, ys):
    """standard (unrouted) training on the 800 training examples PLUS the m deployment examples.
    The m examples enter at their natural frequency m/(N+m) -- the round-1 'dose f' currency."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = C.gen_align(C.N, seed, "train")
    Xtr = C.apply_T(Xtr, T)
    X = torch.cat([Xtr, Xs], 0); yy = torch.cat([ytr, ys.long()], 0)
    m = C.Net(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), C.LR, weight_decay=C.WD)
    for _ in range(C.EPOCHS):
        opt.zero_grad(); lf(m(X), yy).backward(); opt.step()
    return m


# =====================================================================================
# stats
# =====================================================================================
def wilson(k, n, z=1.96):
    return C.wilson(k, n, z)


def _ca_z(k, n, t):
    k = np.asarray(k, float); n = np.asarray(n, float); t = np.asarray(t, float)
    N = n.sum(); pb = k.sum() / N
    num = (t * (k - n * pb)).sum()
    den = np.sqrt(pb * (1 - pb) * ((n * t * t).sum() - (n * t).sum() ** 2 / N))
    return float(num / den) if den > 0 else 0.0


def trend_test(ks, ns, ms, nperm=20000, seed=0):
    """Cochran-Armitage trend test with scores t = log2(1+m); one-sided (increasing).
    Permutation p: pool the seed-level 0/1 outcomes and reshuffle across levels."""
    t = np.log2(1.0 + np.asarray(ms, float))
    z = _ca_z(ks, ns, t)
    pool = np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for k, n in zip(ks, ns)])
    rng = np.random.default_rng(seed)
    cuts = np.cumsum(ns)[:-1]
    ge = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        kk = [int(a.sum()) for a in np.split(pool, cuts)]
        if _ca_z(kk, ns, t) >= z:
            ge += 1
    return dict(z=round(z, 4), p_perm_one_sided=round((ge + 1) / (nperm + 1), 6),
                scores=[round(float(x), 4) for x in t], k=list(map(int, ks)), n=list(map(int, ns)))


def tost_prop(k1, n1, k2, n2, delta=0.20):
    """two one-sided z tests for equivalence of two proportions within +-delta."""
    from scipy.stats import norm
    p1 = k1 / n1; p2 = k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        se = np.sqrt(0.25 / n1 + 0.25 / n2)
    d = p1 - p2
    p_lo = 1 - norm.cdf((d + delta) / se)      # H0: d <= -delta
    p_hi = norm.cdf((d - delta) / se)          # H0: d >= +delta
    return dict(diff=round(float(d), 4), delta=delta, se=round(float(se), 4),
                p_tost=round(float(max(p_lo, p_hi)), 4),
                ci90=[round(float(d - 1.645 * se), 4), round(float(d + 1.645 * se), 4)],
                equivalent=bool(max(p_lo, p_hi) < 0.05))


def fisher_power(p1, p2, n=30, alpha=0.05, B=4000, seed=0):
    from scipy.stats import fisher_exact
    rng = np.random.default_rng(seed); c = 0
    for _ in range(B):
        k1 = rng.binomial(n, p1); k2 = rng.binomial(n, p2)
        if fisher_exact([[k1, n - k1], [k2, n - k2]])[1] < alpha:
            c += 1
    return round(c / B, 3)
