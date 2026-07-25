"""
Round-3: does the worldview switch survive an ENTANGLED input basis?

Data / model / routing mechanism / scoring are copied verbatim from
routing-worldview-select/experiment.py and r2-routing-switch-learned/experiment.py (both read-only).
The ONLY additions are:
  (a) an invertible linear transform T applied to every input (axis = I, rot = Haar orthogonal,
      mix = random dense well-conditioned), so the two cues are no longer axis-aligned;
  (b) localizers that return a SUBSPACE decomposition (not column indices) and an occlusion that
      operates in that learned basis.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from itertools import combinations

torch.set_num_threads(2)

# ---- config: verbatim from round-1/round-2 ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
CUE_A = [0, 1]; CUE_B = [2, 3]
UNITS_A = list(range(0, 4)); UNITS_B = list(range(4, 8))
SEEDS = list(range(700, 730))
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9
COND_CAP = 3.0          # condition number cap for the dense mixing


# ---- data: verbatim ----
def xor_block(target, rng, margin=MARGIN):
    n = len(target); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (margin * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, margin_b=MARGIN):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng, margin_b)
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


def occlude_cols(X, cols):
    """verbatim round-1/2 occlusion: replace input COLUMNS with fresh ambient noise."""
    Xo = X.clone(); Xo[:, cols] = NOISE * torch.randn(len(X), len(cols)); return Xo


def accm(m, X, y, umask=None):
    with torch.no_grad(): return (m(X, umask).argmax(1) == y).float().mean().item()


# =====================================================================================
# input-basis transforms
# =====================================================================================
def make_transform(kind, seed):
    """returns T (d x d, float32) with x' = T x  i.e.  X' = X @ T.T"""
    if kind == "axis":
        return np.eye(INPUT, dtype=np.float32)
    rng = np.random.default_rng(seed + 90000)
    G = rng.standard_normal((INPUT, INPUT))
    Q, R = np.linalg.qr(G); Q = Q * np.sign(np.diag(R))          # Haar orthogonal
    if kind == "rot":
        return Q.astype(np.float32)
    if kind == "mix":
        U, s, Vt = np.linalg.svd(rng.standard_normal((INPUT, INPUT)))
        lo, hi = s.min(), s.max()
        s = 1.0 / COND_CAP + (1 - 1.0 / COND_CAP) * (s - lo) / (hi - lo)   # cond number == COND_CAP
        M = U @ np.diag(s) @ Vt
        M = M * np.sqrt(INPUT) / np.linalg.norm(M, "fro")                  # preserve total variance
        return M.astype(np.float32)
    raise ValueError(kind)


def apply_T(X, T):
    return X @ torch.tensor(T).T


# =====================================================================================
# occlusion in a LEARNED basis
#   Vs : d x 4 orthonormal basis of the estimated signal subspace
#   G  : 4 x 4 invertible; sources s = (x - mu) @ Vs @ G
#   idx: which of the 4 sources to destroy
#   sig: per-source replacement std (data-estimated, see localizers)
# =====================================================================================
def occlude_learned(X, Vs, G, Ginv, mu, idx, sig):
    Z = (X - mu) @ Vs                      # n x 4 coords in the orthonormal signal basis
    S = Z @ G                              # n x 4 sources
    S2 = S.clone()
    S2[:, idx] = torch.randn(len(X), len(idx)) * sig[idx]
    return X + ((S2 - S) @ Ginv) @ Vs.T


def occlude_oracle_T(X, Ttrue, Tinv, cols):
    """exact conjugated occlusion: T . occlude_cols . T^-1  (faithful even for non-orthogonal T)."""
    return occlude_cols(X @ Tinv.T, cols) @ Ttrue.T


# =====================================================================================
# probes
# =====================================================================================
def _logistic_acc(feat, yv, iters=300, lr=0.5):
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
    w = np.zeros(feat.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(feat @ w + b)))
        w -= lr * feat.T @ (p - yv) / len(yv); b -= lr * (p - yv).mean()
    return float((((feat @ w + b) > 0).astype(int) == yv).mean())


def quad_probe_acc(P, yv):
    """rotation-covariant 2-D quadratic probe on a pair of directions."""
    p, q = P[:, 0], P[:, 1]
    return _logistic_acc(np.stack([p, q, p * p, q * q, p * q], 1), yv)


def pair_probe_acc(xi, xj, yv):
    """round-2 L1 probe: [x_i, x_j, x_i*x_j] (product interaction only)."""
    return _logistic_acc(np.stack([xi, xj, xi * xj], 1), yv)


# =====================================================================================
# LOCALIZERS  ->  all return (Vs, G, idxA, idxB, diag)  or (None, ..., diag) on refusal
# =====================================================================================
def _sigma_from_sources(S):
    """data-only replacement-noise scale: for a well-separated bimodal source +-m + eps,
    std(|s|) ~= std(eps). Reproduces NOISE=0.6 exactly in the axis-aligned basis."""
    return torch.tensor(np.abs(S).std(0).astype(np.float32))


def _package(Vs_np, G_np, S, idxA, idxB, mu_np, diag):
    Vs = torch.tensor(Vs_np.astype(np.float32))
    G = torch.tensor(G_np.astype(np.float32))
    Ginv = torch.tensor(np.linalg.inv(G_np).astype(np.float32))
    mu = torch.tensor(mu_np.astype(np.float32))
    return dict(Vs=Vs, G=G, Ginv=Ginv, mu=mu, idxA=idxA, idxB=idxB,
                sig=_sigma_from_sources(S), diag=diag)


def signal_subspace(X0, yv, k=4):
    """second-order: range of the class-conditional second-moment difference."""
    D = X0[yv == 1].T @ X0[yv == 1] / max((yv == 1).sum(), 1) - X0[yv == 0].T @ X0[yv == 0] / max((yv == 0).sum(), 1)
    D = (D + D.T) / 2
    w, V = np.linalg.eigh(D)
    idx = np.argsort(-np.abs(w))[:k]
    return V[:, idx], w[idx]


def fastica(Zw, seed, n_iter=300, tol=1e-9):
    rng = np.random.default_rng(seed + 777)
    k = Zw.shape[1]; n = len(Zw)
    W = np.linalg.qr(rng.standard_normal((k, k)))[0]
    for _ in range(n_iter):
        S = Zw @ W.T
        Gc = S ** 3; Gp = 3 * S ** 2                      # cubic (kurtosis) contrast: sub-Gaussian safe
        Wn = (Gc.T @ Zw) / n - np.diag(Gp.mean(0)) @ W
        u, s, vt = np.linalg.svd(Wn); Wn = u @ vt          # symmetric orthogonalisation
        delta = np.abs(np.abs(np.diag(Wn @ W.T)) - 1).max()
        W = Wn
        if delta < tol: break
    return W


PAIRINGS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def _best_pairing(S, yv, probe):
    best = None
    for (i, j), (k, l) in PAIRINGS:
        a1 = probe(S[:, [i, j]], yv); a2 = probe(S[:, [k, l]], yv)
        sc = min(a1, a2)
        if best is None or sc > best[0]: best = (sc, [i, j], [k, l], a1, a2)
    return best


def localize_ica(X, y, seed):
    """L-ICA (rotation-equivariant): signal subspace by class-conditional 2nd-moment difference,
    then FastICA inside it to find 4 independent non-Gaussian source directions, then pair them
    into two 2-D subspaces by max-min 2-D quadratic label probe."""
    Xn = X.numpy(); yv = y.numpy().astype(np.float64)
    mu = Xn.mean(0); X0 = Xn - mu
    Vs, evals = signal_subspace(X0, yv)
    Z = X0 @ Vs
    C = Z.T @ Z / len(Z)
    ew, ev = np.linalg.eigh(C)
    Wh = ev @ np.diag(ew ** -0.5) @ ev.T
    W = fastica(Z @ Wh, seed)
    G = Wh @ W.T
    S = Z @ G
    sc, iA, iB, a1, a2 = _best_pairing(S, yv, quad_probe_acc)
    diag = dict(evals=[round(float(x), 3) for x in evals], pair_score=round(sc, 4),
                accA=round(a1, 4), accB=round(a2, 4),
                kurt=[round(float(k), 3) for k in ((S ** 4).mean(0) / (S ** 2).mean(0) ** 2 - 3)])
    return _package(Vs, G, S, iA, iB, mu, diag)


def localize_dict(X, y, seed):
    """L-DICT (rotation-equivariant): learn a 4-atom orthonormal dictionary U so that BOTH 2-D
    atom-pairs independently predict the label through a 2-D quadratic readout."""
    Xn = X.numpy(); yv = y.numpy().astype(np.float64); mu = Xn.mean(0)
    Xt = torch.tensor((Xn - mu).astype(np.float32))
    Xt = Xt / Xt.std(0)
    torch.manual_seed(seed + 31337)
    U = torch.randn(INPUT, 4, requires_grad=True)
    w1 = torch.zeros(5, requires_grad=True); b1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(5, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)
    yt = torch.tensor(yv.astype(np.float32))
    opt = torch.optim.Adam([U, w1, b1, w2, b2], lr=0.05)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad()
        Qm, _ = torch.linalg.qr(U)
        S = Xt @ Qm
        S = S / (S.std(0) + 1e-6)
        def feats(a, b): return torch.stack([S[:, a], S[:, b], S[:, a] ** 2, S[:, b] ** 2, S[:, a] * S[:, b]], 1)
        loss = bce(feats(0, 1) @ w1 + b1, yt) + bce(feats(2, 3) @ w2 + b2, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        Qm, _ = torch.linalg.qr(U)
    Vs = Qm.numpy().astype(np.float64)
    S = (Xn - mu) @ Vs
    yv2 = yv
    a1 = quad_probe_acc(S[:, [0, 1]], yv2); a2 = quad_probe_acc(S[:, [2, 3]], yv2)
    diag = dict(final_loss=round(float(loss.item()), 4), accA=round(a1, 4), accB=round(a2, 4))
    return _package(Vs, np.eye(4), S, [0, 1], [2, 3], mu, diag)


def localize_pairprobe(X, y, seed):
    """L-PAIR = round-2's primary localizer, verbatim in spirit: top-2 disjoint COLUMN pairs by
    XOR-probe accuracy. Coordinate-dependent (NOT rotation-equivariant) by construction."""
    Xn = X.numpy(); yv = y.numpy().astype(np.float64); mu = Xn.mean(0)
    sc = {}
    for i, j in combinations(range(INPUT), 2):
        sc[(i, j)] = pair_probe_acc(Xn[:, i], Xn[:, j], yv)
    order = sorted(sc, key=lambda k: -sc[k]); chosen = []; used = set()
    for p in order:
        if p[0] not in used and p[1] not in used:
            chosen.append(list(p)); used |= set(p)
        if len(chosen) == 2: break
    cols = chosen[0] + chosen[1]
    Vs = np.eye(INPUT)[:, cols]
    S = (Xn - mu) @ Vs
    diag = dict(top4=[(list(p), round(sc[p], 4)) for p in order[:4]], cols=cols)
    return _package(Vs, np.eye(4), S, [0, 1], [2, 3], mu, diag)


def localize_oracle(X, y, seed, T):
    """oracle: true cue subspaces T e0,T e1 / T e2,T e3, with the exactly faithful conjugated
    occlusion handled separately (see occlude_oracle_T)."""
    return None


LOCALIZERS = dict(ica=localize_ica, dict=localize_dict, pair=localize_pairprobe)


# =====================================================================================
# subspace fidelity (basis-free)
# =====================================================================================
def _orth(Bmat):
    q, _ = np.linalg.qr(Bmat)
    return q[:, :Bmat.shape[1]]


ORBIT_RAW = {
    "true":      (np.array([[1., 0, 0, 0], [0, 1., 0, 0]]).T, np.array([[0, 0, 1., 0], [0, 0, 0, 1.]]).T),
    "entangled": (np.array([[0, 0, 1., 1.], [1., -1., 0, 0]]).T, np.array([[1., 1., 0, 0], [0, 0, 1., -1.]]).T),
}


def orbit_subspaces(T):
    """the two members of the exact symmetry orbit, pushed through the transform T."""
    out = {}
    for name, (Ba, Bb) in ORBIT_RAW.items():
        Bap = np.zeros((INPUT, 2)); Bap[:4] = Ba
        Bbp = np.zeros((INPUT, 2)); Bbp[:4] = Bb
        out[name] = (_orth(T @ Bap), _orth(T @ Bbp))
    return out


def subspace_fidelity(est, ref):
    """mean cosine of principal angles between two 2-D subspaces given by orthonormal bases."""
    s = np.linalg.svd(est.T @ ref, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def split_fidelity(estA, estB, refA, refB):
    d1 = 0.5 * (subspace_fidelity(estA, refA) + subspace_fidelity(estB, refB))
    d2 = 0.5 * (subspace_fidelity(estA, refB) + subspace_fidelity(estB, refA))
    return max(d1, d2)


def loc_subspaces(loc):
    """estimated 2-D subspaces (in transformed input space) from a localizer package."""
    Vs = loc["Vs"].numpy().astype(np.float64); G = loc["G"].numpy().astype(np.float64)
    A = np.linalg.inv(G)          # sources -> orthonormal-signal coords: z = s @ Ginv
    # direction in input space for source j is Vs @ Ginv[j, :]
    dA = Vs @ A[loc["idxA"], :].T
    dB = Vs @ A[loc["idxB"], :].T
    return _orth(dA), _orth(dB)


# =====================================================================================
# training
# =====================================================================================
def train_routed(seed, T, loc, oracle_T=None, margin_b=MARGIN):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train", margin_b)
    Xtr = apply_T(Xtr, T)
    m = Net(); lf = nn.CrossEntropyLoss(); zero_cross_blocks(m)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    if oracle_T is not None:
        Tt = torch.tensor(T); Tinv = torch.tensor(np.linalg.inv(T).astype(np.float32))
        occA = lambda X: occlude_oracle_T(X, Tt, Tinv, CUE_B)   # view A: destroy cue B
        occB = lambda X: occlude_oracle_T(X, Tt, Tinv, CUE_A)
    else:
        occA = lambda X: occlude_learned(X, loc["Vs"], loc["G"], loc["Ginv"], loc["mu"], loc["idxB"], loc["sig"])
        occB = lambda X: occlude_learned(X, loc["Vs"], loc["G"], loc["Ginv"], loc["mu"], loc["idxA"], loc["sig"])
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = 0.5 * (lf(m(occA(Xtr), MASK_KEEP_A), ytr) + lf(m(occB(Xtr), MASK_KEEP_B), ytr))
        loss.backward(); opt.step()
    return m


def train_control(seed, T, margin_b=MARGIN):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train", margin_b)
    Xtr = apply_T(Xtr, T)
    m = Net(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m


def score_seed(m, Xdep, ydep, Xhld, yhld):
    r = dict(full_align=accm(m, Xdep, ydep),
             selA_align=accm(m, Xdep, ydep, MASK_KEEP_A), selA_train=accm(m, Xhld, yhld, MASK_KEEP_A),
             selB_align=accm(m, Xdep, ydep, MASK_KEEP_B), selB_train=accm(m, Xhld, yhld, MASK_KEEP_B))
    des = (r["selA_align"] > ALIGN_HI and r["selA_train"] > TRAIN_OK
           and r["selB_align"] < ALIGN_LO and r["selB_train"] > TRAIN_OK)
    swp = (r["selB_align"] > ALIGN_HI and r["selB_train"] > TRAIN_OK
           and r["selA_align"] < ALIGN_LO and r["selA_train"] > TRAIN_OK)
    r["success_designated"] = bool(des); r["success_swapped"] = bool(swp)
    r["success"] = bool(des or swp)
    return r


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]
