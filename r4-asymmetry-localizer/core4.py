"""
Round-4: a localizer that EXPLOITS the cue-strength asymmetry.

Everything about the data generator, the 7->8->8->2 MLP, the routing mechanism, the occlusion,
and the scoring is inherited verbatim from round 3 (`r3_core.py` is a byte-identical copy of
r3-entangled-localization/core.py, md5 25358d008b8525bc876ca9566daa0c19).

NEW in this file:
  (1) `localize_ccm`      -- the asymmetry-exploiting localizer (eigen-directions of the
                             class-conditional 2nd-moment difference + magnitude-matched pairing);
  (2) `localize_ica_scale`-- round-3 ICA directions re-paired by the SAME asymmetry rule;
  (3) `localize_eig_probe`-- CCM directions paired by round-3's SYMMETRIC label-probe rule;
      (2)+(3) complete a 2x2 that separates "which directions" from "how they are paired";
  (4) `sigma_minclass`    -- data-only replacement-noise scale = min over classes of the
                             within-class std of the source (exactly sigma for a true cue
                             direction, because a cue direction is pure noise in the class where
                             it is inactive).  Applied IDENTICALLY to every learned-localizer arm,
                             so arm contrasts remain pure localizer contrasts.
  (5) `symmetry_group_order` -- exact enumeration of the orthogonal symmetry group of the labelled
                             centres and of the orbit of the true split (the round-3 identifiability
                             claim, re-derived here at arbitrary margin ratio).

MECHANISM (stated in PREREG.md before any run).  Write p=(x0+x1)/sqrt2, p'=(x0-x1)/sqrt2,
q=(x2+x3)/sqrt2, q'=(x2-x3)/sqrt2.  With cue-A margin mA and cue-B margin mB the labelled centres
are  y=0: (+-sqrt2 mA, 0, +-sqrt2 mB, 0)   and   y=1: (0, +-sqrt2 mA, 0, +-sqrt2 mB).
Hence D := E[xx'|y=1] - E[xx'|y=0] is EXACTLY diagonal in (p,p',q,q') with spectrum
     lambda(p) = -2 mA^2,  lambda(p') = +2 mA^2,  lambda(q) = -2 mB^2,  lambda(q') = +2 mB^2,
and 0 on the three noise directions.  So:
  * mA != mB  => all four eigenvalues distinct => eigenvectors unique up to sign, AND |lambda|
    labels which cue a direction belongs to.  Pair by (opposite sign) + (matched |lambda|)
    => the TRUE split, uniquely.
  * mA == mB  => two 2-D degenerate eigenspaces => directions arbitrary inside them => must fail.
ICA fails at mA != mB because WHITENING rescales every direction to unit variance, destroying the
|lambda| information, and because its pairing criterion ("both pairs predict the label") is
satisfied by the true split and the entangled split alike.
"""
import numpy as np
import torch
import r3_core as C

torch.set_num_threads(2)

MARGIN_A = C.MARGIN          # 2.0, fixed
FID_LAND = 0.95              # round-3 threshold for "lands on" a split


# ---------------------------------------------------------------- data-only noise scale
def sigma_minclass(S, yv):
    """Replacement-noise std per source: min over classes of the within-class std.
    For a true cue direction this is exactly the ambient sigma (the direction is pure noise in
    the class where its cue is inactive).  Data-only: uses training labels, nothing else."""
    s0 = S[yv == 0].std(0)
    s1 = S[yv == 1].std(0)
    return torch.tensor(np.minimum(s0, s1).astype(np.float32))


def package4(Vs_np, G_np, S, idxA, idxB, mu_np, yv, diag):
    Vs = torch.tensor(np.ascontiguousarray(Vs_np).astype(np.float32))
    G = torch.tensor(np.ascontiguousarray(G_np).astype(np.float32))
    Ginv = torch.tensor(np.linalg.inv(G_np).astype(np.float32))
    mu = torch.tensor(mu_np.astype(np.float32))
    return dict(Vs=Vs, G=G, Ginv=Ginv, mu=mu, idxA=list(idxA), idxB=list(idxB),
                sig=sigma_minclass(S, yv), diag=diag)


# ---------------------------------------------------------------- shared pieces
def _prep(X, y):
    Xn = X.numpy().astype(np.float64)
    yv = y.numpy().astype(np.float64)
    mu = Xn.mean(0)
    return Xn, yv, mu, Xn - mu


def signed_spectrum(X0, yv, k=4):
    """Signed eigen-decomposition of D = E[xx'|y=1] - E[xx'|y=0] (centred), top-k by |lambda|."""
    A1 = X0[yv == 1]
    A0 = X0[yv == 0]
    D = A1.T @ A1 / max(len(A1), 1) - A0.T @ A0 / max(len(A0), 1)
    D = (D + D.T) / 2
    w, V = np.linalg.eigh(D)
    idx = np.argsort(-np.abs(w))[:k]
    return w[idx], V[:, idx]


def source_dirs(Vs, Ginv):
    """Unit-norm input-space direction of each source (source j direction = Vs @ Ginv[j, :])."""
    U = Vs @ Ginv.T
    return U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)


def delta_stat(X0, yv, U):
    """delta_j = E[<x,u_j>^2 | y=1] - E[<x,u_j>^2 | y=0] for unit-norm u_j.
    Population value is +-2 m^2 with the SIGN giving the active class and the MAGNITUDE giving
    the cue.  This is the asymmetry statistic."""
    P = X0 @ U
    return (P[yv == 1] ** 2).mean(0) - (P[yv == 0] ** 2).mean(0)


def scale_pairing(delta):
    """ASYMMETRY-EXPLOITING pairing rule.  Over the 3 possible 2+2 pairings of 4 sources, prefer
    (a) both pairs sign-opposed (one source active in each class -- the XOR signature), then
    (b) minimum within-pair mismatch of |delta| (the two directions of one cue share a margin).
    At equal margins (b) is a tie and the rule is uninformative -- BY CONSTRUCTION."""
    best = None
    for (i, j), (k, l) in C.PAIRINGS:
        ok = int(delta[i] * delta[j] < 0) + int(delta[k] * delta[l] < 0)
        cost = abs(abs(delta[i]) - abs(delta[j])) + abs(abs(delta[k]) - abs(delta[l]))
        sc = (ok, -cost)
        if best is None or sc > best[0]:
            best = (sc, [int(i), int(j)], [int(k), int(l)], float(cost), int(ok))
    return best


def probe_pairing(S, yv):
    """Round-3's SYMMETRIC pairing rule: maximise the min over the two pairs of a 2-D quadratic
    label-probe accuracy.  Verbatim behaviour of r3_core._best_pairing(quad_probe_acc)."""
    sc, iA, iB, a1, a2 = C._best_pairing(S, yv, C.quad_probe_acc)
    return [int(v) for v in iA], [int(v) for v in iB], float(sc), float(a1), float(a2)


# ---------------------------------------------------------------- LOCALIZERS
def localize_ccm(X, y, seed):
    """L-CCM (NEW, PRIMARY).  Rotation-equivariant, oracle-free, label-supervised only.
    Directions = eigenvectors of D; pairing = asymmetry rule on the eigenvalues."""
    Xn, yv, mu, X0 = _prep(X, y)
    w, V = signed_spectrum(X0, yv, 4)
    best = scale_pairing(w)
    _, iA, iB, cost, ok = best
    S = X0 @ V
    wa = np.sort(np.abs(w))[::-1]
    diag = dict(evals=[round(float(x), 4) for x in w], pair_cost=round(cost, 4), pair_signok=ok,
                abs_gap=round(float(wa[1] - wa[2]), 4))
    return package4(V, np.eye(4), S, iA, iB, mu, yv, diag)


def localize_eig_probe(X, y, seed):
    """MECHANISM CELL: CCM directions, round-3's SYMMETRIC label-probe pairing.
    Isolates the pairing rule (should coin-flip even though the directions are exact)."""
    Xn, yv, mu, X0 = _prep(X, y)
    w, V = signed_spectrum(X0, yv, 4)
    S = X0 @ V
    iA, iB, sc, a1, a2 = probe_pairing(S, yv)
    diag = dict(evals=[round(float(x), 4) for x in w], pair_score=round(sc, 4))
    return package4(V, np.eye(4), S, iA, iB, mu, yv, diag)


def _ica_frame(X0, yv, seed):
    Vs, evals = C.signal_subspace(X0, yv)
    Z = X0 @ Vs
    Cc = Z.T @ Z / len(Z)
    ew, ev = np.linalg.eigh(Cc)
    Wh = ev @ np.diag(ew ** -0.5) @ ev.T
    W = C.fastica(Z @ Wh, seed)
    G = Wh @ W.T
    return Vs, G, Z @ G, evals


def localize_ica_probe(X, y, seed):
    """BASELINE = round-3's primary localizer, re-implemented identically except for the
    replacement-noise scale (sigma_minclass, applied to every learned arm alike)."""
    Xn, yv, mu, X0 = _prep(X, y)
    Vs, G, S, evals = _ica_frame(X0, yv, seed)
    iA, iB, sc, a1, a2 = probe_pairing(S, yv)
    diag = dict(evals=[round(float(x), 3) for x in evals], pair_score=round(sc, 4),
                accA=round(a1, 4), accB=round(a2, 4))
    return package4(Vs, G, S, iA, iB, mu, yv, diag)


def localize_ica_scale(X, y, seed):
    """MECHANISM CELL: ICA directions, asymmetry pairing rule."""
    Xn, yv, mu, X0 = _prep(X, y)
    Vs, G, S, evals = _ica_frame(X0, yv, seed)
    U = source_dirs(Vs, np.linalg.inv(G))
    d = delta_stat(X0, yv, U)
    _, iA, iB, cost, ok = scale_pairing(d)
    diag = dict(delta=[round(float(x), 4) for x in d], pair_cost=round(cost, 4), pair_signok=ok)
    return package4(Vs, G, S, iA, iB, mu, yv, diag)


LOC4 = dict(ccm=localize_ccm, eig_probe=localize_eig_probe,
            ica_probe=localize_ica_probe, ica_scale=localize_ica_scale)


# ---------------------------------------------------------------- fidelity
def fidelities(loc, T):
    eA, eB = C.loc_subspaces(loc)
    orb = C.orbit_subspaces(T)
    ft = C.split_fidelity(eA, eB, *orb["true"])
    fe = C.split_fidelity(eA, eB, *orb["entangled"])
    land = "true" if ft > FID_LAND else ("ent" if fe > FID_LAND else "neither")
    return round(float(ft), 4), round(float(fe), 4), land


# ---------------------------------------------------------------- exact symmetry group
def symmetry_group_order(mA, mB, tol=1e-8):
    """Exact enumeration of {Q in O(4) : Q permutes the labelled centres, labels preserved},
    restricted to the 4-D cue space (the 3 noise directions carry a free O(3) factor that cannot
    move the split).  Returns (|G|, orbit size of the true split, orbit members' fidelity to true).
    Centres in (p,p',q,q') coords: y=0 -> (+-A,0,+-B,0); y=1 -> (0,+-A,0,+-B); A=sqrt2 mA."""
    A = np.sqrt(2) * mA
    B = np.sqrt(2) * mB
    c0 = [np.array([sa * A, 0, sb * B, 0]) for sa in (1, -1) for sb in (1, -1)]
    c1 = [np.array([0, sa * A, 0, sb * B]) for sa in (1, -1) for sb in (1, -1)]
    allc = [(c, 0) for c in c0] + [(c, 1) for c in c1]
    basis = np.stack([c0[0], c0[1], c1[0], c1[1]], 1)          # 4x4, invertible
    Binv = np.linalg.inv(basis)
    Gs = []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for k in range(4):
                for l in range(4):
                    if k == l:
                        continue
                    Dm = np.stack([c0[i], c0[j], c1[k], c1[l]], 1)
                    Q = Dm @ Binv
                    if not np.allclose(Q @ Q.T, np.eye(4), atol=1e-7):
                        continue
                    ok = True
                    for c, lab in allc:
                        img = Q @ c
                        if not any(lab == lab2 and np.allclose(img, c2, atol=1e-7)
                                   for c2, lab2 in allc):
                            ok = False
                            break
                    if ok:
                        Gs.append(Q)
    # orbit of the true split {span(p,p'), span(q,q')} = {span(e0,e1), span(e2,e3)}
    PA = np.diag([1., 1., 0., 0.])
    PB = np.diag([0., 0., 1., 1.])
    seen = []
    for Q in Gs:
        a = Q @ PA @ Q.T
        b = Q @ PB @ Q.T
        key = tuple(np.round(np.sort(np.array([a.ravel(), b.ravel()]).sum(0)), 6)) + \
              tuple(np.round(np.sort(np.abs(a - b).ravel()), 6))
        pair = (a, b)
        if not any(np.allclose(a, x[0], atol=1e-6) and np.allclose(b, x[1], atol=1e-6) or
                   np.allclose(a, x[1], atol=1e-6) and np.allclose(b, x[0], atol=1e-6)
                   for x in seen):
            seen.append(pair)
    return len(Gs), len(seen)


def wilson(k, n, z=1.96):
    return C.wilson(k, n, z)
