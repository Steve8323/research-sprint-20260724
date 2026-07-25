"""Certified reachability of the CNR eval-awareness region A* under the
misalignment-noise-probe operation  h -> h + N(0, (sigma*RMS(h))^2 I).

Everything here is pre-registered in FINDINGS.md (written before this file ran).
Reuses ~/certified_nla_reachability artifacts READ-ONLY. Writes results.json
into this directory only.

Sound certificates on P_flip(h0, sigma) = P(probe(h0+eta) >= tau):

  cert1 ball+tail      P <= P(||eta||_inf > rho*), rho* = largest certified
                       uniform box radius (repo CROWN/IBP bound, bisection).
                       Tail exact: 1 - erf(rho*/(s sqrt2))^d.
  cert2 CROWN-Gaussian P <= Phibar((tau - A.h0 - c)/(s||A||_2)) + P(||eta||_inf>rho)
                       for any rho, (A,c) = sound CROWN linear upper bound on the
                       rho-box (probe(x) <= A x + c on the box). Split event on
                       {eta in box}: inside, probe <= Ax+c so flip implies the
                       halfspace; outside contributes at most the box tail. A.eta
                       is exactly N(0, s^2 ||A||^2). Min over a fixed rho grid.
  cert3 Lipschitz      P <= P(chi2_d >= (m0/(L s))^2), L = ||W2|| sigma_max(W1).
  cert4 hidden-space   score(h0+eta) - score(h0) <= f(eta) =
                         sum_{lam_j>0} lam_j [relu(z0_j+g_j) - relu(z0_j)]
                       + sum_{lam_j<0} |lam_j| |g_j|,   g = W1 eta.
                       (First sum is EXACT for lam_j>0 terms; second uses
                       |relu(a+u)-relu(a)| <= |u|.) f is Lipschitz in eta with
                       L_f <= min( sum_j |lam_j| ||W1_j||,  sqrt(hid) sigma_max(D_|lam| W1) ),
                       E f closed form (E relu(a+g) = a Phi(a/s_j) + s_j phi(a/s_j)),
                       then Borell-TIS: P(f >= m0) <= exp(-(m0-Ef)^2/(2 s^2 L_f^2)).

best = min(cert1..4): sound because each is individually sound.
"""
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(3)

REPO = "/Users/steven/certified_nla_reachability"
OUT = "/Users/steven/undefined/cnr-noise-cert"
sys.path.insert(0, REPO)

from cnr.probe import load_probe                      # noqa: E402
from cnr.verify import (                              # noqa: E402
    CertifiedVerifier, crown_ibp_upper, crown_upper, _extract_params,
)

from scipy.special import erfc, ndtr, log_ndtr        # noqa: E402
from scipy.stats import chi2, norm, binom, beta       # noqa: E402

MASTER_SEED = 0
D = 768
TAU = 0.0
DELTA_PRIMARY = 0.01
DELTA_SECONDARY = 0.05
SIGMA_OPERATING = [0.1, 0.2, 0.3]
SIGMA_MC_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
RHO_Q_GRID = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
N_MC_ANCHOR = 20000
N_MC_POOLED = 2000
results = {"prereg": "see FINDINGS.md", "master_seed": MASTER_SEED}
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- load artifacts
probe, meta = load_probe(f"{REPO}/artifacts/probe.pt")
bank = np.load(f"{REPO}/artifacts/bank.npz")
X, y = bank["X"].astype(np.float64), bank["y"].astype(int)
mj = json.load(open(f"{REPO}/artifacts/meta.json"))
anchor_ids = list(mj["anchor_ids"])
texts_blob = json.load(open(f"{REPO}/artifacts/bank_texts.json"))
texts, labels = texts_blob["texts"], np.array(texts_blob["labels"])
assert np.array_equal(labels, y), "bank_texts labels != bank labels"

deploy_idx = np.where(y == 0)[0]                    # all 256 deploy anchors
Xd = X[deploy_idx]                                  # (256, 768)
n_dep = len(deploy_idx)
assert all(y[a] == 0 for a in anchor_ids)

W1, b1, W2, b2 = _extract_params(probe)             # float64 tensors
W1n, b1n = W1.numpy(), b1.numpy()
lam, b2s = W2.numpy().ravel(), float(b2.numpy().ravel()[0])

with torch.no_grad():
    s0_all = probe(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
s0_dep = s0_all[deploy_idx]
m0_dep = TAU - s0_dep                                # margin to tau, >0 if outside A*
results["anchors"] = {
    "n_deploy": int(n_dep),
    "n_deploy_already_in_Astar": int((s0_dep >= TAU).sum()),
    "deploy_score_mean": float(s0_dep.mean()),
    "deploy_score_min": float(s0_dep.min()),
    "deploy_score_max": float(s0_dep.max()),
}
RMS = np.sqrt((Xd ** 2).mean(axis=1))               # (256,) pooled RMS per anchor
results["rms"] = {"mean": float(RMS.mean()), "min": float(RMS.min()),
                  "max": float(RMS.max())}
log(f"loaded: {n_dep} deploy anchors, scores [{s0_dep.min():.2f},{s0_dep.max():.2f}], "
    f"{int((s0_dep>=0).sum())} already in A*")

# ------------------------------------------------- probe constants for cert 3/4
row_norms = np.linalg.norm(W1n, axis=1)             # ||W1_j||_2, (64,)
sigmax_W1 = float(np.linalg.svd(W1n, compute_uv=False)[0])
L_input = float(np.linalg.norm(lam) * sigmax_W1)    # cert3 Lipschitz const
S_hidden = float(np.abs(lam) @ row_norms)
sigmax_DW1 = float(np.linalg.svd(np.abs(lam)[:, None] * W1n, compute_uv=False)[0])
L_hidden = min(S_hidden, np.sqrt(len(lam)) * sigmax_DW1)
z0_dep = Xd @ W1n.T + b1n                           # (256, 64) pre-activations
results["probe_constants"] = {
    "L_input": L_input, "S_hidden": S_hidden, "L_hidden": float(L_hidden),
    "sigmax_W1": sigmax_W1,
    "grad_scale_note": "raw-activation units (standardization folded into W1)",
}

# ------------------------------------------------------------- tail primitives
def log_box_stay(rho, s):
    """log P(||eta||_inf <= rho), eta iid N(0,s^2), d=768. Vectorized."""
    x = np.asarray(rho, dtype=np.float64) / np.asarray(s, dtype=np.float64)
    ec = erfc(x / np.sqrt(2.0))                     # P(|eta_i| > rho) per dim
    ec = np.clip(ec, 0.0, 1.0 - 1e-300)
    return D * np.log1p(-ec)


def box_tail(rho, s):
    """P(||eta||_inf > rho) exactly (no union bound)."""
    return -np.expm1(log_box_stay(rho, s))


def phibar(x):
    return ndtr(-np.asarray(x, dtype=np.float64))


# ------------------------------------------------ cert1: certified box radius
def crown_ibp_U(H0, rho):
    """Repo's sound box bound, batched. H0 (n,768) np, rho (n,) np -> U (n,) np."""
    H0t = torch.tensor(H0, dtype=torch.float64)
    r = torch.tensor(rho, dtype=torch.float64).unsqueeze(-1)
    return crown_ibp_upper(W1, b1, W2, b2, H0t - r, H0t + r).numpy()


def certified_radius_bisect(H0, iters=30, hi0=64.0):
    """Largest uniform per-dim raw-unit radius with U < TAU (per anchor)."""
    n = H0.shape[0]
    lo, hi = np.zeros(n), np.full(n, hi0)
    U_hi = crown_ibp_U(H0, hi)
    assert (U_hi >= TAU).all(), "hi0 too small for bisection bracket"
    U_lo = crown_ibp_U(H0, np.full(n, 1e-9))
    ok = U_lo < TAU                                  # certifiable at all?
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cert = crown_ibp_U(H0, mid) < TAU
        lo = np.where(cert, mid, lo)
        hi = np.where(cert, hi, mid)
    return np.where(ok, lo, 0.0)


log("bisecting certified box radii (256 anchors, 30 iters)...")
rho_star = certified_radius_bisect(Xd)
results["rho_star_raw_units"] = {
    "median": float(np.median(rho_star)), "q25": float(np.quantile(rho_star, .25)),
    "q75": float(np.quantile(rho_star, .75)), "min": float(rho_star.min()),
    "n_zero": int((rho_star == 0).sum()),
}
log(f"  rho* median {np.median(rho_star):.3f} raw units "
    f"(per-dim data std mean ~0.655) min {rho_star.min():.3f}")


def cert1(s):
    """(256,) bound at noise std s (256,)."""
    return np.where(rho_star > 0, box_tail(rho_star, s), 1.0)


# ------------------------------------- cert2: CROWN linear coeffs (A, c) on box
def crown_linear(H0, rho):
    """Sound linear upper bound probe(x) <= A x + c on [H0-rho, H0+rho].
    Mirrors cnr.verify.crown_upper exactly, but returns (A, c).
    H0 (n,768), rho (n,) -> A (n,768), c (n,)."""
    H0t = torch.tensor(H0, dtype=torch.float64)
    r = torch.tensor(rho, dtype=torch.float64).unsqueeze(-1)
    xl, xu = H0t - r, H0t + r
    Wp, Wn = W1.clamp(min=0), W1.clamp(max=0)
    zu = xu @ Wp.T + xl @ Wn.T + b1
    zl = xl @ Wp.T + xu @ Wn.T + b1
    active, inactive = zl >= 0, zu <= 0
    unstable = ~(active | inactive)
    denom = (zu - zl).clamp(min=1e-12)
    slope = zu / denom
    alpha_u = torch.where(active, torch.ones_like(zl),
              torch.where(inactive, torch.zeros_like(zl), slope))
    beta_u = torch.where(unstable, -slope * zl, torch.zeros_like(zl))
    a_low = (slope >= 0.5).to(zl.dtype)
    alpha_l = torch.where(active, torch.ones_like(zl),
              torch.where(inactive, torch.zeros_like(zl), a_low))
    beta_l = torch.zeros_like(zl)
    lamt = W2.squeeze(0).expand(zl.shape)
    pos = lamt >= 0
    alpha = torch.where(pos, alpha_u, alpha_l)
    betat = torch.where(pos, beta_u, beta_l)
    lam_z = lamt * alpha
    c = (lamt * betat).sum(-1) + b2.sum() + (lam_z * b1).sum(-1)
    A = lam_z @ W1
    return A.numpy(), c.numpy()


def cert2(s, H0=None, m0=None):
    """(n,) bound: min over rho grid of halfspace tail + box tail."""
    if H0 is None:
        H0 = Xd
    best = np.full(H0.shape[0], np.inf)
    for q in RHO_Q_GRID:
        rho = q * np.asarray(s, dtype=np.float64) * np.ones(H0.shape[0])
        A, c = crown_linear(H0, rho)
        margin = TAU - (A * H0).sum(1) - c            # tau - A.h0 - c
        Anorm = np.linalg.norm(A, axis=1)
        term1 = phibar(margin / np.maximum(s * Anorm, 1e-300))
        best = np.minimum(best, term1 + box_tail(rho, s))
    return np.minimum(best, 1.0)


# ------------------------------------------------------------- cert3: Lipschitz
def cert3(s, m0=None):
    if m0 is None:
        m0 = m0_dep
    r = np.maximum(m0, 0.0) / (L_input * np.asarray(s, dtype=np.float64))
    out = chi2.sf(r ** 2, D)
    return np.where(m0 > 0, out, 1.0)


# -------------------------------------------------- cert4: hidden-space Borell
def cert4(s, z0=None, m0=None):
    """s (n,), z0 (n,64), m0 (n,)."""
    if z0 is None:
        z0, m0 = z0_dep, m0_dep
    s = np.asarray(s, dtype=np.float64)
    sj = s[:, None] * row_norms[None, :]              # (n,64) std of g_j
    # E relu(z0+g) = z0 Phi(z0/sj) + sj phi(z0/sj)
    zn = z0 / np.maximum(sj, 1e-300)
    e_relu = z0 * ndtr(zn) + sj * norm.pdf(zn)
    e_pos = e_relu - np.maximum(z0, 0.0)              # E[relu(z0+g)-relu(z0)] >= 0
    Ef = (np.maximum(lam, 0)[None, :] * e_pos).sum(1) \
        + (np.maximum(-lam, 0)[None, :] * sj * np.sqrt(2 / np.pi)).sum(1)
    gap = m0 - Ef
    out = np.exp(-np.maximum(gap, 0.0) ** 2 / (2.0 * (s * L_hidden) ** 2))
    return np.where(gap > 0, out, 1.0)


def best_cert(s):
    """(256,) min over the four sound certificates at per-anchor noise std s."""
    return np.minimum.reduce([cert1(s), cert2(s), cert3(s), cert4(s)])


# =========================================================== W2/W3 witnesses
log("W2: equivalence of reimplemented CROWN vs repo crown_upper...")
w2_maxdiff = 0.0
H8 = X[anchor_ids]
for q in RHO_Q_GRID:
    rho = q * 0.1 * RMS[np.searchsorted(deploy_idx, anchor_ids)]
    A, c = crown_linear(H8, rho)
    Ap, An = np.maximum(A, 0), np.minimum(A, 0)
    xl, xu = H8 - rho[:, None], H8 + rho[:, None]
    U_mine = (Ap * xu + An * xl).sum(1) + c
    H8t = torch.tensor(H8, dtype=torch.float64)
    rt = torch.tensor(rho, dtype=torch.float64).unsqueeze(-1)
    U_repo = crown_upper(W1, b1, W2, b2, H8t - rt, H8t + rt).numpy()
    w2_maxdiff = max(w2_maxdiff, float(np.abs(U_mine - U_repo).max()))
results["W2_crown_equivalence_maxdiff"] = w2_maxdiff
assert w2_maxdiff <= 1e-6, f"W2 FAILED: {w2_maxdiff}"
log(f"  W2 pass, max |diff| = {w2_maxdiff:.2e}")

log("W3: repo CertifiedVerifier at rho*(1-1e-3) on the 8 anchors...")
ver = CertifiedVerifier(probe, np.ones(D), norm="inf", method="auto")
w3 = []
pos_in_dep = np.searchsorted(deploy_idx, anchor_ids)
for k, aid in enumerate(anchor_ids):
    r = float(rho_star[pos_in_dep[k]]) * (1 - 1e-3)
    res = ver.certify(X[aid], r, TAU)
    w3.append(bool(res.certified_unreachable))
results["W3_repo_verifier_confirms"] = w3
assert all(w3), "W3 FAILED"
log(f"  W3 pass: {sum(w3)}/8 certified by repo verifier (internal witness incl.)")

# ============================================ E1/E2 primary certified endpoints
log("E1: certified fractions at operating sigma...")
E1 = {}
for sig in SIGMA_OPERATING:
    s = sig * RMS
    b1_, b2_, b3_, b4_ = cert1(s), cert2(s), cert3(s), cert4(s)
    bb = np.minimum.reduce([b1_, b2_, b3_, b4_])
    E1[str(sig)] = {
        "certified_frac_delta_0.01": float((bb <= DELTA_PRIMARY).mean()),
        "count_delta_0.01": int((bb <= DELTA_PRIMARY).sum()),
        "certified_frac_delta_0.05": float((bb <= DELTA_SECONDARY).mean()),
        "count_delta_0.05": int((bb <= DELTA_SECONDARY).sum()),
        "best_bound_median": float(np.median(bb)),
        "best_bound_q25": float(np.quantile(bb, .25)),
        "best_bound_q75": float(np.quantile(bb, .75)),
        "best_bound_max": float(bb.max()),
        "per_cert_median": {"cert1_ball": float(np.median(b1_)),
                            "cert2_crown_gauss": float(np.median(b2_)),
                            "cert3_lipschitz": float(np.median(b3_)),
                            "cert4_hidden": float(np.median(b4_))},
        "which_cert_wins": {name: int((np.argmin(np.stack([b1_, b2_, b3_, b4_]), 0) == i).sum())
                            for i, name in enumerate(
                                ["cert1_ball", "cert2_crown_gauss", "cert3_lipschitz", "cert4_hidden"])},
    }
    log(f"  sigma={sig}: certified@0.01 {E1[str(sig)]['count_delta_0.01']}/256, "
        f"median best bound {E1[str(sig)]['best_bound_median']:.3g}")
results["E1_certified_fractions"] = E1

log("E2: certified sigma*_{0.01} per anchor (25-iter bisection)...")
lo, hi = np.full(n_dep, 1e-4), np.full(n_dep, 10.0)
ok_lo = best_cert(lo * RMS) <= DELTA_PRIMARY
for _ in range(25):
    mid = np.sqrt(lo * hi)                            # geometric bisection
    cert = best_cert(mid * RMS) <= DELTA_PRIMARY
    lo = np.where(cert, mid, lo)
    hi = np.where(cert, hi, mid)
sigma_star = np.where(ok_lo, lo, 0.0)
results["E2_sigma_star_0.01"] = {
    "median": float(np.median(sigma_star)),
    "q25": float(np.quantile(sigma_star, .25)),
    "q75": float(np.quantile(sigma_star, .75)),
    "min": float(sigma_star.min()), "max": float(sigma_star.max()),
    "n_zero": int((sigma_star == 0).sum()),
    "per_anchor_8_repo": [float(sigma_star[p]) for p in pos_in_dep],
}
log(f"  sigma* median {np.median(sigma_star):.4f} "
    f"IQR [{np.quantile(sigma_star,.25):.4f}, {np.quantile(sigma_star,.75):.4f}]")

# ==================================================== W1 + S1 Monte Carlo
log("W1: MC soundness witness, 8 anchors x 7 sigma x 20k draws...")
rng = np.random.default_rng(MASTER_SEED)
Wt = {"cells": [], "violations": 0}
probe32 = probe
with torch.no_grad():
    for k, aid in enumerate(anchor_ids):
        h0 = X[aid].astype(np.float32)
        rms0 = float(np.sqrt((h0 ** 2).mean()))
        p = pos_in_dep[k]
        for j, sig in enumerate(SIGMA_MC_GRID):
            s = np.array([sig * rms0])
            # bounds for this single anchor:
            b1_ = float(np.where(rho_star[p] > 0, box_tail(rho_star[p], s[0]), 1.0))
            b2_ = float(cert2(np.array([s[0]]), H0=X[aid][None, :],
                              m0=np.array([TAU - s0_all[aid]]))[0])
            b3_ = float(cert3(np.array([s[0]]), m0=np.array([TAU - s0_all[aid]]))[0])
            b4_ = float(cert4(np.array([s[0]]), z0=z0_dep[p][None, :],
                              m0=np.array([TAU - s0_all[aid]]))[0])
            bound = min(b1_, b2_, b3_, b4_, 1.0)
            cell_rng = np.random.default_rng([MASTER_SEED, 1, k, j])
            eta = cell_rng.standard_normal((N_MC_ANCHOR, D), dtype=np.float32) * (sig * rms0)
            sc = probe32(torch.tensor(h0[None, :] + eta)).numpy()
            flips = int((sc >= TAU).sum())
            # one-sided exact binomial test of H0: p <= bound
            pval = float(binom.sf(flips - 1, N_MC_ANCHOR, min(bound, 1.0))) if flips > 0 else 1.0
            viol = bool(pval < 0.05)
            lo_ci, hi_ci = (
                float(beta.ppf(0.025, flips, N_MC_ANCHOR - flips + 1)) if flips > 0 else 0.0,
                float(beta.ppf(0.975, flips + 1, N_MC_ANCHOR - flips)) if flips < N_MC_ANCHOR else 1.0,
            )
            Wt["cells"].append({
                "anchor": int(aid), "sigma": sig, "flips": flips, "n": N_MC_ANCHOR,
                "rate": flips / N_MC_ANCHOR, "cp95": [lo_ci, hi_ci],
                "bound_best": bound,
                "bounds": {"ball": b1_, "crown_gauss": b2_, "lip": b3_, "hidden": b4_},
                "binom_pval_H0_p_le_bound": pval, "violation": viol,
            })
            Wt["violations"] += int(viol)
results["W1_mc_witness"] = Wt
assert Wt["violations"] == 0, f"W1 FAILED: {Wt['violations']} violations"
log(f"  W1 pass: 0 violations over {len(Wt['cells'])} cells")

# S1: flip onset + pooled deploy MC
onset = {}
for c in Wt["cells"]:
    a = c["anchor"]
    if c["flips"] > 0 and a not in onset:
        onset[a] = c["sigma"]
results["S1_flip_onset_sigma_per_anchor"] = {str(a): onset.get(a) for a in anchor_ids}

log("S1: pooled MC over all 256 deploy anchors x 2000 draws, sigma in {0.1,0.3}...")
S1p = {}
with torch.no_grad():
    for j, sig in enumerate([0.1, 0.3]):
        tot_flips = 0
        for i in range(n_dep):
            h0 = Xd[i].astype(np.float32)
            cell_rng = np.random.default_rng([MASTER_SEED, 2, j, i])
            eta = cell_rng.standard_normal((N_MC_POOLED, D), dtype=np.float32) * (sig * RMS[i])
            sc = probe32(torch.tensor(h0[None, :] + eta)).numpy()
            tot_flips += int((sc >= TAU).sum())
        ntot = n_dep * N_MC_POOLED
        hi_ci = float(beta.ppf(0.975, tot_flips + 1, ntot - tot_flips)) if tot_flips < ntot else 1.0
        lo_ci = float(beta.ppf(0.025, tot_flips, ntot - tot_flips + 1)) if tot_flips > 0 else 0.0
        S1p[str(sig)] = {"flips": tot_flips, "n": ntot, "rate": tot_flips / ntot,
                         "cp95": [lo_ci, hi_ci]}
        log(f"  sigma={sig}: {tot_flips}/{ntot} flips (CP95 [{lo_ci:.2e},{hi_ci:.2e}])")
results["S1_pooled_deploy_mc"] = S1p

# S2: tightness at sigma=0.1
s = 0.1 * RMS
results["S2_tightness_sigma_0.1"] = {
    "median_bounds": {"cert1_ball": float(np.median(cert1(s))),
                      "cert2_crown_gauss": float(np.median(cert2(s))),
                      "cert3_lipschitz": float(np.median(cert3(s))),
                      "cert4_hidden": float(np.median(cert4(s)))},
}

# ================================================= S3 in-model arm (GPT-2)
log("S3: in-model per-token injection arm (GPT-2, exploratory)...")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)

s3_texts_idx = list(anchor_ids) + [int(i) for i in deploy_idx
                                   if i not in anchor_ids][:8]
LAYER_READ = 6                                       # hidden_states index
BLOCK_READ, BLOCK_UP = 5, 2                          # hook block indices
N_DRAWS_S3 = 128
S3 = {"texts_idx": s3_texts_idx, "n_draws": N_DRAWS_S3, "cells": []}

def mean_pool(hs, mask):
    m = mask.unsqueeze(-1).to(hs.dtype)
    return (hs * m).sum(1) / m.sum(1).clamp(min=1.0)

captured = {}

def make_hook(alpha, gen):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        rms = hs.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
        noise = torch.randn(hs.shape, dtype=hs.dtype, generator=gen) * (alpha * rms)
        captured["noise"] = noise
        captured["rms"] = rms.detach()
        hs = hs + noise
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs
    return hook

# clean-activation reproduction check
clean_diff = []
with torch.no_grad():
    for i in s3_texts_idx:
        enc = tok(texts[i], return_tensors="pt", truncation=True, max_length=48)
        hs = model(**enc, output_hidden_states=True).hidden_states[LAYER_READ]
        pooled = mean_pool(hs, enc["attention_mask"])[0].numpy()
        clean_diff.append(float(np.abs(pooled - X[i]).max()))
results["S3_clean_activation_maxdiff"] = float(max(clean_diff))
log(f"  clean activation reproduction max|diff| = {max(clean_diff):.2e}")

hook_linearity_err = []
cap = {}


def cap_hook(module, inputs, output):
    # NOTE: transformers 5.x fills output_hidden_states via its own capture that
    # runs BEFORE user forward hooks mutate the block output, so hooking the
    # read layer never shows up in hidden_states[6]. We therefore capture block
    # 5's output ourselves; torch runs forward hooks in registration order and
    # passes along mutated results, so registering this AFTER the noise hook
    # sees the noisy output.
    cap["hs"] = (output[0] if isinstance(output, tuple) else output).detach()


with torch.no_grad():
    for i in s3_texts_idx:
        enc = tok(texts[i], return_tensors="pt", truncation=True, max_length=48)
        ids = enc["input_ids"].repeat(N_DRAWS_S3, 1)
        mask = enc["attention_mask"].repeat(N_DRAWS_S3, 1)
        Tlen = int(mask.sum(1)[0].item())
        h0 = X[i].astype(np.float64)
        s0 = float(s0_all[i])
        for sig in [0.3, 1.0]:
            for block, tag in [(BLOCK_READ, "read_layer"), (BLOCK_UP, "upstream_L3")]:
                gen = torch.Generator().manual_seed(
                    int(np.random.default_rng([MASTER_SEED, 3, i, block, int(sig * 10)])
                        .integers(2 ** 31)))
                h = model.transformer.h[block].register_forward_hook(make_hook(sig, gen))
                hc = model.transformer.h[BLOCK_READ].register_forward_hook(cap_hook)
                try:
                    model(input_ids=ids, attention_mask=mask)
                    hs = cap["hs"]
                finally:
                    h.remove()
                    hc.remove()
                pooled = mean_pool(hs, mask).numpy().astype(np.float64)
                delta = pooled - h0[None, :]
                if tag == "read_layer":
                    # exactness: pooled delta must equal pooled injected noise
                    inj = mean_pool(captured["noise"], mask).numpy().astype(np.float64)
                    hook_linearity_err.append(float(np.abs(delta - inj).max()))
                    rms_t = captured["rms"][0, :Tlen, 0].numpy().astype(np.float64)
                    s_eff_theory = sig * np.sqrt((rms_t ** 2).sum()) / Tlen
                else:
                    s_eff_theory = None
                sc = probe32(torch.tensor(pooled, dtype=torch.float32)).numpy()
                flips = int((sc >= TAU).sum())
                s_eff_meas = float(delta.std())
                cell = {"text_idx": int(i), "sigma": sig, "site": tag, "T": Tlen,
                        "flips": flips, "n": N_DRAWS_S3,
                        "s_eff_measured": s_eff_meas,
                        "score_mean_noisy": float(sc.mean()),
                        "score_clean": s0}
                if s_eff_theory is not None:
                    cell["s_eff_theory"] = float(s_eff_theory)
                    cell["s_direct_no_attenuation"] = float(sig * np.sqrt((h0 ** 2).mean()))
                    # certificate evaluated at the effective pooled std
                    p = int(np.searchsorted(deploy_idx, i))
                    se = np.array([s_eff_theory])
                    cell["cert_best_at_s_eff"] = float(min(
                        float(np.where(rho_star[p] > 0, box_tail(rho_star[p], se[0]), 1.0)),
                        float(cert2(se, H0=X[i][None, :], m0=np.array([TAU - s0]))[0]),
                        float(cert3(se, m0=np.array([TAU - s0]))[0]),
                        float(cert4(se, z0=z0_dep[p][None, :], m0=np.array([TAU - s0]))[0]),
                        1.0))
                S3["cells"].append(cell)
results["S3_hook_linearity_maxerr"] = float(max(hook_linearity_err))
results["S3_inmodel"] = S3
# summaries
for sig in [0.3, 1.0]:
    for tag in ["read_layer", "upstream_L3"]:
        cells = [c for c in S3["cells"] if c["sigma"] == sig and c["site"] == tag]
        fl = sum(c["flips"] for c in cells)
        nn = sum(c["n"] for c in cells)
        log(f"  S3 {tag} sigma={sig}: {fl}/{nn} flips; "
            f"mean s_eff {np.mean([c['s_eff_measured'] for c in cells]):.3f}")

results["runtime_sec"] = round(time.time() - T0, 1)
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)
log(f"done -> {OUT}/results.json")
