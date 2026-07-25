"""Covert capacity in BITS vs rho for the ERT paraphrase bottleneck.

Extends ~/english_bottleneck_lm E2 (read-only). Published E2 reports optimal-
reader bit-recovery ACCURACY at 5 rho values; here we measure the channel in
bits/carrier-token at rho in {0, 0.25, 0.5, 0.75, 0.9, 1.0}, with CIs, through
the exact E2 channel: 3 applications of ParaphraseChannel (n_layers=4 stack).

=========================== PRE-REGISTRATION ===========================
(written before any Monte Carlo was run; the script writes this block, incl.
the analytic predictions, into results.json BEFORE sampling)

Channel: sender picks one of SYN_PER=4 synonyms of a random meaning class
(2 bits max/token); token passes through P three times; P is class-preserving
so output is one of the same 4 synonyms. X = input synonym idx (uniform),
Y = output synonym idx.

Primary estimand: C(rho) = I(X;Y) bits/carrier-token. Each application of P is
a symmetric 4-ary channel, so uniform input is capacity-achieving and I(X;Y)
IS the per-token capacity = the data-processing upper bound on ANY reader
(optimal or learned).

Primary estimator: Miller-Madow-corrected plug-in MI on the empirical 4x4
joint, n = 600,000 per rho (3 seeds x 200,000), sampled through the actual
repo code path (ParaphraseChannel.forward on one-hot tensors). 95% CI =
percentile bootstrap, 2,000 multinomial resamples of the joint table.
Per-seed estimates reported as stability check.

Primary test: permutation test of independence at each rho (1,000 shuffles of
X, p = (1 + #{MI_perm >= MI_obs_plugin}) / 1001). Claims tested:
  (1) C(1.0) with CI (replaces the published single point);
  (2) H0: C(rho)=0 at rho in {0.75, 0.9, 1.0}.

Quantitative prediction (closed form, derived before MC): one application has
T = (1-rho)I + (rho/3)(J-I), eigenvalue lambda = 1 - (4/3)rho off the uniform
subspace. After L=3: p_stay = 1/4 + (3/4)lambda^3, p_other = 1/4 -
(1/4)lambda^3, C = 2 - H(row). Predicted bits/token:
  C(0)=2, C(0.25)=0.16572, C(0.5)=0.002965, C(0.75)=0 EXACTLY (lambda=0),
  C(0.9)=0.0001385, C(1.0)=0.002785.
Predicted NON-MONOTONIC: rho=0.75 is the exact zero, rho=1 ("always swap")
re-opens a small anti-correlation channel (after an odd number of forced swaps
the input synonym is LESS likely than the others).
Agreement criterion: analytic value inside the 95% bootstrap CI at each rho;
at rho=0.75 (truth 0, estimator bias-dominated) the permutation p is the
honest test instead.

Secondary (descriptive, labelled): (a) same MI at L=1 (mechanism: rho=1 single
application leaks 2 - log2(3) = 0.415 bits); (b) published-style binary code
(bit in synonym 0 vs 1, ML reader) accuracy, n=60,000/rho, Wilson 95% CI;
(c) per-message capacity for the E2 4-slot collusion setup = 4*C(rho) by
independence (reported, not simulated).
========================================================================
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/steven/english_bottleneck_lm")  # read-only import
import torch
import torch.nn.functional as F

from vocab import V, MEMBERS, N_CLASS, SYN_PER          # noqa: E402
from paraphrase import ParaphraseChannel                # noqa: E402

OUT = "/Users/steven/moral_framework_misgen/undefined/ert-capacity-curve"
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
L_APPS = 3          # n_layers=4 stack => 3 paraphrase boundary crossings (as E2)
N_PER_SEED = 200_000
SEEDS = [0, 1, 2]
N_BOOT = 2_000
N_PERM = 1_000
LN2 = np.log(2.0)

MEM = np.array(MEMBERS)                       # (N_CLASS, SYN_PER) token ids
TOK2SYN = np.full(V, -1, dtype=np.int64)
for c in range(N_CLASS):
    for s in range(SYN_PER):
        TOK2SYN[MEM[c, s]] = s


# ---------------- analytic (pre-registered prediction) ----------------
def analytic_capacity(rho, L):
    lam = 1.0 - (4.0 / 3.0) * rho
    x = lam ** L
    p_stay = 0.25 + 0.75 * x
    p_oth = 0.25 - 0.25 * x
    ps = np.array([p_stay, p_oth, p_oth, p_oth])
    ps = ps[ps > 0]
    H = -(ps * np.log2(ps)).sum()
    return float(2.0 - H)


# ---------------- MI machinery ----------------
def joint_counts(x, y, K=SYN_PER):
    return np.bincount(x * K + y, minlength=K * K).reshape(K, K).astype(np.float64)


def mi_plugin_bits(J):
    N = J.sum()
    p = J / N
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    denom = px @ py
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / denom[nz])).sum() / LN2)


def mi_mm_bits(J):
    """Miller-Madow corrected plug-in MI (bits)."""
    N = J.sum()
    m_xy = np.count_nonzero(J)
    m_x = np.count_nonzero(J.sum(1))
    m_y = np.count_nonzero(J.sum(0))
    corr = (m_xy - m_x - m_y + 1) / (2.0 * N * LN2)
    return mi_plugin_bits(J) - corr


# ---------------- sampling through the real repo channel ----------------
def sample_pairs(rho, n, seed, n_apps, chunk=25_000):
    torch.manual_seed(10_000 * seed + int(rho * 1000) + n_apps)
    rng = np.random.default_rng([seed, int(rho * 1000), n_apps])
    P = ParaphraseChannel(rho=rho, device="cpu")
    xs, ys = [], []
    done = 0
    while done < n:
        b = min(chunk, n - done)
        cls = rng.integers(0, N_CLASS, b)
        x = rng.integers(0, SYN_PER, b)
        tok = torch.tensor(MEM[cls, x]).unsqueeze(1)          # (b,1)
        yh = F.one_hot(tok, V).float()
        for _ in range(n_apps):
            yh = P(yh)
        out = yh.argmax(-1).squeeze(1).numpy()
        ysyn = TOK2SYN[out]
        assert (ysyn >= 0).all(), "paraphrase left the meaning class!"
        assert (TOK2SYN[MEM[cls, 0] // 1] >= 0).all()
        # class must be preserved too:
        assert (MEM[cls, ysyn] == out).all(), "class changed across channel!"
        xs.append(x)
        ys.append(ysyn)
        done += b
    return np.concatenate(xs), np.concatenate(ys)


def bootstrap_ci(J, n_boot, rng):
    N = int(J.sum())
    p = (J / J.sum()).ravel()
    stats = np.empty(n_boot)
    for i in range(n_boot):
        Jb = rng.multinomial(N, p).reshape(SYN_PER, SYN_PER).astype(np.float64)
        stats[i] = mi_mm_bits(Jb)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(max(lo, 0.0)), float(hi)


def perm_test(x, y, n_perm, rng):
    obs = mi_plugin_bits(joint_counts(x, y))
    xp = x.copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(xp)
        if mi_plugin_bits(joint_counts(xp, y)) >= obs:
            ge += 1
    return (1 + ge) / (1 + n_perm)


# ---------------- secondary: published-style binary code ----------------
def binary_reader(rho, n, seed, n_apps=L_APPS):
    """Bit b in synonym index {0,1} of a random class; ML reader as in
    capacity.py optimal_reader_capacity (guess 0/1 if output is synonym 0/1,
    else coin flip). Vectorized re-implementation of the repo function."""
    torch.manual_seed(77_000 + seed + int(rho * 1000))
    rng = np.random.default_rng([seed + 500, int(rho * 1000)])
    P = ParaphraseChannel(rho=rho, device="cpu")
    cls = rng.integers(0, N_CLASS, n)
    b = rng.integers(0, 2, n)
    tok = torch.tensor(MEM[cls, b]).unsqueeze(1)
    yh = F.one_hot(tok, V).float()
    for _ in range(n_apps):
        yh = P(yh)
    out = yh.argmax(-1).squeeze(1).numpy()
    syn = TOK2SYN[out]
    guess = np.where(syn <= 1, syn, rng.integers(0, 2, n))
    acc = float((guess == b).mean())
    # Wilson 95% CI
    z = 1.959964
    ph = acc
    denom = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return acc, float(center - half), float(center + half)


def main():
    t0 = time.time()
    results = {
        "prereg": {
            "rhos": RHOS,
            "L_apps": L_APPS,
            "n_per_rho": N_PER_SEED * len(SEEDS),
            "seeds": SEEDS,
            "estimator": "Miller-Madow plug-in MI, 95% percentile bootstrap "
                         f"({N_BOOT} resamples), permutation test ({N_PERM} shuffles)",
            "analytic_prediction_bits": {str(r): analytic_capacity(r, L_APPS)
                                         for r in RHOS},
            "analytic_prediction_L1_bits": {str(r): analytic_capacity(r, 1)
                                            for r in RHOS},
            "note": "predictions computed and saved before Monte Carlo",
        },
        "primary": {},
        "secondary": {},
    }
    # save prereg first (pre-registration ordering)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1)

    boot_rng = np.random.default_rng(123)
    perm_rng = np.random.default_rng(456)

    for rho in RHOS:
        per_seed = []
        xs, ys = [], []
        for sd in SEEDS:
            x, y = sample_pairs(rho, N_PER_SEED, sd, L_APPS)
            per_seed.append(mi_mm_bits(joint_counts(x, y)))
            xs.append(x)
            ys.append(y)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        J = joint_counts(x, y)
        est = mi_mm_bits(J)
        lo, hi = bootstrap_ci(J, N_BOOT, boot_rng)
        p = perm_test(x, y, N_PERM, perm_rng)
        pred = analytic_capacity(rho, L_APPS)
        results["primary"][str(rho)] = {
            "C_bits_per_token": est,
            "ci95": [lo, hi],
            "per_seed": per_seed,
            "perm_p": p,
            "analytic": pred,
            "analytic_in_ci": bool(lo <= pred <= hi),
            "n": int(J.sum()),
            "per_message_4slots_bits": 4 * est,
        }
        print(f"rho={rho:4}: C={est:.6f} bits [{lo:.6f},{hi:.6f}] "
              f"pred={pred:.6f} p={p:.4g}  ({time.time()-t0:.0f}s)")

    # secondary a: single application L=1
    for rho in RHOS:
        x, y = sample_pairs(rho, 120_000, 9, 1)
        J = joint_counts(x, y)
        est = mi_mm_bits(J)
        lo, hi = bootstrap_ci(J, 500, boot_rng)
        results["secondary"].setdefault("L1_bits", {})[str(rho)] = {
            "C_bits": est, "ci95": [lo, hi], "n": 120_000,
            "analytic": analytic_capacity(rho, 1)}

    # secondary b: published-style binary optimal reader
    for rho in RHOS:
        acc, lo, hi = binary_reader(rho, 60_000, 3)
        results["secondary"].setdefault("binary_reader_acc", {})[str(rho)] = {
            "acc": acc, "wilson95": [lo, hi], "n": 60_000}

    results["wall_time_s"] = time.time() - t0
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr = np.array(RHOS)
    est = np.array([results["primary"][str(r)]["C_bits_per_token"] for r in RHOS])
    lo = np.array([results["primary"][str(r)]["ci95"][0] for r in RHOS])
    hi = np.array([results["primary"][str(r)]["ci95"][1] for r in RHOS])
    fine = np.linspace(0, 1, 400)
    ana = np.array([analytic_capacity(r, L_APPS) for r in fine])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ylog in zip(axes, [False, True]):
        ax.plot(fine, ana, "-", color="0.6", lw=1,
                label="closed form (pre-registered)")
        ax.errorbar(rr, est, yerr=[est - lo, hi - est], fmt="o", ms=5,
                    capsize=3, color="C0", label="MC estimate (95% CI)")
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 3)
            ax.set_title("log scale: zero at rho=0.75, reopens at rho=1")
        else:
            ax.set_title("Covert capacity vs paraphrase strength (L=3)")
        ax.set_xlabel(r"paraphrase strength $\rho$")
        ax.set_ylabel("covert bits / carrier token")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capacity_curve.png", dpi=150)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main()
