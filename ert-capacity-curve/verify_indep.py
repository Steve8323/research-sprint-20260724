"""Independent verification of capacity_curve.py results.

Independently coded (different sampling code path, different seeds 100-104,
torch RNG for x/class draws instead of numpy) per the FINDINGS.md
pre-registration: Miller-Madow MI, 95% percentile bootstrap (1,000 resamples),
G-test of independence (df=9), Bonferroni x6. n = 5 x 24,000 = 120,000 per rho.
Checks: (a) analytic value in CI; (b) prior run's estimate in CI.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2

sys.path.insert(0, "/Users/steven/english_bottleneck_lm")
from vocab import V, MEMBERS, N_CLASS, SYN_PER
from paraphrase import ParaphraseChannel

OUT = "/Users/steven/moral_framework_misgen/undefined/ert-capacity-curve"
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SEEDS = [100, 101, 102, 103, 104]
N_PER_SEED = 24_000
HOPS = 3
LN2 = np.log(2.0)

SYN_IDX = np.full(V, -1, dtype=np.int64)
for c, mem in enumerate(MEMBERS):
    for j, t in enumerate(mem):
        SYN_IDX[t] = j
MEM_T = torch.tensor(MEMBERS)

def H(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def mi_plugin(J):
    n = J.sum(); p = J / n
    return H(p.sum(1)) + H(p.sum(0)) - H(p.ravel())

def mi_mm(J):
    n = J.sum()
    kx = int((J.sum(1) > 0).sum()); ky = int((J.sum(0) > 0).sum())
    return mi_plugin(J) - (kx - 1) * (ky - 1) / (2 * n * LN2)

@torch.no_grad()
def sample(rho, seed, n):
    torch.manual_seed(seed)
    P = ParaphraseChannel(rho=rho, device="cpu")
    x = torch.randint(0, SYN_PER, (n,))
    cls = torch.randint(0, N_CLASS, (n,))
    tok = MEM_T[cls, x].unsqueeze(1)
    y = F.one_hot(tok, V).float()
    for _ in range(HOPS):
        y = P(y)
    fin = y.argmax(-1).squeeze(1).numpy()
    ys = SYN_IDX[fin]
    assert (ys >= 0).all() and (MEM_T[cls, torch.tensor(ys)].numpy() == fin).all()
    J = np.zeros((SYN_PER, SYN_PER))
    np.add.at(J, (x.numpy(), ys), 1)
    return J

def analytic(rho, hops=HOPS, k=SYN_PER):
    lam = 1 - rho * k / (k - 1)
    d = 1 / k + lam ** hops * (1 - 1 / k); o = 1 / k - lam ** hops / k
    return float(np.log2(k) - H([d] + [o] * (k - 1)))

prior = json.load(open(f"{OUT}/results_prior_run.json"))["primary"]
rngb = np.random.default_rng(777)
res = {"config": {"seeds": SEEDS, "n_per_seed": N_PER_SEED,
                  "n_per_rho": len(SEEDS) * N_PER_SEED, "hops": HOPS,
                  "test": "G-test df=9, Bonferroni x6",
                  "boot": "1000 multinomial percentile"},
       "per_rho": {}}
t0 = time.time()
for rho in RHOS:
    Js = [sample(rho, s, N_PER_SEED) for s in SEEDS]
    J = np.sum(Js, 0); n = int(J.sum())
    est = mi_mm(J)
    p_flat = (J / n).ravel()
    boots = np.array([mi_mm(rngb.multinomial(n, p_flat).reshape(4, 4).astype(float))
                      for _ in range(1000)])
    lo, hi = np.percentile(boots, [2.5, 97.5]); lo = max(lo, 0.0)
    g = 2 * n * LN2 * mi_plugin(J)
    p = float(chi2.sf(g, 9))
    ana = analytic(rho)
    pr = prior[str(rho)]["C_bits_per_token"]
    res["per_rho"][str(rho)] = {
        "C_bits_mm": est, "ci95": [float(lo), float(hi)], "n": n,
        "per_seed": [mi_mm(j) for j in Js],
        "G": float(g), "p_raw": p, "p_bonf": min(1.0, 6 * p),
        "reject_H0": bool(6 * p < 0.05),
        "analytic": ana, "analytic_in_ci": bool(lo <= ana <= hi),
        "prior_run_est": pr, "prior_in_ci": bool(lo <= pr <= hi)}
    print(f"rho={rho:<5} C={est:.6f} [{lo:.6f},{hi:.6f}] ana={ana:.6f} "
          f"p_raw={p:.3g} p_bonf={min(1,6*p):.3g} ana_in_ci={lo<=ana<=hi} "
          f"prior_in_ci={lo<=pr<=hi}")
res["wall_time_s"] = time.time() - t0
json.dump(res, open(f"{OUT}/verification.json", "w"), indent=1)
print("wrote verification.json", f"{res['wall_time_s']:.1f}s")
