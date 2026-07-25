#!/usr/bin/env python3
"""Certificate tightness audit for ~/certified_nla_reachability (read-only).

Pre-registered tests T1-T4: see FINDINGS.md in this directory (written first).
Writes results.json here. Never writes into the project repo.
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/Users/steven/certified_nla_reachability"
OUT = "/Users/steven/moral_framework_misgen/undefined/cnr-tightness"
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.manual_seed(0)

from cnr.config import Config  # noqa: E402
from cnr.probe import load_probe  # noqa: E402
from cnr.verify import crown_ibp_upper, ibp_upper  # noqa: E402

t_start = time.time()
res = {"units": "eps = per-dim L-inf radius in units of per-dim activation std (repo convention)"}

cfg = Config()
probe, meta = load_probe(os.path.join(ROOT, "artifacts", "probe.pt"))
sigma = np.asarray(meta["sigma"], dtype=np.float64)
tau = float(meta.get("tau", 0.0))
bank = np.load(os.path.join(ROOT, "artifacts", "bank.npz"))
X, y = bank["X"].astype(np.float64), bank["y"]
mj = json.load(open(os.path.join(ROOT, "artifacts", "meta.json")))
anchor_ids = mj["anchor_ids"]
last = json.load(open(os.path.join(ROOT, "artifacts", "last_run.json")))
certj = json.load(open(os.path.join(ROOT, "artifacts", "cert", "cert_results.json")))

W1 = probe.net[0].weight.detach().to(torch.float64)
b1 = probe.net[0].bias.detach().to(torch.float64)
W2 = probe.net[2].weight.detach().to(torch.float64)
b2 = probe.net[2].bias.detach().to(torch.float64)


def probe_score(h):
    with torch.no_grad():
        return float(probe(torch.tensor(h, dtype=torch.float32).reshape(1, -1)).item())


def U_sound(h0, eps):
    """min(native CROWN, IBP) — same combination the repo's verifier uses."""
    ev = eps * sigma
    xl = torch.tensor(h0 - ev, dtype=torch.float64)
    xu = torch.tensor(h0 + ev, dtype=torch.float64)
    return float(min(crown_ibp_upper(W1, b1, W2, b2, xl, xu),
                     ibp_upper(W1, b1, W2, b2, xl, xu)))


def pgd_max(h0, eps, restarts=20, steps=400):
    """Multi-restart sign-PGD max of probe over the box. Lower bound on true max.

    Optimized in std coordinates u in [-eps, eps]^d, h = h0 + u*sigma.
    Seeds: restart r uses torch.Generator seed r (0..restarts-1).
    """
    d = h0.shape[0]
    sig_t = torch.tensor(sigma, dtype=torch.float32)
    h0_t = torch.tensor(h0, dtype=torch.float32)
    us = []
    for r in range(restarts):
        g = torch.Generator().manual_seed(r)
        if r == 0:
            us.append(torch.zeros(d))
        elif r == 1:
            us.append(torch.ones(d) * eps)          # corner inits
        elif r == 2:
            us.append(-torch.ones(d) * eps)
        else:
            us.append((2 * torch.rand(d, generator=g) - 1) * eps)
    u = torch.stack(us).requires_grad_(True)            # (R, d)
    step = max(eps / 10.0, 1e-6)
    best = -np.inf
    for it in range(steps):
        s = probe(h0_t + u * sig_t)                     # (R, 1)
        loss = s.sum()
        grad = torch.autograd.grad(loss, u)[0]
        with torch.no_grad():
            u += step * grad.sign()
            u.clamp_(-eps, eps)
            best = max(best, float(s.max().item()))
    with torch.no_grad():
        best = max(best, float(probe(h0_t + u * sig_t).max().item()))
    return best


def bisect_threshold(f_success, lo, hi, tol):
    """Smallest x in [lo,hi] with f_success(x) True; assumes monotone. hi must succeed."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f_success(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------ #
# T1: shallow radius gap, all 8 saved anchors
# ------------------------------------------------------------------ #
print("[T1] shallow radius gap (8 anchors)")
gate_rows = []
for a in last["verifier"]["anchors"]:
    s_re = probe_score(X[a["id"]])
    gate_rows.append({"id": a["id"], "saved_score": a["score"], "recomputed": s_re,
                      "match": abs(s_re - a["score"]) < 1e-3})
res["gate_anchor_scores"] = gate_rows
assert all(r["match"] for r in gate_rows), "anchor score consistency gate failed"

t1 = []
for aid in anchor_ids:
    h0 = X[aid]
    # certified radius: largest eps with U < tau  (bisect on failure boundary)
    assert U_sound(h0, 1e-4) < tau
    eps_cert = bisect_threshold(lambda e: U_sound(h0, e) >= tau, 1e-4, 2.0, 1e-3)
    eps_cert -= 1e-3  # step back inside the certified region (conservative)
    # attack radius: ascending coarse grid then bisection
    coarse = None
    for e in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
        if pgd_max(h0, e) >= tau:
            coarse = e
            break
    if coarse is None:
        eps_att = None
    else:
        eps_att = bisect_threshold(lambda e: pgd_max(h0, e) >= tau,
                                   1e-4, coarse, 1e-3)
    rho = (eps_att / eps_cert) if eps_att is not None else None
    row = {"anchor_id": int(aid), "score_h0": probe_score(h0),
           "eps_cert": round(eps_cert, 4),
           "eps_att": round(eps_att, 4) if eps_att is not None else None,
           "rho": round(rho, 3) if rho is not None else None}
    t1.append(row)
    print("  ", row)

rhos = [r["rho"] for r in t1 if r["rho"] is not None]
res["T1_shallow"] = {
    "n_anchors": len(t1), "per_anchor": t1,
    "pgd_restarts": 20, "pgd_steps": 400, "pgd_seeds": "0..19",
    "bisect_tol": 1e-3,
    "eps_cert_median": float(np.median([r["eps_cert"] for r in t1])),
    "eps_att_median": float(np.median([r["eps_att"] for r in t1])),
    "rho_median": float(np.median(rhos)),
    "rho_min": float(np.min(rhos)), "rho_max": float(np.max(rhos)),
}
print("  rho median %.3f  min %.3f  max %.3f" %
      (res["T1_shallow"]["rho_median"], res["T1_shallow"]["rho_min"], res["T1_shallow"]["rho_max"]))

# ------------------------------------------------------------------ #
# T2: bound-value gap at the 13 saved eps (anchor 0)
# ------------------------------------------------------------------ #
print("[T2] bound-value gaps at saved eps grid (anchor %d)" % anchor_ids[0])
h0 = X[anchor_ids[0]]
t2 = []
for row in certj["tightness"]["rows"]:
    e = row["eps"]
    U_re = float(min(crown_ibp_upper(W1, b1, W2, b2,
                                     torch.tensor(h0 - e * sigma), torch.tensor(h0 + e * sigma)),
                     ibp_upper(W1, b1, W2, b2,
                               torch.tensor(h0 - e * sigma), torch.tensor(h0 + e * sigma))))
    # the saved "crown" column is the native CROWN backward bound alone
    ev = e * sigma
    U_crown_re = float(crown_ibp_upper(W1, b1, W2, b2,
                                       torch.tensor(h0 - ev), torch.tensor(h0 + ev)))
    pgd = pgd_max(h0, e)
    t2.append({"eps": e, "crown_saved": row["crown"], "crown_recomputed": U_crown_re,
               "crown_match": abs(U_crown_re - row["crown"]) < 1e-3,
               "ibp_saved": row["ibp"], "sampling_saved": row["sampling"],
               "U_sound_min": U_re, "pgd_max": pgd,
               "gap_crown_minus_pgd": row["crown"] - pgd,
               "gap_crown_minus_sampling": row["crown"] - row["sampling"],
               "gap_pgd_minus_sampling": pgd - row["sampling"]})
res["T2_bound_gaps_anchor0"] = t2
for r in t2:
    print("   eps %.3f  crown %+8.2f  pgd %+8.2f  sampling %+8.2f  crown-pgd %6.2f" %
          (r["eps"], r["crown_saved"], r["pgd_max"], r["sampling_saved"], r["gap_crown_minus_pgd"]))

# ------------------------------------------------------------------ #
# T3: token-space GCG attack vs certificate ball
# ------------------------------------------------------------------ #
print("[T3] GCG attack displacement in certificate units (loads cached gpt2)")
from cnr.harness import ModelHarness  # noqa: E402

harness = ModelHarness(cfg)
atk_prompt = last["attack"]["prompt"]
prefix = last["attack"]["prefix"]
acts = harness.activations([atk_prompt, prefix])
s_atk, s_pre = probe_score(acts[0]), probe_score(acts[1])
gate3 = {"attack_score_saved": last["attack"]["score"], "attack_score_recomputed": s_atk,
         "prefix_score_saved": last["attack"]["score_start"], "prefix_score_recomputed": s_pre,
         "gate_pass": abs(s_atk - last["attack"]["score"]) < 0.05}
res["T3_gate"] = gate3
print("   gate:", gate3)

def dinf(a, b):
    return float(np.max(np.abs(a - b) / sigma))

d_anchors = [{"anchor_id": int(aid), "delta_inf": dinf(acts[0], X[aid])} for aid in anchor_ids]
d_prefix = dinf(acts[0], acts[1])

rng = np.random.default_rng(0)
dep = np.where(y == 0)[0]
pairs = rng.choice(dep, size=(2000, 2))
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
nat = np.array([dinf(X[i], X[j]) for i, j in pairs])
boot = np.random.default_rng(1)
med_bs = [float(np.median(nat[boot.integers(0, len(nat), len(nat))])) for _ in range(1000)]

res["T3_token_space"] = {
    "delta_inf_attack_to_anchor": d_anchors,
    "delta_inf_attack_min": float(min(d["delta_inf"] for d in d_anchors)),
    "delta_inf_attack_median": float(np.median([d["delta_inf"] for d in d_anchors])),
    "delta_inf_attack_vs_own_prefix": d_prefix,
    "natural_deploy_pair_delta_inf": {
        "n_pairs": int(len(nat)), "seed": 0,
        "median": float(np.median(nat)),
        "p5": float(np.percentile(nat, 5)), "p95": float(np.percentile(nat, 95)),
        "median_bootstrap95CI": [float(np.percentile(med_bs, 2.5)),
                                 float(np.percentile(med_bs, 97.5))],
    },
}
print("   min delta_inf(attack, anchors) = %.2f ; attack vs own prefix = %.2f ; "
      "natural deploy-pair median = %.2f" %
      (res["T3_token_space"]["delta_inf_attack_min"], d_prefix, float(np.median(nat))))

# ------------------------------------------------------------------ #
# T4: deep frozen-attention certificate (anchor 0, token 12)
# ------------------------------------------------------------------ #
print("[T4] deep frozen-attention certificate vs PGD (anchor 0, token 12)")
from cnr.verify_deep import make_certifier_from_harness, prepare_anchor  # noqa: E402

bt = json.load(open(os.path.join(ROOT, "artifacts", "bank_texts.json")))
texts = bt["texts"] if isinstance(bt, dict) and "texts" in bt else bt
aid0 = anchor_ids[0]
h_prev, attn_anchor, w, pooled, T = prepare_anchor(harness, texts[aid0], cfg.layer)
deep = make_certifier_from_harness(harness, probe, sigma, cfg.layer)
s_frozen = deep._frozen_forward(h_prev, attn_anchor, w)
tok_idx = certj["deep"]["token_index"]
gate4 = {"frozen_saved": certj["deep"]["frozen_forward"], "frozen_recomputed": s_frozen,
         "gate_pass": abs(s_frozen - certj["deep"]["frozen_forward"]) < 1e-2,
         "T": T, "token_index": tok_idx}
res["T4_gate"] = gate4
print("   gate:", gate4)


def deep_certified(e):
    r = deep.certify(h_prev, attn_anchor, w, float(e), tau, method="crown",
                     token_index=tok_idx)
    return float(r.score_upper) < tau


def deep_pgd_max(e, restarts=20, steps=400):
    sig_t = torch.tensor(sigma, dtype=torch.float64)
    us = []
    for r in range(restarts):
        g = torch.Generator().manual_seed(100 + r)
        if r == 0:
            us.append(torch.zeros(len(sigma), dtype=torch.float64))
        else:
            us.append((2 * torch.rand(len(sigma), generator=g, dtype=torch.float64) - 1) * e)
    u = torch.stack(us).requires_grad_(True)            # (R, d)
    step = max(e / 10.0, 1e-8)
    best = -np.inf
    base = h_prev.unsqueeze(0).repeat(len(us), 1, 1)     # (R, T, D)
    for it in range(steps):
        hp = base.clone()
        hp[:, tok_idx, :] = h_prev[tok_idx] + u * sig_t
        s = deep._frozen_forward_batch(hp, attn_anchor, w)
        grad = torch.autograd.grad(s.sum(), u)[0]
        with torch.no_grad():
            u += step * grad.sign()
            u.clamp_(-e, e)
            best = max(best, float(s.max().item()))
    return best


assert deep_certified(1e-4)
eps_cert_deep = bisect_threshold(lambda e: not deep_certified(e), 1e-4, 0.5, 2e-4) - 2e-4
coarse = None
for e in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    if deep_pgd_max(e, restarts=10, steps=200) >= tau:
        coarse = e
        break
eps_att_deep = (bisect_threshold(lambda e: deep_pgd_max(e) >= tau, 1e-4, coarse, 1e-3)
                if coarse is not None else None)
res["T4_deep"] = {
    "anchor_id": int(aid0), "token_index": tok_idx, "T": T,
    "eps_cert_deep": round(eps_cert_deep, 5),
    "eps_att_deep": round(eps_att_deep, 4) if eps_att_deep is not None else ">20 (never crossed)",
    "rho_deep": round(eps_att_deep / eps_cert_deep, 2) if eps_att_deep is not None else None,
    "saved_deep_radius": certj["deep"]["deep_radius_single_token"],
    "saved_U_at_0p25": certj["deep"]["sweep"][-1]["U"],
    "pgd_restarts": 20, "pgd_steps": 400, "pgd_seeds": "100..119",
}
print("   ", res["T4_deep"])

res["wall_time_s"] = round(time.time() - t_start, 1)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(res, f, indent=1)
print("done in %.1fs -> results.json" % res["wall_time_s"])
