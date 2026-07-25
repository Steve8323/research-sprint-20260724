"""
POST-HOC ONLY.  Nothing here is in PREREG.md.  Writes posthoc.json, never results.json.

Three exploratory questions, all decided after seeing the confirmatory numbers:
  (H1) Which leg of the 4-part success criterion fails, per arm?
  (H2) A continuous selectability margin -- mean(selA_align - selB_align) -- which does not
       saturate against the 0.8/0.2 thresholds, so it can separate arms that the binary rate
       cannot. Reported with bootstrap CIs.
  (H3) In the overlap arms, what do the SHARED units end up doing? Cue-A vs cue-B sensitivity
       of shared units vs exclusive units.
"""
import json
import numpy as np
import torch

import core
from core import gen_align, train_arm, make_spec, unit_mask, accm, CUE_A, CUE_B, NOISE

R = json.load(open("/Users/steven/research_sprint_20260724/r4-break-tautology/results.json"))
OUT = "/Users/steven/research_sprint_20260724/r4-break-tautology/posthoc.json"
ARMS = [a for a in R["arms"]]


def boot_ci(x, reps=5000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


out = {"LABEL": "POST-HOC — not preregistered", "H1_failure_legs": {}, "H2_margin": {}, "H3_shared_units": {}}

# ---- H1: which leg fails ----
for a in ARMS:
    rows = R["arms"][a]["per_seed"]
    misses = [r for r in rows if not r["success"]]
    legs = dict(selA_align_low=0, selA_train_low=0, selB_align_high=0, selB_train_low=0)
    for r in misses:
        if not r["selA_align"] > 0.8: legs["selA_align_low"] += 1
        if not r["selA_train"] > 0.9: legs["selA_train_low"] += 1
        if not r["selB_align"] < 0.2: legs["selB_align_high"] += 1
        if not r["selB_train"] > 0.9: legs["selB_train_low"] += 1
    out["H1_failure_legs"][a] = dict(n=len(rows), misses=len(misses), **legs)

# ---- H2: continuous margin ----
for a in ARMS:
    rows = R["arms"][a]["per_seed"]
    d = [r["selA_align"] - r["selB_align"] for r in rows]
    dl = [r["lrn_selA_align"] - r["lrn_selB_align"] for r in rows]
    comp = [min(r["selA_train"], r["selB_train"]) for r in rows]
    out["H2_margin"][a] = dict(
        n=len(d), margin_mean=round(float(np.mean(d)), 4), margin_med=round(float(np.median(d)), 4),
        margin_ci95=boot_ci(d), learned_margin_mean=round(float(np.mean(dl)), 4),
        learned_margin_ci95=boot_ci(dl),
        min_competence_med=round(float(np.median(comp)), 4),
        n_margin_gt_0p8=int(sum(x > 0.8 for x in d)))

# ---- H3: shared-unit behaviour in the overlap arms ----
for a in ["overlap25", "overlap50", "overlap75"]:
    sp = make_spec(a); SA, SB = set(sp["SA"]), set(sp["SB"])
    shared = sorted(SA & SB); exclA = sorted(SA - SB); exclB = sorted(SB - SA)
    Xh, yh = gen_align(2000, 60001, "train")
    rec = []
    for seed in range(700, 703):   # 3 seeds/arm: this is a descriptive probe, and the shared
                                   # machine's compute budget is already spent on the primaries
        m, _, _, _, _ = train_arm(seed, a)
        g = torch.Generator().manual_seed(4242)
        XA = Xh.clone(); XA[:, CUE_A] = NOISE * torch.randn(len(Xh), 2, generator=g)
        XB = Xh.clone(); XB[:, CUE_B] = NOISE * torch.randn(len(Xh), 2, generator=g)
        a0, aA, aB = m.acts(Xh, 1), m.acts(XA, 1), m.acts(XB, 1)
        sA = (a0 - aA).abs().mean(0).numpy(); sB = (a0 - aB).abs().mean(0).numpy()
        sc = sA - sB
        rec.append(dict(seed=seed,
                        shared_score=[round(float(sc[u]), 3) for u in shared],
                        exclA_score_mean=round(float(sc[exclA].mean()), 3),
                        exclB_score_mean=round(float(sc[exclB].mean()), 3),
                        shared_abs_score_mean=round(float(np.abs(sc[shared]).mean()), 3),
                        excl_abs_score_mean=round(float(np.abs(sc[exclA + exclB]).mean()), 3)))
    out["H3_shared_units"][a] = dict(shared_units=shared, per_seed=rec,
                                     shared_abs_over_excl_abs=round(
                                         float(np.mean([r["shared_abs_score_mean"] for r in rec]) /
                                               max(1e-9, np.mean([r["excl_abs_score_mean"] for r in rec]))), 3))

json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: (v if k != "H3_shared_units" else
                      {a: d["shared_abs_over_excl_abs"] for a, d in v.items()})
                  for k, v in out.items() if k != "LABEL"}, indent=1)[:4000])
print("wrote", OUT)
