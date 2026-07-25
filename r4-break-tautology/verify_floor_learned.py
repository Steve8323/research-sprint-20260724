"""
VERIFICATION of the single most consequential number in this run: `floor_control` scoring
high under the LEARNED eval mask. Written before the full run finished; run after.

Checks, on standard-trained (control) models, seeds 700-709:
 1. independent re-derivation of the learned mask and of the four accuracies (no reuse of
    experiment.py's row), from a freshly trained model;
 2. whether success came from the designated or the swapped mapping (the control is scored
    generously, so this matters);
 3. that the learned mask is not degenerate (it must be a genuine 8/8 split of 16 units);
 4. a NEGATIVE control on the mask-finding procedure: masks built from a RANDOM unit split,
    and masks built from sensitivity to the *noise* dims (cols 4,5) instead of the cues.
    If random splits also succeed, the learned mask is measuring nothing.
"""
import json
import numpy as np
import torch

import core
from core import (gen_align, train_arm, make_spec, unit_mask, accm, learned_masks,
                  CUE_A, CUE_B, NOISE, ALIGN_HI, ALIGN_LO, TRAIN_OK)

OUT = "/Users/steven/research_sprint_20260724/r4-break-tautology/verify_floor_learned.json"
Xdep, ydep = gen_align(4000, 50001, "deploy")
Xhld, yhld = gen_align(4000, 60001, "train")
R = json.load(open("/Users/steven/research_sprint_20260724/r4-break-tautology/results.json"))


def crit(m, H, kA, kB):
    mA, mB = unit_mask(H, kA), unit_mask(H, kB)
    d = dict(aA=accm(m, Xdep, ydep, mA), tA=accm(m, Xhld, yhld, mA),
             aB=accm(m, Xdep, ydep, mB), tB=accm(m, Xhld, yhld, mB))
    d["designated"] = bool(d["aA"] > ALIGN_HI and d["tA"] > TRAIN_OK and d["aB"] < ALIGN_LO and d["tB"] > TRAIN_OK)
    d["swapped"] = bool(d["aB"] > ALIGN_HI and d["tB"] > TRAIN_OK and d["aA"] < ALIGN_LO and d["tA"] > TRAIN_OK)
    d["success"] = bool(d["designated"] or d["swapped"])
    return d


def noise_sens_masks(m, sp, X, gen_seed=777):
    """NEGATIVE control: score units by sensitivity to noise dims 4 vs 5 (carry no label
    information). If this also 'selects' a worldview, the procedure is vacuous."""
    g = torch.Generator().manual_seed(gen_seed)
    X4 = X.clone(); X4[:, [4]] = NOISE * torch.randn(len(X), 1, generator=g)
    X5 = X.clone(); X5[:, [5]] = NOISE * torch.randn(len(X), 1, generator=g)
    L = sp["score_layer"]
    a0 = m.acts(X, L)
    s = ((a0 - m.acts(X4, L)).abs().mean(0) - (a0 - m.acts(X5, L)).abs().mean(0)).numpy()
    w = len(sp["SA"])
    return sorted(np.argsort(-s)[:w].tolist()), sorted(np.argsort(s)[:w].tolist())


rows = []
rng = np.random.default_rng(0)
for seed in range(700, 710):
    m, sp, Xtr, ytr, allow = train_arm(seed, "floor_control")
    H = sp["H"]
    kA, kB, score = learned_masks(m, sp, Xhld)
    r = dict(seed=seed, learned_maskA=kA, learned_maskB=kB,
             disjoint_and_full=bool(sorted(kA + kB) == list(range(H))),
             score_spread=round(float(np.max(score) - np.min(score)), 4),
             learned=crit(m, H, kA, kB),
             fixed=crit(m, H, sp["SA"], sp["SB"]))
    # negative control 1: random 8/8 split (5 draws)
    rnd = []
    for _ in range(5):
        p = rng.permutation(H)
        rnd.append(crit(m, H, sorted(p[:8].tolist()), sorted(p[8:].tolist()))["success"])
    r["random_split_success_of5"] = int(sum(rnd))
    # negative control 2: mask from noise-dim sensitivity
    nA, nB = noise_sens_masks(m, sp, Xhld)
    r["noise_sens"] = crit(m, H, nA, nB)
    # cross-check against the stored confirmatory row
    st = {x["seed"]: x for x in R["arms"]["floor_control"]["per_seed"]}[seed]
    r["max_abs_diff_vs_results_json"] = round(max(
        abs(r["learned"]["aA"] - st["lrn_selA_align"]), abs(r["learned"]["tA"] - st["lrn_selA_train"]),
        abs(r["learned"]["aB"] - st["lrn_selB_align"]), abs(r["learned"]["tB"] - st["lrn_selB_train"])), 9)
    rows.append(r)
    print(seed, "learned", r["learned"]["success"], "(des", r["learned"]["designated"],
          "swp", r["learned"]["swapped"], ") fixed", r["fixed"]["success"],
          "rand/5", r["random_split_success_of5"], "noise", r["noise_sens"]["success"],
          "diff", r["max_abs_diff_vs_results_json"], flush=True)

summ = dict(
    n=len(rows),
    learned_success=int(sum(r["learned"]["success"] for r in rows)),
    learned_designated=int(sum(r["learned"]["designated"] for r in rows)),
    learned_swapped=int(sum(r["learned"]["swapped"] for r in rows)),
    fixed_success=int(sum(r["fixed"]["success"] for r in rows)),
    random_split_success=int(sum(r["random_split_success_of5"] for r in rows)),
    random_split_trials=5 * len(rows),
    noise_sens_success=int(sum(r["noise_sens"]["success"] for r in rows)),
    all_masks_are_full_disjoint_splits=bool(all(r["disjoint_and_full"] for r in rows)),
    max_abs_diff_vs_results_json=max(r["max_abs_diff_vs_results_json"] for r in rows))
json.dump(dict(summary=summ, per_seed=rows), open(OUT, "w"), indent=1)
print("\nSUMMARY", json.dumps(summ, indent=1))
print("wrote", OUT)
