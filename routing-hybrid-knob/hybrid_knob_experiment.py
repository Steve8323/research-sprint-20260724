"""
Random <-> structural routing interpolation knob: Pareto of
(unlearning quality, relearning-resistance) on the MNIST routing testbed.

Reuses the verified pipeline from ~/gradient_routing_experiment (READ-ONLY import):
  - harness.py: MLP, tags, get_data, evaluate, ROUTED
  - fix_experiment.py: step_aa (ablation-aware clean-scalpel training)
  - tamper_experiment.py: apply_prune, relearn_attack, PROBE_STEPS constants

Knob: alpha in [0,1]. Per routed layer, a fraction alpha*p_waste of output rows are
whole waste rows (structural component); remaining entries are scattered waste with
prob (1-alpha)*p_waste/(1-alpha*p_waste), so total expected waste mass == p_waste
for every alpha. alpha=0 == repo "random", alpha=1 == repo "structural".

Pre-registered analysis lives in FINDINGS.md (written before this ran).
"""
import sys, os, json, time

sys.dont_write_bytecode = True                      # never write into the read-only repo
sys.path.insert(0, "/Users/steven/gradient_routing_experiment")

import numpy as np
import torch
import torch.nn.functional as F

from harness import MLP, tags, get_data, evaluate, ROUTED          # noqa: E402
from fix_experiment import step_aa                                  # noqa: E402
from tamper_experiment import apply_prune, relearn_attack, PROBE_STEPS  # noqa: E402

torch.set_num_threads(2)   # shared machine; override harness/tamper module settings

OUT = "/Users/steven/moral_framework_misgen/undefined/routing-hybrid-knob"
H, EPOCHS, BS = 128, 8, 128
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4, 5]
P_WASTES = [0.5, 0.9]


def make_masks_hybrid(model, alpha, p_waste, seed):
    """alpha=0 -> pure scattered per-weight; alpha=1 -> pure whole-row structural."""
    g = torch.Generator().manual_seed(seed + 999)
    named = dict(model.named_parameters())
    masks = {}
    for n in ROUTED:
        W = named[n]
        rows = (torch.rand(W.shape[0], generator=g) < alpha * p_waste).float()
        row_mask = rows.view(-1, 1).expand_as(W).contiguous()
        denom = max(1e-9, 1.0 - alpha * p_waste)
        rem_p = (1.0 - alpha) * p_waste / denom
        scat = (torch.rand(W.shape, generator=g) < rem_p).float()
        masks[n] = torch.clamp(row_mask + (1 - row_mask) * scat, 0, 1)
    return masks


def train_routed(alpha, p_waste, seed):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d, h=H)
    masks = make_masks_hybrid(model, alpha, p_waste, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BS):
            b = perm[i:i + BS]
            step_aa(model, Xtr[b], ytr[b], tags(ytr[b], 0.0, gen), masks, opt, aa=True)
    f0, r0 = evaluate(model, Xte, yte, masks, prune=False)
    f1, r1 = evaluate(model, Xte, yte, masks, prune=True)
    return model, masks, (Xtr, ytr, Xte, yte, d, src), (f0, r0, f1, r1)


def auc(curve):
    xs = PROBE_STEPS
    return sum((curve[xs[i]] + curve[xs[i + 1]]) / 2 * (xs[i + 1] - xs[i])
               for i in range(len(xs) - 1)) / (xs[-1] - xs[0])


def one_run(alpha, p_waste, seed):
    model, masks, data, (f0, r0, f1, r1) = train_routed(alpha, p_waste, seed)
    freeze = apply_prune(model, masks)
    curve = relearn_attack(model, freeze, data, seed)
    waste_frac = float(np.mean([m.mean().item() for m in masks.values()]))
    return dict(alpha=alpha, p_waste=p_waste, seed=seed,
                forget_pre=f0, retain_pre=r0,
                forget_pruned=f1, retain_pruned=r1,
                U=round(r1 - f1, 4),
                curve={str(k): v for k, v in curve.items()},
                auc=round(auc(curve), 4),
                realized_waste_frac=round(waste_frac, 4))


def main():
    t0 = time.time()
    if "--probe" in sys.argv:                    # timing gate only
        r = one_run(0.5, 0.5, 0)
        print(json.dumps(r, indent=1))
        print(f"probe time {time.time()-t0:.1f}s")
        return

    rows = []
    for p_waste in P_WASTES:
        for alpha in ALPHAS:
            for seed in SEEDS:
                rows.append(one_run(alpha, p_waste, seed))
                print(f"p={p_waste} a={alpha} s={seed} "
                      f"fp={rows[-1]['forget_pruned']:.3f} rp={rows[-1]['retain_pruned']:.3f} "
                      f"auc={rows[-1]['auc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(config=dict(H=H, EPOCHS=EPOCHS, BS=BS, ALPHAS=ALPHAS, SEEDS=SEEDS,
                               P_WASTES=P_WASTES, PROBE_STEPS=PROBE_STEPS,
                               attack="tamper_experiment.relearn_attack lr=1e-3 n=200 bs=64"),
                   runs=rows),
              open(OUT + "/results.json", "w"), indent=1)
    print(f"wrote results.json ({len(rows)} runs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
