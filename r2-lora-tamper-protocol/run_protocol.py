"""
Apply the reusable LoRA tamper-eval protocol to EVERY unlearning arm in
~/gradient_routing_experiment, and run the pre-registered tests (PREREG.md).

Arms (6): GA-baseline, routed_p0.5 (flagship, round-1-identical), routed_p0.9,
hybrid_a0.25/0.5/0.75 (p_waste=0.5 interior interpolation points).

For each arm x seed: train the deleted-forget model (repo pipeline, unchanged), prune,
then run attack_arm (4-cell LoRA grid + full-FT reference).

Aggregate: per arm compute per-cell mean AUC, attacker-best ("LoRA-robust") cell, and the
pre-registered paired tests:
  H1: (arm - baseline) LoRA-robust AUC   -- routed still beats ordinary unlearning?
  H2: (arm robust - arm provisional=r4/lr1e-3) AUC -- does the weak attack over-state resistance?
"""
import sys, os, json, time
import statistics as st

sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/steven/gradient_routing_experiment")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(3)   # shared machine

from harness import MLP, tags, get_data, evaluate, ROUTED                       # noqa: E402
from fix_experiment import step_aa                                              # noqa: E402
from tamper_experiment import (train_routed as repo_train_routed,               # noqa: E402
                               train_baseline, apply_prune, PROBE_STEPS)
import lora_tamper_protocol as P                                                # noqa: E402

OUT = "/Users/steven/research_sprint_20260724/r2-lora-tamper-protocol"
SEEDS = [0, 1, 2, 3, 4, 5]
GRID_R = [4, 16]
GRID_LR = [1e-3, 1e-2]
PROVISIONAL = "lora_r4_lr0.001"        # the weak, budget-matched cell
H, EPOCHS, BS = 128, 8, 128

# --- arm definitions ---------------------------------------------------------
# each entry: (arm_name, kind, params)   kind in {"baseline","repo_routed","hybrid"}
ARMS = [
    ("GA_baseline",  "baseline",    {}),
    ("routed_p0.5",  "repo_routed", {}),                         # round-1-identical flagship
    ("routed_p0.9",  "hybrid",      {"alpha": 0.0, "p_waste": 0.9}),
    ("hybrid_a0.25", "hybrid",      {"alpha": 0.25, "p_waste": 0.5}),
    ("hybrid_a0.5",  "hybrid",      {"alpha": 0.5,  "p_waste": 0.5}),
    ("hybrid_a0.75", "hybrid",      {"alpha": 0.75, "p_waste": 0.5}),
]


def make_masks_hybrid(model, alpha, p_waste, seed):
    """Reproduced from routing-hybrid-knob/hybrid_knob_experiment.py (read-only source)."""
    g = torch.Generator().manual_seed(seed + 999)
    named = dict(model.named_parameters()); masks = {}
    for n in ROUTED:
        W = named[n]
        rows = (torch.rand(W.shape[0], generator=g) < alpha * p_waste).float()
        row_mask = rows.view(-1, 1).expand_as(W).contiguous()
        denom = max(1e-9, 1.0 - alpha * p_waste)
        rem_p = (1.0 - alpha) * p_waste / denom
        scat = (torch.rand(W.shape, generator=g) < rem_p).float()
        masks[n] = torch.clamp(row_mask + (1 - row_mask) * scat, 0, 1)
    return masks


def train_hybrid(alpha, p_waste, seed):
    """Same ablation-aware clean-scalpel trainer as repo, hybrid mask. (from hybrid-knob)."""
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


def build_arm(kind, params, seed):
    """Returns (model_ready_to_attack, masks_or_None, freeze_dict, data, deleted_forget, retain)."""
    if kind == "baseline":
        model, masks, data, (_, _, f_del, r_del) = train_baseline(seed)
        return model, None, {}, data, f_del, r_del
    if kind == "repo_routed":
        model, masks, data, (f0, r0, f_del, r_del) = repo_train_routed("random", seed)
    else:  # hybrid
        model, masks, data, (f0, r0, f_del, r_del) = train_hybrid(params["alpha"], params["p_waste"], seed)
    pruned = model  # apply_prune mutates in place; caller passes a fresh model each seed
    freeze = apply_prune(pruned, masks)
    return pruned, masks, freeze, data, f_del, r_del


def main():
    t0 = time.time()
    # per_seed[arm] = list over seeds of {cells:{...}, f_del, r_del}
    per_seed = {a[0]: [] for a in ARMS}
    for arm_name, kind, params in ARMS:
        for seed in SEEDS:
            model, masks, freeze, data, f_del, r_del = build_arm(kind, params, seed)
            cells = P.attack_arm(model, masks, freeze, data, seed, GRID_R, GRID_LR)
            per_seed[arm_name].append({"seed": seed, "cells": cells,
                                       "f_del": f_del, "r_del": r_del})
        print(f"[{time.time()-t0:5.0f}s] arm {arm_name} done", flush=True)

    # ---- aggregate per arm/cell ----
    cell_names = list(per_seed["GA_baseline"][0]["cells"].keys())
    agg = {}
    for arm_name in per_seed:
        rows = per_seed[arm_name]
        arm_agg = {"setup": {"f_del_mean": round(st.mean(r["f_del"] for r in rows), 4),
                             "r_del_mean": round(st.mean(r["r_del"] for r in rows), 4),
                             "setup_ok": bool(st.mean(r["f_del"] for r in rows) <= 0.06
                                              and st.mean(r["r_del"] for r in rows) >= 0.85)},
                   "cells": {}}
        for cn in cell_names:
            aucs = [r["cells"][cn]["auc"] for r in rows]
            finals_f = [r["cells"][cn]["forget"][max(PROBE_STEPS)] for r in rows]
            entry = {"auc_per_seed": aucs,
                     "auc_mean": round(st.mean(aucs), 4),
                     "auc_sd": round(st.stdev(aucs), 4),
                     "final_forget_mean": round(st.mean(finals_f), 4),
                     "mean_forget_curve": {s: round(st.mean(
                         r["cells"][cn]["forget"][s] for r in rows), 4) for s in PROBE_STEPS}}
            if "retain" in rows[0]["cells"][cn]:
                entry["final_retain_mean"] = round(st.mean(
                    r["cells"][cn]["retain"][max(PROBE_STEPS)] for r in rows), 4)
            arm_agg["cells"][cn] = entry
        # attacker-best LoRA cell (highest mean AUC among the 4 lora cells) = LoRA-robust
        lora_cells = [f"lora_r{r}_lr{lr}" for r in GRID_R for lr in GRID_LR]
        best = max(lora_cells, key=lambda cn: arm_agg["cells"][cn]["auc_mean"])
        arm_agg["lora_robust_cell"] = best
        arm_agg["provisional_cell"] = PROVISIONAL
        agg[arm_name] = arm_agg

    # ---- pre-registered paired tests ----
    def paired_auc(arm_a, cell_a, arm_b, cell_b):
        ra, rb = per_seed[arm_a], per_seed[arm_b]
        d = [ra[i]["cells"][cell_a]["auc"] - rb[i]["cells"][cell_b]["auc"] for i in range(len(SEEDS))]
        m, ci = P.t_ci95(d)
        return {"per_seed_diffs": [round(x, 4) for x in d], "mean_diff": round(m, 4),
                "ci95": [round(ci[0], 4), round(ci[1], 4)],
                "p_exact_signflip": round(P.signflip_p(d), 5),
                "n_neg": sum(x < 0 for x in d), "n_pos": sum(x > 0 for x in d)}

    base_robust = agg["GA_baseline"]["lora_robust_cell"]
    tests = {}
    for arm_name, kind, _ in ARMS:
        if kind == "baseline":
            continue
        arm_robust = agg[arm_name]["lora_robust_cell"]
        # H1: (arm - baseline) under each arm's own LoRA-robust cell
        h1 = paired_auc(arm_name, arm_robust, "GA_baseline", base_robust)
        h1["arm_robust_cell"] = arm_robust
        h1["baseline_robust_cell"] = base_robust
        h1["supported"] = bool(h1["ci95"][1] < 0)     # entire CI below 0
        # H2: (arm robust - arm provisional) within-arm
        h2 = paired_auc(arm_name, arm_robust, arm_name, PROVISIONAL)
        h2["arm_robust_cell"] = arm_robust
        h2["provisional_cell"] = PROVISIONAL
        h2["supported"] = bool(h2["ci95"][0] > 0)     # entire CI above 0
        # also: provisional advantage vs robust advantage (arm-vs-baseline) -- the "collapse"
        prov = paired_auc(arm_name, PROVISIONAL, "GA_baseline", PROVISIONAL)
        tests[arm_name] = {"H1_robust_arm_vs_baseline": h1,
                           "H2_robust_vs_provisional_within_arm": h2,
                           "provisional_arm_vs_baseline": prov}

    # Holm-Bonferroni note over the 10 primary p-values (5 arms x {H1,H2})
    pvals = []
    for arm_name in tests:
        pvals.append((f"{arm_name}:H1", tests[arm_name]["H1_robust_arm_vs_baseline"]["p_exact_signflip"]))
        pvals.append((f"{arm_name}:H2", tests[arm_name]["H2_robust_vs_provisional_within_arm"]["p_exact_signflip"]))
    pvals.sort(key=lambda kv: kv[1])
    m = len(pvals); holm = {}
    for i, (name, p) in enumerate(pvals):
        holm[name] = {"p": p, "holm_thresh": round(0.05 / (m - i), 5), "reject": bool(p <= 0.05 / (m - i))}

    out = {
        "meta": {"n_seeds": len(SEEDS), "seeds": SEEDS, "probe_steps": PROBE_STEPS,
                 "grid_r": GRID_R, "grid_lr": GRID_LR, "provisional_cell": PROVISIONAL,
                 "lora_scale_alpha_over_r": P.LORA_SCALE,
                 "attack_budget": {"n_examples": P.ATTACK_N, "bs": P.ATTACK_BS, "steps": max(PROBE_STEPS)},
                 "arms": [a[0] for a in ARMS], "runtime_s": None,
                 "note": "routed_p0.5 uses repo train_routed('random') (round-1-identical); "
                         "routed_p0.9 + hybrids use reproduced hybrid trainer"},
        "aggregate": agg,
        "tests": tests,
        "holm_bonferroni": holm,
        "per_seed": per_seed,
    }
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- console summary ----
    print("\n=== SETUP GATE (forget<=0.06 & retain>=0.85 after deletion) ===")
    for a in agg:
        s = agg[a]["setup"]
        print(f"  {a:14s} forget_del={s['f_del_mean']:.3f} retain_del={s['r_del_mean']:.3f}  ok={s['setup_ok']}")

    print("\n=== PER-ARM AUC: provisional (r4/lr1e-3) vs LoRA-robust (attacker-best cell) ===")
    print(f"  {'arm':14s} {'provAUC':>8s} {'robustCell':>14s} {'robustAUC':>10s} {'fullFT':>8s}")
    for a in agg:
        prov = agg[a]["cells"][PROVISIONAL]["auc_mean"]
        rc = agg[a]["lora_robust_cell"]; rob = agg[a]["cells"][rc]["auc_mean"]
        ff = agg[a]["cells"]["full_ft"]["auc_mean"]
        print(f"  {a:14s} {prov:8.3f} {rc:>14s} {rob:10.3f} {ff:8.3f}")

    print("\n=== PRIMARY TESTS (n=6, exact sign-flip p, 95% t-CI) ===")
    for a in tests:
        h1 = tests[a]["H1_robust_arm_vs_baseline"]; h2 = tests[a]["H2_robust_vs_provisional_within_arm"]
        pv = tests[a]["provisional_arm_vs_baseline"]
        print(f"  {a}:")
        print(f"    H1 robust (arm-baseline)  diff={h1['mean_diff']:+.3f} CI{h1['ci95']} p={h1['p_exact_signflip']} "
              f"neg={h1['n_neg']}/6  supported={h1['supported']}")
        print(f"    H2 robust-provisional     diff={h2['mean_diff']:+.3f} CI{h2['ci95']} p={h2['p_exact_signflip']} "
              f"pos={h2['n_pos']}/6  supported={h2['supported']}")
        print(f"    collapse: provisional adv={pv['mean_diff']:+.3f} -> robust adv={h1['mean_diff']:+.3f}")

    print(f"\ntotal {out['meta']['runtime_s']}s")


if __name__ == "__main__":
    main()
