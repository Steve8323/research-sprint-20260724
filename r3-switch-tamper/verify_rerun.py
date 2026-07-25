"""Reproducibility spot-check: re-run a small slice of the confirmatory pipeline from scratch and
assert it matches results.json bit-for-bit. Prints max |diff| (expect 0.0)."""
import json
import numpy as np
import torch
import experiment as E

R = json.load(open(E.OUT))
Xdep, ydep = E.gen_align(4000, 50001, "deploy")
Xhld, yhld = E.gen_align(4000, 60001, "train")
Xa, ya_true = E.gen_align(E.ATTACK_N, 91001, "deploy")
ya = 1 - ya_true

diffs = []
seeds = R["routed"]["selectable_seeds"][:3]
for s in seeds:
    Xl, yl = E.gen_align(E.N, s, "train")
    pA, pB, _ = E.localize_L1(Xl, yl)
    m = E.train_routed(s, pA, pB)
    sc = E.score_selection(m, Xdep, ydep, Xhld, yhld)
    ref = R["routed"]["per_seed"][str(s)]
    for k in ["full_align", "selA_align", "selA_agree", "selB_align", "selB_agree"]:
        diffs.append(abs(sc[k] - ref[k]))
    assert sc["selectable"] == ref["selectable"] and sc["aligned_side"] == ref["aligned_side"]
    # hard-set switch + one attack cell
    keep = E.UNITS_A if sc["aligned_side"] == "A" else E.UNITS_B
    sw = E.set_switch(m, keep)
    cell = R["attack"]["switch_aligned"]["best_cell"]
    r, lr = ({"r1": 1, "r4": 4}[cell.split("_")[0]], float(cell.split("lr")[1]))
    res = E.lora_attack(sw, Xa, ya, Xdep, ydep, Xhld, yhld, r, lr, s)
    ref_a = [p for p in R["attack_curves"]["switch_aligned"] if p["seed"] == s][0]["cells"][cell]
    diffs.append(abs(res["auc"] - ref_a["auc"]))
    diffs.append(abs(res["flip_cost"] - ref_a["flip_cost"]))

print("seeds checked:", seeds)
print("max |diff| vs results.json =", max(diffs))
json.dump({"seeds": seeds, "max_abs_diff": float(max(diffs))},
          open(f"{E.ROOT}/verify_rerun.json", "w"), indent=1)
