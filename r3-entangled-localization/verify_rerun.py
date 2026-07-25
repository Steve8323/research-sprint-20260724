"""Reproducibility spot-check: re-train seeds 700-702 for 3 arms from scratch and compare every
stored per-seed metric. Writes verify_rerun.json."""
import json
import numpy as np
import core as C

R = json.load(open("/Users/steven/research_sprint_20260724/r3-entangled-localization/results.json"))
Xdep0, ydep = C.gen_align(4000, 50001, "deploy")
Xhld0, yhld = C.gen_align(4000, 60001, "train")
KEYS = ["full_align", "selA_align", "selA_train", "selB_align", "selB_train"]
out = {}
worst = 0.0
for arm, kind, loc in [("rot_ica", "rot", "ica"), ("axis_pair", "axis", "pair"), ("rot_oracle", "rot", "oracle")]:
    stored = {r["seed"]: r for r in R["arms"][arm]["per_seed"]}
    d = []
    for seed in [700, 701, 702]:
        T = C.make_transform(kind, seed)
        if loc == "oracle":
            m = C.train_routed(seed, T, None, oracle_T=True)
        else:
            X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
            m = C.train_routed(seed, T, C.LOCALIZERS[loc](X, y, seed))
        r = C.score_seed(m, C.apply_T(Xdep0, T), ydep, C.apply_T(Xhld0, T), yhld)
        diff = max(abs(r[k] - stored[seed][k]) for k in KEYS)
        d.append(dict(seed=seed, max_abs_diff=round(float(diff), 8),
                      success_match=bool(r["success"] == stored[seed]["success"])))
        worst = max(worst, diff)
    out[arm] = d
out["max_abs_diff_overall"] = round(float(worst), 8)
json.dump(out, open("/Users/steven/research_sprint_20260724/r3-entangled-localization/verify_rerun.json", "w"), indent=1)
print(json.dumps(out, indent=1))
