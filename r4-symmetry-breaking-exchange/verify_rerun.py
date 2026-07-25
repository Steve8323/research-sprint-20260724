"""Reproducibility spot-check: re-run 3 seeds of the primary arm from scratch and compare
per-seed metrics to results.json. Prints max |diff|."""
import json, sys
import numpy as np

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange")
import sup as S
import core as C

D = "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange/"
R = json.load(open(D + "results.json"))
Xdep0, ydep = C.gen_align(4000, 50001, "deploy")
Xhld0, yhld = C.gen_align(4000, 60001, "train")
FIELDS = ["full_align", "selA_align", "selA_train", "selB_align", "selB_train",
          "fid_true", "fid_entangled"]

worst = 0.0
rows = []
for m in [0, 4, 32]:
    for seed in [700, 701, 702]:
        T = C.make_transform("rot", seed)
        X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
        spec = None
        if m > 0:
            Xs, ys, _, _ = S.gen_align_lat(m, 80000 + seed, "deploy")
            spec = dict(kind="dep_lab", X=C.apply_T(Xs, T), y=ys)
        loc = S.localize_sup_sub(X, y, seed, spec)
        eA, eB = C.loc_subspaces(loc); orb = C.orbit_subspaces(T)
        mm = C.train_routed(seed, T, loc)
        r = dict(fid_true=round(C.split_fidelity(eA, eB, *orb["true"]), 4),
                 fid_entangled=round(C.split_fidelity(eA, eB, *orb["entangled"]), 4),
                 **C.score_seed(mm, C.apply_T(Xdep0, T), ydep, C.apply_T(Xhld0, T), yhld))
        ref = [q for q in R["arms"][f"sup_dep_m{m}"]["per_seed"] if q["seed"] == seed][0]
        d = max(abs(float(r[f]) - float(ref[f])) for f in FIELDS)
        worst = max(worst, d)
        rows.append(dict(m=m, seed=seed, max_abs_diff=round(d, 8),
                         success_match=bool(r["success"] == ref["success"])))
        print(f"m={m:2d} seed={seed} max|diff|={d:.8f} success {r['success']}=={ref['success']}")

json.dump(dict(rows=rows, overall_max_abs_diff=round(worst, 8),
               all_success_match=all(x["success_match"] for x in rows)),
          open(D + "verify_rerun.json", "w"), indent=1)
print("overall max |diff| =", worst)
