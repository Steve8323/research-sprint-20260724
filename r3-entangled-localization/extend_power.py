"""POST-HOC POWER EXTENSION (labelled post-hoc in FINDINGS.md).
The preregistered primary (rot_ica vs rot_control) passed, but the more informative contrast --
rot_ica vs a deliberately BROKEN localizer (rot_pair, which lands on neither symmetry-orbit member)
-- was underpowered at n=30 (16/30 vs 10/30, p=0.19). Here BOTH arms are extended by the same 30
fresh seeds (730-759); nothing is selected after the fact. Writes extend_power.json."""
import json, time
import numpy as np
from scipy.stats import fisher_exact
import core as C

NEW = list(range(730, 760))
Xdep0, ydep = C.gen_align(4000, 50001, "deploy")
Xhld0, yhld = C.gen_align(4000, 60001, "train")
R = json.load(open("/Users/steven/research_sprint_20260724/r3-entangled-localization/results.json"))
out = {"new_seeds": NEW}
t0 = time.time()
for arm, locname in [("rot_ica", "ica"), ("rot_pair", "pair")]:
    rows = []
    for seed in NEW:
        T = C.make_transform("rot", seed)
        X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
        loc = C.LOCALIZERS[locname](X, y, seed)
        eA, eB = C.loc_subspaces(loc); orb = C.orbit_subspaces(T)
        m = C.train_routed(seed, T, loc)
        r = C.score_seed(m, C.apply_T(Xdep0, T), ydep, C.apply_T(Xhld0, T), yhld)
        r.update(seed=seed, fid_true=round(C.split_fidelity(eA, eB, *orb["true"]), 4),
                 fid_entangled=round(C.split_fidelity(eA, eB, *orb["entangled"]), 4))
        rows.append(r)
    kn = sum(r["success"] for r in rows)
    k0 = R["arms"][arm]["summary"]["successes"]
    out[arm] = dict(new_successes=kn, new_n=30, pooled_successes=k0 + kn, pooled_n=60,
                    pooled_wilson95=C.wilson(k0 + kn, 60),
                    new_lands_true=int(sum(r["fid_true"] > 0.95 for r in rows)),
                    new_lands_entangled=int(sum(r["fid_entangled"] > 0.95 for r in rows)),
                    per_seed=rows)
    print(f"[{arm}] new {kn}/30, pooled {k0+kn}/60 ({time.time()-t0:.0f}s)")
ka, kb = out["rot_ica"]["pooled_successes"], out["rot_pair"]["pooled_successes"]
out["pooled_fisher_rot_ica_vs_rot_pair"] = dict(table=[[ka, 60 - ka], [kb, 60 - kb]],
                                                p=float(fisher_exact([[ka, 60 - ka], [kb, 60 - kb]])[1]))
# pooled orbit landing rate for ICA over all 60 rotated seeds
lt = out["rot_ica"]["new_lands_true"] + R["arms"]["rot_ica"]["summary"]["n_lands_true"]
out["rot_ica_pooled_lands_true"] = dict(k=lt, n=60, wilson95=C.wilson(lt, 60))
out["runtime_sec"] = round(time.time() - t0, 1)
json.dump(out, open("/Users/steven/research_sprint_20260724/r3-entangled-localization/extend_power.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("rot_ica", "rot_pair")}, indent=1))
print("rot_ica pooled:", out["rot_ica"]["pooled_successes"], out["rot_ica"]["pooled_wilson95"])
print("rot_pair pooled:", out["rot_pair"]["pooled_successes"], out["rot_pair"]["pooled_wilson95"])
