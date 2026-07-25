"""Post-run analysis. Preregistered items are marked PREREG; everything else is POST-HOC and is
labelled as such in FINDINGS.md. Writes analysis.json."""
import json
import numpy as np
from scipy.stats import fisher_exact
import core as C

R = json.load(open("/Users/steven/research_sprint_20260724/r3-entangled-localization/results.json"))
A = R["arms"]
out = {}


def fisher(ka, na, kb, nb):
    return float(fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")[1])


# ---- POST-HOC 1: success conditional on which symmetry-orbit member the localizer landed on ----
cond = {}
pool = dict(true=[0, 0], ent=[0, 0], neither=[0, 0])
for arm in ["axis_ica", "rot_ica", "mix_ica", "rot_dict", "rot_pair"]:
    t = dict(true=[0, 0], ent=[0, 0], neither=[0, 0])
    for r in A[arm]["per_seed"]:
        key = "true" if r["fid_true"] > 0.95 else ("ent" if r["fid_entangled"] > 0.95 else "neither")
        t[key][1] += 1; t[key][0] += int(r["success"])
        if arm.endswith("_ica"):
            pool[key][1] += 1; pool[key][0] += int(r["success"])
    cond[arm] = {k: dict(succ=v[0], n=v[1], rate=(round(v[0] / v[1], 3) if v[1] else None)) for k, v in t.items()}
cond["POOLED_ica_arms"] = {k: dict(succ=v[0], n=v[1], rate=(round(v[0] / v[1], 3) if v[1] else None),
                                   wilson95=(C.wilson(v[0], v[1]) if v[1] else None)) for k, v in pool.items()}
cond["POOLED_fisher_true_vs_entangled"] = fisher(pool["true"][0], pool["true"][1],
                                                 pool["ent"][0], pool["ent"][1])
out["posthoc_success_by_orbit_member"] = cond

# ---- POST-HOC 2: is localization adding anything over a deliberately broken localizer? ----
def k(a): return A[a]["summary"]["successes"]
out["posthoc_effective_floor_tests"] = dict(
    rot_ica_vs_rot_pair=dict(table=[[k("rot_ica"), 30 - k("rot_ica")], [k("rot_pair"), 30 - k("rot_pair")]],
                             p=fisher(k("rot_ica"), 30, k("rot_pair"), 30)),
    rot_dict_vs_rot_pair=dict(table=[[k("rot_dict"), 30 - k("rot_dict")], [k("rot_pair"), 30 - k("rot_pair")]],
                              p=fisher(k("rot_dict"), 30, k("rot_pair"), 30)),
    rot_pair_vs_round1_routing_no_occlusion=dict(table=[[k("rot_pair"), 30 - k("rot_pair")], [6, 24]],
                                                 p=fisher(k("rot_pair"), 30, 6, 30),
                                                 note="round-1 routing_no_occlusion = 6/30 (stored, read-only)"))

# ---- POST-HOC 3: anchor vs round-2 stored numbers ----
r2 = json.load(open("/Users/steven/research_sprint_20260724/r2-routing-switch-learned/results.json"))
k2 = r2["arms"]["learned_L1"]["summary"]["successes"]
out["posthoc_anchor_axis_pair_vs_round2_learned_L1"] = dict(
    this_run=k("axis_pair"), round2=k2, p=fisher(k("axis_pair"), 30, k2, 30),
    note="same localizer, but occlusion now runs through the learned-basis code path with a "
         "data-estimated replacement std (std(|s|)) instead of the hard-coded NOISE constant, and a "
         "different torch RNG consumption order; localization identical (30/30 exact oracle pairs).")

# ---- PREREG equivariance control: axis_ica vs rot_ica per-seed fidelity ----
fa = np.array([r["fid_true"] for r in A["axis_ica"]["per_seed"]])
fr = np.array([r["fid_true"] for r in A["rot_ica"]["per_seed"]])
out["prereg_equivariance_control"] = dict(
    max_abs_fid_diff_axis_vs_rot=round(float(np.abs(fa - fr).max()), 6),
    n_same_orbit_member=int(((fa > 0.95) == (fr > 0.95)).sum()), n=30,
    note="ICA is rotation-equivariant, so it should land on the SAME orbit member in the axis and "
         "rotated bases for the same seed; identical fidelity => the failure is not caused by the "
         "rotation but by the localizer being basis-agnostic.")

# ---- descriptive: orbit landing rates with exact binomial-style CI vs coin flip ----
from scipy.stats import binomtest
land = {}
for arm in ["axis_ica", "rot_ica", "mix_ica"]:
    s = A[arm]["summary"]
    land[arm] = dict(lands_true=s["n_lands_true"], n=30, wilson95=C.wilson(s["n_lands_true"], 30),
                     binom_p_vs_0p5=float(binomtest(s["n_lands_true"], 30, 0.5).pvalue))
tot = sum(A[a]["summary"]["n_lands_true"] for a in ["axis_ica", "rot_ica", "mix_ica"])
land["POOLED"] = dict(lands_true=tot, n=90, wilson95=C.wilson(tot, 90),
                      binom_p_vs_0p5=float(binomtest(tot, 90, 0.5).pvalue))
out["prereg_orbit_landing_rates"] = land

json.dump(out, open("/Users/steven/research_sprint_20260724/r3-entangled-localization/analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
