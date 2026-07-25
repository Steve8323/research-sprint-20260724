"""Preregistered statistics for r4-asymmetry-localizer. Writes analysis.json.
Everything here maps 1:1 onto PREREG.md; anything post-hoc is keyed with a POSTHOC_ prefix."""
import json
import numpy as np
from scipy.stats import fisher_exact, binomtest
import r3_core as C

P = "/Users/steven/research_sprint_20260724/r4-asymmetry-localizer/"
R = json.load(open(P + "results.json"))
A = R["arms"]
out = {}
rng = np.random.default_rng(7)


def kn(name):
    return A[name]["successes"], A[name]["n"]


def fisher(n1, n2, label=None):
    k1, m1 = kn(n1) if isinstance(n1, str) else n1
    k2, m2 = kn(n2) if isinstance(n2, str) else n2
    tab = [[k1, m1 - k1], [k2, m2 - k2]]
    return dict(table=tab, a=f"{k1}/{m1}", b=f"{k2}/{m2}",
                p=float(fisher_exact(tab)[1]))


def newcombe(k1, m1, k2, m2, z=1.96):
    """95% score (Newcombe method 10) CI on the risk difference p1 - p2."""
    def wil(k, m):
        p = k / m
        d = 1 + z * z / m
        c = (p + z * z / (2 * m)) / d
        h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
        return max(0.0, c - h), min(1.0, c + h)
    l1, u1 = wil(k1, m1)
    l2, u2 = wil(k2, m2)
    d = k1 / m1 - k2 / m2
    lo = d - np.sqrt((k1 / m1 - l1) ** 2 + (u2 - k2 / m2) ** 2)
    hi = d + np.sqrt((u1 - k1 / m1) ** 2 + (k2 / m2 - l2) ** 2)
    return [round(float(lo), 4), round(float(hi), 4)]


def power_sim(p1, p2, n1, n2, alpha, reps=3000):
    k1 = rng.binomial(n1, p1, reps)
    k2 = rng.binomial(n2, p2, reps)
    return round(float(np.mean([fisher_exact([[a, n1 - a], [b, n2 - b]])[1] < alpha
                                for a, b in zip(k1, k2)])), 3)


# ----------------------------------------------------------------- PRIMARY
pri = fisher("r205_ccm", "r205_ica")
k1, m1 = kn("r205_ccm")
kc, mc = kn("r205_control")
cond1 = pri["p"] < 0.05 and k1 / m1 > kn("r205_ica")[0] / kn("r205_ica")[1]
cond2 = k1 >= 35
flo = fisher("r205_ccm", "r205_control")
cond3 = flo["p"] < 0.01
out["PRIMARY"] = {"test": "Fisher exact 2-sided, r205_ccm vs r205_ica", **pri,
    "cond1_p_lt_0.05_and_ccm_higher": bool(cond1),
    "cond2_ccm_ge_35_of_50": bool(cond2),
    "cond3_vs_control_p_lt_0.01": {**flo, "pass": bool(cond3)},
    "risk_diff_ci95": newcombe(k1, m1, *kn("r205_ica")),
    "DECISION": "THE ASYMMETRY LOCALIZER WORKS" if (cond1 and cond2 and cond3)
                else "THE ASYMMETRY IS NOT EXPLOITABLE BY THIS LOCALIZER"}

# vs the stored round-3 anchors (read-only)
out["PRIMARY_vs_round3_stored"] = dict(
    vs_axis_ica_13_of_30=fisher((k1, m1), (13, 30)),
    vs_rot_ica_pooled_31_of_60=fisher((k1, m1), (31, 60)),
    vs_rot_pair_broken_split_floor_18_of_60=fisher((k1, m1), (18, 60)),
    vs_standard_training_floor_0_of_30=fisher((k1, m1), (0, 30)))

# ----------------------------------------------------------------- S1 localization
sa = R["sweepA"]
c205 = sa["2.05"]
out["S1_localization_r2.05"] = {
    ln: {"lands_true": c205[ln]["lands_true"], "n": c205[ln]["n"],
         "wilson95": c205[ln]["wilson95_true"],
         "binom_vs_0.5_one_sided": float(binomtest(c205[ln]["lands_true"], c205[ln]["n"], 0.5,
                                                   alternative="greater").pvalue)}
    for ln in c205}
out["S1_ccm_vs_ica_probe"] = fisher((c205["ccm"]["lands_true"], c205["ccm"]["n"]),
                                    (c205["ica_probe"]["lands_true"], c205["ica_probe"]["n"]))

# ----------------------------------------------------------------- S2 threshold
def threshold(cells, key="wilson_lb"):
    for r in sorted(cells, key=float):
        c = cells[r]
        if key == "wilson_lb" and c["wilson95_true"][0] > 0.90:
            return float(r)
        if key == "point" and c["rate_true"] >= 0.90:
            return float(r)
    return None


ccmA = {r: sa[r]["ccm"] for r in sa}
out["S2_margin_threshold_N800"] = {
    "curve": {r: dict(rate_true=ccmA[r]["rate_true"], wilson95=ccmA[r]["wilson95_true"],
                      lands_ent=ccmA[r]["lands_ent"], lands_neither=ccmA[r]["lands_neither"])
              for r in sorted(ccmA, key=float)},
    "threshold_wilson_lb_gt_0.90": threshold(ccmA, "wilson_lb"),
    "threshold_point_ge_0.90": threshold(ccmA, "point")}

# ----------------------------------------------------------------- S3 scaling law
sb = R["sweepB_scaling"]
th = {}
for N, cells in sb.items():
    t = threshold(cells, "point")
    th[N] = dict(r_star=t, r2m1=round(t ** 2 - 1, 5) if t else None,
                 curve={r: cells[r]["rate_true"] for r in sorted(cells, key=float)})
base = th["800"]["r2m1"]
rat = {}
if base:
    for N in ["200", "3200"]:
        v = th[N]["r2m1"]
        rat[N] = round(v / base, 3) if v else None
out["S3_scaling_law"] = dict(
    thresholds=th, ratio_to_N800=rat,
    prediction=dict(N200_in=[1.4, 2.8], N3200_in=[0.35, 0.71], point=[2.0, 0.5]),
    PASS_N200=(rat.get("200") is not None and 1.4 <= rat["200"] <= 2.8),
    PASS_N3200=(rat.get("3200") is not None and 0.35 <= rat["3200"] <= 0.71))

# --------------------------------- POST-HOC refinement of S3 (labelled; prereg used the >=0.90
# grid point, which is right-censored at N=200 because the grid stops at r=1.25).
def cross50(cells):
    rs = sorted(cells, key=float)
    prev = None
    for r in rs:
        v = cells[r]["rate_true"]
        if v >= 0.5 and prev is not None:
            r0, v0 = prev
            return float(r0) + (float(r) - float(r0)) * (0.5 - v0) / (v - v0 + 1e-12)
        prev = (r, v)
    return None


c50 = {N: cross50(sb[N]) for N in sb}
b50 = c50["800"] ** 2 - 1 if c50["800"] else None
out["POSTHOC_S3_50pct_crossing"] = dict(
    note="POST-HOC: linear interpolation of the 50% lands-true crossing, used because the "
         "preregistered >=0.90 threshold is right-censored at N=200 (grid stops at r=1.25).",
    r_star={N: round(v, 4) if v else None for N, v in c50.items()},
    r2m1={N: round(v ** 2 - 1, 5) if v else None for N, v in c50.items()},
    ratio_to_N800={N: round((c50[N] ** 2 - 1) / b50, 3) for N in ["200", "3200"] if c50[N] and b50},
    prediction_point=dict(N200=2.0, N3200=0.5))

# ----------------------------------------------------------------- S4 2x2 mechanism
scale_k = c205["ccm"]["lands_true"] + c205["ica_scale"]["lands_true"]
scale_n = c205["ccm"]["n"] + c205["ica_scale"]["n"]
probe_k = c205["eig_probe"]["lands_true"] + c205["ica_probe"]["lands_true"]
probe_n = c205["eig_probe"]["n"] + c205["ica_probe"]["n"]
out["S4_2x2_mechanism_r2.05"] = dict(
    cells={ln: f'{c205[ln]["lands_true"]}/{c205[ln]["n"]}' for ln in c205},
    pooled_by_pairing_rule=dict(asymmetry=f"{scale_k}/{scale_n}", probe=f"{probe_k}/{probe_n}",
                                **fisher((scale_k, scale_n), (probe_k, probe_n))),
    pooled_by_direction_finder=dict(
        eigen=f'{c205["ccm"]["lands_true"] + c205["eig_probe"]["lands_true"]}/{2*c205["ccm"]["n"]}',
        ica=f'{c205["ica_probe"]["lands_true"] + c205["ica_scale"]["lands_true"]}/{2*c205["ccm"]["n"]}',
        **fisher((c205["ccm"]["lands_true"] + c205["eig_probe"]["lands_true"], 2 * c205["ccm"]["n"]),
                 (c205["ica_probe"]["lands_true"] + c205["ica_scale"]["lands_true"], 2 * c205["ccm"]["n"]))),
    falsifier_ica_scale_lands_true=f'{c205["ica_scale"]["lands_true"]}/{c205["ica_scale"]["n"]}')

# ----------------------------------------------------------------- S5 negative control
if "r100_ccm" in A:
    out["S5_negative_control"] = dict(
        r100_ccm=f'{kn("r100_ccm")[0]}/{kn("r100_ccm")[1]}',
        vs_r205_ccm=fisher("r100_ccm", "r205_ccm"),
        vs_r100_ica=fisher("r100_ccm", "r100_ica") if "r100_ica" in A else None,
        landings_r100_ccm=dict(true=A["r100_ccm"]["lands_true"], ent=A["r100_ccm"]["lands_ent"],
                               neither=A["r100_ccm"]["lands_neither"]))

# ----------------------------------------------------------------- S6 selectability threshold
out["S6_selectability_vs_r"] = {a: dict(r=A[a]["r"], succ=f'{A[a]["successes"]}/{A[a]["n"]}',
                                        wilson95=A[a]["wilson95"],
                                        lands_true=A[a]["lands_true"])
                                for a in ["r100_ccm", "r110_ccm", "r125_ccm", "r205_ccm"] if a in A}

# ----------------------------------------------------------------- S7 already deterministic
out["S7"] = dict(group=R["S7_symmetry_group"], spectrum=R["S7_D_spectrum_n2e5_r2.05"],
                 PASS=all((v["group_order"] == 64 and v["orbit_of_true_split"] == 2) if float(k) == 1.0
                          else (v["group_order"] == 16 and v["orbit_of_true_split"] == 1)
                          for k, v in R["S7_symmetry_group"].items()))

# ----------------------------------------------------------------- nulls: bounds + power
nulls = {}
if "r205_oracle" in A:
    ko, mo = kn("r205_oracle")
    f = fisher("r205_ccm", "r205_oracle")
    nulls["ccm_vs_oracle_r2.05"] = {**f, "risk_diff_ci95": newcombe(k1, m1, ko, mo),
        "power_vs_20pt_diff_alpha0.05": power_sim(0.87, 0.67, m1, mo, 0.05),
        "note": "equivalence bound = risk_diff_ci95"}
if "r100_ccm" in A and "r100_control" in A:
    nulls["r100_control_check"] = dict(succ=f'{kn("r100_control")[0]}/{kn("r100_control")[1]}')
out["NULLS_with_bounds_and_power"] = nulls

# success conditional on landing (mechanism, preregistered as part of S1/S4 reporting)
out["success_given_landing"] = {a: dict(succ_given_true=A[a]["succ_given_true"],
                                        lands=dict(true=A[a]["lands_true"], ent=A[a]["lands_ent"],
                                                   neither=A[a]["lands_neither"]))
                                for a in A if A[a]["lands_true"] is not None}

out["dropped_arms"] = R.get("dropped_arms", [])
out["runtime_sec"] = R.get("runtime_sec")
json.dump(out, open(P + "analysis.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("S2_margin_threshold_N800", "S3_scaling_law")}, indent=1, default=str))
print("\n--- S2 ---");  print(json.dumps(out["S2_margin_threshold_N800"], indent=1, default=str))
print("\n--- S3 ---");  print(json.dumps(out["S3_scaling_law"], indent=1, default=str))
