"""Round-4 analysis: preregistered primary tests, deliverable k*, secondaries, controls, and
clearly separated post-hoc items. Reads results.json (+ sources.json if present)."""
import json, os, sys
import numpy as np
from scipy.stats import fisher_exact, binomtest

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange")
import sup as S
import core as C

D = "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange/"
R = json.load(open(D + "results.json"))
A = R["arms"]
M = R["config"]["M_GRID"]
AUG = R["config"]["AUG_GRID"]
out = {"prereg": {}, "secondary": {}, "controls": {}, "post_hoc": {}}


def cell(arm, field="successes"):
    s = A[arm]["summary"]
    return s[field], s["n"]


def fisher(k1, n1, k2, n2):
    o, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    return dict(table=[[k1, n1 - k1], [k2, n2 - k2]], p=float(p))


# ---------------- PRIMARY 1 + 2 -------------------------------------------------------------
ks_loc = [A[f"sup_dep_m{m}"]["summary"]["n_lands_true"] for m in M]
ks_sel = [A[f"sup_dep_m{m}"]["summary"]["successes"] for m in M]
ns = [A[f"sup_dep_m{m}"]["summary"]["n"] for m in M]
out["prereg"]["PRIMARY1_localization_trend"] = S.trend_test(ks_loc, ns, M)
out["prereg"]["PRIMARY2_selectability_trend"] = S.trend_test(ks_sel, ns, M)

curve = []
for i, m in enumerate(M):
    s = A[f"sup_dep_m{m}"]["summary"]
    curve.append(dict(m=m, n=ns[i], lands_true=ks_loc[i], lands_true_wilson95=s["lands_true_wilson95"],
                      lands_entangled=s["n_lands_entangled"], lands_neither=s["n_lands_neither"],
                      selectable=ks_sel[i], sel_wilson95=s["wilson95"],
                      designated=s.get("successes_designated"),
                      fid_true_mean=s["fid_true_mean"], full_align_mean=s["full_align_mean"]))
out["prereg"]["exchange_rate_curve"] = curve

k_sel = next((m for i, m in enumerate(M) if A[f"sup_dep_m{m}"]["summary"]["wilson95"][0] >= 0.50), None)
k_loc = next((m for i, m in enumerate(M)
              if A[f"sup_dep_m{m}"]["summary"]["lands_true_wilson95"][0] >= 0.75), None)
out["prereg"]["k_star_sel"] = k_sel
out["prereg"]["k_star_loc"] = k_loc
p1 = out["prereg"]["PRIMARY1_localization_trend"]["p_perm_one_sided"]
p2 = out["prereg"]["PRIMARY2_selectability_trend"]["p_perm_one_sided"]
if p1 < 0.05 and p2 < 0.05 and k_sel is not None and k_sel <= 8:
    dec = "SUPPORTED"
elif p1 < 0.05 and p2 < 0.05 and k_sel is not None:
    dec = "NARROWED (trend holds, k*_sel > 8)"
else:
    dec = "HONEST NEGATIVE"
out["prereg"]["DECISION"] = dec

# per-level Fisher vs m=0 (SECONDARY / descriptive per PREREG; Bonferroni over 6)
k0s, n0s = cell("sup_dep_m0"); k0l = A["sup_dep_m0"]["summary"]["n_lands_true"]
out["secondary"]["per_level_vs_m0"] = [
    dict(m=m, sel=fisher(A[f"sup_dep_m{m}"]["summary"]["successes"], 30, k0s, 30),
         loc=fisher(A[f"sup_dep_m{m}"]["summary"]["n_lands_true"], 30, k0l, 30),
         bonferroni_alpha=round(0.05 / (len(M) - 1), 4)) for m in M[1:]]

# ---------------- S1: the same m examples spent on data instead of on a switch ---------------
s1 = []
for m in AUG:
    ka, na = A[f"augment_m{m}"]["summary"]["successes"], A[f"augment_m{m}"]["summary"]["n"]
    ks, nsel = A[f"sup_dep_m{m}"]["summary"]["successes"], A[f"sup_dep_m{m}"]["summary"]["n"]
    row = dict(m=m, augment_aligned=ka, augment_wilson95=A[f"augment_m{m}"]["summary"]["wilson95"],
               augment_switch=A[f"augment_m{m}"]["summary"]["successes_switch"],
               augment_align_mean=A[f"augment_m{m}"]["summary"]["full_align_mean"],
               sup_selectable=ks, fisher=fisher(ks, nsel, ka, na),
               tost_delta0p20=S.tost_prop(ks, nsel, ka, na, 0.20),
               fisher_power_vs_delta0p20=S.fisher_power(ks / nsel, min(1.0, ks / nsel + 0.20), 30, 0.05))
    s1.append(row)
out["secondary"]["S1_switch_vs_just_training_on_them"] = s1
out["secondary"]["S1_augment_trend"] = S.trend_test(
    [A[f"augment_m{m}"]["summary"]["successes"] for m in AUG], [30] * len(AUG), AUG)

# ---------------- S4: does labelled supervision also pick the switch POSITION? ----------------
out["secondary"]["S4_position_identification"] = [
    dict(m=m, designated=A[f"sup_dep_m{m}"]["summary"].get("successes_designated"),
         either=A[f"sup_dep_m{m}"]["summary"]["successes"], n=30) for m in M]

# ---------------- S5: is the passing level at the oracle ceiling? ----------------------------
ko, no = cell("rot_oracle")
if k_sel is not None:
    kk, nn = cell(f"sup_dep_m{k_sel}")
    out["secondary"]["S5_vs_oracle_at_kstar"] = dict(
        m=k_sel, fisher=fisher(kk, nn, ko, no), tost_delta0p20=S.tost_prop(kk, nn, ko, no, 0.20))
best_m = max(M, key=lambda m: A[f"sup_dep_m{m}"]["summary"]["successes"])
kb, nb = cell(f"sup_dep_m{best_m}")
out["secondary"]["S5_vs_oracle_at_best_m"] = dict(
    m=best_m, fisher=fisher(kb, nb, ko, no), tost_delta0p20=S.tost_prop(kb, nb, ko, no, 0.20))

# ---------------- controls / replication anchors ---------------------------------------------
out["controls"] = dict(
    rot_oracle=dict(k=ko, n=no, round3_expected=26, matches=bool(ko == 26)),
    rot_control=dict(k=cell("rot_control")[0], n=30, round3_expected=0,
                     matches=bool(cell("rot_control")[0] == 0)),
    unsup_ica=dict(k=cell("unsup_ica")[0], n=30, round3_expected=16,
                   matches=bool(cell("unsup_ica")[0] == 16),
                   lands_true=A["unsup_ica"]["summary"]["n_lands_true"], round3_lands_true=None),
    generator_identity_max_abs_diff=R["generator_identity_max_abs_diff"],
    m0_floor_vs_ica=fisher(k0s, 30, cell("unsup_ica")[0], 30))

# ---------------- secondary sources ----------------------------------------------------------
if os.path.exists(D + "sources.json"):
    SR = json.load(open(D + "sources.json"))
    SA = SR["arms"]
    src = {}
    for tag, grid, key in [("unlab", SR["config"]["UNLAB_GRID"], "unlab_m"),
                           ("concept", SR["config"]["CONCEPT_GRID"], "concept_m"),
                           ("agree_ctrl", SR["config"]["AGREE_GRID"], "agree_m"),
                           ("dsup_dict_family", SR["config"]["DSUP_GRID"], "dsup_m")]:
        ks = [SA[f"{key}{m}"]["summary"]["n_lands_true"] for m in grid]
        nn = [SA[f"{key}{m}"]["summary"]["n"] for m in grid]
        rows = [dict(m=m, lands_true=SA[f"{key}{m}"]["summary"]["n_lands_true"],
                     wilson95=SA[f"{key}{m}"]["summary"]["lands_true_wilson95"],
                     lands_entangled=SA[f"{key}{m}"]["summary"]["n_lands_entangled"],
                     lands_neither=SA[f"{key}{m}"]["summary"]["n_lands_neither"],
                     fid_true_mean=SA[f"{key}{m}"]["summary"]["fid_true_mean"]) for m in grid]
        entry = dict(curve=rows, k_star_loc=next((m for m in grid if
                     SA[f"{key}{m}"]["summary"]["lands_true_wilson95"][0] >= 0.75), None))
        if len(grid) >= 3:
            entry["trend"] = S.trend_test(ks, nn, grid)
        if tag == "agree_ctrl" or tag == "dsup_dict_family":
            entry["vs_dep_lab_same_m"] = [
                dict(m=m, this_arm=SA[f"{key}{m}"]["summary"]["n_lands_true"],
                     dep_lab=A[f"sup_dep_m{m}"]["summary"]["n_lands_true"],
                     fisher=fisher(SA[f"{key}{m}"]["summary"]["n_lands_true"], 30,
                                   A[f"sup_dep_m{m}"]["summary"]["n_lands_true"], 30),
                     tost_delta0p20_vs_m0=S.tost_prop(SA[f"{key}{m}"]["summary"]["n_lands_true"], 30,
                                                      k0l, 30, 0.20)) for m in grid if m in M]
        src[tag] = entry
    asym = []
    for rho in SR["config"]["ASYM_GRID"]:
        s = SA[f"asym_rho{rho}"]["summary"]
        asym.append(dict(rho=rho, lands_true=s["n_lands_true"], n=s["n"],
                         wilson95=s["lands_true_wilson95"],
                         lands_entangled=s["n_lands_entangled"], lands_neither=s["n_lands_neither"],
                         binom_p_vs_0p5=float(binomtest(s["n_lands_true"], s["n"], 0.5).pvalue)))
    src["asym"] = dict(curve=asym, trend=S.trend_test(
        [a["lands_true"] for a in asym], [a["n"] for a in asym],
        [r - 1.0 for r in SR["config"]["ASYM_GRID"]]))
    out["secondary"]["S2_S3_other_sources"] = src

# ---------------- POST-HOC (labelled) ---------------------------------------------------------
cond = {"true": [0, 0], "entangled": [0, 0], "neither": [0, 0]}
for m in M:
    for r in A[f"sup_dep_m{m}"]["per_seed"]:
        k = "true" if r["fid_true"] > 0.95 else ("entangled" if r["fid_entangled"] > 0.95 else "neither")
        cond[k][1] += 1; cond[k][0] += int(r["success"])
out["post_hoc"]["success_by_orbit_member_pooled_over_m"] = {
    k: dict(successes=v[0], n=v[1], rate=round(v[0] / v[1], 4) if v[1] else None,
            wilson95=C.wilson(v[0], v[1]) if v[1] else None) for k, v in cond.items()}
if cond["true"][1] and cond["entangled"][1]:
    out["post_hoc"]["true_vs_entangled_fisher"] = fisher(cond["true"][0], cond["true"][1],
                                                         cond["entangled"][0], cond["entangled"][1])
out["post_hoc"]["note"] = ("orbit-member conditioning is post-hoc here (it replicates round-3's "
                           "post-hoc conditional); everything under 'prereg'/'secondary'/'controls' "
                           "is from PREREG.md + AMENDMENT 1.")

json.dump(out, open(D + "analysis.json", "w"), indent=1)
print(json.dumps({k: out["prereg"][k] for k in
                  ["PRIMARY1_localization_trend", "PRIMARY2_selectability_trend",
                   "k_star_sel", "k_star_loc", "DECISION"]}, indent=1))
print("\ncurve:")
for c in curve:
    print(f"  m={c['m']:2d}  lands_true {c['lands_true']:2d}/30 {c['lands_true_wilson95']}  "
          f"sel {c['selectable']:2d}/30 {c['sel_wilson95']}  des {c['designated']}  "
          f"T/E/N {c['lands_true']}/{c['lands_entangled']}/{c['lands_neither']}")
print("\ncontrols:", json.dumps(out["controls"], indent=1))
print("\nS1 (switch vs just training on the same m):")
for r in s1:
    print(f"  m={r['m']:2d}  sup_selectable {r['sup_selectable']:2d}/30 | augment_aligned "
          f"{r['augment_aligned']:2d}/30 (mean align {r['augment_align_mean']}, switch "
          f"{r['augment_switch']}/30)  fisher p={r['fisher']['p']:.4g}")
print("\nwrote analysis.json")
