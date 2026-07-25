"""POST-HOC analyses (NOT in PREREG.md). Reads results.json, writes posthoc.json.
Kept in a separate file/output so nothing here can be mistaken for a preregistered number."""
import json
import numpy as np
import experiment as E

R = json.load(open(E.OUT))
P = {"note": "Everything in this file is POST-HOC (chosen after seeing results.json)."}

# P1 — capacity control on the flip-cost metric (prereg only tested it on AUC)
P["P1_flip_switch_vs_h4"] = E.mwu(R["attack"]["switch_aligned"]["best_flip"],
                                  R["attack"]["h4_aligned"]["best_flip"])
P["P1_flip_h4_vs_corrected"] = E.mwu(R["attack"]["h4_aligned"]["best_flip"],
                                     R["attack"]["corrected_f5"]["best_flip"])

# P2 — Holm-Bonferroni over the four non-shift primaries
pr = {"A1": R["attack_tests"]["PRIMARY_A1_auc_switch_vs_corrected"]["p_perm"],
      "A2": R["attack_tests"]["PRIMARY_A2_flip_switch_vs_corrected"]["p_perm"],
      "D1": R["drift"]["D1_selectability"]["PRIMARY_mcnemar_N2000_vs_0"]["p"],
      "D2": R["drift"]["D2_position"]["PRIMARY_mcnemar_N2000_vs_0"]["p"]}
order = sorted(pr, key=lambda k: pr[k]); m = len(order); holm = {}
prev = 0.0
for i, k in enumerate(order):
    thr = 0.05 / (m - i)
    adj = max(prev, min(1.0, pr[k] * (m - i))); prev = adj
    holm[k] = dict(p=pr[k], holm_threshold=round(thr, 5), p_holm_adjusted=round(adj, 6),
                   rejected_at_0p05=bool(adj < 0.05))
P["P2_holm_over_primaries"] = holm

# P3 — stability with a DIFFICULTY-MATCHED functionality bar instead of the fixed 0.9
#      (bar = the full-width unrouted control's own mean agreeing accuracy under that shift, -0.05)
p3 = {}
for name, d in R["stability"].items():
    bar = d["control_agree_mean"] - 0.05
    k = sum(1 for r in d["per_seed"]
            if r["aligned_side_align"] > E.ALIGN_HI and r["mis_side_align"] < E.ALIGN_LO
            and r["aligned_side_agree"] > bar and r["mis_side_agree"] > bar)
    p3[name] = dict(matched_bar=round(bar, 4), retained=k, n=d["n"],
                    retention=round(k / d["n"], 4), wilson95=E.wilson(k, d["n"]),
                    prereg_retention=d["retention"])
P["P3_stability_difficulty_matched_bar"] = p3
# direction-only: does the aligned side ever fall below the misaligned side?
P["P3b_direction_preserved"] = {name: dict(n=d["n"], gap_positive=sum(r["gap"] > 0 for r in d["per_seed"]),
                                           mean_gap=d["mean_gap"], ci=d["ci_gap"])
                                for name, d in R["stability"].items()}

# P4 — flip cost WITHOUT the "still functional" requirement (deploy alignment < 0.2 alone)
def flip_nofunc(curves, cell):
    out = []
    for p in curves:
        f = E.ATTACK_STEPS + 1
        for pt in p["curves"][cell]:
            if pt["deploy_align"] < E.ALIGN_LO:
                f = pt["step"]; break
        out.append(f)
    return out


p4 = {arm: flip_nofunc(R["attack_curves"][arm], R["attack"][arm]["best_cell"])
      for arm in ["switch_aligned", "corrected_f5", "aligned_chance", "h4_aligned"]}
P["P4_flipcost_no_functionality_requirement"] = dict(
    per_arm=p4,
    medians={a: float(np.median(v)) for a, v in p4.items()},
    n_censored={a: int(sum(x > E.ATTACK_STEPS for x in v)) for a, v in p4.items()},
    switch_vs_corrected=E.mwu(p4["switch_aligned"], p4["corrected_f5"]))

# P5 — D1 early erosion: selectability after only 100 agreeing steps (5% of training budget)
P["P5_drift_early"] = dict(
    N100=R["drift"]["D1_selectability"]["summary"]["100"],
    mcnemar_N100_vs_0=E.mcnemar([r["probes"]["0"]["selectable"] for r in R["drift"]["D1_selectability"]["per_seed"]],
                                [r["probes"]["100"]["selectable"] for r in R["drift"]["D1_selectability"]["per_seed"]]))

json.dump(P, open(f"{E.ROOT}/posthoc.json", "w"), indent=1)
for k, v in P.items():
    if k != "note": print(k, "->", json.dumps(v)[:600])
