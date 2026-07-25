"""analysis.py -- zero-cost re-analysis of results.json. NO new training.

Two jobs:
1. Apply the PREREGISTERED COMPETENCE GATE that experiment.py did not enforce in its success flag.
   PREREG.md says: "seeds with held_full <= 0.9 are reported and counted as non-successes (never
   silently dropped)". experiment.py's `success` used only selA_held/selB_held. This script
   recomputes the gated counts and re-runs the primary/secondary Fisher tests on them; the gated
   numbers are the ones reported as primary in FINDINGS.md.
2. Cross-round comparison against round-1 (routing-worldview-select/results.json) and the
   preregistered secondary descriptives.

Anything marked POST-HOC below was not in PREREG.md.
"""
import json
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency

R3 = "/Users/steven/research_sprint_20260724/r3-lm-organism/results.json"
R1 = "/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"
TRAIN_OK, HI, LO, NEVAL = 0.9, 0.8, 0.2, 4000

res = json.load(open(R3))
out = {}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# ---- 1. preregistered competence gate ----
gated = {}
for arm, v in res["arms"].items():
    rows = v["per_seed"]
    n = len(rows)
    k_ungated = sum(r["success"] for r in rows)
    k_gated = sum(r["success"] and r["held_full"] > TRAIN_OK for r in rows)
    fails = [r["seed"] for r in rows if r["held_full"] <= TRAIN_OK]
    gated[arm] = dict(n=n, successes_ungated=k_ungated, successes_gated=k_gated,
                      wilson95_gated=wilson(k_gated, n),
                      seeds_failing_competence_gate=fails,
                      successes_designated_gated=sum(r["success_designated"] and r["held_full"] > TRAIN_OK
                                                     for r in rows))
out["competence_gated_counts"] = gated


def fisher(a, b, key="successes_gated"):
    ka, na = gated[a][key], gated[a]["n"]
    kb, nb = gated[b][key], gated[b]["n"]
    odds, p = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
    return dict(arms=[a, b], table=[[ka, na - ka], [kb, nb - kb]], p=float(p))


out["primary2_gated"] = fisher("routed", "control")
kr = gated["routed"]["successes_gated"]
out["primary2_gated"]["criterion_p_lt_0p01"] = bool(out["primary2_gated"]["p"] < 0.01)
out["primary2_gated"]["criterion_routed_ge_8_of_12"] = bool(kr >= 8)
out["primary2_gated"]["decision"] = ("H2 SUPPORTED" if (out["primary2_gated"]["p"] < 0.01 and kr >= 8)
                                     else "H2 NOT SUPPORTED")
out["secondary_gated"] = [fisher("routed", "occlusion_only"), fisher("occlusion_only", "control")]

# ---- 2. un-ablated deploy distribution per arm (preregistered secondary) ----
# The per-arm chi-square homogeneity test is PREREGISTERED for `control` only (Primary 1a);
# running it on `routed` and `occlusion_only` as well is POST-HOC and labeled as such.
dist = {}
for arm, v in res["arms"].items():
    dep = np.array([r["deploy_full"] for r in v["per_seed"]])
    corr = np.round(dep * NEVAL).astype(int)
    chi2, p, dof, _ = chi2_contingency(np.stack([corr, NEVAL - corr], 1))
    R = np.array([r["R_full"] for r in v["per_seed"]])
    dist[arm] = dict(deploy_full=[round(float(x), 4) for x in dep],
                     mean=round(float(dep.mean()), 4), std=round(float(dep.std(ddof=1)), 4),
                     min=round(float(dep.min()), 4), max=round(float(dep.max()), 4),
                     n_hi=int((dep > HI).sum()), n_lo=int((dep < LO).sum()),
                     n_mid=int(((dep >= LO) & (dep <= HI)).sum()),
                     chi2_homogeneity=float(chi2), p_homogeneity=float(p), dof=int(dof),
                     reliance_R_mean=round(float(R.mean()), 4),
                     reliance_R_absmean=round(float(np.abs(R).mean()), 4),
                     reliance_R_std=round(float(R.std(ddof=1)), 4),
                     prereg="Primary 1a" if arm == "control" else "POST-HOC (chi-square on this arm)")
out["unablated_deploy_distribution_per_arm"] = dist

# ---- 3. ablation cleanliness (preregistered descriptive: selA/selB per arm) ----
clean = {}
for arm, v in res["arms"].items():
    rows = v["per_seed"]
    clean[arm] = dict(
        selA_deploy=dict(min=round(min(r["selA_deploy"] for r in rows), 4),
                         med=round(float(np.median([r["selA_deploy"] for r in rows])), 4),
                         max=round(max(r["selA_deploy"] for r in rows), 4)),
        selB_deploy=dict(min=round(min(r["selB_deploy"] for r in rows), 4),
                         med=round(float(np.median([r["selB_deploy"] for r in rows])), 4),
                         max=round(max(r["selB_deploy"] for r in rows), 4)),
        selA_held_min=round(min(r["selA_held"] for r in rows), 4),
        selB_held_min=round(min(r["selB_held"] for r in rows), 4))
out["ablation_cleanliness"] = clean

# ---- 4. cross-round comparison with the round-1 MLP toy (read-only) ----
r1 = json.load(open(R1))
out["round1_mlp_comparison"] = {a: dict(successes=v["summary"]["successes"], n=v["summary"]["n"],
                                        full_align_hi=v["summary"]["full_nHi"],
                                        full_align_lo=v["summary"]["full_nLo"],
                                        full_align_mid=v["summary"]["full_nMid"],
                                        full_align_mean=v["summary"]["full_align_mean"])
                               for a, v in r1["arms"].items()}

res["post_run_analysis"] = out
json.dump(res, open(R3, "w"), indent=1)
print(json.dumps(out, indent=1))
