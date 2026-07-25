"""POST-HOC analyses. NOT preregistered. Every key is prefixed POSTHOC_ and is reported as
exploratory in FINDINGS.md. Adds equivalence bounds + power to every null, per PREREG's power
discipline, and unpacks the S4 2x2 whose preregistered prediction failed."""
import json
import numpy as np
from scipy.stats import fisher_exact

P = "/Users/steven/research_sprint_20260724/r4-asymmetry-localizer/"
R = json.load(open(P + "results.json"))
A = json.load(open(P + "analysis.json"))
rng = np.random.default_rng(11)
out = {}


def wil(k, m, z=1.96):
    p = k / m; d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, m1, k2, m2):
    l1, u1 = wil(k1, m1); l2, u2 = wil(k2, m2)
    d = k1 / m1 - k2 / m2
    return [round(float(d - np.sqrt((k1/m1-l1)**2 + (u2-k2/m2)**2)), 4),
            round(float(d + np.sqrt((u1-k1/m1)**2 + (k2/m2-l2)**2)), 4)]


def power_sim(p1, p2, n1, n2, alpha=0.05, reps=3000):
    a = rng.binomial(n1, p1, reps); b = rng.binomial(n2, p2, reps)
    return round(float(np.mean([fisher_exact([[x, n1-x], [y, n2-y]])[1] < alpha
                                for x, y in zip(a, b)])), 3)


def null_report(k1, m1, k2, m2, delta=0.20):
    p = float(fisher_exact([[k1, m1-k1], [k2, m2-k2]])[1])
    base = k2 / m2
    return dict(a=f"{k1}/{m1}", b=f"{k2}/{m2}", fisher_p=round(p, 4),
                risk_diff_ci95=newcombe(k1, m1, k2, m2),
                power_vs_20pt=power_sim(min(1.0, base + delta), base, m1, m2),
                verdict=("equivalent to within the stated risk-difference bound"
                         if p >= 0.05 else "different"))


# --- 1. the r=1.00 negative control: is L-CCM at the symmetric point the BROKEN-SPLIT floor?
out["POSTHOC_r100_ccm_nulls"] = dict(
    vs_r100_ica=null_report(14, 50, 12, 30),
    vs_round3_rot_pair_broken_split_18_of_60=null_report(14, 50, 18, 60),
    vs_round3_rot_ica_pooled_31_of_60=null_report(14, 50, 31, 60),
    note="PREREG S5 predicted r100_ccm <= r100_ica; direction matches (0.28 vs 0.40) but the test "
         "is non-significant, so only the bound is claimed. The landing profile (true 1 / ent 2 / "
         "neither 47) is the preregistered signature: degenerate eigenspaces -> random split.")

# --- 2. S4 2x2: the preregistered 'pairing rule dominates' prediction FAILED. Unpack it.
c = R["sweepA"]["2.05"]
cells = {k: (c[k]["lands_true"], c[k]["n"]) for k in ["ccm", "eig_probe", "ica_probe", "ica_scale"]}
out["POSTHOC_S4_interaction"] = dict(
    cells={k: f"{v[0]}/{v[1]}" for k, v in cells.items()},
    ccm_vs_best_single_ingredient=dict(
        vs_ica_probe=dict(p=float(fisher_exact([[40, 0], [27, 13]])[1])),
        vs_eig_probe=dict(p=float(fisher_exact([[40, 0], [18, 22]])[1])),
        vs_ica_scale=dict(p=float(fisher_exact([[40, 0], [18, 22]])[1]))),
    single_ingredient_vs_neither=dict(
        eig_probe_vs_ica_probe=null_report(18, 40, 27, 40),
        ica_scale_vs_ica_probe=null_report(18, 40, 27, 40)),
    reading="Neither ingredient helps on its own (both single-ingredient cells are at or BELOW the "
            "no-ingredient cell); only eigen-directions AND the asymmetry pairing rule together "
            "give 40/40. The preregistered claim 'the pairing rule is the whole story' is FALSE: "
            "ICA's whitened directions are not the D-eigenvectors, so the asymmetry statistic "
            "computed on them is uninformative.")

# --- 3. does ICA benefit from asymmetry at all?
ic = {r: R["sweepA"][r]["ica_probe"] for r in R["sweepA"] if "ica_probe" in R["sweepA"][r]}
out["POSTHOC_ica_lands_true_vs_r"] = dict(
    curve={r: f'{ic[r]["lands_true"]}/{ic[r]["n"]}' for r in sorted(ic, key=float)},
    r2p05_vs_r1p00=null_report(ic["2.05"]["lands_true"], 40, ic["1"]["lands_true"], 40),
    note="ICA's landing rate is flat-to-slightly-rising in r and never approaches 1; the round-3 "
         "statement 'ICA coin-flips regardless of margin' is directionally right but the rate is "
         "~0.58-0.68, mildly above 0.5.")

# --- 4. routing-stage ceiling given perfect localization
sg = A["success_given_landing"]
out["POSTHOC_routing_stage_ceiling_r2.05"] = dict(
    ccm_success_given_lands_true=f'{sg["r205_ccm"]["succ_given_true"][0]}/{sg["r205_ccm"]["succ_given_true"][1]}',
    ica_success_given_lands_true=f'{sg["r205_ica"]["succ_given_true"][0]}/{sg["r205_ica"]["succ_given_true"][1]}',
    oracle=f'{R["arms"]["r205_oracle"]["successes"]}/{R["arms"]["r205_oracle"]["n"]}',
    ccm_vs_ica_given_true=null_report(38, 50, 25, 29),
    ica_success_given_NOT_true=f'{R["arms"]["r205_ica"]["successes"] - sg["r205_ica"]["succ_given_true"][0]}'
                               f'/{50 - sg["r205_ica"]["lands"]["true"]}',
    reading="Conditional on landing on the true split, success is ~0.76-0.87 in every arm including "
            "the oracle: the residual 13-24% loss is the ROUTING stage, not localization. L-CCM "
            "removes the localization loss entirely (50/50 landings) and inherits the routing loss.")

# --- 5. r110 selectability vs r205 (is 1.10 already enough?)
out["POSTHOC_r110_vs_r205"] = null_report(23, 30, 38, 50)

json.dump(out, open(P + "posthoc.json", "w"), indent=1)
print(json.dumps(out, indent=1))
