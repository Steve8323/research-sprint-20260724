"""POST-HOC / EXPLORATORY (not in PREREG.md). No training: reads the two confirmatory
results files and re-describes the same models. Every number produced here is labeled
exploratory in FINDINGS.md.

E1. Purity decomposition -- what mode does each arm actually produce (pure-true /
    pure-proxy / mixed / non-deterministic)? Sharper than "routed are more deterministic".
E2. Is patchwork-ness just "intermediate deploy alignment"?
E3. Direction structure of the per-input confidence monitor (M2): is the useless median
    hiding usable per-model signal with an unpredictable SIGN?
"""
import json
import numpy as np
from scipy.stats import fisher_exact, spearmanr

DIR = "/Users/steven/research_sprint_20260724/r4-patchwork-prereg"
OUT = f"{DIR}/results_exploratory.json"


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def modes(rows, seeds, align_key):
    out = dict(pure_true=0, pure_proxy=0, mixed=0, nondeterministic=0)
    for s in seeds:
        r = rows[str(s)] if str(s) in rows else rows[s]
        t, p = r["n_det_true"], r["n_det_proxy"]
        if t >= 1 and p >= 1: out["mixed"] += 1
        elif t >= 1: out["pure_true"] += 1
        elif p >= 1: out["pure_proxy"] += 1
        else: out["nondeterministic"] += 1
    return out


def substrate(res, arm_key_r, arm_key_c, seeds_r, seeds_c, align_key, name):
    rows_r = res["per_seed"][arm_key_r]; rows_c = res["per_seed"][arm_key_c]
    g = lambda rows, s, k: (rows[str(s)] if str(s) in rows else rows[s])[k]
    e = {}
    e["E1_mode_census"] = dict(routed=modes(rows_r, seeds_r, align_key),
                               control=modes(rows_c, seeds_c, align_key),
                               n_routed=len(seeds_r), n_control=len(seeds_c))
    # E2
    al_r = np.array([g(rows_r, s, align_key) for s in seeds_r])
    pw_r = np.array([g(rows_r, s, "patchwork") for s in seeds_r], dtype=bool)
    inter = (al_r >= 0.2) & (al_r <= 0.8)
    e["E2_patchwork_vs_alignment"] = dict(
        routed_mean_align_patchwork=round(float(al_r[pw_r].mean()), 4) if pw_r.any() else None,
        routed_mean_align_nonpatchwork=round(float(al_r[~pw_r].mean()), 4) if (~pw_r).any() else None,
        routed_intermediate_align_n=int(inter.sum()),
        patchwork_given_intermediate=f"{int((pw_r & inter).sum())}/{int(inter.sum())}",
        patchwork_given_committed=f"{int((pw_r & ~inter).sum())}/{int((~inter).sum())}",
        fisher_p=float(fisher_exact([[int((pw_r & inter).sum()), int((~pw_r & inter).sum())],
                                     [int((pw_r & ~inter).sum()), int((~pw_r & ~inter).sum())]])[1]),
        spearman_align_vs_ndet=None)
    # how many patterns are ambiguous vs deterministic per arm
    e["E2b_pattern_mix"] = dict(
        routed_median_ambiguous=float(np.median([8 - g(rows_r, s, "n_det_true") - g(rows_r, s, "n_det_proxy")
                                                 if name == "mlp" else g(rows_r, s, "n_ambig") for s in seeds_r])),
        control_median_ambiguous=float(np.median([8 - g(rows_c, s, "n_det_true") - g(rows_c, s, "n_det_proxy")
                                                  if name == "mlp" else g(rows_c, s, "n_ambig") for s in seeds_c])))
    # E3
    a = [g(rows_r, s, "auroc_conf") for s in seeds_r
         if g(rows_r, s, "patchwork") and g(rows_r, s, "m2_eligible")]
    a = np.array([x for x in a if x is not None], dtype=float)
    if len(a):
        e["E3_monitor_direction"] = dict(
            n=len(a), median=round(float(np.median(a)), 4), mean=round(float(a.mean()), 4),
            sd=round(float(a.std(ddof=1)), 4) if len(a) > 1 else None,
            frac_abs_dev_gt_0p10=round(float(np.mean(np.abs(a - 0.5) > 0.10)), 4),
            frac_abs_dev_gt_0p20=round(float(np.mean(np.abs(a - 0.5) > 0.20)), 4),
            frac_above_0p5=round(float(np.mean(a > 0.5)), 4),
            sign_binomial_wilson=wilson(int((a > 0.5).sum()), len(a)),
            note="per-model deviations are large but the SIGN is ~a coin flip, so no fixed monitor "
                 "direction generalises across models")
        al = np.array([g(rows_r, s, align_key) for s in seeds_r
                       if g(rows_r, s, "patchwork") and g(rows_r, s, "m2_eligible")], dtype=float)
        if len(al) == len(a) and len(a) > 2:
            sp = spearmanr(al, a)
            e["E3_monitor_direction"]["spearman_auroc_vs_deploy_alignment"] = dict(
                rho=round(float(sp.statistic), 4), p=float(sp.pvalue), n=len(a))
    return e


def main():
    out = {"label": "POST-HOC / EXPLORATORY (not preregistered)"}
    mlp = json.load(open(f"{DIR}/results_mlp.json"))
    out["mlp"] = substrate(mlp, "routed", "control", mlp["S"], mlp["config"]["seeds"], "full_align", "mlp")
    try:
        tf = json.load(open(f"{DIR}/results_tf.json"))
        seeds_r = [s for s in tf["config"]["seeds"]
                   if (tf["per_seed"]["routed"][str(s)] if str(s) in tf["per_seed"]["routed"]
                       else tf["per_seed"]["routed"][s])["competent"]]
        out["transformer"] = substrate(tf, "routed", "control", seeds_r, tf["config"]["seeds"],
                                       "deploy_full", "tf")
    except FileNotFoundError:
        out["transformer"] = "results_tf.json not present"
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
