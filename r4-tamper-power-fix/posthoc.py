"""EVERYTHING IN THIS FILE IS POST-HOC (chosen after seeing results.json). Kept physically separate
from the preregistered claims. Motivated by one preregistered observation: in C5 the two
preregistered summaries disagree (Welch CI on the mean excludes 0, MWU on ranks does not)."""
import json, numpy as np
from scipy import stats
R = json.load(open("results.json"))
F, P = R["attack_functional"], R["attack_prereg"]
out = {"note": "POST-HOC. Not preregistered. Descriptive only; no verdict is revised on this basis."}

def arr(D, a, f="best_auc9"): return np.array(D[a][f], float)

# 1. is the switch a location shift or a heavy resistant tail?
het = {}
for tag, D in [("func", F), ("prereg", P)]:
    for a in D:
        v = arr(D, a)
        het[f"{tag}:{a}"] = dict(n=len(v), mean=round(float(v.mean()), 4),
                                 median=round(float(np.median(v)), 4),
                                 sd=round(float(v.std(ddof=1)), 4),
                                 iqr=[round(float(np.percentile(v, 25)), 4), round(float(np.percentile(v, 75)), 4)],
                                 frac_strong_resist_auc_lt_0p6=round(float((v < 0.6).mean()), 4),
                                 n_strong_resist=int((v < 0.6).sum()))
out["heterogeneity"] = het
sw, h4, co = arr(F, "switch_aligned"), arr(F, "h4_aligned"), arr(F, "corrected_f5")
out["levene_func_switch_vs_corrected"] = dict(
    stat=round(float(stats.levene(sw, co, center="median")[0]), 4),
    p=round(float(stats.levene(sw, co, center="median")[1]), 6),
    sd_ratio=round(float(sw.std(ddof=1) / co.std(ddof=1)), 3))
out["levene_func_switch_vs_h4"] = dict(
    stat=round(float(stats.levene(sw, h4, center="median")[0]), 4),
    p=round(float(stats.levene(sw, h4, center="median")[1]), 6),
    sd_ratio=round(float(sw.std(ddof=1) / h4.std(ddof=1)), 3))
# does the capacity control ALSO have a resistant tail? (i.e. is the tail itself capacity?)
k1, k2 = int((sw < 0.6).sum()), int((h4 < 0.6).sum())
out["strong_resisters_switch_vs_h4_FUNC"] = dict(
    k_switch=k1, k_h4=k2, n=len(sw),
    fisher_p=round(float(stats.fisher_exact([[k1, len(sw) - k1], [k2, len(h4) - k2]])[1]), 6))
k3 = int((co < 0.6).sum())
out["strong_resisters_switch_vs_corrected_FUNC"] = dict(
    k_switch=k1, k_corrected=k3, n=len(sw),
    fisher_p=round(float(stats.fisher_exact([[k1, len(sw) - k1], [k3, len(co) - k3]])[1]), 8))

# 2. what does the attacker's product look like? (collateral at step 100)
coll = {}
for tag, D in [("func", F), ("prereg", P)]:
    for a in D:
        v = np.array(D[a]["best_final_agree"], float)
        coll[f"{tag}:{a}"] = dict(mean_agree_at_100=round(float(v.mean()), 4),
                                  frac_still_competent_gt0p9=round(float((v > 0.9).mean()), 4))
out["collateral"] = coll

# 3. attacker-strength ladder for the switch-vs-capacity contrast (sign of the difference)
ladder = {}
for name, f in [("prereg_LoRA_best", ("best_auc9", P)), ("func_LoRA_best", ("best_auc9", F)),
                ("prereg_fullft", ("fullft_auc9", P)), ("func_fullft", ("fullft_auc9", F)),
                ("prereg_fullft_jit", ("fullftjit_auc9", P)), ("func_fullft_jit", ("fullftjit_auc9", F))]:
    fld, D = f
    if "switch_aligned" in D and "h4_aligned" in D:
        a, b = np.array(D["switch_aligned"][fld], float), np.array(D["h4_aligned"][fld], float)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        ladder[name] = dict(switch=round(float(a.mean()), 4), h4=round(float(b.mean()), 4),
                            diff=round(float(a.mean() - b.mean()), 4), mwu_p=round(float(u.pvalue), 6))
out["attacker_strength_ladder_switch_minus_h4"] = ladder
json.dump(out, open("posthoc.json", "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v) if not isinstance(v, str) else v)
