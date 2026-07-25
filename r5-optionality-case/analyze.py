"""Analysis for r5-optionality-case. Reads results.json, writes analysis.json + prints tables.
Every test here is the one named in PREREG.md; post-hoc items are prefixed POSTHOC.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r5-optionality-case")
import opt as O

R = json.load(open("/Users/steven/research_sprint_20260724/r5-optionality-case/results.json"))
rows = R["per_seed"]; n = len(rows)
A = {}


def col(key):
    return np.array([r[key] for r in rows])


def rate(key):
    v = col(key).astype(bool)
    k = int(v.sum())
    return dict(k=k, n=n, rate=round(k / n, 4), wilson95=O.wilson(k, n))


def compare(ka, kb, label):
    a, b = col(ka).astype(bool), col(kb).astype(bool)
    d = dict(a=label[0], b=label[1], a_rate=round(a.mean(), 4), b_rate=round(b.mean(), 4),
             mcnemar=O.mcnemar_exact(a, b), boot95=O.boot_paired(a, b),
             tost_d15=O.tost_paired(a, b, 0.15))
    return d


# ---------------------------------------------------------------- anchors
A["anchors"] = dict(
    A0_score_maxdiff=float(max(r["A0_score_maxdiff"] for r in rows)),
    A0_masks_match=bool(all(r["A0_maskA_match"] and r["A0_maskB_match"] for r in rows)),
    A1_ord_learned_either=rate("A1_ord_learned_either"),
    A1_ord_learned_designated=rate("A1_ord_learned_designated"),
    A1_target_round4="51/60",
    A2_rout_true_designated=rate("A2_rout_true_designated"),
    NEG_single_random_mask=rate("NEG_rand_single_TRUE"))

# ---------------------------------------------------------------- per-candidate marginals
cand_names = list(rows[0]["rout_cand"].keys())
marg = {}
for tag in ("ord", "rout"):
    marg[tag] = {}
    for cname in rows[0][f"{tag}_cand"]:
        okT = sum(1 for r in rows if r[f"{tag}_cand"][cname][0] > 0.8 and r[f"{tag}_cand"][cname][1] > 0.9)
        okP = sum(1 for r in rows if r[f"{tag}_cand"][cname][0] < 0.2 and r[f"{tag}_cand"][cname][1] > 0.9)
        al = np.array([r[f"{tag}_cand"][cname][0] for r in rows])
        marg[tag][cname] = dict(TRUE=[okT, n], PROXY=[okP, n], mean_align=round(float(al.mean()), 4))
A["candidate_marginals"] = marg

# ---------------------------------------------------------------- PRIMARY 1 (S1)
A["S1_primary"] = dict(
    target_TRUE=compare("S1_rout_TRUE_ok", "S1_ord_TRUE_ok", ("ROUT+select", "ORD+select")),
    target_PROXY=compare("S1_rout_PROXY_ok", "S1_ord_PROXY_ok", ("ROUT+select", "ORD+select")),
    rout_TRUE=rate("S1_rout_TRUE_ok"), ord_TRUE=rate("S1_ord_TRUE_ok"),
    rout_PROXY=rate("S1_rout_PROXY_ok"), ord_PROXY=rate("S1_ord_PROXY_ok"),
    picks_ord_TRUE={p: sum(1 for r in rows if r["S1_ord_TRUE_pick"] == p) for p in
                    sorted({r["S1_ord_TRUE_pick"] for r in rows})},
    picks_rout_TRUE={p: sum(1 for r in rows if r["S1_rout_TRUE_pick"] == p) for p in
                     sorted({r["S1_rout_TRUE_pick"] for r in rows})})
p1 = A["S1_primary"]["target_TRUE"]
A["S1_primary"]["DECISION"] = ("ROUTING WINS" if (p1["mcnemar"]["p_exact"] < 0.0167 and p1["boot95"]["ci"][0] > 0)
                               else "ROUTING LOSES" if p1["boot95"]["ci"][1] < 0
                               else ("NO DIFFERENCE (equivalent at d=0.15)" if p1["tost_d15"]["equivalent"]
                                     else "INCONCLUSIVE (CI straddles 0, TOST fails)"))

A["S1a_designated_m0"] = {k: rate(k) for k in
                          ["S1a_ord_TRUE_ok", "S1a_ord_PROXY_ok", "S1a_rout_TRUE_ok", "S1a_rout_PROXY_ok"]}
A["S1a_designated_m0"]["cmp_TRUE"] = compare("S1a_rout_TRUE_ok", "S1a_ord_TRUE_ok", ("ROUT_true", "ORD_learned"))
A["S1a_designated_m0"]["cmp_PROXY"] = compare("S1a_rout_PROXY_ok", "S1a_ord_PROXY_ok", ("ROUT_true", "ORD_learned"))

A["S1c_flip_TOY_ARTIFACT"] = {k: rate(k) for k in ["S1c_flip_TRUE_ok", "S1c_flip_PROXY_ok"]}

# ---------------------------------------------------------------- S1-b blind
blind = {}
for m in R["config"]["m_grid_blind"]:
    for tgt in ("TRUE", "PROXY"):
        blind[f"m{m}_{tgt}"] = dict(rout=rate(f"S1b_rout_m{m}_{tgt}_ok"), ord=rate(f"S1b_ord_m{m}_{tgt}_ok"),
                                    cmp=compare(f"S1b_rout_m{m}_{tgt}_ok", f"S1b_ord_m{m}_{tgt}_ok",
                                                ("ROUT_blind", "ORD_blind")))
blind["ord_option_exists_TRUE"] = rate("S1b_ord_exists_TRUE")
blind["ord_option_exists_PROXY"] = rate("S1b_ord_exists_PROXY")
blind["ord_n_good_masks_mean"] = round(float(col("S1b_ord_n_good_TRUE").mean()), 3)
A["S1b_blind"] = blind

# ---------------------------------------------------------------- PRIMARY 2 (S2)
def stage_stats(key):
    M = np.array([r[key] for r in rows], bool)          # n x 4
    k = int(M.sum()); tot = M.size
    per_stage = [int(M[:, j].sum()) for j in range(M.shape[1])]
    seedmean = M.mean(1)
    allfour = int((M.sum(1) == M.shape[1]).sum())
    return dict(k=k, n=tot, rate=round(k / tot, 4), wilson95=O.wilson(k, tot),
                per_stage=per_stage, all_stages_ok=[allfour, len(rows)],
                seed_mean=round(float(seedmean.mean()), 4)), seedmean


s2 = {}
s2["rout_select"], sm_rout = stage_stats("S2_rout_select_stages")
s2["ord_select"], sm_ord = stage_stats("S2_ord_select_stages")
for mode in ("mix", "pure"):
    for k in R["config"]["k_ladder"]:
        s2[f"ft{mode}_k{k}"], sm = stage_stats(f"S2_ft{mode}_k{k}_stages")
        s2[f"ft{mode}_k{k}"]["cost_examples_per_change"] = k * (808 if mode == "mix" else 8)
        s2[f"ft{mode}_k{k}"]["total_cost_3changes"] = int(O.COST_ORD_TRAIN + 3 * k * (808 if mode == "mix" else 8))
        s2[f"ft{mode}_k{k}"]["boot_vs_rout"] = O.boot_paired(sm, sm_rout)
        s2[f"ft{mode}_k{k}"]["hold_final_mean"] = round(float(np.mean([r[f"S2_ft{mode}_k{k}_hold"][-1] for r in rows])), 4)
s2["rout_select"]["total_cost_3changes"] = int(O.COST_ROUT_TRAIN)
s2["ord_select"]["total_cost_3changes"] = int(O.COST_ORD_TRAIN)
# smallest k in the ladder whose Wilson LB >= rout point estimate
target = s2["rout_select"]["rate"]
s2["kstar_mix"] = next((k for k in R["config"]["k_ladder"]
                        if s2[f"ftmix_k{k}"]["wilson95"][0] >= target), None)
s2["kstar_pure"] = next((k for k in R["config"]["k_ladder"]
                         if s2[f"ftpure_k{k}"]["wilson95"][0] >= target), None)
s2["still_switchable_after_ftmix_k20"] = rate("S2_ftmix_k20_still_switchable")
# NOTE (disclosed in FINDINGS): the prereg sentence "smallest k whose Wilson-95 LB >= the ROUT
# point estimate" is UNSATISFIABLE when the routed arm is at 1.000, because a Wilson LB is never 1.
# Both operationalizations of "reaches the ROUT rate" are therefore reported; the natural one
# (rate >= rout rate) is used for the decision, which is the conservative direction (it can only
# make routing lose).
ks_lit = s2["kstar_mix"]
ks_nat = next((k for k in R["config"]["k_ladder"] if s2[f"ftmix_k{k}"]["rate"] >= s2["rout_select"]["rate"]), None)
s2["kstar_mix_natural_reading"] = ks_nat
s2["DECISION_literal_wilsonLB_rule"] = ("ROUTING LOSES S2" if (ks_lit is not None and
                                        s2[f"ftmix_k{ks_lit}"]["total_cost_3changes"] < O.COST_ROUT_TRAIN)
                                        else "ROUTING WINS S2 (rule unsatisfiable at rate 1.0)")
s2["DECISION"] = ("ROUTING LOSES S2" if (ks_nat is not None and
                                         s2[f"ftmix_k{ks_nat}"]["total_cost_3changes"] < O.COST_ROUT_TRAIN)
                  else "ROUTING WINS S2")
s2["note_ord_select_is_cheapest"] = dict(rate=s2["ord_select"]["rate"], cost=int(O.COST_ORD_TRAIN))
A["S2_primary"] = s2

# ---------------------------------------------------------------- PRIMARY 3 (S3)
s3 = {}
for tag in ("ord", "rout"):
    for rule in ("R1", "R2", "R3"):
        r_ = rate(f"S3_{tag}_{rule}_ok")
        r_["p_vs_half"] = round(O.binom_vs_half(r_["k"], n), 4)
        r_["equivalent_to_chance_d15"] = bool(r_["wilson95"][0] > 0.35 and r_["wilson95"][1] < 0.65)
        r_["pickA_rate"] = round(float(col(f"S3_{tag}_{rule}_pickA").mean()), 4)
        s3[f"{tag}_{rule}"] = r_
    s3[f"{tag}_option_exists"] = rate(f"S3_{tag}_exists")
r4_ = rate("S3_ord_R4_ok")
r4_["p_vs_half"] = round(O.binom_vs_half(r4_["k"], n), 4)
r4_["mean_align"] = round(float(col("S3_ord_R4_align").mean()), 4)
r4_["mean_align_before"] = round(float(col("S3_ord_asis_align").mean()), 4)
s3["ord_R4_selftrain"] = r4_
sig = [k for k, v in s3.items() if isinstance(v, dict) and v.get("p_vs_half", 1) < 0.0042
       and v["rate"] > 0.5 and k.startswith("rout")]
s3["DECISION"] = "ROUTING WINS S3" if sig else "NO UNLABELLED RULE BEATS CHANCE"
A["S3_primary"] = s3

# ---------------------------------------------------------------- cost ledger
A["cost_ledger"] = {
    "ORD_train_example_forwards": O.COST_ORD_TRAIN,
    "ROUT_train_example_forwards": O.COST_ROUT_TRAIN,
    "routing_premium_pct": 100.0,
    "ftmix_step_cost": 808, "ftpure_step_cost": 8,
    "mask_derivation_needs": "cue oracle + 4000 unlabelled train-mode inputs (no labels, no gradients)",
}

# ---------------------------------------------------------------- POST-HOC addendum
add = {}
for k in ("S1_ord_exists_TRUE", "S1_ord_exists_PROXY", "S1_rout_exists_TRUE", "S1_rout_exists_PROXY"):
    add[k] = rate(k)
occB = np.array([r["ord_cand"]["occB"] for r in rows]); occA = np.array([r["ord_cand"]["occA"] for r in rows])
okT = (occB[:, 0] > 0.8) & (occB[:, 1] > 0.9); okP = (occA[:, 0] < 0.2) & (occA[:, 1] > 0.9)
add["POSTHOC_ord_occB_designated_TRUE"] = [int(okT.sum()), n]
add["POSTHOC_ord_occA_designated_PROXY"] = [int(okP.sum()), n]
add["POSTHOC_mcnemar_rout_vs_ordoccB_TRUE"] = O.mcnemar_exact(col("S1a_rout_TRUE_ok").astype(bool), okT)
add["S2_discordant_k100_vs_rout"] = O.mcnemar_exact(
    np.array([all(r["S2_ftmix_k100_stages"]) for r in rows]),
    np.array([all(r["S2_rout_select_stages"]) for r in rows]))
add["rout_unablated_align_mean"] = round(float(np.mean([r["rout_cand"]["none"][0] for r in rows])), 4)
add["ord_unablated_align_mean"] = round(float(np.mean([r["ord_cand"]["none"][0] for r in rows])), 4)
add["ord_unablated_committed_either"] = int(sum(1 for r in rows if r["ord_cand"]["none"][0] > 0.8
                                                or r["ord_cand"]["none"][0] < 0.2))
add["rout_unablated_committed_either"] = int(sum(1 for r in rows if r["rout_cand"]["none"][0] > 0.8
                                                 or r["rout_cand"]["none"][0] < 0.2))
A["ADDENDUM_posthoc"] = add

json.dump(A, open("/Users/steven/research_sprint_20260724/r5-optionality-case/analysis.json", "w"), indent=1)

# ---------------------------------------------------------------- print
print("== ANCHORS", json.dumps(A["anchors"], indent=1))
print("\n== S1 PRIMARY (no-retrain, cue oracle, m=8)")
for k in ("rout_TRUE", "ord_TRUE", "rout_PROXY", "ord_PROXY"):
    print(f"  {k:12s} {A['S1_primary'][k]['k']}/{n} {A['S1_primary'][k]['wilson95']}")
print("  cmp TRUE :", json.dumps(A["S1_primary"]["target_TRUE"]))
print("  cmp PROXY:", json.dumps(A["S1_primary"]["target_PROXY"]))
print("  DECISION:", A["S1_primary"]["DECISION"])
print("  picks ord:", A["S1_primary"]["picks_ord_TRUE"], " rout:", A["S1_primary"]["picks_rout_TRUE"])
print("\n== S1a designated, zero labels")
for k, v in A["S1a_designated_m0"].items():
    print("  ", k, json.dumps(v))
print("\n== S1b BLIND")
for k, v in A["S1b_blind"].items():
    print("  ", k, json.dumps(v))
print("\n== S1c FLIP (toy artifact)", json.dumps(A["S1c_flip_TOY_ARTIFACT"]))
print("\n== S2 PRIMARY (4 stages = 3 changes)")
for k, v in A["S2_primary"].items():
    print("  ", k, json.dumps(v))
print("\n== S3 PRIMARY")
for k, v in A["S3_primary"].items():
    print("  ", k, json.dumps(v))
print("\n== candidate marginals")
for tag in ("ord", "rout"):
    print(" ", tag, json.dumps(A["candidate_marginals"][tag]))
