"""POST-HOC / EXPLORATORY (not in PREREG.md).

The preregistered patchwork census PASSED in the MLP (29/46 vs 0/60). Inspecting the
per-pattern profiles afterwards showed that in that substrate the census cannot separate
"per-input mosaic of two worldviews" from "confident class collapse": the 8 patterns split
into two groups of 4 by the label v, and within a v-group all four patterns carry the SAME
cue semantics (they differ only in which of the two bit-orderings encodes the XOR value).
r3 showed every routed column implements its cue's XOR exactly (44/44), so a routed model is
invariant within a v-group by construction -- therefore the ONLY profile split it can ever
produce is the v-split, which is exactly the signature of "always emit class c".

This script measures what the un-ablated routed model actually is:
  A1. majority-class rate on cue-disagreement (deploy) inputs, and its confidence,
  A2. un-ablated held-out AGREEING accuracy (is it still competent on-distribution?),
  A3. does the readout bias decide? (Delta = Delta_A + Delta_B + bias, exact for routed nets)
Also re-derives the strict WITHIN-v criterion from the saved profiles for both substrates
(transformer via a counts-based lower bound that is valid without retraining).
"""
import json, time
import numpy as np
import torch
from scipy.stats import fisher_exact

import mlp_stage as M

DIR = "/Users/steven/research_sprint_20260724/r4-patchwork-prereg"
OUT = f"{DIR}/results_posthoc.json"
V8 = [0, 0, 0, 0, 1, 1, 1, 1]


def within_v_exact(ps, vlab):
    """proven per-input mixing: same label, different input configuration, different cue followed"""
    for v in (0, 1):
        sub = [ps[k] for k in range(len(ps)) if vlab[k] == v]
        if any(p > 0.9 for p in sub) and any(p < 0.1 for p in sub):
            return True
    return False


def within_v_lower_bound(a, b, half):
    """counts-only PROOF of within-v mixing (valid without the per-pattern profile):
    under a pure v-split each v-group is homogeneous, so (det_true, det_proxy) must satisfy
    det_true = 0, or det_proxy = 0, or both <= half. Anything else proves mixing."""
    return bool(a > 0 and b > 0 and (a > half or b > half))


def class_bias(ps, vlab):
    q = np.array([ps[k] if vlab[k] == 1 else 1 - ps[k] for k in range(len(ps))])  # P(pred==1)
    return float(abs(q.mean() - 0.5) * 2), float(max(q.mean(), 1 - q.mean()))


def main():
    t0 = time.time()
    mlp = json.load(open(f"{DIR}/results_mlp.json"))
    S = mlp["S"]; seeds = mlp["config"]["seeds"]; rows = mlp["per_seed"]
    out = {"label": "POST-HOC / EXPLORATORY (not preregistered)"}

    # ---------- A0: criterion decomposition from saved profiles ----------
    def dec(arm, ss):
        pw = wv = 0; bias = []
        for s in ss:
            ps = rows[arm][str(s)]["p_patterns"]
            a = sum(p > 0.9 for p in ps); b = sum(p < 0.1 for p in ps)
            if a >= 1 and b >= 1: pw += 1
            if within_v_exact(ps, V8): wv += 1
            bias.append(class_bias(ps, V8))
        return dict(n=len(ss), patchwork=pw, within_v_genuine=wv,
                    median_class_bias=round(float(np.median([x[0] for x in bias])), 4),
                    median_majority_class_rate_over_patterns=round(float(np.median([x[1] for x in bias])), 4))
    dr, dc = dec("routed", S), dec("control", seeds)
    fe = fisher_exact([[dr["within_v_genuine"], dr["n"] - dr["within_v_genuine"]],
                       [dc["within_v_genuine"], dc["n"] - dc["within_v_genuine"]]], alternative="greater")
    out["A0_mlp_criterion_decomposition"] = dict(
        routed_S=dr, control=dc, within_v_fisher_one_sided_p=float(fe[1]),
        note="patchwork census replicates; the strict within-v (same-label) criterion does not")

    # ---------- transformer: counts-only lower bound ----------
    try:
        tf = json.load(open(f"{DIR}/results_tf.json"))
        tseeds = tf["config"]["seeds"]
        def tdec(arm, only_competent):
            n = wv = pw = 0
            for s in tseeds:
                r = tf["per_seed"][arm][str(s)]
                if only_competent and not r["competent"]: continue
                n += 1
                a, b = r["n_det_true"], r["n_det_proxy"]
                if a >= 1 and b >= 1: pw += 1
                if within_v_lower_bound(a, b, 30): wv += 1
            return dict(n=n, patchwork=pw, within_v_proven_lower_bound=wv)
        tr, tc = tdec("routed", True), tdec("control", False)
        fet = fisher_exact([[tr["within_v_proven_lower_bound"], tr["n"] - tr["within_v_proven_lower_bound"]],
                            [tc["within_v_proven_lower_bound"], tc["n"] - tc["within_v_proven_lower_bound"]]],
                           alternative="greater")
        fet2 = fisher_exact([[tc["within_v_proven_lower_bound"], tc["n"] - tc["within_v_proven_lower_bound"]],
                             [tr["within_v_proven_lower_bound"], tr["n"] - tr["within_v_proven_lower_bound"]]],
                            alternative="greater")
        out["A0b_transformer_within_v_lower_bound"] = dict(
            routed=tr, control=tc,
            fisher_routed_gt_control_p=float(fet[1]), fisher_control_gt_routed_p=float(fet2[1]),
            note="lower bound: counts alone prove mixing only when one det-count exceeds 30 while "
                 "the other is > 0; the true rates are >= these")
    except FileNotFoundError:
        out["A0b_transformer_within_v_lower_bound"] = "results_tf.json missing"

    # ---------- A1-A3: retrain the switchable routed seeds and measure the real mode ----------
    Xdep, ydep = M.gen_align(4000, M.DEPLOY_SEED, "deploy")
    Xhld, yhld = M.gen_align(4000, M.HELD_SEED, "train")
    yd = ydep.numpy()
    per = {}
    for s in S:
        m = M.train_routed(s, M.CUE_A, M.CUE_B)
        dA, dB, db, dfull = M.decompose(m, Xdep)
        with torch.no_grad():
            lg = m(Xdep); conf = torch.softmax(lg, 1).max(1).values.numpy()
            pred = lg.argmax(1).numpy()
        maj = float(max((pred == 0).mean(), (pred == 1).mean()))
        per[s] = dict(
            held_agree_unablated=M.accm(m, Xhld, yhld),
            deploy_align=M.accm(m, Xdep, ydep),
            majority_class_rate=maj,
            conf_on_majority_class=float(conf[pred == np.argmax([(pred == 0).mean(), (pred == 1).mean()])].mean()),
            logit_bias=db,
            mean_absdA=float(np.abs(dA).mean()), mean_absdB=float(np.abs(dB).mean()),
            opp_sign_rate=float((np.sign(dA) != np.sign(dB)).mean()),
            mean_abs_column_sum=float(np.abs(dA + dB).mean()),
            bias_decides_rate=float((np.sign(dfull) == np.sign(db)).mean()),
            bias_bigger_than_column_sum=float((np.abs(db) > np.abs(dA + dB)).mean()))
    g = lambda k: [per[s][k] for s in S]
    out["A1_A3_unablated_routed_mode"] = dict(
        n=len(S),
        median_held_agree_unablated=round(float(np.median(g("held_agree_unablated"))), 4),
        min_held_agree_unablated=round(float(np.min(g("held_agree_unablated"))), 4),
        frac_competent_gt_0p9=round(float(np.mean(np.array(g("held_agree_unablated")) > 0.9)), 4),
        median_deploy_align=round(float(np.median(g("deploy_align"))), 4),
        median_majority_class_rate=round(float(np.median(g("majority_class_rate"))), 4),
        iqr_majority_class_rate=[round(float(np.percentile(g("majority_class_rate"), 25)), 4),
                                 round(float(np.percentile(g("majority_class_rate"), 75)), 4)],
        median_conf_on_majority_class=round(float(np.median(g("conf_on_majority_class"))), 4),
        median_opp_sign_rate=round(float(np.median(g("opp_sign_rate"))), 4),
        median_abs_logit_bias=round(float(np.median(np.abs(g("logit_bias")))), 4),
        median_mean_abs_column_sum=round(float(np.median(g("mean_abs_column_sum"))), 4),
        median_bias_decides_rate=round(float(np.median(g("bias_decides_rate"))), 4),
        median_bias_bigger_than_column_sum=round(float(np.median(g("bias_bigger_than_column_sum"))), 4),
        per_seed=per)
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "A1_A3_unablated_routed_mode"}, indent=1))
    print(json.dumps({k: v for k, v in out["A1_A3_unablated_routed_mode"].items() if k != "per_seed"}, indent=1))
    print("wrote", OUT, out["runtime_sec"], "s")


if __name__ == "__main__":
    main()
