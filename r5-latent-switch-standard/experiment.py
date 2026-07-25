"""experiment.py -- CONFIRMATORY run for PREREG.md (r5-latent-switch-standard).

Every model trained here is STANDARD-trained (no routing, no occluded loss, no training mask).
The only manipulation is at SCORING time: a data-chosen ablation mask vs a random balanced mask
given 10x the search freedom.

Arms, in the preregistered priority order:
  TF-A   transformer, agreeing data, seeds 700-729                 (PRIMARY)
  MLP-W  MLP width sweep H in {8,16,32,64}, seeds 700-729          (S1, anchor A0 at H=16)
  TF-N   transformer, DISAGREEING data f=0.25, seeds 700-715       (S2)
  MLP-N  MLP disagreeing f in {0.05,0.25} x H in {16,64}           (S2)
  MLP-K3 MLP k=3 cues, H in {24,48}                                (S3)

Writes results.json.
"""
import json, sys, time
import numpy as np
import torch
from scipy.stats import fisher_exact, binomtest, norm

DIR = "/Users/steven/research_sprint_20260724/r5-latent-switch-standard"
sys.path.insert(0, DIR)
import core_tf as CT
import core_mlp as CM

OUT = f"{DIR}/results.json"
DEV = torch.device("mps")
KRAND = 10                                   # random control gets 10 draws; method gets 1
TF_SEEDS = list(range(700, 730))
TF_N_SEEDS = list(range(700, 716))
MLP_SEEDS = list(range(700, 730))
TF_BUDGET_STOP = 420.0
ARM_SKIP_AFTER = 780.0
T0 = time.time()


# ------------------------------- statistics -------------------------------
def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar(meth, rand):
    """Exact McNemar: two-sided binomial on discordant pairs. meth/rand are bool lists."""
    b = int(sum(1 for m, r in zip(meth, rand) if m and not r))      # method only
    c = int(sum(1 for m, r in zip(meth, rand) if r and not m))      # control only
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return dict(b_method_only=b, c_control_only=c, n=len(meth), p=float(p))


def newcombe(k1, n1, k2, n2, z=1.96):
    """95% CI for p1 - p2 (Newcombe hybrid score)."""
    l1, u1 = wilson(k1, n1, z); l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - z * np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2) / z
    hi = d + z * np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2) / z
    return [round(max(-1.0, lo), 4), round(min(1.0, hi), 4)]


def cochran_armitage(ks, ns, scores):
    """One-sided (increasing) trend test on proportions."""
    ks = np.array(ks, float); ns = np.array(ns, float); s = np.array(scores, float)
    N = ns.sum(); K = ks.sum(); pbar = K / N
    sbar = (ns * s).sum() / N
    T = (ks * (s - sbar)).sum()
    var = pbar * (1 - pbar) * (ns * (s - sbar) ** 2).sum()
    z = T / np.sqrt(var) if var > 0 else 0.0
    return dict(z=round(float(z), 4), p_one_sided_increasing=float(1 - norm.cdf(z)))


def holm(named_ps):
    order = sorted(named_ps.items(), key=lambda kv: kv[1])
    out = {}; mx = 0.0
    for i, (nm, p) in enumerate(order):
        adj = min(1.0, (len(order) - i) * p)
        mx = max(mx, adj); out[nm] = round(mx, 6)
    return out


# ------------------------------- transformer arm -------------------------------
def run_tf(seeds, f, tag):
    """Standard-trained transformers; data-chosen mask vs random-best-of-10."""
    Xd, yd, _ = [t.to(DEV) for t in CT.gen(CT.NEVAL, CT.DEPLOY_SEED, "deploy")]
    Xh, yh, chh = [t.to(DEV) for t in CT.gen(CT.NEVAL, CT.HELD_SEED, "train")]  # agreeing, f=0
    fields = [CT.CUE_A_FIELD, CT.CUE_B_FIELD, CT.DIST_1_FIELD, CT.DIST_2_FIELD]
    fixA = CT.make_mask(DEV, CT.HEADS_A, CT.UNITS_A)
    fixB = CT.make_mask(DEV, CT.HEADS_B, CT.UNITS_B)
    rows, stopped = [], None
    for si, seed in enumerate(seeds):
        if tag == "TF-A" and time.time() - T0 > TF_BUDGET_STOP and si >= 1:
            stopped = f"preregistered budget rule fired at {time.time()-T0:.0f}s after {si} seeds"
            print("  STOP:", stopped, flush=True)
            break
        m, ids, y = CT.train_standard(seed, DEV, f)
        sens = CT.cue_sensitivity(m, Xh, chh, fields)
        (hA, uA), (hB, uB), (hD1, uD1), (hD2, uD2) = sens

        # --- data-chosen mask (held-out AGREEING data only) ---
        hsplitA, hsplitB = CT.split_from_score(hA - hB, CT.NHEAD // 2)
        usplitA, usplitB = CT.split_from_score(uA - uB, CT.DFF // 2)
        mA = CT.make_mask(DEV, hsplitA, usplitA); mB = CT.make_mask(DEV, hsplitB, usplitB)
        learned = CT.crit(m, mA, mB, Xd, yd, Xh, yh)

        # --- S4 distractor-cue negative control (same procedure, label-free fields) ---
        dhA, dhB = CT.split_from_score(hD1 - hD2, CT.NHEAD // 2)
        duA, duB = CT.split_from_score(uD1 - uD2, CT.DFF // 2)
        dist = CT.crit(m, CT.make_mask(DEV, dhA, duA), CT.make_mask(DEV, dhB, duB),
                       Xd, yd, Xh, yh)

        # --- round-3's arbitrary fixed partition (anchor A1 + the old floor) ---
        fixed = CT.crit(m, fixA, fixB, Xd, yd, Xh, yh)

        # --- FAIR CONTROL: random balanced masks, best of KRAND ---
        rng = np.random.default_rng([seed, 11, si])
        rnd_hits, rnd_any = 0, False
        for _ in range(KRAND):
            ph = rng.permutation(CT.NHEAD); pu = rng.permutation(CT.DFF)
            c = CT.crit(m, CT.make_mask(DEV, ph[:2], pu[:64]),
                        CT.make_mask(DEV, ph[2:], pu[64:]), Xd, yd, Xh, yh)
            rnd_hits += int(c["success"]); rnd_any = rnd_any or c["success"]

        r = dict(seed=seed,
                 held_full=CT.acc(m, Xh, yh), deploy_full=CT.acc(m, Xd, yd),
                 train_acc=CT.acc(m, ids, y),
                 learned=learned, fixed=fixed, distractor=dist,
                 rand_hits_of_K=rnd_hits, rand_best=bool(rnd_any),
                 learned_headsA=hsplitA, learned_unitsA_n=len(usplitA),
                 head_score=[round(float(x), 5) for x in (hA - hB)],
                 unit_score_absmean=round(float(np.abs(uA - uB).mean()), 6))
        rows.append(r)
        print(f"  {tag} s{seed} learned={learned['success']} (des {learned['designated']}) "
              f"fixed={fixed['success']} rand{rnd_hits}/{KRAND} dist={dist['success']} "
              f"dep_full={r['deploy_full']:.3f} [{time.time()-T0:.0f}s]", flush=True)
    return rows, stopped


def tf_summary(rows):
    n = len(rows)
    L = [r["learned"]["success"] for r in rows]
    R = [r["rand_best"] for r in rows]
    F = [r["fixed"]["success"] for r in rows]
    Dm = [r["distractor"]["success"] for r in rows]
    Ld = [r["learned"]["designated"] for r in rows]
    return dict(n=n, learned_k=int(sum(L)), learned_rate=round(sum(L) / max(1, n), 4),
                learned_wilson=wilson(sum(L), n),
                learned_designated_k=int(sum(Ld)),
                rand_best_k=int(sum(R)), rand_best_wilson=wilson(sum(R), n),
                rand_per_draw_k=int(sum(r["rand_hits_of_K"] for r in rows)),
                rand_per_draw_n=KRAND * n,
                fixed_k=int(sum(F)), distractor_k=int(sum(Dm)),
                mcnemar_learned_vs_randbest=mcnemar(L, R),
                mcnemar_learned_vs_distractor=mcnemar(L, Dm),
                mcnemar_learned_vs_fixed=mcnemar(L, F),
                newcombe_learned_minus_randbest=newcombe(sum(L), n, sum(R), n),
                held_full_min=round(min(r["held_full"] for r in rows), 4),
                n_competence_fail=int(sum(1 for r in rows if r["held_full"] <= CT.TRAIN_OK)),
                deploy_full=dict(
                    mean=round(float(np.mean([r["deploy_full"] for r in rows])), 4),
                    n_hi=int(sum(1 for r in rows if r["deploy_full"] > 0.8)),
                    n_lo=int(sum(1 for r in rows if r["deploy_full"] < 0.2)),
                    n_mid=int(sum(1 for r in rows if 0.2 <= r["deploy_full"] <= 0.8))),
                failure_clauses=fail_clauses(rows))


def fail_clauses(rows):
    """Pre-stated TF diagnostic: which clause of the criterion failed, per failing seed."""
    c = dict(A_deploy_not_hi=0, A_held_low=0, B_deploy_not_lo=0, B_held_low=0,
             either_half_incompetent=0, n_fail=0)
    for r in rows:
        d = r["learned"]
        if d["success"]: continue
        c["n_fail"] += 1
        # scored under whichever mapping is closer (designated orientation reported)
        c["A_deploy_not_hi"] += int(not (d["aA"] > CT.ALIGN_HI))
        c["A_held_low"] += int(not (d["tA"] > CT.TRAIN_OK))
        c["B_deploy_not_lo"] += int(not (d["aB"] < CT.ALIGN_LO))
        c["B_held_low"] += int(not (d["tB"] > CT.TRAIN_OK))
        c["either_half_incompetent"] += int(d["tA"] <= CT.TRAIN_OK or d["tB"] <= CT.TRAIN_OK)
    return c


# ------------------------------- MLP arms (k=2) -------------------------------
def run_mlp2(seeds, H, f, tag):
    inp = CM.inp_dim(2)
    Xd, yd = CM.gen_align(CM.NEVAL, CM.DEPLOY_SEED, "deploy")
    Xh, yh = CM.gen_align(CM.NEVAL, CM.HELD_SEED, "train")
    cc = CM.cue_cols(2); nz = CM.noise_cols(2)
    rows = []
    for si, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, ytr = CM.gen_align(CM.N, seed, "train", f)
        m = CM.train_standard(seed, H, inp, Xtr, ytr)
        kA, kB, score = CM.learned_masks_2(m, Xh, 2)
        learned = CM.crit2(m, H, kA, kB, Xd, yd, Xh, yh)
        # fixed arbitrary partition (round-4's original floor scoring)
        fixed = CM.crit2(m, H, list(range(H // 2)), list(range(H // 2, H)), Xd, yd, Xh, yh)
        # S4 distractor control: noise columns 4 vs 5
        sN = CM.sensitivity(m, Xh, [[nz[0]], [nz[1]]], 1)
        sc = (sN[0] - sN[1]).numpy()
        dA = sorted(np.argsort(-sc)[:H // 2].tolist()); dB = sorted(np.argsort(sc)[:H // 2].tolist())
        dist = CM.crit2(m, H, dA, dB, Xd, yd, Xh, yh)
        # FAIR CONTROL: random balanced split, best of KRAND
        rng = np.random.default_rng([seed, 22, H, int(100 * f)])
        hits, anyh = 0, False
        for _ in range(KRAND):
            p = rng.permutation(H)
            c = CM.crit2(m, H, sorted(p[:H // 2].tolist()), sorted(p[H // 2:].tolist()),
                         Xd, yd, Xh, yh)
            hits += int(c["success"]); anyh = anyh or c["success"]
        rows.append(dict(seed=seed, H=H, f=f,
                         train_acc=CM.accm(m, Xtr, ytr), full_deploy=CM.accm(m, Xd, yd),
                         full_held=CM.accm(m, Xh, yh),
                         learned=learned, fixed=fixed, distractor=dist,
                         rand_hits_of_K=hits, rand_best=bool(anyh),
                         learned_maskA=kA,
                         mask_is_full_split=bool(sorted(kA + kB) == list(range(H)))))
    print(f"  {tag} H={H} f={f}: learned {sum(r['learned']['success'] for r in rows)}/{len(rows)}"
          f"  randbest {sum(r['rand_best'] for r in rows)}/{len(rows)}"
          f"  fixed {sum(r['fixed']['success'] for r in rows)}/{len(rows)}"
          f"  dist {sum(r['distractor']['success'] for r in rows)}/{len(rows)} [{time.time()-T0:.0f}s]",
          flush=True)
    return rows


def mlp_summary(rows):
    n = len(rows)
    L = [r["learned"]["success"] for r in rows]; R = [r["rand_best"] for r in rows]
    F = [r["fixed"]["success"] for r in rows]; D = [r["distractor"]["success"] for r in rows]
    Ld = [r["learned"]["designated"] for r in rows]
    return dict(n=n, learned_k=int(sum(L)), learned_rate=round(sum(L) / n, 4),
                learned_wilson=wilson(sum(L), n), learned_designated_k=int(sum(Ld)),
                rand_best_k=int(sum(R)), rand_best_wilson=wilson(sum(R), n),
                rand_per_draw_k=int(sum(r["rand_hits_of_K"] for r in rows)),
                rand_per_draw_n=KRAND * n,
                fixed_k=int(sum(F)), distractor_k=int(sum(D)),
                mcnemar_learned_vs_randbest=mcnemar(L, R),
                mcnemar_learned_vs_distractor=mcnemar(L, D),
                newcombe_learned_minus_randbest=newcombe(sum(L), n, sum(R), n),
                all_masks_full_splits=bool(all(r["mask_is_full_split"] for r in rows)),
                n_competence_fail=int(sum(1 for r in rows if r["full_held"] <= CM.TRAIN_OK)))


# ------------------------------- MLP arm (k=3) -------------------------------
def run_mlp3(seeds, H, tag, k=3):
    inp = CM.inp_dim(k)
    Xh, yh = CM.gen_k(CM.NEVAL, CM.HELD_SEED, k, "train")
    probes = [CM.gen_k(CM.NEVAL, CM.DEPLOY_SEED + j, k, ("probe", j)) for j in range(k)]
    rows = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, ytr = CM.gen_k(CM.N, seed, k, "train")
        m = CM.train_standard(seed, H, inp, Xtr, ytr)
        groups, S = CM.learned_partition_k(m, Xh, k)
        sizes = [len(g) for g in groups]
        learned = CM.critk(m, H, groups, Xh, yh, probes)
        # fixed arbitrary partition into k equal blocks (the round-3 control scoring)
        w = H // k
        fixedg = [list(range(i * w, (i + 1) * w)) for i in range(k)]
        fixed = CM.critk(m, H, fixedg, Xh, yh, probes)
        # FAIR CONTROL: random partition with the SAME group sizes, best of KRAND
        rng = np.random.default_rng([seed, 33, H, k])
        hits, anyh = 0, False
        for _ in range(KRAND):
            p = rng.permutation(H).tolist(); gs, o = [], 0
            for s in sizes:
                gs.append(sorted(p[o:o + s])); o += s
            c = CM.critk(m, H, gs, Xh, yh, probes)
            hits += int(c["success"]); anyh = anyh or c["success"]
        rows.append(dict(seed=seed, H=H, k=k, group_sizes=sizes,
                         train_acc=CM.accm(m, Xtr, ytr), full_held=CM.accm(m, Xh, yh),
                         learned=learned, fixed=fixed,
                         rand_hits_of_K=hits, rand_best=bool(anyh)))
    print(f"  {tag} k={k} H={H}: learned {sum(r['learned']['success'] for r in rows)}/{len(rows)}"
          f"  randbest {sum(r['rand_best'] for r in rows)}/{len(rows)}"
          f"  fixed {sum(r['fixed']['success'] for r in rows)}/{len(rows)} [{time.time()-T0:.0f}s]",
          flush=True)
    return rows


def mlp3_summary(rows):
    n = len(rows)
    L = [r["learned"]["success"] for r in rows]; R = [r["rand_best"] for r in rows]
    F = [r["fixed"]["success"] for r in rows]
    return dict(n=n, learned_k=int(sum(L)), learned_rate=round(sum(L) / n, 4),
                learned_wilson=wilson(sum(L), n),
                learned_designated_k=int(sum(r["learned"]["designated"] for r in rows)),
                rand_best_k=int(sum(R)), rand_per_draw_k=int(sum(r["rand_hits_of_K"] for r in rows)),
                rand_per_draw_n=KRAND * n, fixed_k=int(sum(F)),
                mcnemar_learned_vs_randbest=mcnemar(L, R),
                newcombe_learned_minus_randbest=newcombe(sum(L), n, sum(R), n),
                mean_group_sizes=[round(float(np.mean([r["group_sizes"][i] for r in rows])), 2)
                                  for i in range(len(rows[0]["group_sizes"]))],
                n_empty_group=int(sum(1 for r in rows if r["learned"].get("empty_group"))))


# ------------------------------- anchors -------------------------------
def anchor_mlp(rows_H16):
    ref = json.load(open("/Users/steven/research_sprint_20260724/r4-break-tautology/results.json"))
    st = {x["seed"]: x for x in ref["arms"]["floor_control"]["per_seed"]}
    dmax, same = 0.0, True
    for r in rows_H16:
        s = st.get(r["seed"])
        if s is None: continue
        dmax = max(dmax,
                   abs(r["learned"]["aA"] - s["lrn_selA_align"]), abs(r["learned"]["tA"] - s["lrn_selA_train"]),
                   abs(r["learned"]["aB"] - s["lrn_selB_align"]), abs(r["learned"]["tB"] - s["lrn_selB_train"]),
                   abs(r["fixed"]["aA"] - s["selA_align"]), abs(r["fixed"]["tA"] - s["selA_train"]),
                   abs(r["fixed"]["aB"] - s["selB_align"]), abs(r["fixed"]["tB"] - s["selB_train"]))
        same = same and (r["learned"]["success"] == s["lrn_success"]) and (r["fixed"]["success"] == s["success"])
    return dict(n_compared=sum(1 for r in rows_H16 if r["seed"] in st),
                max_abs_diff=round(float(dmax), 9), identical_success_flags=bool(same),
                anchored=bool(dmax < 1e-6 and same))


def anchor_tf(rows):
    ref = json.load(open("/Users/steven/research_sprint_20260724/r3-lm-organism/results.json"))
    st = {x["seed"]: x for x in ref["arms"]["control"]["per_seed"]}
    dmax, nc = 0.0, 0
    for r in rows:
        s = st.get(r["seed"])
        if s is None: continue
        nc += 1
        dmax = max(dmax,
                   abs(r["fixed"]["aA"] - s["selA_deploy"]), abs(r["fixed"]["tA"] - s["selA_held"]),
                   abs(r["fixed"]["aB"] - s["selB_deploy"]), abs(r["fixed"]["tB"] - s["selB_held"]),
                   abs(r["deploy_full"] - s["deploy_full"]), abs(r["held_full"] - s["held_full"]))
    return dict(n_compared=nc, max_abs_diff=round(float(dmax), 9),
                anchored_at_1e3=bool(dmax < 1e-3))


# ------------------------------- main -------------------------------
def main():
    res = dict(config=dict(
        prereg="PREREG.md stamped Sat Jul 25 13:58:30 EDT 2026",
        K_random_draws=KRAND, tf_seeds=TF_SEEDS, tf_n_seeds=TF_N_SEEDS, mlp_seeds=MLP_SEEDS,
        tf_params=CT.nparams(CT.TinyTF()), tf_steps=CT.STEPS, mlp_epochs=CM.EPOCHS,
        thresholds=dict(hi=CM.ALIGN_HI, lo=CM.ALIGN_LO, train_ok=CM.TRAIN_OK),
        neval=CM.NEVAL, device=str(DEV)), arms={}, tests={})

    print("== TF-A (PRIMARY): transformer, agreeing data ==", flush=True)
    tfa, stopped = run_tf(TF_SEEDS, 0.0, "TF-A")
    res["arms"]["TF-A"] = dict(per_seed=tfa, summary=tf_summary(tfa), early_stop=stopped)
    res["anchor_A1_transformer"] = anchor_tf(tfa)

    print("== MLP-W: width sweep ==", flush=True)
    res["arms"]["MLP-W"] = {}
    widths = [8, 16, 32, 64]
    for H in widths:
        rows = run_mlp2(MLP_SEEDS, H, 0.0, "MLP-W")
        res["arms"]["MLP-W"][f"H{H}"] = dict(per_seed=rows, summary=mlp_summary(rows))
    res["anchor_A0_mlp"] = anchor_mlp(res["arms"]["MLP-W"]["H16"]["per_seed"])

    if time.time() - T0 < ARM_SKIP_AFTER:
        print("== TF-N: transformer, DISAGREEING data f=0.25 ==", flush=True)
        tfn, _ = run_tf(TF_N_SEEDS, 0.25, "TF-N")
        res["arms"]["TF-N"] = dict(per_seed=tfn, summary=tf_summary(tfn))
    else:
        res["arms"]["TF-N"] = dict(skipped=f"budget rule at {time.time()-T0:.0f}s")

    if time.time() - T0 < ARM_SKIP_AFTER:
        print("== MLP-N: disagreeing data ==", flush=True)
        res["arms"]["MLP-N"] = {}
        for H in [16, 64]:
            for f in [0.05, 0.25]:
                rows = run_mlp2(MLP_SEEDS, H, f, "MLP-N")
                res["arms"]["MLP-N"][f"H{H}_f{f}"] = dict(per_seed=rows, summary=mlp_summary(rows))
    else:
        res["arms"]["MLP-N"] = dict(skipped=f"budget rule at {time.time()-T0:.0f}s")

    if time.time() - T0 < ARM_SKIP_AFTER:
        print("== MLP-K3: three-way ==", flush=True)
        res["arms"]["MLP-K3"] = {}
        for H in [24, 48]:
            rows = run_mlp3(MLP_SEEDS, H, "MLP-K3")
            res["arms"]["MLP-K3"][f"H{H}"] = dict(per_seed=rows, summary=mlp3_summary(rows))
    else:
        res["arms"]["MLP-K3"] = dict(skipped=f"budget rule at {time.time()-T0:.0f}s")

    # ---------------- preregistered tests ----------------
    sa = res["arms"]["TF-A"]["summary"]
    mc = sa["mcnemar_learned_vs_randbest"]
    if sa["n"] < 20:
        decision = "NOT RUN (fewer than 20 seeds completed)"
    elif mc["p"] < 0.01 and sa["learned_k"] >= sa["n"] / 2:
        decision = "LATENT SWITCH PRESENT"
    elif mc["p"] < 0.01:
        decision = "PARTIAL"
    else:
        decision = "ABSENT"
    res["tests"]["PRIMARY_TF"] = dict(
        arm="TF-A", n=sa["n"], learned_k=sa["learned_k"], rand_best_k=sa["rand_best_k"],
        mcnemar=mc, learned_wilson=sa["learned_wilson"],
        newcombe_diff=sa["newcombe_learned_minus_randbest"],
        criterion_p_lt_0p01=bool(mc["p"] < 0.01),
        criterion_k_ge_half=bool(sa["learned_k"] >= sa["n"] / 2),
        decision=decision)

    # S1 width trend
    ws = res["arms"]["MLP-W"]
    ks = [ws[f"H{H}"]["summary"]["learned_k"] for H in widths]
    ns = [ws[f"H{H}"]["summary"]["n"] for H in widths]
    res["tests"]["S1_width_trend"] = dict(
        widths=widths, k=ks, n=ns, scores=[int(np.log2(H)) for H in widths],
        **cochran_armitage(ks, ns, [np.log2(H) for H in widths]),
        per_width_mcnemar={f"H{H}": ws[f"H{H}"]["summary"]["mcnemar_learned_vs_randbest"]
                           for H in widths},
        per_width_rand_per_draw={f"H{H}": [ws[f"H{H}"]["summary"]["rand_per_draw_k"],
                                           ws[f"H{H}"]["summary"]["rand_per_draw_n"]]
                                 for H in widths})

    # S2 specificity (disagreeing data)
    s2 = {}
    if "summary" in res["arms"].get("TF-N", {}):
        a, b = sa, res["arms"]["TF-N"]["summary"]
        p = fisher_exact([[a["learned_k"], a["n"] - a["learned_k"]],
                          [b["learned_k"], b["n"] - b["learned_k"]]])[1]
        s2["TF_f0_vs_f025"] = dict(f0=[a["learned_k"], a["n"]], f025=[b["learned_k"], b["n"]],
                                   p=float(p), f025_wilson=b["learned_wilson"],
                                   passes=bool(p < 0.05 and b["learned_wilson"][1] < 0.5))
    if "H16_f0.25" in res["arms"].get("MLP-N", {}):
        for H in [16, 64]:
            a = ws[f"H{H}"]["summary"]
            for f in [0.05, 0.25]:
                b = res["arms"]["MLP-N"][f"H{H}_f{f}"]["summary"]
                p = fisher_exact([[a["learned_k"], a["n"] - a["learned_k"]],
                                  [b["learned_k"], b["n"] - b["learned_k"]]])[1]
                s2[f"MLP_H{H}_f0_vs_f{f}"] = dict(
                    f0=[a["learned_k"], a["n"]], fpos=[b["learned_k"], b["n"]], p=float(p),
                    fpos_wilson=b["learned_wilson"],
                    passes=bool(p < 0.05 and b["learned_wilson"][1] < 0.5))
    res["tests"]["S2_specificity"] = s2

    # S3 multiway
    if "H24" in res["arms"].get("MLP-K3", {}):
        res["tests"]["S3_multiway"] = {
            f"H{H}": res["arms"]["MLP-K3"][f"H{H}"]["summary"]["mcnemar_learned_vs_randbest"]
            for H in [24, 48]}

    # S4 distractor negative control (all k=2 arms)
    s4 = dict(TF_A=dict(learned=sa["learned_k"], distractor=sa["distractor_k"], n=sa["n"],
                        mcnemar=sa["mcnemar_learned_vs_distractor"]))
    for H in widths:
        u = ws[f"H{H}"]["summary"]
        s4[f"MLP_H{H}"] = dict(learned=u["learned_k"], distractor=u["distractor_k"], n=u["n"],
                               mcnemar=u["mcnemar_learned_vs_distractor"])
    res["tests"]["S4_distractor_control"] = s4

    # S5 designated-only
    res["tests"]["S5_designated_only"] = dict(
        TF_A=[sa["learned_designated_k"], sa["n"]],
        **{f"MLP_H{H}": [ws[f"H{H}"]["summary"]["learned_designated_k"],
                         ws[f"H{H}"]["summary"]["n"]] for H in widths})

    # Holm within the secondary family
    fam = {"S1_trend": res["tests"]["S1_width_trend"]["p_one_sided_increasing"]}
    for nm, d in s2.items(): fam["S2_" + nm] = d["p"]
    if "S3_multiway" in res["tests"]:
        for nm, d in res["tests"]["S3_multiway"].items(): fam["S3_" + nm] = d["p"]
    for H in widths:
        fam[f"S1_mcnemar_H{H}"] = ws[f"H{H}"]["summary"]["mcnemar_learned_vs_randbest"]["p"]
    res["tests"]["holm_adjusted_secondary"] = holm(fam)

    res["runtime_sec"] = round(time.time() - T0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nPRIMARY:", json.dumps(res["tests"]["PRIMARY_TF"], indent=1))
    print("runtime", res["runtime_sec"], "s -> wrote", OUT)


if __name__ == "__main__":
    main()
