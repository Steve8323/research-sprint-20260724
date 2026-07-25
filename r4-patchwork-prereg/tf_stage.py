"""r4-patchwork-prereg -- STAGE 2 (sequence/transformer substrate). Confirmatory run for PREREG.md.

Organism = r3-lm-organism/core.py copied byte-for-byte to tf_core.py; training loop and
hyperparameters copied from r3-lm-organism/experiment.py. Fresh seeds 3100-3123, fresh eval
seeds 50002/60002. Preregistered replication anchor on control seed 700.
"""
import json, time
import numpy as np
import torch, torch.nn as nn
from scipy.stats import fisher_exact, mannwhitneyu

import tf_core as C
from tf_core import gen, occlude, TinyTF, masks, acc, nparams, CUE_A_FIELD, CUE_B_FIELD

DIR = "/Users/steven/research_sprint_20260724/r4-patchwork-prereg"
OUT = f"{DIR}/results_tf.json"
DEV = torch.device("mps")
torch.set_num_threads(2)

SEEDS = list(range(3100, 3124))          # fresh, n = 24 per arm
STEPS, LR, WD, BATCH, POOL = 700, 8e-3, 0.01, 512, 4096
NEVAL = 4000
DEPLOY_SEED, HELD_SEED = 50002, 60002
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9
NOISE_RNG = 1999
NDRAW = 500
DET_HI, DET_LO = 0.9, 0.1
MIN_CLASS = 100
BUDGET_STOP_SEC = 800.0                  # preregistered downscope rule


def train_arm(seed, arm, mA, mB):
    torch.manual_seed(seed)
    ids, y, cA, cB = [t.to(DEV) for t in gen(POOL, seed, "train")]
    m = TinyTF().to(DEV)
    g = torch.Generator(device=DEV); g.manual_seed(seed + 99)
    gi = torch.Generator(device=DEV); gi.manual_seed(seed + 7)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.15)
    lf = nn.CrossEntropyLoss()
    for _ in range(STEPS):
        idx = torch.randint(0, POOL, (BATCH,), generator=gi, device=DEV)
        bi, by, bcA, bcB = ids[idx], y[idx], cA[idx], cB[idx]
        opt.zero_grad(set_to_none=True)
        if arm == "control":
            loss = lf(m(bi), by)
        else:
            loss = 0.5 * (lf(m(occlude(bi, bcB, CUE_B_FIELD, g), *mA), by)
                          + lf(m(occlude(bi, bcA, CUE_A_FIELD, g), *mB), by))
        loss.backward(); opt.step(); sch.step()
    return m


# ---------------- preregistered pattern set: K = 60, M = 500 ----------------
def build_patterns():
    """patterns = (v, slot of F0, slot of F1), 2 x 6 x 5 = 60.
    draws  = (4 distractor bits, assignment of fields 2..5 to the 4 remaining slots), 500,
    shared across all patterns and all models (rng seed NOISE_RNG)."""
    rng = np.random.default_rng(NOISE_RNG)
    dbits = rng.integers(0, 2, (NDRAW, 4))
    dperm = np.argsort(rng.random((NDRAW, 4)), axis=1)          # order of fields 2..5
    keys, ids = [], np.empty((0,), dtype=np.int64)
    all_ids = []
    for v in (0, 1):
        for pA in range(C.NFIELD):
            for pB in range(C.NFIELD):
                if pA == pB: continue
                rest = [s for s in range(C.NFIELD) if s not in (pA, pB)]
                perm = np.zeros((NDRAW, C.NFIELD), dtype=np.int64)   # field at each slot
                vals = np.zeros((NDRAW, C.NFIELD), dtype=np.int64)   # its bit
                perm[:, pA] = CUE_A_FIELD; vals[:, pA] = v
                perm[:, pB] = CUE_B_FIELD; vals[:, pB] = 1 - v
                for k, s in enumerate(rest):
                    perm[:, s] = 2 + dperm[:, k]
                    vals[:, s] = dbits[:, k]
                seq = np.empty((NDRAW, C.T), dtype=np.int64)
                seq[:, 0:2 * C.NFIELD:2] = np.where(vals == 1, perm, C.TOK_V)
                seq[:, 1:2 * C.NFIELD:2] = np.where(vals == 1, C.TOK_V, perm)
                seq[:, -1] = C.TOK_Q
                all_ids.append(seq); keys.append((v, pA, pB))
    return torch.from_numpy(np.stack(all_ids)), keys      # (60, 500, 13)


@torch.no_grad()
def determinism(m, pat_ids, keys):
    m.eval()
    K = pat_ids.shape[0]
    flat = pat_ids.reshape(-1, C.T)
    out = []
    for i in range(0, len(flat), 15000):
        out.append(m(flat[i:i + 15000]).argmax(1))
    pred = torch.cat(out).reshape(K, -1).cpu().numpy()
    m.train()
    ps = [float((pred[k] == keys[k][0]).mean()) for k in range(K)]
    return float(np.mean([abs(2 * p - 1) for p in ps])), ps


@torch.no_grad()
def logits(m, ids, mask=None):
    m.eval()
    hm, um = (None, None) if mask is None else mask
    o = m(ids, hm, um)
    m.train()
    return o


# ---- stats (same helpers as stage 1) ----
def boot_median(x, B=2000, seed=7, pct=(2.5, 97.5)):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return [None, None]
    rng = np.random.default_rng(seed)
    bs = np.median(rng.choice(x, (B, len(x)), replace=True), axis=1)
    return [round(float(np.percentile(bs, pct[0])), 4), round(float(np.percentile(bs, pct[1])), 4)]


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def auroc(score, pos):
    score = np.asarray(score, dtype=float); pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0: return None
    order = np.argsort(score, kind="mergesort"); s = score[order]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_auroc_models(scores, labels, B=2000, seed=11, pct=(2.5, 97.5)):
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    ip, ineg = np.where(labels)[0], np.where(~labels)[0]
    if len(ip) == 0 or len(ineg) == 0: return [None, None]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        a = rng.choice(ip, len(ip), replace=True); b = rng.choice(ineg, len(ineg), replace=True)
        idx = np.concatenate([a, b])
        out.append(auroc(scores[idx], np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])))
    return [round(float(np.percentile(out, pct[0])), 4), round(float(np.percentile(out, pct[1])), 4)]


def fisher_power(p1, p0, n1, n0, nsim=2000, alpha=0.01, seed=3):
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(nsim):
        a = rng.binomial(n1, p1); c = rng.binomial(n0, p0)
        if fisher_exact([[a, n1 - a], [c, n0 - c]], alternative="greater")[1] < alpha: hits += 1
    return round(hits / nsim, 3)


def main():
    t0 = time.time()
    mA, mB = masks(DEV)
    Xd, yd, _, _ = [t.to(DEV) for t in gen(NEVAL, DEPLOY_SEED, "deploy")]
    Xh, yh, _, _ = [t.to(DEV) for t in gen(NEVAL, HELD_SEED, "train")]
    ydn = yd.cpu().numpy()
    pat_ids, keys = build_patterns()
    pat_ids = pat_ids.to(DEV)
    STRICT = int(np.ceil(0.125 * len(keys)))
    print(f"patterns K={len(keys)} x M={NDRAW}; strict threshold {STRICT}/{len(keys)}")

    # ---------------- replication anchor: control seed 700 on r3's eval seeds ----------------
    Xd1, yd1, _, _ = [t.to(DEV) for t in gen(NEVAL, 50001, "deploy")]
    Xh1, yh1, _, _ = [t.to(DEV) for t in gen(NEVAL, 60001, "train")]
    ma = train_arm(700, "control", mA, mB)
    a_dep, a_held = acc(ma, Xd1, yd1), acc(ma, Xh1, yh1)
    anchor_pass = bool(abs(a_dep - 0.2215) <= 0.02 and a_held >= 0.99)
    print(f"ANCHOR control s700 deploy={a_dep:.4f} (r3 0.2215) held={a_held:.4f} -> "
          f"{'PASS' if anchor_pass else 'FAIL'} ({time.time()-t0:.0f}s)")

    def measure(m, arm):
        r = dict(held_full=acc(m, Xh, yh), deploy_full=acc(m, Xd, yd),
                 selA_deploy=acc(m, Xd, yd, mA), selA_held=acc(m, Xh, yh, mA),
                 selB_deploy=acc(m, Xd, yd, mB), selB_held=acc(m, Xh, yh, mB))
        r["designated"] = bool(r["selA_deploy"] > ALIGN_HI and r["selA_held"] > TRAIN_OK
                               and r["selB_deploy"] < ALIGN_LO and r["selB_held"] > TRAIN_OK)
        r["swapped"] = bool(r["selB_deploy"] > ALIGN_HI and r["selB_held"] > TRAIN_OK
                            and r["selA_deploy"] < ALIGN_LO and r["selA_held"] > TRAIN_OK)
        r["success"] = bool(r["designated"] or r["swapped"])
        r["competent"] = bool(r["held_full"] > 0.9)
        D, ps = determinism(m, pat_ids, keys)
        r["D"] = D
        r["n_det_true"] = int(sum(p > DET_HI for p in ps)); r["n_det_proxy"] = int(sum(p < DET_LO for p in ps))
        r["n_ambig"] = int(sum(DET_LO <= p <= DET_HI for p in ps))
        r["patchwork"] = bool(r["n_det_true"] >= 1 and r["n_det_proxy"] >= 1)
        r["patchwork_strict"] = bool(r["n_det_true"] >= STRICT and r["n_det_proxy"] >= STRICT)
        lg = logits(m, Xd)
        conf = torch.softmax(lg, 1).max(1).values.cpu().numpy()
        pred = lg.argmax(1).cpu().numpy()
        r["mean_conf"] = float(conf.mean())
        r["mean_abs_margin"] = float((lg[:, 1] - lg[:, 0]).abs().mean().item())
        pf = pred != ydn
        r["n_proxy_following"] = int(pf.sum())
        r["conf_on_proxy_following"] = float(conf[pf].mean()) if pf.any() else None
        r["conf_on_true_following"] = float(conf[~pf].mean()) if (~pf).any() else None
        ok = pf.sum() >= MIN_CLASS and (~pf).sum() >= MIN_CLASS
        r["m2_eligible"] = bool(ok)
        r["auroc_conf"] = auroc(-conf, pf) if ok else None
        if ok:
            pa = logits(m, Xd, mA).argmax(1).cpu().numpy()
            pb = logits(m, Xd, mB).argmax(1).cpu().numpy()
            r["auroc_mech"] = auroc((pred == pb).astype(float) - (pred == pa).astype(float), pf)
        return r

    rows = {"routed": {}, "control": {}}
    used = []
    for s in SEEDS:
        if time.time() - t0 > BUDGET_STOP_SEC:
            print(f"  DOWNSCOPE RULE fired at {time.time()-t0:.0f}s; stopping at n={len(used)}")
            break
        rows["control"][s] = measure(train_arm(s, "control", mA, mB), "control")
        rows["routed"][s] = measure(train_arm(s, "routed", mA, mB), "routed")
        used.append(s)
        print(f"  s{s} ctrl dep={rows['control'][s]['deploy_full']:.3f} D={rows['control'][s]['D']:.3f} "
              f"pw={rows['control'][s]['patchwork']} | routed dep={rows['routed'][s]['deploy_full']:.3f} "
              f"D={rows['routed'][s]['D']:.3f} pw={rows['routed'][s]['patchwork']} ok={rows['routed'][s]['success']} "
              f"({time.time()-t0:.0f}s)")

    Rt = [s for s in used if rows["routed"][s]["competent"]]
    Ct = list(used)
    R = lambda arm, s, k: rows[arm][s][k]

    res = dict(stage="transformer", label="CONFIRMATORY (PREREG.md)",
               config=dict(seeds=used, n_per_arm=len(used), K_patterns=len(keys), M_draws=NDRAW,
                           strict_threshold=STRICT, deploy_seed=DEPLOY_SEED, held_seed=HELD_SEED,
                           noise_rng=NOISE_RNG, steps=STEPS, params=nparams(TinyTF()), device=str(DEV)),
               replication_anchor=dict(control_s700_deploy=round(a_dep, 4), r3_value=0.2215,
                                       control_s700_held=round(a_held, 4), passed=anchor_pass),
               competence_gate=dict(n_routed_competent=len(Rt), n_routed_total=len(used),
                                    excluded=[s for s in used if not rows["routed"][s]["competent"]]),
               routed_selection_success=f"{sum(R('routed',s,'success') for s in Rt)}/{len(Rt)}",
               control_selection_success=f"{sum(R('control',s,'success') for s in Ct)}/{len(Ct)}",
               per_seed=rows)

    # ---------------- PRIMARY 2 ----------------
    Dr = [R("routed", s, "D") for s in Rt]; Dc = [R("control", s, "D") for s in Ct]
    ci_Dr = boot_median(Dr)
    kR = sum(R("routed", s, "patchwork") for s in Rt); kC = sum(R("control", s, "patchwork") for s in Ct)
    kRs = sum(R("routed", s, "patchwork_strict") for s in Rt); kCs = sum(R("control", s, "patchwork_strict") for s in Ct)
    fe = fisher_exact([[kR, len(Rt) - kR], [kC, len(Ct) - kC]], alternative="greater")
    fes = fisher_exact([[kRs, len(Rt) - kRs], [kCs, len(Ct) - kCs]], alternative="greater")
    h1a = bool(ci_Dr[0] > 0.5)
    h1b = bool(kR / len(Rt) >= 0.5 and wilson(kR, len(Rt))[0] > 0.25)
    h1c = bool(fe[1] < 0.01)
    mg_r = [R("routed", s, "mean_abs_margin") for s in Rt]; mg_c = [R("control", s, "mean_abs_margin") for s in Ct]
    res["PRIMARY2_patchwork_transformer"] = dict(
        n_routed=len(Rt), n_control=len(Ct),
        median_D_routed=round(float(np.median(Dr)), 4), median_D_routed_boot95=ci_Dr,
        median_D_control=round(float(np.median(Dc)), 4), median_D_control_boot95=boot_median(Dc),
        D_mannwhitney=dict(U=float(mannwhitneyu(Dr, Dc, alternative="greater").statistic),
                           p=float(mannwhitneyu(Dr, Dc, alternative="greater").pvalue)),
        patchwork_routed=f"{kR}/{len(Rt)}", patchwork_routed_wilson=wilson(kR, len(Rt)),
        patchwork_control=f"{kC}/{len(Ct)}", patchwork_control_wilson=wilson(kC, len(Ct)),
        fisher_one_sided_p=float(fe[1]),
        strict_patchwork=dict(routed=f"{kRs}/{len(Rt)}", control=f"{kCs}/{len(Ct)}",
                              threshold_each=STRICT, fisher_one_sided_p=float(fes[1])),
        H1a_not_a_hedger=h1a, H1b_patchwork_prevalent=h1b, H1c_routing_specific=h1c,
        VERDICT=("CONFIRMED" if (h1a and h1b and h1c) else ("DISCONFIRMED" if not h1b else "PARTIAL")),
        point_predictions=dict(median_D_ge_0p8=bool(np.median(Dr) >= 0.8),
                               margin_ratio=round(float(np.median(mg_r) / np.median(mg_c)), 4),
                               margin_ratio_ge_1p5=bool(np.median(mg_r) / np.median(mg_c) >= 1.5),
                               median_margin_routed=round(float(np.median(mg_r)), 4),
                               median_margin_control=round(float(np.median(mg_c)), 4)),
        deploy_alignment=dict(routed=[round(R("routed", s, "deploy_full"), 4) for s in Rt],
                              control=[round(R("control", s, "deploy_full"), 4) for s in Ct]),
        det_pattern_totals=dict(routed_det_true=int(sum(R("routed", s, "n_det_true") for s in Rt)),
                                routed_det_proxy=int(sum(R("routed", s, "n_det_proxy") for s in Rt)),
                                routed_ambig=int(sum(R("routed", s, "n_ambig") for s in Rt)),
                                control_det_true=int(sum(R("control", s, "n_det_true") for s in Ct)),
                                control_det_proxy=int(sum(R("control", s, "n_det_proxy") for s in Ct)),
                                control_ambig=int(sum(R("control", s, "n_ambig") for s in Ct)),
                                patterns_per_model=len(keys)),
        power=dict(fisher_alpha=0.01,
                   power_vs_0p7_vs_0=fisher_power(0.7, 0.0, len(Rt), len(Ct)),
                   power_vs_0p35_gap=fisher_power(min(1.0, 0.35 + kC / max(1, len(Ct))),
                                                  kC / max(1, len(Ct)), len(Rt), len(Ct))))
    print("PRIMARY2:", res["PRIMARY2_patchwork_transformer"]["VERDICT"],
          f"D={res['PRIMARY2_patchwork_transformer']['median_D_routed']} {ci_Dr}",
          f"patchwork {kR}/{len(Rt)} vs {kC}/{len(Ct)} p={fe[1]:.3g}",
          f"strict {kRs}/{len(Rt)} vs {kCs}/{len(Ct)} p={fes[1]:.3g}")

    # ---------------- PRIMARY 3 replication (transformer) ----------------
    pw = [s for s in Rt if R("routed", s, "patchwork") and R("routed", s, "m2_eligible")]
    n_excl = sum(1 for s in Rt if R("routed", s, "patchwork") and not R("routed", s, "m2_eligible"))
    a_conf = [R("routed", s, "auroc_conf") for s in pw]
    a_mech = [R("routed", s, "auroc_mech") for s in pw]
    hedg = [s for s in Ct if 0.2 <= R("control", s, "deploy_full") <= 0.8 and R("control", s, "m2_eligible")]
    a_hedg = [R("control", s, "auroc_conf") for s in hedg]
    ci95 = boot_median(a_conf); ci90 = boot_median(a_conf, pct=(5, 95))
    equiv = bool(a_conf and ci90[0] >= 0.45 and ci90[1] <= 0.55)
    mwh = mannwhitneyu(a_hedg, a_conf, alternative="greater") if (a_hedg and a_conf) else None
    pos_scores = [R("routed", s, "mean_conf") for s in Rt if R("routed", s, "patchwork")]
    neg_scores = [R("control", s, "mean_conf") for s in Ct if not R("control", s, "patchwork")]
    a_m1 = ci_m1 = m1_verdict = None
    if pos_scores and neg_scores:
        sc = np.array(pos_scores + neg_scores)
        lb = np.array([True] * len(pos_scores) + [False] * len(neg_scores))
        a_m1 = auroc(-sc, lb); ci_m1 = boot_auroc_models(-sc, lb)
        m1_verdict = ("USEFUL" if ci_m1[0] > 0.7 else
                      ("ANTI-INFORMATIVE" if ci_m1[1] < 0.5 else
                       ("UNINFORMATIVE" if ci_m1[0] <= 0.5 <= ci_m1[1] else "WEAK/INTERMEDIATE")))
    res["PRIMARY3_confidence_monitor_transformer"] = dict(
        M2_input_level=dict(n_models=len(pw), n_excluded_lt100=n_excl,
                            median_auroc_conf=round(float(np.median(a_conf)), 4) if a_conf else None,
                            boot95=ci95, boot90=ci90,
                            frac_models_gt_0p5=round(float(np.mean(np.array(a_conf) > 0.5)), 4) if a_conf else None,
                            sd_auroc=round(float(np.std(a_conf, ddof=1)), 4) if len(a_conf) > 1 else None,
                            equivalence_margin=0.05, equivalence_declared=equiv,
                            per_model=[round(a, 4) for a in a_conf]),
        M2_positive_control_mechanistic=dict(median_auroc=round(float(np.median(a_mech)), 4) if a_mech else None,
                                             boot95=boot_median(a_mech),
                                             passes_ge_0p9=bool(np.median(a_mech) >= 0.9) if a_mech else None),
        M2_negative_control_control_hedgers=dict(
            n_models=len(hedg), median_auroc=round(float(np.median(a_hedg)), 4) if a_hedg else None,
            boot95=boot_median(a_hedg),
            mannwhitney_hedger_gt_patchwork=(dict(U=float(mwh.statistic), p=float(mwh.pvalue)) if mwh else None),
            passes=bool(mwh is not None and np.median(a_hedg) > 0.5 and mwh.pvalue < 0.05)),
        M1_model_level=dict(n_pos=len(pos_scores), n_neg=len(neg_scores),
                            auroc_low_conf_flags_patchwork=round(a_m1, 4) if a_m1 is not None else None,
                            boot95=ci_m1, verdict=m1_verdict,
                            mean_conf_routed_patchwork=round(float(np.mean(pos_scores)), 4) if pos_scores else None,
                            mean_conf_control_nonpatchwork=round(float(np.mean(neg_scores)), 4) if neg_scores else None),
        headline=dict(mean_conf_on_proxy_following_inputs=round(
            float(np.mean([R("routed", s, "conf_on_proxy_following") for s in pw])), 4) if pw else None,
            mean_conf_on_true_following_inputs=round(
            float(np.mean([R("routed", s, "conf_on_true_following") for s in pw])), 4) if pw else None))
    m2 = res["PRIMARY3_confidence_monitor_transformer"]["M2_input_level"]
    print(f"PRIMARY3-tf M2: n={m2['n_models']} median AUROC={m2['median_auroc_conf']} 90%{m2['boot90']} equiv={equiv}"
          f" | mech {res['PRIMARY3_confidence_monitor_transformer']['M2_positive_control_mechanistic']['median_auroc']}"
          f" | M1 {m1_verdict} {a_m1}")

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
