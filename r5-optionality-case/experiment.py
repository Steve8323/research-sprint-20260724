"""r5-optionality-case -- confirmatory run. See PREREG.md (stamped 2026-07-25 13:55:32 EDT).

Three deployment regimes, one primary each:
  S1 NO-RETRAIN      routed vs ordinary under the same no-gradient toolbox
  S2 UNKNOWN-TARGET  switch-by-ablation vs switch-by-fine-tune over 3 target changes
  S3 UNLABELLED-ONLY unlabelled selection rules vs chance
Writes only inside this directory.
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r5-optionality-case")
import opt as O
import core as C

OUT = "/Users/steven/research_sprint_20260724/r5-optionality-case/results.json"
M_PRIMARY = 8
M_GRID_BLIND = [1, 4, 8]
K_LADDER = [0, 5, 20, 100]
TARGETS = ["TRUE", "PROXY", "TRUE", "PROXY"]      # 4 stages = 3 changes


def build_cands(model, sp, Xh, blind_rng=None):
    """Cue-oracle candidate list in the prereg order, plus (for routed) the true partition."""
    H = sp["H"]
    score = O.sens_score(model, sp, Xh)
    cs = [O.Cand("none"),
          O.Cand("maskA_w8", O.topk_mask(score, H, 8, +1)),
          O.Cand("maskB_w8", O.topk_mask(score, H, 8, -1)),
          O.Cand("maskA_w4", O.topk_mask(score, H, 4, +1)),
          O.Cand("maskB_w4", O.topk_mask(score, H, 4, -1)),
          O.Cand("occB", view="occB"),
          O.Cand("occA", view="occA")]
    if sp["routed"]:
        cs += [O.Cand("trueA", C.unit_mask(H, sp["SA"])),
               O.Cand("trueB", C.unit_mask(H, sp["SB"]))]
    return cs, score


def random_cands(H, rng, n_pairs):
    """n_pairs random 8/8 splits, each contributing both signs -> 2*n_pairs candidates."""
    out = []
    for i in range(n_pairs):
        idx = rng.permutation(H)
        a, b = sorted(idx[:H // 2].tolist()), sorted(idx[H // 2:].tolist())
        out.append(O.Cand(f"rand{i}a", C.unit_mask(H, a)))
        out.append(O.Cand(f"rand{i}b", C.unit_mask(H, b)))
    return out


def metrics_for(model, dv, hv, sv, yd, yh, ys, cands):
    al, ho, sc = [], [], []
    for c in cands:
        a, h, s = O.cand_metrics(model, dv, hv, sv, yd, yh, ys, c)
        al.append(a); ho.append(h); sc.append(s)
    return np.array(al), np.array(ho), sc


def run_seed(seed, dep, hld, sel, n_pairs):
    (Xd, yd), (Xh, yh), (Xs, ys) = dep, hld, sel
    r = {"seed": seed}

    ord_m, sp_o, Xtr, ytr, _ = C.train_arm(seed, "floor_control")
    rout_m, sp_r, _, _, _ = C.train_arm(seed, "ceiling_disjoint_w8")

    dA, dB = O.occluded_views(Xd, seed)
    hA, hB = O.occluded_views(Xh, seed + 1)
    sA, sB = O.occluded_views(Xs, seed + 2)
    dv = {"raw": Xd, "occA": dA, "occB": dB}
    hv = {"raw": Xh, "occA": hA, "occB": hB}
    sv = {"raw": Xs, "occA": sA, "occB": sB}

    co, score_o = build_cands(ord_m, sp_o, Xh)
    cr, score_r = build_cands(rout_m, sp_r, Xh)

    # --- anchor A0: my mask == core.learned_masks at w=8 (same scoring set Xh) --------------
    kA, kB, sc_ref = C.learned_masks(ord_m, sp_o, Xh)
    r["A0_score_maxdiff"] = float(np.abs(np.array(sc_ref) - score_o).max())
    r["A0_maskA_match"] = bool(sorted(np.nonzero(co[1].umask.numpy())[0].tolist()) == sorted(kA))
    r["A0_maskB_match"] = bool(sorted(np.nonzero(co[2].umask.numpy())[0].tolist()) == sorted(kB))

    ao, ho_, so = metrics_for(ord_m, dv, hv, sv, yd, yh, ys, co)
    ar, hr_, sr = metrics_for(rout_m, dv, hv, sv, yd, yh, ys, cr)
    r["ord_cand"] = {c.name: [round(a, 4), round(h, 4)] for c, a, h in zip(co, ao, ho_)}
    r["rout_cand"] = {c.name: [round(a, 4), round(h, 4)] for c, a, h in zip(cr, ar, hr_)}

    # --- anchors A1/A2: reproduce round-4's bidirectional switch criterion ------------------
    def bidir(al, hh, ia, ib, either):
        des = (al[ia] > C.ALIGN_HI and hh[ia] > C.TRAIN_OK
               and al[ib] < C.ALIGN_LO and hh[ib] > C.TRAIN_OK)
        swp = (al[ib] > C.ALIGN_HI and hh[ib] > C.TRAIN_OK
               and al[ia] < C.ALIGN_LO and hh[ia] > C.TRAIN_OK)
        return bool(des or swp) if either else bool(des)
    r["A1_ord_learned_either"] = bidir(ao, ho_, 1, 2, True)
    r["A1_ord_learned_designated"] = bidir(ao, ho_, 1, 2, False)
    r["A2_rout_true_designated"] = bidir(ar, hr_, 7, 8, False)

    # ================= S1 PRIMARY: no-retrain, cue oracle, m=8, target TRUE ================
    for tag, cands, al, hh, sc in (("ord", co, ao, ho_, so), ("rout", cr, ar, hr_, sr)):
        for tgt in ("TRUE", "PROXY"):
            i, selacc = O.select(cands, sc, M_PRIMARY, tgt)
            r[f"S1_{tag}_{tgt}_pick"] = cands[i].name
            r[f"S1_{tag}_{tgt}_ok"] = O.success(al[i], hh[i], tgt)
            r[f"S1_{tag}_{tgt}_align"] = round(float(al[i]), 4)
        # option existence within the cue-oracle candidate set
        r[f"S1_{tag}_exists_TRUE"] = bool(any(O.success(a, h, "TRUE") for a, h in zip(al, hh)))
        r[f"S1_{tag}_exists_PROXY"] = bool(any(O.success(a, h, "PROXY") for a, h in zip(al, hh)))

    # S1-a: designated candidate, zero labels
    r["S1a_ord_TRUE_ok"] = O.success(ao[1], ho_[1], "TRUE")
    r["S1a_ord_PROXY_ok"] = O.success(ao[2], ho_[2], "PROXY")
    r["S1a_rout_TRUE_ok"] = O.success(ar[7], hr_[7], "TRUE")
    r["S1a_rout_PROXY_ok"] = O.success(ar[8], hr_[8], "PROXY")

    # S1-c: TOY-ARTIFACT logit flip (ordinary model, m=8 chooses flip vs no-flip)
    cflip = [O.Cand("none"), O.Cand("flip", flip=True)]
    af, hf, sf = metrics_for(ord_m, dv, hv, sv, yd, yh, ys, cflip)
    for tgt in ("TRUE", "PROXY"):
        i, _ = O.select(cflip, sf, M_PRIMARY, tgt)
        r[f"S1c_flip_{tgt}_ok"] = O.success(af[i], hf[i], tgt)

    # ================= S1-b: blind (no cue oracle at deployment) ===========================
    rng = np.random.default_rng(seed * 17 + 3)
    cb_ord = [O.Cand("none")] + random_cands(sp_o["H"], rng, n_pairs)
    ab, hb, sb = metrics_for(ord_m, dv, hv, sv, yd, yh, ys, cb_ord)
    cb_rout = [cr[0], cr[7], cr[8]]
    abr, hbr, sbr = metrics_for(rout_m, dv, hv, sv, yd, yh, ys, cb_rout)
    for m in M_GRID_BLIND:
        for tgt in ("TRUE", "PROXY"):
            i, _ = O.select(cb_ord, sb, m, tgt)
            r[f"S1b_ord_m{m}_{tgt}_ok"] = O.success(ab[i], hb[i], tgt)
            j, _ = O.select(cb_rout, sbr, m, tgt)
            r[f"S1b_rout_m{m}_{tgt}_ok"] = O.success(abr[j], hbr[j], tgt)
    r["S1b_ord_exists_TRUE"] = bool(any(O.success(a, h, "TRUE") for a, h in zip(ab, hb)))
    r["S1b_ord_exists_PROXY"] = bool(any(O.success(a, h, "PROXY") for a, h in zip(ab, hb)))
    r["S1b_ord_n_good_TRUE"] = int(sum(O.success(a, h, "TRUE") for a, h in zip(ab, hb)))
    # NEG control: one random mask, no selection
    r["NEG_rand_single_TRUE"] = O.success(ab[1], hb[1], "TRUE")

    # ================= S2: unknown target, 4 stages ========================================
    Xsup, ysup = Xs[:M_PRIMARY], ys[:M_PRIMARY]
    for tag, cands, al, hh, sc in (("ord_select", co, ao, ho_, so), ("rout_select", cr, ar, hr_, sr)):
        oks = []
        for tgt in TARGETS:
            i, _ = O.select(cands, sc, M_PRIMARY, tgt)
            oks.append(O.success(al[i], hh[i], tgt))
        r[f"S2_{tag}_stages"] = oks

    for mode in ("mix", "pure"):
        for k in K_LADDER:
            cur = ord_m; oks = []; holds = []; aligns = []
            for tgt in TARGETS:
                ylab = ysup if tgt == "TRUE" else (1 - ysup)
                if mode == "mix":
                    Xft = torch.cat([Xtr, Xsup], 0); yft = torch.cat([ytr, ylab], 0)
                else:
                    Xft, yft = Xsup, ylab
                cur = O.finetune(cur, Xft, yft, k)
                a = C.accm(cur, Xd, yd); h = C.accm(cur, Xh, yh)
                oks.append(O.success(a, h, tgt)); holds.append(round(h, 4)); aligns.append(round(a, 4))
            r[f"S2_ft{mode}_k{k}_stages"] = oks
            r[f"S2_ft{mode}_k{k}_hold"] = holds
            r[f"S2_ft{mode}_k{k}_align"] = aligns
            if mode == "mix" and k == 20:
                # secondary: is the fine-tuned model still switchable afterwards?
                sc2 = O.sens_score(cur, sp_o, Xh)
                mA = O.topk_mask(sc2, sp_o["H"], 8, +1); mB = O.topk_mask(sc2, sp_o["H"], 8, -1)
                aa, ab_ = C.accm(cur, Xd, yd, mA), C.accm(cur, Xd, yd, mB)
                ha, hb_ = C.accm(cur, Xh, yh, mA), C.accm(cur, Xh, yh, mB)
                r["S2_ftmix_k20_still_switchable"] = bidir(np.array([aa, ab_]), np.array([ha, hb_]), 0, 1, True)

    # ================= S3: unlabelled only =================================================
    coin = np.random.default_rng(seed * 31 + 5).integers(0, 2)
    for tag, model, cands, al, hh, ia, ib in (("ord", ord_m, co, ao, ho_, 1, 2),
                                              ("rout", rout_m, cr, ar, hr_, 7, 8)):
        with torch.no_grad():
            base = model(Xd).argmax(1)
            agr, conf = [], []
            for i in (ia, ib):
                lo = model(dv[cands[i].view], cands[i].umask)
                agr.append((lo.argmax(1) == base).float().mean().item())
                conf.append(torch.softmax(lo, 1).max(1).values.mean().item())
        pickR1 = ia if coin == 0 else ib
        pickR2 = ia if agr[0] >= agr[1] else ib
        pickR3 = ia if conf[0] >= conf[1] else ib
        for rule, p in (("R1", pickR1), ("R2", pickR2), ("R3", pickR3)):
            r[f"S3_{tag}_{rule}_ok"] = O.success(al[p], hh[p], "TRUE")
            r[f"S3_{tag}_{rule}_pickA"] = bool(p == ia)
        r[f"S3_{tag}_agree"] = [round(x, 4) for x in agr]
        r[f"S3_{tag}_conf"] = [round(x, 4) for x in conf]
        r[f"S3_{tag}_exists"] = bool(O.success(al[ia], hh[ia], "TRUE") or O.success(al[ib], hh[ib], "TRUE"))
    ent = O.entropy_min(ord_m, Xd[:2000], 100)
    r["S3_ord_R4_ok"] = O.success(C.accm(ent, Xd, yd), C.accm(ent, Xh, yh), "TRUE")
    r["S3_ord_R4_align"] = round(C.accm(ent, Xd, yd), 4)
    r["S3_ord_asis_align"] = round(float(ao[0]), 4)
    return r


def main():
    t0 = time.time()
    seeds = O.SEEDS
    n_pairs = O.N_RAND // 2
    if "--pilot" in sys.argv:
        seeds = seeds[:3]
    dep, hld, sel = O.eval_sets()
    rows = []
    out = dict(config=dict(seeds=seeds, m_primary=M_PRIMARY, m_grid_blind=M_GRID_BLIND,
                           k_ladder=K_LADDER, targets=TARGETS, n_rand=2 * n_pairs,
                           cost_ord_train=O.COST_ORD_TRAIN, cost_rout_train=O.COST_ROUT_TRAIN,
                           thresholds=dict(hi=C.ALIGN_HI, lo=C.ALIGN_LO, ok=C.TRAIN_OK)),
               per_seed=rows)
    for i, s in enumerate(seeds):
        rows.append(run_seed(s, dep, hld, sel, n_pairs))
        if (i + 1) % 5 == 0 or i == len(seeds) - 1:
            out["runtime_sec"] = round(time.time() - t0, 1)
            json.dump(out, open(OUT if "--pilot" not in sys.argv else OUT + ".pilot", "w"), indent=1)
            print(f"  seed {s} ({i+1}/{len(seeds)}) {time.time()-t0:.0f}s", flush=True)
    print("wrote", OUT, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
