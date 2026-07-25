"""experiment.py -- CONFIRMATORY run for PREREG.md (r3-lm-organism).

Three arms x 12 seeds (700-711) on the tiny-transformer sequence organism in core.py:
  control        : standard training, ablations applied post hoc
  routed         : occlude the other cue + forward-ablate the other group (gradient routing)
  occlusion_only : the same occluded losses, no ablation masks (method control)

Everything scored exactly as preregistered. Writes results.json.
"""
import json, time
import numpy as np
import torch, torch.nn as nn
from scipy.stats import fisher_exact, chi2_contingency

from core import (gen, occlude, TinyTF, masks, acc, nparams, CUE_A_FIELD, CUE_B_FIELD,
                  D, NLAYER, NHEAD, DFF, T, V_VOCAB, HEADS_A, HEADS_B)

OUT = "/Users/steven/research_sprint_20260724/r3-lm-organism/results.json"
DEV = torch.device("mps")
SEEDS = list(range(700, 712))
STEPS, LR, WD, BATCH, POOL = 700, 8e-3, 0.01, 512, 4096
NEVAL = 4000
DEPLOY_SEED, HELD_SEED = 50001, 60001
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9
BUDGET_SKIP_SEC = 700.0   # preregistered downscope rule for the occlusion_only arm
torch.set_num_threads(2)


def flip_bit(ids, chunk):
    """Deterministically flip a field's bit by swapping the two tokens of its chunk."""
    out = ids.clone()
    r = torch.arange(len(ids), device=ids.device)
    a = out[r, chunk].clone()
    out[r, chunk] = out[r, chunk + 1]
    out[r, chunk + 1] = a
    return out


@torch.no_grad()
def preds(m, ids, mask=None):
    m.eval()
    hm, um = (None, None) if mask is None else mask
    p = m(ids, hm, um).argmax(1)
    m.train()
    return p


def reliance(m, ids, cA, cB, mask=None):
    """Counterfactual cue-reliance probe (preregistered secondary)."""
    p0 = preds(m, ids, mask)
    fa = (preds(m, flip_bit(ids, cA), mask) != p0).float().mean().item()
    fb = (preds(m, flip_bit(ids, cB), mask) != p0).float().mean().item()
    return fa, fb, (fa - fb) / (fa + fb + 1e-9)


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
        elif arm == "routed":
            loss = 0.5 * (lf(m(occlude(bi, bcB, CUE_B_FIELD, g), *mA), by)
                          + lf(m(occlude(bi, bcA, CUE_A_FIELD, g), *mB), by))
        elif arm == "occlusion_only":
            loss = 0.5 * (lf(m(occlude(bi, bcB, CUE_B_FIELD, g)), by)
                          + lf(m(occlude(bi, bcA, CUE_A_FIELD, g)), by))
        else:
            raise ValueError(arm)
        loss.backward(); opt.step(); sch.step()
    return m, ids, y


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def main():
    t0 = time.time()
    mA, mB = masks(DEV)
    Xd, yd, cAd, cBd = [t.to(DEV) for t in gen(NEVAL, DEPLOY_SEED, "deploy")]
    Xh, yh, cAh, cBh = [t.to(DEV) for t in gen(NEVAL, HELD_SEED, "train")]

    res = {"config": dict(seeds=SEEDS, steps=STEPS, lr=LR, wd=WD, batch=BATCH, pool=POOL,
                          d=D, layers=NLAYER, heads=NHEAD, dff=DFF, T=T, vocab=V_VOCAB,
                          params=nparams(TinyTF()), heads_A=HEADS_A, heads_B=HEADS_B,
                          n_eval=NEVAL, deploy_seed=DEPLOY_SEED, held_seed=HELD_SEED,
                          thresholds=dict(hi=ALIGN_HI, lo=ALIGN_LO, train_ok=TRAIN_OK),
                          device=str(DEV)),
           "arms": {}}

    arms = ["control", "routed", "occlusion_only"]
    for arm in arms:
        if arm == "occlusion_only" and time.time() - t0 > BUDGET_SKIP_SEC:
            res["occlusion_only_skipped"] = f"budget rule fired at {time.time()-t0:.0f}s"
            print("SKIP occlusion_only (preregistered budget rule)")
            break
        rows = []
        for seed in SEEDS:
            m, ids, y = train_arm(seed, arm, mA, mB)
            fa0, fb0, r0 = reliance(m, Xh, cAh, cBh, None)
            faA, fbA, rA = reliance(m, Xh, cAh, cBh, mA)
            faB, fbB, rB = reliance(m, Xh, cAh, cBh, mB)
            r = dict(seed=seed,
                     train_acc=acc(m, ids, y),
                     held_full=acc(m, Xh, yh),
                     deploy_full=acc(m, Xd, yd),
                     selA_held=acc(m, Xh, yh, mA), selA_deploy=acc(m, Xd, yd, mA),
                     selB_held=acc(m, Xh, yh, mB), selB_deploy=acc(m, Xd, yd, mB),
                     flipA_full=round(fa0, 4), flipB_full=round(fb0, 4), R_full=round(r0, 4),
                     flipA_selA=round(faA, 4), flipB_selA=round(fbA, 4), R_selA=round(rA, 4),
                     flipA_selB=round(faB, 4), flipB_selB=round(fbB, 4), R_selB=round(rB, 4))
            des = (r["selA_deploy"] > ALIGN_HI and r["selA_held"] > TRAIN_OK
                   and r["selB_deploy"] < ALIGN_LO and r["selB_held"] > TRAIN_OK)
            swp = (r["selB_deploy"] > ALIGN_HI and r["selB_held"] > TRAIN_OK
                   and r["selA_deploy"] < ALIGN_LO and r["selA_held"] > TRAIN_OK)
            r["success_designated"] = bool(des)
            r["success"] = bool(des or swp)
            rows.append(r)
            print(f"  [{arm}] s{seed} held={r['held_full']:.3f} dep={r['deploy_full']:.3f} "
                  f"selA(dep/held)={r['selA_deploy']:.3f}/{r['selA_held']:.3f} "
                  f"selB={r['selB_deploy']:.3f}/{r['selB_held']:.3f} "
                  f"R={r['R_full']:+.2f} ok={r['success']} ({time.time()-t0:.0f}s)")
        k = sum(x["success"] for x in rows); n = len(rows)
        dep = np.array([x["deploy_full"] for x in rows])
        res["arms"][arm] = dict(
            summary=dict(n=n, successes=k, rate=round(k / n, 4), wilson95=wilson(k, n),
                         successes_designated=sum(x["success_designated"] for x in rows),
                         deploy_full_mean=round(float(dep.mean()), 4),
                         deploy_full_std=round(float(dep.std(ddof=1)), 4),
                         deploy_full_min=round(float(dep.min()), 4),
                         deploy_full_max=round(float(dep.max()), 4),
                         n_hi=int((dep > ALIGN_HI).sum()), n_lo=int((dep < ALIGN_LO).sum()),
                         n_mid=int(((dep >= ALIGN_LO) & (dep <= ALIGN_HI)).sum()),
                         held_full_min=round(float(min(x["held_full"] for x in rows)), 4),
                         n_failed_competence=int(sum(x["held_full"] <= TRAIN_OK for x in rows)),
                         R_full_mean=round(float(np.mean([x["R_full"] for x in rows])), 4),
                         R_selA_mean=round(float(np.mean([x["R_selA"] for x in rows])), 4),
                         R_selB_mean=round(float(np.mean([x["R_selB"] for x in rows])), 4)),
            per_seed=rows)
        s = res["arms"][arm]["summary"]
        print(f"[{arm}] success {k}/{n} wilson={s['wilson95']} deploy hi/lo/mid="
              f"{s['n_hi']}/{s['n_lo']}/{s['n_mid']} mean={s['deploy_full_mean']} "
              f"({time.time()-t0:.0f}s)")

    # ---------- PRIMARY 1: seed lottery on the control arm ----------
    dep = np.array([x["deploy_full"] for x in res["arms"]["control"]["per_seed"]])
    corr = np.round(dep * NEVAL).astype(int)
    table = np.stack([corr, NEVAL - corr], 1)
    chi2, p_hom, dof, _ = chi2_contingency(table)
    n_hi, n_lo = int((dep > ALIGN_HI).sum()), int((dep < ALIGN_LO).sum())
    ok1a = bool(p_hom < 1e-3)
    ok1b = bool((n_hi + n_lo) >= 3 and n_hi >= 1 and n_lo >= 1)
    res["primary1_seed_lottery"] = dict(
        deploy_full=[round(float(x), 4) for x in dep],
        chi2=float(chi2), dof=int(dof), p_homogeneity=float(p_hom),
        n_hi=n_hi, n_lo=n_lo, n_mid=int(len(dep) - n_hi - n_lo),
        spread=dict(min=round(float(dep.min()), 4), max=round(float(dep.max()), 4),
                    std=round(float(dep.std(ddof=1)), 4)),
        criterion_1a_p_lt_1e3=ok1a, criterion_1b_commitment=ok1b,
        decision=("REPRODUCES" if (ok1a and ok1b) else
                  ("PARTIAL: dispersion without both-sided commitment" if ok1a else
                   "DOES NOT REPRODUCE")))

    # ---------- PRIMARY 2: routing -> selectability ----------
    def fisher(a, b):
        ka = res["arms"][a]["summary"]["successes"]; na = res["arms"][a]["summary"]["n"]
        kb = res["arms"][b]["summary"]["successes"]; nb = res["arms"][b]["summary"]["n"]
        odds, p = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
        return dict(arms=[a, b], table=[[ka, na - ka], [kb, nb - kb]], p=float(p))

    p2 = fisher("routed", "control")
    kr = res["arms"]["routed"]["summary"]["successes"]
    p2["criterion_p_lt_0p01"] = bool(p2["p"] < 0.01)
    p2["criterion_routed_ge_8_of_12"] = bool(kr >= 8)
    p2["decision"] = ("H2 SUPPORTED" if (p2["criterion_p_lt_0p01"] and p2["criterion_routed_ge_8_of_12"])
                      else "H2 NOT SUPPORTED (preregistered failure)")
    res["primary2_routing_selectability"] = p2

    res["secondary_tests"] = []
    if "occlusion_only" in res["arms"]:
        res["secondary_tests"].append(fisher("routed", "occlusion_only"))
        res["secondary_tests"].append(fisher("occlusion_only", "control"))

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nPRIMARY 1 (seed lottery, control):", json.dumps(res["primary1_seed_lottery"], indent=1))
    print("PRIMARY 2 (routed vs control):", json.dumps(p2, indent=1))
    for t in res["secondary_tests"]:
        print("SECONDARY", t["arms"], "p =", t["p"], t["table"])
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
