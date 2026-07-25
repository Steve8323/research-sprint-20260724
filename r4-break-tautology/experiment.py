"""
r4-break-tautology -- confirmatory run.  Everything here is specified in PREREG.md
(stamped Sat Jul 25 11:10:15 EDT 2026, before this file was executed).

Arms, in run order (most load-bearing first so a partial run is still interpretable):
  ceiling_disjoint_w8, floor_control, anchor_disjoint_w4,
  overlap25, overlap50, shared_trunk, soft25, overlap75, soft50
"""
import json, time, sys
import numpy as np
import torch
from scipy.stats import fisher_exact, norm

import core
from core import (gen_align, train_arm, make_spec, unit_mask, accm, learned_masks,
                  jaccard, allowed_l2, occlude, CUE_A, CUE_B, N,
                  ALIGN_HI, ALIGN_LO, TRAIN_OK)

OUT = "/Users/steven/research_sprint_20260724/r4-break-tautology/results.json"

SEEDS60 = list(range(700, 760))
SEEDS30 = list(range(700, 730))
VARIANTS = ["overlap25", "overlap50", "overlap75", "shared_trunk", "soft25", "soft50"]
ARM_ORDER = ["ceiling_disjoint_w8", "floor_control", "anchor_disjoint_w4",
             "overlap25", "overlap50", "shared_trunk", "soft25", "overlap75", "soft50"]
ALPHA_PRIMARY = 0.05 / 6
DELTA_EQUIV = 0.20


# ---------------- stats helpers ----------------
def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def newcombe_diff(k1, n1, k2, n2):
    """Newcombe hybrid-score 95% CI for p1 - p2."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    lo = (p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return [round(float(max(-1.0, lo)), 4), round(float(min(1.0, hi)), 4)]


def fisher(k1, n1, k2, n2):
    odds, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    return float(p)


def cochran_armitage(counts, ns, doses):
    """One-sided (decreasing-in-dose) trend test."""
    counts = np.array(counts, float); ns = np.array(ns, float); x = np.array(doses, float)
    Ntot = ns.sum(); R = counts.sum(); pbar = R / Ntot
    xbar = (ns * x).sum() / Ntot
    T = (counts * (x - xbar)).sum()
    var = pbar * (1 - pbar) * ((ns * x * x).sum() - (ns * x).sum() ** 2 / Ntot)
    if var <= 0:
        return dict(z=None, p_one_sided_decreasing=None, note="zero variance (degenerate counts)")
    z = T / np.sqrt(var)
    return dict(z=round(float(z), 4), p_one_sided_decreasing=round(float(norm.cdf(z)), 6))


def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(idx):
        v = (m - r) * pvals[i]; run = max(run, v); adj[i] = min(1.0, run)
    return [round(float(a), 6) for a in adj]


# ---------------- gradient trace (SECONDARY S3) ----------------
def gradient_trace(name, seed=700):
    """How many parameters receive nonzero gradient from BOTH worldview views?
    Measured at initialization, where the structural mask -- not convergence -- decides."""
    sp = make_spec(name); H, SA, SB = sp["H"], sp["SA"], sp["SB"]
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = core.Net(H, mask_l1=sp["mask_l1"])
    allow = allowed_l2(H, [SA, SB])
    if sp["zero_cross"]:
        with torch.no_grad():
            m.l2.weight[~allow] = 0.0
    lf = torch.nn.CrossEntropyLoss()
    trA = unit_mask(H, SA, SB, sp["eps"]); trB = unit_mask(H, SB, SA, sp["eps"])

    def grads(view):
        m.zero_grad()
        if not sp["routed"]:
            loss = lf(m(Xtr), ytr)
        elif view == "A":
            loss = lf(m(occlude(Xtr, CUE_B), trA), ytr)
        else:
            loss = lf(m(occlude(Xtr, CUE_A), trB), ytr)
        loss.backward()
        return {n_: (p.grad.detach().clone() != 0) for n_, p in m.named_parameters()}

    torch.manual_seed(seed + 1); gA = grads("A")
    torch.manual_seed(seed + 1); gB = grads("B")
    per, shared, total = {}, 0, 0
    for k in gA:
        both = int((gA[k] & gB[k]).sum().item())
        per[k] = dict(shared=both, total=int(gA[k].numel()))
        shared += both; total += gA[k].numel()
    return dict(arm=name, shared_params=int(shared), total_params=int(total), per_tensor=per)


# ---------------- per-arm run ----------------
def score_row(m, sp, seed, Xdep, ydep, Xhld, yhld, Xtr):
    H = sp["H"]
    mkA = unit_mask(H, sp["SA"]); mkB = unit_mask(H, sp["SB"])   # EVAL masks are always hard
    r = dict(seed=seed,
             full_align=accm(m, Xdep, ydep),
             selA_align=accm(m, Xdep, ydep, mkA), selA_train=accm(m, Xhld, yhld, mkA),
             selB_align=accm(m, Xdep, ydep, mkB), selB_train=accm(m, Xhld, yhld, mkB))
    des = (r["selA_align"] > ALIGN_HI and r["selA_train"] > TRAIN_OK
           and r["selB_align"] < ALIGN_LO and r["selB_train"] > TRAIN_OK)
    swp = (r["selB_align"] > ALIGN_HI and r["selB_train"] > TRAIN_OK
           and r["selA_align"] < ALIGN_LO and r["selA_train"] > TRAIN_OK)
    r["success_designated"] = bool(des); r["success_either"] = bool(des or swp)
    r["success"] = bool(des) if sp["routed"] else bool(des or swp)
    # pre-stated failure diagnostic: was the miss a competence (capacity) miss?
    r["competence_miss"] = bool(r["selA_train"] <= TRAIN_OK or r["selB_train"] <= TRAIN_OK)

    # --- learned eval mask (train mask != eval mask) ---
    kA, kB, score = learned_masks(m, sp, Xhld)
    lA = unit_mask(H, kA); lB = unit_mask(H, kB)
    r["lrn_selA_align"] = accm(m, Xdep, ydep, lA); r["lrn_selA_train"] = accm(m, Xhld, yhld, lA)
    r["lrn_selB_align"] = accm(m, Xdep, ydep, lB); r["lrn_selB_train"] = accm(m, Xhld, yhld, lB)
    ldes = (r["lrn_selA_align"] > ALIGN_HI and r["lrn_selA_train"] > TRAIN_OK
            and r["lrn_selB_align"] < ALIGN_LO and r["lrn_selB_train"] > TRAIN_OK)
    lswp = (r["lrn_selB_align"] > ALIGN_HI and r["lrn_selB_train"] > TRAIN_OK
            and r["lrn_selA_align"] < ALIGN_LO and r["lrn_selA_train"] > TRAIN_OK)
    r["lrn_success"] = bool(ldes) if sp["routed"] else bool(ldes or lswp)
    r["lrn_maskA"] = kA; r["lrn_jaccard_A"] = round(jaccard(kA, sp["SA"]), 4)
    r["lrn_exact_A"] = bool(sorted(kA) == sorted(sp["SA"]))
    return r


def run_arm(name, seeds, Xdep, ydep, Xhld, yhld):
    sp = make_spec(name); rows = []
    t0 = time.time()
    for seed in seeds:
        m, sp_, Xtr, ytr, allow = train_arm(seed, name)
        r = score_row(m, sp, seed, Xdep, ydep, Xhld, yhld, Xtr)
        r["train_acc"] = accm(m, Xtr, ytr)
        with torch.no_grad():
            r["cross_block_maxabs"] = round(float(m.l2.weight[~allow].abs().max().item()) if (~allow).any() else 0.0, 6)
        rows.append(r)
    k = sum(r["success"] for r in rows); n = len(rows)
    kl = sum(r["lrn_success"] for r in rows)
    fa = np.array([r["full_align"] for r in rows])
    misses = [r for r in rows if not r["success"]]
    summ = dict(n=n, successes=k, rate=round(k / n, 4), wilson95=wilson(k, n),
                learned_successes=kl, learned_rate=round(kl / n, 4), learned_wilson95=wilson(kl, n),
                learned_mask_exact_match=int(sum(r["lrn_exact_A"] for r in rows)),
                learned_mask_jaccard_mean=round(float(np.mean([r["lrn_jaccard_A"] for r in rows])), 4),
                full_align_mean=round(float(fa.mean()), 4), full_align_std=round(float(fa.std()), 4),
                selA_align_med=round(float(np.median([r["selA_align"] for r in rows])), 4),
                selB_align_med=round(float(np.median([r["selB_align"] for r in rows])), 4),
                selA_train_med=round(float(np.median([r["selA_train"] for r in rows])), 4),
                selB_train_med=round(float(np.median([r["selB_train"] for r in rows])), 4),
                n_misses=len(misses),
                n_misses_competence=int(sum(r["competence_miss"] for r in misses)),
                cross_block_maxabs_med=round(float(np.median([r["cross_block_maxabs"] for r in rows])), 6),
                H=sp["H"], w=len(sp["SA"]), overlap_units=len(set(sp["SA"]) & set(sp["SB"])),
                runtime_sec=round(time.time() - t0, 1))
    print(f"[{name:20s}] true-mask {k}/{n}  learned-mask {kl}/{n}  "
          f"(miss/comp {summ['n_misses']}/{summ['n_misses_competence']})  {summ['runtime_sec']}s", flush=True)
    return dict(summary=summ, per_seed=rows)


def main():
    Xdep, ydep = gen_align(4000, 50001, "deploy")
    Xhld, yhld = gen_align(4000, 60001, "train")
    res = dict(prereg="PREREG.md stamped Sat Jul 25 11:10:15 EDT 2026",
               config=dict(EPOCHS=core.EPOCHS, LR=core.LR, WD=core.WD, N=core.N,
                           MARGIN=core.MARGIN, NOISE=core.NOISE,
                           thresholds=dict(hi=ALIGN_HI, lo=ALIGN_LO, train_ok=TRAIN_OK),
                           seeds60=[700, 759], seeds30=[700, 729],
                           alpha_primary=ALPHA_PRIMARY, delta_equiv=DELTA_EQUIV),
               arms={}, grad_trace={})
    t0 = time.time()
    for name in ARM_ORDER:
        seeds = SEEDS30 if name == "anchor_disjoint_w4" else SEEDS60
        res["arms"][name] = run_arm(name, seeds, Xdep, ydep, Xhld, yhld)
        res["grad_trace"][name] = gradient_trace(name)
        res["elapsed_sec"] = round(time.time() - t0, 1)
        json.dump(res, open(OUT, "w"), indent=1)   # incremental: partial runs stay usable

    # ---------- ANCHOR A0 ----------
    try:
        pub = json.load(open("/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"))
        pubrows = {r["seed"]: r for r in pub["arms"]["routed"]["per_seed"]}
        mine = {r["seed"]: r for r in res["arms"]["anchor_disjoint_w4"]["per_seed"]}
        keys = ["selA_align", "selA_train", "selB_align", "selB_train"]
        d = max(abs(pubrows[s][k] - mine[s][k]) for s in SEEDS30 for k in keys)
        same = all(bool(pubrows[s]["success"]) == bool(mine[s]["success"]) for s in SEEDS30)
        res["anchor_A0"] = dict(max_abs_diff=round(float(d), 9),
                                per_seed_success_identical=bool(same),
                                round1_successes=int(sum(pubrows[s]["success"] for s in SEEDS30)),
                                my_successes=int(sum(mine[s]["success"] for s in SEEDS30)),
                                anchored=bool(d < 1e-6 and same))
    except Exception as e:
        res["anchor_A0"] = dict(anchored=False, error=str(e))
    print("ANCHOR:", res["anchor_A0"], flush=True)

    # ---------- PRIMARY ----------
    fk, fn = (res["arms"]["floor_control"]["summary"]["successes"],
              res["arms"]["floor_control"]["summary"]["n"])
    ck, cn = (res["arms"]["ceiling_disjoint_w8"]["summary"]["successes"],
              res["arms"]["ceiling_disjoint_w8"]["summary"]["n"])
    prim = {}
    for v in VARIANTS:
        s = res["arms"][v]["summary"]; k, n = s["successes"], s["n"]
        p = fisher(k, n, fk, fn)
        broke = res["grad_trace"][v]["shared_params"] > res["grad_trace"]["ceiling_disjoint_w8"]["shared_params"]
        if not broke:
            verdict = "VOID (no tautology break: shared params not > ceiling)"
        elif p < ALPHA_PRIMARY and s["wilson95"][0] > 0.20:
            verdict = "SURVIVES"
        elif p < ALPHA_PRIMARY:
            verdict = "WEAKENED"
        else:
            verdict = "BROKEN"
        prim[v] = dict(k=k, n=n, rate=s["rate"], wilson95=s["wilson95"],
                       floor=[fk, fn], p_vs_floor=p, alpha=ALPHA_PRIMARY,
                       passes_primary=bool(p < ALPHA_PRIMARY),
                       shared_params=res["grad_trace"][v]["shared_params"],
                       verdict=verdict)
    res["primary"] = prim

    # ---------- S1: distance from ceiling ----------
    s1 = {}
    for v in VARIANTS:
        s = res["arms"][v]["summary"]; k, n = s["successes"], s["n"]
        ci = newcombe_diff(ck, cn, k, n)
        s1[v] = dict(ceiling=[ck, cn], variant=[k, n], p_vs_ceiling=fisher(k, n, ck, cn),
                     newcombe95_ceiling_minus_variant=ci, delta_equiv=DELTA_EQUIV,
                     equivalent_to_ceiling=bool(ci[1] < DELTA_EQUIV),
                     note=("equivalence asserted only when CI upper < delta; "
                           "otherwise this is a BOUND, not a null"))
    res["S1_vs_ceiling"] = s1

    # ---------- S2: overlap dose-response ----------
    doses = [0.0, 0.25, 0.50, 0.75]
    arms_d = ["ceiling_disjoint_w8", "overlap25", "overlap50", "overlap75"]
    cs = [res["arms"][a]["summary"]["successes"] for a in arms_d]
    ns = [res["arms"][a]["summary"]["n"] for a in arms_d]
    res["S2_trend"] = dict(arms=arms_d, doses=doses, successes=cs, ns=ns,
                           **cochran_armitage(cs, ns, doses))

    # ---------- S4: learned eval mask ----------
    routed = [a for a in ARM_ORDER if make_spec(a)["routed"]]
    lf_k = res["arms"]["floor_control"]["summary"]["learned_successes"]
    lf_n = res["arms"]["floor_control"]["summary"]["n"]
    ps = [fisher(res["arms"][a]["summary"]["learned_successes"], res["arms"][a]["summary"]["n"], lf_k, lf_n)
          for a in routed]
    adj = holm(ps)
    res["S4_learned_mask"] = dict(
        floor_learned=[lf_k, lf_n],
        arms={a: dict(k=res["arms"][a]["summary"]["learned_successes"],
                      n=res["arms"][a]["summary"]["n"],
                      rate=res["arms"][a]["summary"]["learned_rate"],
                      wilson95=res["arms"][a]["summary"]["learned_wilson95"],
                      exact_mask_match=res["arms"][a]["summary"]["learned_mask_exact_match"],
                      jaccard_mean=res["arms"][a]["summary"]["learned_mask_jaccard_mean"],
                      p_raw=p, p_holm=q, passes=bool(q < 0.05))
              for a, p, q in zip(routed, ps, adj)})

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n=== PRIMARY ===")
    for v, d in prim.items():
        print(f"  {v:14s} {d['k']:2d}/{d['n']}  rate {d['rate']:.3f} {d['wilson95']}  "
              f"p_vs_floor={d['p_vs_floor']:.3g}  shared={d['shared_params']}  -> {d['verdict']}")
    print("S2 trend:", res["S2_trend"])
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
