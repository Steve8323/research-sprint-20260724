"""POST-HOC (labelled as such in FINDINGS.md): two extensions of the preregistered S1-b blind arm.

(1) LABEL-EXCHANGE CURVE. The prereg capped the blind label budget at m = 8. The blind control
    fails there not because the option is missing (it exists in 57/60 seeds) but because 8 labels
    cannot identify which of 200 random masks is the right one. Extend m to the full 32-example
    selection pool and read off how many labels the blind control needs to catch routing.
(2) TIE-BREAK ROBUSTNESS. The preregistered rule breaks ties by candidate order, which put `none`
    first; at m = 1 that produced a degenerate 0/60 for the routed PROXY target. Re-run every
    blind selection with a seeded RANDOM tie-break among the argmax set.

Nothing outside this directory is written. Models are re-derived deterministically (same seeds).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r5-optionality-case")
import opt as O
import core as C
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location(
    "r5experiment", "/Users/steven/research_sprint_20260724/r5-optionality-case/experiment.py")
_ex = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_ex)
build_cands, random_cands, metrics_for = _ex.build_cands, _ex.random_cands, _ex.metrics_for

M_GRID = [1, 2, 4, 8, 16, 32]
OUT = "/Users/steven/research_sprint_20260724/r5-optionality-case/posthoc.json"


def select_rt(cands, sel_correct, m, target, rng):
    accs = []
    for i in range(len(cands)):
        v = sel_correct[i][:m]
        accs.append(v.mean() if target == "TRUE" else (1.0 - v).mean())
    accs = np.array(accs)
    best = np.flatnonzero(accs >= accs.max() - 1e-12)
    return int(rng.choice(best))


def main():
    t0 = time.time()
    dep, hld, sel = O.eval_sets()
    (Xd, yd), (Xh, yh), (Xs, ys) = dep, hld, sel
    rows = []
    for seed in O.SEEDS:
        ord_m, sp_o, Xtr, ytr, _ = C.train_arm(seed, "floor_control")
        rout_m, sp_r, _, _, _ = C.train_arm(seed, "ceiling_disjoint_w8")
        dA, dB = O.occluded_views(Xd, seed); hA, hB = O.occluded_views(Xh, seed + 1)
        sA, sB = O.occluded_views(Xs, seed + 2)
        dv = {"raw": Xd, "occA": dA, "occB": dB}
        hv = {"raw": Xh, "occA": hA, "occB": hB}
        sv = {"raw": Xs, "occA": sA, "occB": sB}
        cr, _ = build_cands(rout_m, sp_r, Xh)
        rng_m = np.random.default_rng(seed * 17 + 3)          # identical stream to experiment.py
        cb_ord = [O.Cand("none")] + random_cands(sp_o["H"], rng_m, O.N_RAND // 2)
        cb_rout = [cr[0], cr[7], cr[8]]
        ab, hb, sb = metrics_for(ord_m, dv, hv, sv, yd, yh, ys, cb_ord)
        abr, hbr, sbr = metrics_for(rout_m, dv, hv, sv, yd, yh, ys, cb_rout)
        r = {"seed": seed,
             "n_good_TRUE": int(sum(O.success(a, h, "TRUE") for a, h in zip(ab, hb))),
             "n_good_PROXY": int(sum(O.success(a, h, "PROXY") for a, h in zip(ab, hb)))}
        rt = np.random.default_rng(seed * 101 + 9)
        for m in M_GRID:
            for tgt in ("TRUE", "PROXY"):
                i, _ = O.select(cb_ord, sb, m, tgt)
                r[f"ord_prereg_m{m}_{tgt}"] = O.success(ab[i], hb[i], tgt)
                j = select_rt(cb_ord, sb, m, tgt, rt)
                r[f"ord_rt_m{m}_{tgt}"] = O.success(ab[j], hb[j], tgt)
                i2, _ = O.select(cb_rout, sbr, m, tgt)
                r[f"rout_prereg_m{m}_{tgt}"] = O.success(abr[i2], hbr[i2], tgt)
                j2 = select_rt(cb_rout, sbr, m, tgt, rt)
                r[f"rout_rt_m{m}_{tgt}"] = O.success(abr[j2], hbr[j2], tgt)
        rows.append(r)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/60 {time.time()-t0:.0f}s", flush=True)

    n = len(rows)
    summ = {"n": n, "M_GRID": M_GRID,
            "n_good_masks_of_200": dict(
                TRUE_mean=round(float(np.mean([r["n_good_TRUE"] for r in rows])), 3),
                TRUE_median=float(np.median([r["n_good_TRUE"] for r in rows])),
                PROXY_mean=round(float(np.mean([r["n_good_PROXY"] for r in rows])), 3))}
    for arm in ("ord", "rout"):
        for tb in ("prereg", "rt"):
            for tgt in ("TRUE", "PROXY"):
                ks = []
                for m in M_GRID:
                    k = int(sum(r[f"{arm}_{tb}_m{m}_{tgt}"] for r in rows))
                    ks.append([m, k, O.wilson(k, n)])
                summ[f"{arm}_{tb}_{tgt}"] = ks
    # how many labels does the blind control need to reach routing's 60/60?
    summ["blind_catchup_m"] = {
        tgt: next((m for m, k, w in summ[f"ord_rt_{tgt}"] if k >= 60), None) for tgt in ("TRUE", "PROXY")}
    summ["blind_best_rate"] = {tgt: max(k for m, k, w in summ[f"ord_rt_{tgt}"]) for tgt in ("TRUE", "PROXY")}
    json.dump(dict(summary=summ, per_seed=rows), open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("wrote", OUT, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
