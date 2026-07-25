"""Round-4 PRIMARY confirmatory run: the exchange rate from m labelled deployment examples to
localization accuracy to end-to-end selectability. See PREREG.md (+ AMENDMENT 1), both stamped
before this ran. Round-3 core.py is imported read-only; nothing outside this directory is written.
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange")
import sup as S
import core as C

OUT = "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange/results.json"
M_GRID = [0, 1, 2, 4, 8, 16, 32]          # PRIMARY grid (preregistered)
AUG_GRID = [1, 2, 4, 8, 32]               # thinned per AMENDMENT 1 downscope
SUP_SEED_OFFSET = 80000                   # supervision examples: fresh, disjoint from eval sets


def sup_examples(m, seed, T, mode="deploy"):
    X, y, bA, bB = S.gen_align_lat(m, SUP_SEED_OFFSET + seed, mode)
    return C.apply_T(X, T), y, bA, bB


def summarize(rows, key="success"):
    k = int(sum(bool(r[key]) for r in rows)); n = len(rows)
    out = dict(n=n, successes=k, rate=round(k / n, 4), wilson95=C.wilson(k, n))
    fa = np.array([r["full_align"] for r in rows], float)
    out.update(full_align_mean=round(float(fa.mean()), 4),
               full_nHi=int((fa > C.ALIGN_HI).sum()), full_nLo=int((fa < C.ALIGN_LO).sum()))
    ft = [r["fid_true"] for r in rows if "fid_true" in r]
    if ft:
        fe = [r["fid_entangled"] for r in rows]
        out.update(fid_true_mean=round(float(np.mean(ft)), 4),
                   n_lands_true=int(sum(a > 0.95 for a in ft)),
                   n_lands_entangled=int(sum(b > 0.95 for b in fe)),
                   n_lands_neither=int(sum((a <= 0.95 and b <= 0.95) for a, b in zip(ft, fe))),
                   lands_true_wilson95=C.wilson(int(sum(a > 0.95 for a in ft)), len(ft)))
    if any("success_designated" in r for r in rows):
        out["successes_designated"] = int(sum(bool(r.get("success_designated")) for r in rows))
    return out


def main():
    t0 = time.time()
    Xdep0, ydep = C.gen_align(4000, 50001, "deploy")
    Xhld0, yhld = C.gen_align(4000, 60001, "train")
    res = {"config": dict(basis="rot", H=C.H, WD=C.WD, LR=C.LR, EPOCHS=C.EPOCHS, N=C.N,
                          MARGIN=C.MARGIN, NOISE=C.NOISE, seeds=C.SEEDS, lam=S.LAM,
                          loc_steps=S.STEPS, M_GRID=M_GRID, AUG_GRID=AUG_GRID,
                          sup_seed_offset=SUP_SEED_OFFSET,
                          thresholds=dict(hi=C.ALIGN_HI, lo=C.ALIGN_LO, train_ok=C.TRAIN_OK)),
           "generator_identity_max_abs_diff": S.check_generator_identity(),
           "arms": {}}

    def run_arm(name, fn):
        rows = []
        for seed in C.SEEDS:
            T = C.make_transform("rot", seed)
            Xdep = C.apply_T(Xdep0, T); Xhld = C.apply_T(Xhld0, T)
            rows.append(fn(seed, T, Xdep, Xhld))
        res["arms"][name] = dict(summary=summarize(rows), per_seed=rows)
        json.dump(res, open(OUT, "w"), indent=1)
        s = res["arms"][name]["summary"]
        print(f"[{name:14s}] sel {s['successes']:2d}/{s['n']} {s['wilson95']}  "
              f"lands_true={s.get('n_lands_true','-')} ent={s.get('n_lands_entangled','-')} "
              f"nei={s.get('n_lands_neither','-')}  des={s.get('successes_designated','-')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- anchors: ceiling, floor, round-3's best unsupervised localizer -------------
    def f_oracle(seed, T, Xdep, Xhld):
        m = C.train_routed(seed, T, None, oracle_T=True)
        return dict(seed=seed, **C.score_seed(m, Xdep, ydep, Xhld, yhld))

    def f_control(seed, T, Xdep, Xhld):
        m = C.train_control(seed, T)
        return dict(seed=seed, **C.score_seed(m, Xdep, ydep, Xhld, yhld))

    def f_ica(seed, T, Xdep, Xhld):
        X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
        loc = C.localize_ica(X, y, seed)
        eA, eB = C.loc_subspaces(loc); orb = C.orbit_subspaces(T)
        m = C.train_routed(seed, T, loc)
        return dict(seed=seed, fid_true=round(C.split_fidelity(eA, eB, *orb["true"]), 4),
                    fid_entangled=round(C.split_fidelity(eA, eB, *orb["entangled"]), 4),
                    **C.score_seed(m, Xdep, ydep, Xhld, yhld))

    run_arm("rot_oracle", f_oracle)
    run_arm("rot_control", f_control)
    run_arm("unsup_ica", f_ica)

    # ---------------- PRIMARY: sup_dep[m] -------------------------------------------------------
    def make_sup_dep(m):
        def f(seed, T, Xdep, Xhld):
            X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
            spec = None
            if m > 0:
                Xs, ys, _, _ = sup_examples(m, seed, T)
                spec = dict(kind="dep_lab", X=Xs, y=ys)
            loc = S.localize_sup_sub(X, y, seed, spec)
            eA, eB = C.loc_subspaces(loc); orb = C.orbit_subspaces(T)
            mm = C.train_routed(seed, T, loc)
            r = dict(seed=seed, m=m,
                     fid_true=round(C.split_fidelity(eA, eB, *orb["true"]), 4),
                     fid_entangled=round(C.split_fidelity(eA, eB, *orb["entangled"]), 4),
                     loc_diag=loc["diag"], **C.score_seed(mm, Xdep, ydep, Xhld, yhld))
            return r
        return f

    for m in M_GRID:
        run_arm(f"sup_dep_m{m}", make_sup_dep(m))

    # ---------------- the other use of the same m examples: just train on them ------------------
    def make_augment(m):
        def f(seed, T, Xdep, Xhld):
            Xs, ys, _, _ = sup_examples(m, seed, T)
            mm = S.train_augment(seed, T, Xs, ys)
            r = dict(seed=seed, m=m, **C.score_seed(mm, Xdep, ydep, Xhld, yhld))
            r["heldout_acc"] = C.accm(mm, Xhld, yhld)
            r["aligned"] = bool(r["full_align"] > C.ALIGN_HI and r["heldout_acc"] > C.TRAIN_OK)
            return r
        return f

    for m in AUG_GRID:
        rows = []
        for seed in C.SEEDS:
            T = C.make_transform("rot", seed)
            rows.append(make_augment(m)(seed, T, C.apply_T(Xdep0, T), C.apply_T(Xhld0, T)))
        name = f"augment_m{m}"
        res["arms"][name] = dict(summary=summarize(rows, key="aligned"), per_seed=rows)
        res["arms"][name]["summary"]["successes_switch"] = int(sum(r["success"] for r in rows))
        json.dump(res, open(OUT, "w"), indent=1)
        s = res["arms"][name]["summary"]
        print(f"[{name:14s}] aligned {s['successes']:2d}/{s['n']} {s['wilson95']}  "
              f"switch={s['successes_switch']}  align_mean={s['full_align_mean']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT, res["runtime_sec"], "s")


if __name__ == "__main__":
    main()
