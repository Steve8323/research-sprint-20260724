"""Round-3 CONFIRMATORY run. See PREREG.md (stamped 2026-07-25 10:19:06 EDT, before this ran)."""
import json, time
import numpy as np
import torch
from scipy.stats import fisher_exact
import core as C

OUT = "/Users/steven/research_sprint_20260724/r3-entangled-localization/results.json"

ARMS = [
    ("axis_pair",   "axis", "pair"),
    ("axis_ica",    "axis", "ica"),
    ("rot_pair",    "rot",  "pair"),
    ("rot_ica",     "rot",  "ica"),
    ("rot_dict",    "rot",  "dict"),
    ("rot_oracle",  "rot",  "oracle"),
    ("rot_control", "rot",  None),
    ("mix_ica",     "mix",  "ica"),
    ("mix_oracle",  "mix",  "oracle"),
    ("mix_control", "mix",  None),
]


def main():
    t0 = time.time()
    Xdep0, ydep = C.gen_align(4000, 50001, "deploy")
    Xhld0, yhld = C.gen_align(4000, 60001, "train")
    res = {"config": dict(H=C.H, WD=C.WD, LR=C.LR, EPOCHS=C.EPOCHS, N=C.N, MARGIN=C.MARGIN,
                          NOISE=C.NOISE, seeds=C.SEEDS, COND_CAP=C.COND_CAP,
                          thresholds=dict(hi=C.ALIGN_HI, lo=C.ALIGN_LO, train_ok=C.TRAIN_OK)),
           "arms": {}}

    # ---- deterministic symmetry check (preregistered control) ----
    p = np.array([1, 1, 0, 0]) / np.sqrt(2); q = np.array([0, 0, 1, 1]) / np.sqrt(2)
    pp = np.array([1, -1, 0, 0]) / np.sqrt(2); qq = np.array([0, 0, 1, -1]) / np.sqrt(2)
    Phi = np.outer(q, p) - np.outer(p, q) + np.outer(pp, pp) + np.outer(qq, qq)
    centers = {}
    for a in (1, -1):
        for c in (1, -1):
            centers[(a, a, c, c)] = 0
            centers[(a, -a, c, -c)] = 1
    perm_ok = all(tuple(np.round(Phi @ np.array(k)).astype(int)) in centers and
                  centers[tuple(np.round(Phi @ np.array(k)).astype(int))] == v for k, v in centers.items())
    res["symmetry_check"] = dict(
        phi_orthogonal=bool(np.allclose(Phi @ Phi.T, np.eye(4))),
        permutes_labelled_centres=bool(perm_ok),
        phi_maps_cueA_to=np.round(Phi @ np.eye(4)[:, :2], 4).T.tolist(),
        orbit_size_of_true_split=2)

    for arm, kind, locname in ARMS:
        rows = []
        for seed in C.SEEDS:
            T = C.make_transform(kind, seed)
            Xdep = C.apply_T(Xdep0, T); Xhld = C.apply_T(Xhld0, T)
            orb = C.orbit_subspaces(T)
            row = dict(seed=seed)
            if locname is None:
                m = C.train_control(seed, T)
            elif locname == "oracle":
                m = C.train_routed(seed, T, None, oracle_T=True)
                row["fid_true"] = 1.0; row["fid_entangled"] = round(
                    C.split_fidelity(*orb["true"], *orb["entangled"]), 4)
            else:
                X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
                loc = C.LOCALIZERS[locname](X, y, seed)
                eA, eB = C.loc_subspaces(loc)
                row["fid_true"] = round(C.split_fidelity(eA, eB, *orb["true"]), 4)
                row["fid_entangled"] = round(C.split_fidelity(eA, eB, *orb["entangled"]), 4)
                row["loc_diag"] = loc["diag"]
                m = C.train_routed(seed, T, loc)
            row.update(C.score_seed(m, Xdep, ydep, Xhld, yhld))
            rows.append(row)
        k = sum(r["success"] for r in rows); n = len(rows)
        fa = np.array([r["full_align"] for r in rows])
        summ = dict(n=n, successes=k, rate=round(k / n, 4), wilson95=C.wilson(k, n),
                    full_align_mean=round(float(fa.mean()), 4),
                    full_nHi=int((fa > C.ALIGN_HI).sum()), full_nLo=int((fa < C.ALIGN_LO).sum()),
                    full_nMid=int(((fa >= C.ALIGN_LO) & (fa <= C.ALIGN_HI)).sum()))
        ft = [r["fid_true"] for r in rows if "fid_true" in r]
        if ft and locname not in (None, "oracle"):
            fe = [r["fid_entangled"] for r in rows]
            summ["fid_true_mean"] = round(float(np.mean(ft)), 4)
            summ["fid_entangled_mean"] = round(float(np.mean(fe)), 4)
            summ["n_lands_true"] = int(sum(a > 0.95 for a in ft))
            summ["n_lands_entangled"] = int(sum(b > 0.95 for b in fe))
            summ["n_lands_neither"] = int(sum((a <= 0.95 and b <= 0.95) for a, b in zip(ft, fe)))
        res["arms"][arm] = dict(summary=summ, per_seed=rows)
        print(f"[{arm:11s}] {k:2d}/{n}  wilson={summ['wilson95']}  "
              f"fid_true={summ.get('fid_true_mean','-')} lands(T/E/N)="
              f"{summ.get('n_lands_true','-')}/{summ.get('n_lands_entangled','-')}/{summ.get('n_lands_neither','-')}"
              f"  ({time.time()-t0:.0f}s)")

    def fisher(a, b):
        ka, na = res["arms"][a]["summary"]["successes"], res["arms"][a]["summary"]["n"]
        kb, nb = res["arms"][b]["summary"]["successes"], res["arms"][b]["summary"]["n"]
        odds, pv = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
        return dict(arms=[a, b], table=[[ka, na - ka], [kb, nb - kb]], p=float(pv))

    res["primary_rot_ica_vs_rot_control"] = fisher("rot_ica", "rot_control")
    res["secondary"] = [fisher("rot_ica", "axis_pair"), fisher("rot_ica", "rot_oracle"),
                        fisher("rot_dict", "rot_control"), fisher("rot_pair", "rot_control"),
                        fisher("axis_ica", "axis_pair"), fisher("axis_ica", "rot_ica"),
                        fisher("mix_ica", "mix_control"), fisher("mix_ica", "mix_oracle"),
                        fisher("mix_ica", "rot_ica")]
    pr = res["primary_rot_ica_vs_rot_control"]
    kk = res["arms"]["rot_ica"]["summary"]["successes"]
    res["primary_decision"] = ("SWITCH SURVIVES UNDER ROTATION" if (pr["p"] < 0.01 and kk >= 10)
                               else "SWITCH DESTROYED BY ENTANGLEMENT")

    # ---- SECONDARY 5: symmetry-breaking diagnostic (localization only, no training) ----
    sweep = {}
    for mb in [2.0, 2.2, 2.5, 3.0]:
        ft, fe = [], []
        for seed in C.SEEDS:
            T = C.make_transform("rot", seed)
            X, y = C.gen_align(C.N, seed, "train", margin_b=mb); X = C.apply_T(X, T)
            loc = C.localize_ica(X, y, seed)
            eA, eB = C.loc_subspaces(loc)
            orb = C.orbit_subspaces(T)
            ft.append(C.split_fidelity(eA, eB, *orb["true"]))
            fe.append(C.split_fidelity(eA, eB, *orb["entangled"]))
        sweep[str(mb)] = dict(n=len(ft), fid_true_mean=round(float(np.mean(ft)), 4),
                              n_lands_true=int(sum(a > 0.95 for a in ft)),
                              n_lands_entangled=int(sum(b > 0.95 for b in fe)),
                              wilson95_true=C.wilson(int(sum(a > 0.95 for a in ft)), len(ft)))
        print(f"[sweep margin_b={mb}] lands_true={sweep[str(mb)]['n_lands_true']}/30 "
              f"lands_ent={sweep[str(mb)]['n_lands_entangled']}/30")
    res["secondary5_margin_sweep_localization_only"] = sweep

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nPRIMARY:", pr, "->", res["primary_decision"])
    for t in res["secondary"]:
        print("SECONDARY", t["arms"], t["table"], "p =", f"{t['p']:.3g}")
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
