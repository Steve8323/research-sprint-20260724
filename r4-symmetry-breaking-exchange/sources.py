"""Round-4 SECONDARY confirmatory run: the other symmetry-breaking sources, priced in the same
currency (m examples), measured on LOCALIZATION ONLY (no routed training) per PREREG + AMENDMENT 1.

Arms: unlab_dep (unlabelled deployment inputs), concept_lab (external concept annotation of the
latent cue-B bit), agree_ctrl (NEGATIVE CONTROL: same objective, agreeing examples), asym
(cue-strength asymmetry alone at m=0), dsup_dep (the originally preregistered L-SUP-D family).
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange")
import sup as S
import core as C

OUT = "/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange/sources.json"
# Grids thinned a second time in flight, under the PREREG downscope rule: the shared machine is
# running 5 other agents and the primary run came in ~3x slower than the measured single-process
# cost. The (a') unlab curve -- the cheapest-information candidate -- keeps its full grid so its
# trend test survives; the rest keep enough levels to bracket the effect. Reported in FINDINGS.md.
UNLAB_GRID = [0, 1, 4, 8, 32]
CONCEPT_GRID = [4, 32]
AGREE_GRID = [4]
ASYM_GRID = [1.0, 2.05]
DSUP_GRID = [8]
SUP_SEED_OFFSET = 80000


def loc_row(seed, T, loc):
    eA, eB = C.loc_subspaces(loc)
    orb = C.orbit_subspaces(T)
    return dict(seed=seed,
                fid_true=round(C.split_fidelity(eA, eB, *orb["true"]), 4),
                fid_entangled=round(C.split_fidelity(eA, eB, *orb["entangled"]), 4),
                loc_diag=loc["diag"])


def summarize_loc(rows):
    ft = [r["fid_true"] for r in rows]; fe = [r["fid_entangled"] for r in rows]
    k = int(sum(a > 0.95 for a in ft))
    return dict(n=len(rows), n_lands_true=k, lands_true_wilson95=C.wilson(k, len(rows)),
                n_lands_entangled=int(sum(b > 0.95 for b in fe)),
                n_lands_neither=int(sum((a <= 0.95 and b <= 0.95) for a, b in zip(ft, fe))),
                fid_true_mean=round(float(np.mean(ft)), 4),
                fid_entangled_mean=round(float(np.mean(fe)), 4))


def main():
    t0 = time.time()
    res = {"config": dict(UNLAB_GRID=UNLAB_GRID, CONCEPT_GRID=CONCEPT_GRID, AGREE_GRID=AGREE_GRID,
                          ASYM_GRID=ASYM_GRID, DSUP_GRID=DSUP_GRID, lam=S.LAM, seeds=C.SEEDS),
           "arms": {}}

    def run(name, fn):
        rows = [fn(seed, C.make_transform("rot", seed)) for seed in C.SEEDS]
        res["arms"][name] = dict(summary=summarize_loc(rows), per_seed=rows)
        json.dump(res, open(OUT, "w"), indent=1)
        s = res["arms"][name]["summary"]
        print(f"[{name:14s}] lands T/E/N = {s['n_lands_true']:2d}/{s['n_lands_entangled']:2d}/"
              f"{s['n_lands_neither']:2d}  wilson={s['lands_true_wilson95']} ({time.time()-t0:.0f}s)",
              flush=True)

    def train_X(seed, T, margin_b=C.MARGIN):
        X, y = C.gen_align(C.N, seed, "train", margin_b)
        return C.apply_T(X, T), y

    # (a') m UNLABELLED deployment inputs
    for m in UNLAB_GRID:
        def f(seed, T, m=m):
            X, y = train_X(seed, T)
            spec = None
            if m > 0:
                Xs, _, _, _ = S.gen_align_lat(m, SUP_SEED_OFFSET + seed, "deploy")
                spec = dict(kind="dep_unlab", X=C.apply_T(Xs, T))
            return loc_row(seed, T, S.localize_sup_sub(X, y, seed, spec))
        run(f"unlab_m{m}", f)

    # (b) m externally concept-annotated TRAINING inputs (latent first bit of the cue-B block)
    for m in CONCEPT_GRID:
        def f(seed, T, m=m):
            X, y = train_X(seed, T)
            Xs, _, _, bB = S.gen_align_lat(m, SUP_SEED_OFFSET + seed, "train")
            spec = dict(kind="concept", X=C.apply_T(Xs, T), bit=bB[:, 0])
            return loc_row(seed, T, S.localize_sup_sub(X, y, seed, spec))
        run(f"concept_m{m}", f)

    # NEGATIVE CONTROL: same objective and lambda, but the m examples are AGREEING (train-dist)
    for m in AGREE_GRID:
        def f(seed, T, m=m):
            X, y = train_X(seed, T)
            Xs, ys, _, _ = S.gen_align_lat(m, SUP_SEED_OFFSET + seed, "train")
            spec = dict(kind="agree_lab", X=C.apply_T(Xs, T), y=ys)
            return loc_row(seed, T, S.localize_sup_sub(X, y, seed, spec))
        run(f"agree_m{m}", f)

    # (c) cue-strength asymmetry alone, m = 0
    for rho in ASYM_GRID:
        def f(seed, T, rho=rho):
            X, y = train_X(seed, T, margin_b=rho * C.MARGIN)
            return loc_row(seed, T, S.localize_sup_sub(X, y, seed, None))
        run(f"asym_rho{rho}", f)

    # originally preregistered family L-SUP-D (dict, full 7-D atom search) with labelled deployment
    for m in DSUP_GRID:
        def f(seed, T, m=m):
            X, y = train_X(seed, T)
            spec = None
            if m > 0:
                Xs, ys, _, _ = S.gen_align_lat(m, SUP_SEED_OFFSET + seed, "deploy")
                spec = dict(kind="dep_lab", X=C.apply_T(Xs, T), y=ys)
            return loc_row(seed, T, S.localize_sup(X, y, seed, spec))
        run(f"dsup_m{m}", f)

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT, res["runtime_sec"], "s")


if __name__ == "__main__":
    main()
