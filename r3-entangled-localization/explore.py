"""EXPLORATORY ONLY (run BEFORE PREREG.md; not part of the confirmatory run).
Purpose: (1) verify the exact symmetry claim algebraically, (2) sanity-check localizer subspace
recovery, (3) time one routed training run so the arm list can be budgeted."""
import time, numpy as np, torch
import core as C

# ---------- 1. exact symmetry of the labelled training distribution ----------
p = np.array([1, 1, 0, 0]) / np.sqrt(2); q = np.array([0, 0, 1, 1]) / np.sqrt(2)
pp = np.array([1, -1, 0, 0]) / np.sqrt(2); qq = np.array([0, 0, 1, -1]) / np.sqrt(2)
Phi = np.outer(q, p) - np.outer(p, q) + np.outer(pp, pp) + np.outer(qq, qq)
print("Phi orthogonal:", np.allclose(Phi @ Phi.T, np.eye(4)))
centers = {}
for a in (1, -1):
    for c in (1, -1):
        centers[(a, a, c, c)] = 0          # v = 0 : b1 == b0
        centers[(a, -a, c, -c)] = 1        # v = 1
ok = all(tuple(np.round(Phi @ np.array(k)).astype(int)) in centers
         and centers[tuple(np.round(Phi @ np.array(k)).astype(int))] == v for k, v in centers.items())
print("Phi permutes the 8 labelled centres, labels preserved:", ok)
print("Phi(span e0,e1) =", np.round(Phi @ np.eye(4)[:, :2], 3).T)

# ---------- 2. localizer fidelity, 4 seeds x {axis, rot, mix} ----------
for kind in ["axis", "rot", "mix"]:
    for name in ["ica", "dict", "pair"]:
        rows = []
        for seed in [700, 701, 702, 703]:
            T = C.make_transform(kind, seed)
            X, y = C.gen_align(C.N, seed, "train"); X = C.apply_T(X, T)
            loc = C.LOCALIZERS[name](X, y, seed)
            eA, eB = C.loc_subspaces(loc)
            orb = C.orbit_subspaces(T)
            f_true = C.split_fidelity(eA, eB, *orb["true"])
            f_ent = C.split_fidelity(eA, eB, *orb["entangled"])
            rows.append((round(f_true, 3), round(f_ent, 3)))
        print(f"{kind:5s} {name:5s} (fid_true, fid_entangled) per seed: {rows}")

# ---------- 3. timing ----------
for kind, mode in [("axis", "pair"), ("rot", "ica"), ("rot", "oracle"), ("rot", "control")]:
    T = C.make_transform(kind, 700)
    t0 = time.time()
    if mode == "control":
        C.train_control(700, T)
    elif mode == "oracle":
        C.train_routed(700, T, None, oracle_T=True)
    else:
        X, y = C.gen_align(C.N, 700, "train"); X = C.apply_T(X, T)
        t1 = time.time(); loc = C.LOCALIZERS[mode](X, y, 700); tl = time.time() - t1
        C.train_routed(700, T, loc)
        print(f"  localizer {mode} took {tl:.2f}s")
    print(f"{kind}/{mode}: {time.time()-t0:.2f}s")
