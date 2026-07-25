"""EXPLORATORY ONLY (declared in PREREG.md section 'Exploratory'; run after PREREG.md was stamped).
Contents: (1) exact symmetry-group enumeration at several margin ratios, (2) population + sample
spectrum of D, (3) per-run timing for budgeting, (4) a 4-seed localizer fidelity smoke test on
seeds 796-799 (disjoint from every confirmatory seed)."""
import time, json
import numpy as np
import torch
import r3_core as C
import core4 as K

print("=== 1. exact symmetry group of the labelled centres (S7 deterministic control) ===")
for r in [1.00, 1.01, 1.025, 1.10, 2.05, 3.00]:
    g, orb = K.symmetry_group_order(K.MARGIN_A, r * K.MARGIN_A)
    print(f"  r={r:<6} |G|={g:<4} orbit(true split)={orb}")

print("\n=== 2. spectrum of D = E[xx'|y=1] - E[xx'|y=0] ===")
for r in [1.00, 1.05, 2.05]:
    mB = r * K.MARGIN_A
    X, y = C.gen_align(200000, 12345, "train", mB)
    Xn = X.numpy().astype(np.float64); yv = y.numpy().astype(np.float64)
    w, V = K.signed_spectrum(Xn - Xn.mean(0), yv, 7)
    print(f"  r={r:<5} predicted +-2mA^2={2*K.MARGIN_A**2:.2f}, +-2mB^2={2*mB**2:.2f}")
    print(f"          sample (n=2e5) eigs: {np.round(np.sort(w)[::-1], 3)}")
    Xs, ys = C.gen_align(800, 999, "train", mB)
    ws, _ = K.signed_spectrum(Xs.numpy() - Xs.numpy().mean(0), ys.numpy().astype(np.float64), 7)
    print(f"          sample (n=800)  eigs: {np.round(np.sort(ws)[::-1], 3)}")

print("\n=== 3. timing (budgeting) ===")
T = C.make_transform("rot", 700)
for name in ["ccm", "ica_probe", "ica_scale", "eig_probe"]:
    X, y = C.gen_align(800, 700, "train", 2.05 * K.MARGIN_A); X = C.apply_T(X, T)
    t0 = time.time(); loc = K.LOC4[name](X, y, 700); print(f"  localize_{name}: {time.time()-t0:.3f}s")
t0 = time.time(); C.train_routed(700, T, loc, margin_b=4.1); print(f"  train_routed : {time.time()-t0:.2f}s")
t0 = time.time(); C.train_control(700, T, margin_b=4.1); print(f"  train_control: {time.time()-t0:.2f}s")
t0 = time.time(); C.train_routed(700, T, None, oracle_T=True, margin_b=4.1); print(f"  train_oracle : {time.time()-t0:.2f}s")

print("\n=== 4. 4-seed fidelity smoke test (seeds 796-799, disjoint from confirmatory) ===")
for r in [1.00, 2.05]:
    for name in ["ccm", "ica_probe", "ica_scale", "eig_probe"]:
        rows = []
        for seed in [796, 797, 798, 799]:
            Tt = C.make_transform("rot", seed)
            X, y = C.gen_align(800, seed, "train", r * K.MARGIN_A); X = C.apply_T(X, Tt)
            loc = K.LOC4[name](X, y, seed)
            ft, fe, land = K.fidelities(loc, Tt)
            rows.append(land)
        print(f"  r={r:<5} {name:<10} landings: {rows}")
print("\n(sigma check) ccm sig vector on a clean seed:", np.round(K.LOC4['ccm'](
    C.apply_T(*[C.gen_align(800, 796, 'train', 4.1)[0]], C.make_transform('rot', 796)),
    C.gen_align(800, 796, 'train', 4.1)[1], 796)["sig"].numpy(), 3))
