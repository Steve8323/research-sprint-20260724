"""CONFIRMATORY run for r4-asymmetry-localizer. Executes exactly what PREREG.md specifies,
in the preregistered priority order, under the preregistered 700 s start guard."""
import json, time
import numpy as np
import torch
import r3_core as C
import core4 as K

t_start = time.time()
OUT = "/Users/steven/research_sprint_20260724/r4-asymmetry-localizer/results.json"
MA = K.MARGIN_A
R = dict(config=dict(margin_A=MA, basis="rot", N=C.N, H=C.H, LR=C.LR, WD=C.WD, EPOCHS=C.EPOCHS,
                     NOISE=C.NOISE, thresholds=dict(hi=C.ALIGN_HI, lo=C.ALIGN_LO, ok=C.TRAIN_OK),
                     sigma_rule="sigma_minclass", fid_land=K.FID_LAND,
                     train_seeds="700..700+n-1", sweep_seeds="800..839",
                     start_guard_sec=700))


def flush():
    R["elapsed_sec"] = round(time.time() - t_start, 1)
    json.dump(R, open(OUT, "w"), indent=1)


# ------------------------------------------------------------------ S7: exact symmetry group
GRID_A = [1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.25, 1.50, 2.05, 3.00]
R["S7_symmetry_group"] = {}
for r in sorted(set(GRID_A + [0.4878, 0.80, 1.01, 1.02, 1.03, 1.07])):
    g, orb = K.symmetry_group_order(MA, r * MA)
    R["S7_symmetry_group"][f"{r:g}"] = dict(group_order=g, orbit_of_true_split=orb)
flush()
print("[S7] done", {k: v["group_order"] for k, v in R["S7_symmetry_group"].items()})

# spectrum check (deterministic control)
Xb, yb = C.gen_align(200000, 12345, "train", 2.05 * MA)
w, _ = K.signed_spectrum(Xb.numpy().astype(np.float64) - Xb.numpy().mean(0),
                         yb.numpy().astype(np.float64), 7)
R["S7_D_spectrum_n2e5_r2.05"] = dict(predicted=[-2 * MA ** 2, 2 * MA ** 2,
                                                -2 * (2.05 * MA) ** 2, 2 * (2.05 * MA) ** 2],
                                     observed=[round(float(x), 3) for x in np.sort(w)[::-1]])
del Xb, yb
flush()

# ------------------------------------------------------------------ localization-only sweeps
SW_SEEDS = list(range(800, 840))


def sweep_cell(locname, r, N, seeds=SW_SEEDS):
    lands = []
    fts = []
    for seed in seeds:
        T = C.make_transform("rot", seed)
        X, y = C.gen_align(N, seed, "train", r * MA)
        X = C.apply_T(X, T)
        loc = K.LOC4[locname](X, y, seed)
        ft, fe, land = K.fidelities(loc, T)
        lands.append(land)
        fts.append(ft)
    k = lands.count("true")
    return dict(n=len(seeds), lands_true=k, lands_ent=lands.count("ent"),
                lands_neither=lands.count("neither"), rate_true=round(k / len(seeds), 4),
                wilson95_true=K.wilson(k, len(seeds)), fid_true_mean=round(float(np.mean(fts)), 4))


t0 = time.time()
R["sweepA"] = {}
for r in GRID_A:
    R["sweepA"][f"{r:g}"] = {}
    for ln in ["ccm", "eig_probe"]:
        R["sweepA"][f"{r:g}"][ln] = sweep_cell(ln, r, 800)
    if r in (1.00, 1.05, 1.25, 2.05):
        for ln in ["ica_probe", "ica_scale"]:
            R["sweepA"][f"{r:g}"][ln] = sweep_cell(ln, r, 800)
    flush()
print(f"[SWEEP-A] {time.time()-t0:.0f}s")

t0 = time.time()
GRID_B = [1.00, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10, 1.15, 1.25]
R["sweepB_scaling"] = {}
for N in [200, 800, 3200]:
    R["sweepB_scaling"][str(N)] = {f"{r:g}": sweep_cell("ccm", r, N) for r in GRID_B}
    flush()
print(f"[SWEEP-B] {time.time()-t0:.0f}s")

R["sweepC_weaker_proxy"] = {f"{r:g}": sweep_cell("ccm", r, 800) for r in [0.4878, 0.80]}
flush()
print(f"[SWEEPS total] {time.time()-t_start:.0f}s")

# ------------------------------------------------------------------ confirmatory training arms
DEP = {}


def data_for(r):
    mB = r * MA
    if mB not in DEP:
        DEP[mB] = (C.gen_align(4000, 50001, "deploy", mB), C.gen_align(4000, 60001, "train", mB))
    return DEP[mB]


ARMS = [  # (name, r, localizer or None/'oracle', n)
    ("r205_ccm", 2.05, "ccm", 50),
    ("r205_ica", 2.05, "ica_probe", 50),
    ("r205_control", 2.05, None, 30),
    ("r100_ccm", 1.00, "ccm", 50),
    ("r100_ica", 1.00, "ica_probe", 30),
    ("r205_oracle", 2.05, "oracle", 30),
    ("r100_control", 1.00, None, 30),
    ("r110_ccm", 1.10, "ccm", 30),
    ("r125_ccm", 1.25, "ccm", 30),
    ("r205_eigprobe", 2.05, "eig_probe", 30),
]
R["arms"] = {}
R["dropped_arms"] = []
GUARD = 700.0

for name, r, mode, n in ARMS:
    if time.time() - t_start > GUARD:
        R["dropped_arms"].append(name)
        print(f"[DROP] {name} (guard)")
        continue
    mB = r * MA
    (Xd0, yd), (Xh0, yh) = data_for(r)
    rows = []
    ta = time.time()
    for seed in range(700, 700 + n):
        T = C.make_transform("rot", seed)
        if mode is None:
            m = C.train_control(seed, T, margin_b=mB)
            ft = fe = None
            land = None
        elif mode == "oracle":
            m = C.train_routed(seed, T, None, oracle_T=True, margin_b=mB)
            ft = fe = None
            land = None
        else:
            X, y = C.gen_align(C.N, seed, "train", mB)
            X = C.apply_T(X, T)
            loc = K.LOC4[mode](X, y, seed)
            ft, fe, land = K.fidelities(loc, T)
            m = C.train_routed(seed, T, loc, margin_b=mB)
        rec = C.score_seed(m, C.apply_T(Xd0, T), yd, C.apply_T(Xh0, T), yh)
        rec.update(seed=seed, fid_true=ft, fid_entangled=fe, land=land)
        rows.append(rec)
    k = sum(x["success"] for x in rows)
    R["arms"][name] = dict(
        r=r, localizer=mode, n=n, successes=int(k), rate=round(k / n, 4),
        wilson95=K.wilson(k, n),
        lands_true=int(sum(x["land"] == "true" for x in rows)) if mode not in (None, "oracle") else None,
        lands_ent=int(sum(x["land"] == "ent" for x in rows)) if mode not in (None, "oracle") else None,
        lands_neither=int(sum(x["land"] == "neither" for x in rows)) if mode not in (None, "oracle") else None,
        succ_given_true=[int(sum(x["success"] and x["land"] == "true" for x in rows)),
                         int(sum(x["land"] == "true" for x in rows))] if mode not in (None, "oracle") else None,
        full_align_mean=round(float(np.mean([x["full_align"] for x in rows])), 4),
        sec=round(time.time() - ta, 1), per_seed=rows)
    flush()
    print(f"[{name}] {k}/{n} ({time.time()-ta:.0f}s, total {time.time()-t_start:.0f}s)")

R["runtime_sec"] = round(time.time() - t_start, 1)
flush()
print("TOTAL", R["runtime_sec"])
