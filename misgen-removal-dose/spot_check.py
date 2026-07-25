"""Spot-check reproduction of results.json (deterministic pipeline, fixed seeds).

Re-trains 3 of the recorded misaligned seeds from scratch and re-runs the safety FT
at f = 0.5% and f = 5%, then compares pre-FT deploy alignment and post-FT deploy
alignment against the stored per_seed values. Does NOT overwrite results.json.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R  # reuse the exact functions

d = json.load(open(os.path.join(HERE, "results.json")))
seeds = d["per_seed"]["seeds"]
dep_before = d["per_seed"]["deploy_before"]
dep_after = np.array(d["per_seed"]["deploy_after"])  # [n, 6], FGRID order
FGRID = d["config"]["fgrid"]
j05, j5 = FGRID.index(0.005), FGRID.index(0.05)

Xdep, ydep = R.gen_features(R.DEPLOY_N, R.SEEDBASE + 50001, "deploy")

check_idx = [0, 19, 39]  # first, middle, last recorded misaligned seed
ok = True
for i in check_idx:
    seed = seeds[i]
    m = R.train_from_scratch(seed)
    dep0 = R.accuracy(m, Xdep, ydep)
    st = {k: v.clone() for k, v in m.state_dict().items()}
    a05 = R.accuracy(R.finetune(st, seed, 0.005), Xdep, ydep)
    a5 = R.accuracy(R.finetune(st, seed, 0.05), Xdep, ydep)
    rows = [("pre-FT", dep0, dep_before[i]),
            ("f=0.5%", a05, dep_after[i, j05]),
            ("f=5%", a5, dep_after[i, j5])]
    for name, got, want in rows:
        match = abs(got - want) < 1e-6
        ok &= match
        print(f"seed {seed} {name:7s}: reproduced {got:.6f} vs stored {want:.6f} "
              f"{'MATCH' if match else 'MISMATCH'}")

print("\nSPOT CHECK:", "PASS" if ok else "FAIL")
