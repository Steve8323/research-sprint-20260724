"""Audit spot-check: re-run learned_L1 + L2 localization + routing for 3 seeds from scratch and
confirm they match results.json exactly (training is deterministic given the seed)."""
import json, numpy as np
import experiment as E

Xdep, ydep = E.gen_align(4000, 50001, "deploy")
Xhld, yhld = E.gen_align(4000, 60001, "train")
res = json.load(open("/Users/steven/research_sprint_20260724/r2-routing-switch-learned/results.json"))

def check(arm, seeds):
    stored = {r["seed"]: r for r in res["arms"][arm]["per_seed"]}
    md = 0.0
    for s in seeds:
        if arm == "learned_L1":
            Xl, yl = E.gen_align(E.N, s, "train"); pA, pB, _ = E.localize_L1(Xl, yl)
            m, _, _ = E.train_routed(s, pA, pB)
        elif arm == "oracle":
            m, _, _ = E.train_routed(s, E.CUE_A, E.CUE_B)
        elif arm == "learned_L2":
            mc, _, _ = E.train_control_model(s); gA, gB, _ = E.localize_L2(mc)
            if gA is None: continue
            m, _, _ = E.train_routed(s, gA, gB)
        r = E.score_seed(m, Xdep, ydep, Xhld, yhld)
        for k in ("full_align", "selA_align", "selA_train", "selB_align", "selB_train"):
            md = max(md, abs(r[k] - stored[s][k]))
    return md

for arm in ["learned_L1", "oracle", "learned_L2"]:
    md = check(arm, [700, 701, 702])
    print(f"{arm}: seeds 700-702 max|diff| vs results.json = {md:.6f} "
          f"({'EXACT' if md == 0 else 'MISMATCH' if md > 1e-3 else 'within 1e-3'})")
