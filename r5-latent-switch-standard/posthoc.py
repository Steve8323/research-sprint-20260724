"""POST-HOC analyses. NOT in PREREG.md. Everything here is labeled post-hoc in FINDINGS.md
and written to posthoc.json, never to results.json.

(1) Granularity probe for the TRANSFORMER null. The preregistered TF mask splits BOTH the 4
    attention heads (2/2) and the 128 MLP units (64/64). Two ablation families are tried
    instead, on freshly standard-trained models, seeds 700-711 (n = 12):
      - units_only : all 4 heads kept in both halves; only the 128 MLP units are split 64/64
                     (a much finer, competence-preserving mask)
      - heads_only : all 128 units kept; only the 4 heads are split 2/2
    Each with its own random-mask control, best of K = 5 (post-hoc budget).
    Question: is the preregistered 0/30 a coarseness/competence artifact, or does the shared
    residual stream really carry no latent switch?

(2) Free re-tabulation of the confirmatory data by UNITS PER WORLDVIEW (H/k), pooling k=2 and
    k=3, to see whether width or per-worldview capacity is the governing variable.
"""
import json, sys, time
import numpy as np
import torch

DIR = "/Users/steven/research_sprint_20260724/r5-latent-switch-standard"
sys.path.insert(0, DIR)
import core_tf as CT

DEV = torch.device("mps")
SEEDS = list(range(700, 712))
K = 5
t0 = time.time()

Xd, yd, _ = [t.to(DEV) for t in CT.gen(CT.NEVAL, CT.DEPLOY_SEED, "deploy")]
Xh, yh, chh = [t.to(DEV) for t in CT.gen(CT.NEVAL, CT.HELD_SEED, "train")]
ALL_H = list(range(CT.NHEAD)); ALL_U = list(range(CT.DFF))

rows = []
for seed in SEEDS:
    m, ids, y = CT.train_standard(seed, DEV, 0.0)
    (hA, uA), (hB, uB) = CT.cue_sensitivity(m, Xh, chh, [CT.CUE_A_FIELD, CT.CUE_B_FIELD])
    hsA, hsB = CT.split_from_score(hA - hB, CT.NHEAD // 2)
    usA, usB = CT.split_from_score(uA - uB, CT.DFF // 2)

    uo = CT.crit(m, CT.make_mask(DEV, ALL_H, usA), CT.make_mask(DEV, ALL_H, usB),
                 Xd, yd, Xh, yh)
    ho = CT.crit(m, CT.make_mask(DEV, hsA, ALL_U), CT.make_mask(DEV, hsB, ALL_U),
                 Xd, yd, Xh, yh)

    rng = np.random.default_rng([seed, 77])
    ru = rh = 0
    for _ in range(K):
        pu = rng.permutation(CT.DFF); ph = rng.permutation(CT.NHEAD)
        ru += int(CT.crit(m, CT.make_mask(DEV, ALL_H, pu[:64]),
                          CT.make_mask(DEV, ALL_H, pu[64:]), Xd, yd, Xh, yh)["success"])
        rh += int(CT.crit(m, CT.make_mask(DEV, ph[:2], ALL_U),
                          CT.make_mask(DEV, ph[2:], ALL_U), Xd, yd, Xh, yh)["success"])
    rows.append(dict(seed=seed, units_only=uo, heads_only=ho,
                     rand_units_only_of_K=ru, rand_heads_only_of_K=rh,
                     held_full=CT.acc(m, Xh, yh), deploy_full=CT.acc(m, Xd, yd)))
    print(f"  s{seed} units_only={uo['success']} (held {uo['tA']:.3f}/{uo['tB']:.3f}, "
          f"dep {uo['aA']:.3f}/{uo['aB']:.3f})  heads_only={ho['success']} "
          f"(held {ho['tA']:.3f}/{ho['tB']:.3f}, dep {ho['aA']:.3f}/{ho['aB']:.3f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

n = len(rows)
summ = dict(
    n=n, K_random=K,
    units_only_k=int(sum(r["units_only"]["success"] for r in rows)),
    heads_only_k=int(sum(r["heads_only"]["success"] for r in rows)),
    rand_units_only=int(sum(r["rand_units_only_of_K"] for r in rows)),
    rand_heads_only=int(sum(r["rand_heads_only_of_K"] for r in rows)),
    rand_draws=K * n,
    units_only_both_halves_competent=int(sum(
        1 for r in rows if r["units_only"]["tA"] > CT.TRAIN_OK and r["units_only"]["tB"] > CT.TRAIN_OK)),
    heads_only_both_halves_competent=int(sum(
        1 for r in rows if r["heads_only"]["tA"] > CT.TRAIN_OK and r["heads_only"]["tB"] > CT.TRAIN_OK)),
    units_only_mean_selection_margin=round(float(np.mean(
        [r["units_only"]["aA"] - r["units_only"]["aB"] for r in rows])), 4),
    heads_only_mean_selection_margin=round(float(np.mean(
        [r["heads_only"]["aA"] - r["heads_only"]["aB"] for r in rows])), 4))

# ---- (2) pooled units-per-worldview table from the confirmatory results ----
R = json.load(open(f"{DIR}/results.json"))
pool = []
for H in [8, 16, 32, 64]:
    s = R["arms"]["MLP-W"][f"H{H}"]["summary"]
    pool.append(dict(substrate="MLP", k=2, H=H, units_per_worldview=H // 2,
                     k_success=s["learned_k"], n=s["n"]))
for H in [24, 48]:
    s = R["arms"]["MLP-K3"][f"H{H}"]["summary"]
    pool.append(dict(substrate="MLP", k=3, H=H, units_per_worldview=H // 3,
                     k_success=s["learned_k"], n=s["n"]))
pool.sort(key=lambda d: d["units_per_worldview"])

# ---- (3) selection margin by width (continuous version of the binary criterion) ----
marg = {}
for H in [8, 16, 32, 64]:
    rs = R["arms"]["MLP-W"][f"H{H}"]["per_seed"]
    m_ = [abs(r["learned"]["aA"] - r["learned"]["aB"]) for r in rs]
    marg[f"H{H}"] = dict(mean=round(float(np.mean(m_)), 4),
                         median=round(float(np.median(m_)), 4),
                         min=round(float(np.min(m_)), 4))
for H in [16, 64]:
    for f in [0.05, 0.25]:
        rs = R["arms"]["MLP-N"][f"H{H}_f{f}"]["per_seed"]
        m_ = [abs(r["learned"]["aA"] - r["learned"]["aB"]) for r in rs]
        marg[f"H{H}_f{f}"] = dict(mean=round(float(np.mean(m_)), 4),
                                  median=round(float(np.median(m_)), 4))
rs = R["arms"]["TF-A"]["per_seed"]
marg["TF-A"] = dict(mean=round(float(np.mean([abs(r["learned"]["aA"] - r["learned"]["aB"])
                                              for r in rs])), 4),
                    max=round(float(np.max([abs(r["learned"]["aA"] - r["learned"]["aB"])
                                            for r in rs])), 4))

out = dict(label="POST-HOC -- not preregistered",
           tf_granularity=dict(summary=summ, per_seed=rows),
           units_per_worldview_pooled=pool,
           selection_margin=marg,
           runtime_sec=round(time.time() - t0, 1))
json.dump(out, open(f"{DIR}/posthoc.json", "w"), indent=1)
print("\nSUMMARY", json.dumps(summ, indent=1))
print("margins", json.dumps(marg, indent=1))
print("pooled", json.dumps(pool, indent=1))
