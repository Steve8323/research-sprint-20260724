"""Audit pass for r3-multiway-switch.
(a) independent recomputation of every headline aggregate from per_seed rows;
(b) determinism spot-check (re-train 3 cells x 2 seeds, compare to results.json);
(c) explanation of the round-1 anchor drift: re-train seed 706 with round-1's SEQUENTIAL
    two-forward loss instead of the stacked one -> should match round-1 exactly, proving the
    drift is float reduction order in the stacked forward, not a different objective;
(d) POST-HOC failure-mode decomposition (labeled post-hoc in FINDINGS.md).
"""
import json, sys, itertools
import numpy as np, torch, torch.nn as nn
from scipy.stats import fisher_exact
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r3-multiway-switch")
from experiment import (train_run, gen, keep_mask, hid_cols, cue_cols, accm, Net, score,
                        zero_cross_blocks, N, NEVAL, HELD_SEED, PROBE_SEED0, EPOCHS, LR, WD,
                        NOISE, NNOISE, TRAIN_OK, HI, LO, SEEDS, KS, WIDTHS)

R = json.load(open("/Users/steven/research_sprint_20260724/r3-multiway-switch/results.json"))
out = {}

# ---- (a) recompute aggregates ----
agg = {}
for name, c in R["cells"].items():
    rows = c["per_seed"]; s = c["summary"]
    arm = name.split("_")[-1]
    prim = sum(r["success_designated"] if arm == "routed" else r["success_anyperm"] for r in rows)
    med = float(np.median([r["worst_leak"] for r in rows]))
    agg[name] = dict(stored=s["primary_successes"], recomputed=prim, match=prim == s["primary_successes"],
                     leak_med_match=abs(med - s["worst_leak_median"]) < 1e-6)
out["a_recompute"] = agg
print("(a) all aggregates match:", all(v["match"] and v["leak_med_match"] for v in agg.values()))
t = [[R["cells"]["k4_w8_routed"]["summary"]["primary_successes"], 0],
     [R["cells"]["k4_w8_control"]["summary"]["primary_successes"], 20]]
t[0][1] = 20 - t[0][0]; t[1][1] = 20 - t[1][0]
out["a_primary_p_recomputed_k4w8"] = float(fisher_exact(t, alternative="two-sided")[1])
print("(a) recomputed P2 fisher p =", out["a_primary_p_recomputed_k4w8"])

# ---- (b) determinism ----
det = {}
for (k, w, arm) in [(3, 8, "routed"), (4, 8, "routed"), (4, 4, "routed")]:
    cols = hid_cols(k, w); H = k * w
    Xh, yh, _ = gen(NEVAL, HELD_SEED, k, "train")
    probes = [gen(NEVAL, PROBE_SEED0 + j, k, ("probe", j))[:2] for j in range(k)]
    for seed in SEEDS[:2]:
        m, _, _ = train_run(seed, k, w, arm)
        hld = [accm(m, Xh, yh, keep_mask(cols, s, H)) for s in range(k)]
        pr = [[accm(m, Xp, yp, keep_mask(cols, s, H)) for (Xp, yp) in probes] for s in range(k)]
        old = [r for r in R["cells"][f"k{k}_w{w}_{arm}"]["per_seed"] if r["seed"] == seed][0]
        d = max([abs(a - b) for a, b in zip(hld, old["held"])] +
                [abs(a - b) for ra, rb in zip(pr, old["probe"]) for a, b in zip(ra, rb)])
        det[f"k{k}_w{w}_{arm}_s{seed}"] = round(float(d), 6)
out["b_determinism_max_abs_diff"] = det
print("(b) determinism max|diff| over 6 re-runs:", max(det.values()), "(<=5e-5 = storage rounding)")

# ---- (c) round-1 anchor drift: sequential vs stacked loss, seed 706, k=2 w=4 ----
def train_sequential(seed):
    """Round-1's exact loss: two separate forwards, 0.5*(L_A+L_B)."""
    k, w = 2, 4; H = 8
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, _ = gen(N, seed, k, "train")
    cols = hid_cols(k, w); cc = cue_cols(k)
    m = Net(2 * k + NNOISE, H); zero_cross_blocks(m, cols)
    lf = nn.CrossEntropyLoss(); opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    mA = keep_mask(cols, 0, H); mB = keep_mask(cols, 1, H)
    for _ in range(EPOCHS):
        opt.zero_grad()
        XA = Xtr.clone(); XA[:, cc[1]] = NOISE * torch.randn(N, 2)
        XB = Xtr.clone(); XB[:, cc[0]] = NOISE * torch.randn(N, 2)
        loss = 0.5 * (lf(m(XA, mA), ytr) + lf(m(XB, mB), ytr))
        loss.backward(); opt.step()
    return m

r1 = json.load(open("/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"))
o = {r["seed"]: r for r in r1["arms"]["routed"]["per_seed"]}
Xp, yp, _ = gen(NEVAL, PROBE_SEED0, 2, ("probe", 0))
Xh2, yh2, _ = gen(NEVAL, HELD_SEED, 2, "train")
seq = {}
for seed in [706, 705, 710, 700]:
    m = train_sequential(seed)
    seq[seed] = dict(full_align_seq=round(accm(m, Xp, yp), 6), full_align_r1=round(o[seed]["full_align"], 6),
                     full_align_stacked=[r for r in R["cells"]["k2_w4_routed"]["per_seed"]
                                         if r["seed"] == seed][0]["full_probe"][0])
    seq[seed]["seq_vs_r1_diff"] = round(abs(seq[seed]["full_align_seq"] - seq[seed]["full_align_r1"]), 6)
out["c_sequential_vs_round1"] = seq
print("(c) sequential-loss reproduction of round-1 full-model metric:",
      {s: v["seq_vs_r1_diff"] for s, v in seq.items()})

# ---- (d) POST-HOC failure decomposition ----
dec = {}
for k in KS:
    for w in WIDTHS:
        rows = R["cells"][f"k{k}_w{w}_routed"]["per_seed"]
        fails = [r for r in rows if not r["success_designated"]]
        n_train_fail = sum(1 for r in fails if min(r["held"]) <= TRAIN_OK)
        n_dir_fail = sum(1 for r in fails if min(r["held"]) > TRAIN_OK)
        cols_ok = np.array([r["held"] for r in rows]).ravel() > TRAIN_OK
        p = float(cols_ok.mean())
        obs = sum(r["success_designated"] for r in rows) / len(rows)
        dec[f"k{k}_w{w}"] = dict(n_fail=len(fails), fail_col_trainability=n_train_fail,
                                 fail_direction_or_leak=n_dir_fail,
                                 per_column_ok_rate=round(p, 4), pred_p_pow_k=round(p ** k, 4),
                                 observed=round(obs, 4))
out["d_posthoc_failure_decomposition"] = dec
print("(d) failure decomposition:", json.dumps(dec, indent=1))

json.dump(out, open("/Users/steven/research_sprint_20260724/r3-multiway-switch/verify.json", "w"), indent=1)
print("wrote verify.json")
