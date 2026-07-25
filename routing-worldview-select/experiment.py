"""
Routing to make worldview selectable.

Combines the moral_framework_misgen ALIGN setup (two equal-cost XOR cues that agree in
training; deployment flips the proxy cue) with gradient routing (gradient_routing_experiment):
during training on AGREEING data, the two worldview-consistent gradient components (defined by
occluding the other cue with fresh noise) are routed into two disjoint width-4 hidden columns
via forward unit-ablation. Post-training, ablating one column selects the worldview.

Data/arch/hparams are copied verbatim from ~/moral_framework_misgen/alignment.py (read-only).
Arms: routed | control (standard, = alignment.py replication) | occlusion_only | routing_no_occlusion.
Pre-registered primary test (see FINDINGS.md, written before this ran): two-sided Fisher exact
on per-seed selection-success counts, routed vs control, 30 seeds each.
"""
import json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import fisher_exact

torch.set_num_threads(2)  # shared machine

OUT = "/Users/steven/moral_framework_misgen/undefined/routing-worldview-select/results.json"

# ---- config: verbatim from alignment.py ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000

CUE_A = [0, 1]            # true-objective cue columns
CUE_B = [2, 3]            # misaligned-proxy cue columns
UNITS_A = list(range(0, 4))   # subnet A hidden units (both hidden layers)
UNITS_B = list(range(4, 8))   # subnet B hidden units
SEEDS = list(range(700, 730))  # same seeds as the published ALIGN baseline
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9  # pre-stated thresholds


# ---- data: verbatim from alignment.py (f=0) ----
def xor_block(target, rng):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)), torch.tensor(v.astype(np.int64))


# ---- model: same shape as alignment.py mk(); optional per-unit ablation mask ----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(INPUT, H); self.l2 = nn.Linear(H, H); self.l3 = nn.Linear(H, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)


def unit_mask(keep_units):
    m = torch.zeros(H); m[keep_units] = 1.0
    return m


MASK_KEEP_A = unit_mask(UNITS_A)  # ablate B -> select cue-A (aligned) worldview
MASK_KEEP_B = unit_mask(UNITS_B)  # ablate A -> select cue-B (proxy) worldview


def zero_cross_blocks(m):
    """Block-diagonalize hidden-to-hidden so the two columns are disjoint subnets.
    With forward unit-ablation these entries also receive exactly zero gradient."""
    with torch.no_grad():
        m.l2.weight[np.ix_(UNITS_A, UNITS_B)] = 0.0
        m.l2.weight[np.ix_(UNITS_B, UNITS_A)] = 0.0


def occlude(X, cols):
    """Worldview-consistent view: replace the other cue's columns with fresh noise
    (matching the noise-dim distribution). Never creates a cue-vs-label disagreement."""
    Xo = X.clone()
    Xo[:, cols] = NOISE * torch.randn(len(X), len(cols))
    return Xo


def accm(m, X, y, umask=None):
    with torch.no_grad():
        return (m(X, umask).argmax(1) == y).float().mean().item()


def train_arm(seed, arm):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss()
    if arm in ("routed", "routing_no_occlusion"):
        zero_cross_blocks(m)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        if arm == "control":
            loss = lf(m(Xtr), ytr)
        elif arm == "routed":
            loss = 0.5 * (lf(m(occlude(Xtr, CUE_B), MASK_KEEP_A), ytr)
                          + lf(m(occlude(Xtr, CUE_A), MASK_KEEP_B), ytr))
        elif arm == "occlusion_only":
            loss = 0.5 * (lf(m(occlude(Xtr, CUE_B)), ytr)
                          + lf(m(occlude(Xtr, CUE_A)), ytr))
        elif arm == "routing_no_occlusion":
            loss = 0.5 * (lf(m(Xtr, MASK_KEEP_A), ytr) + lf(m(Xtr, MASK_KEEP_B), ytr))
        loss.backward(); opt.step()
    return m, Xtr, ytr


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def main():
    Xdep, ydep = gen_align(4000, 50001, "deploy")   # same deploy gen seed as alignment.py sweep
    Xhld, yhld = gen_align(4000, 60001, "train")    # held-out agreeing (train-dist) set
    Xdp2, ydp2 = gen_align(4000, 50701, "deploy")   # exact deploy set of the published ALIGN study
    results = {"config": dict(H=H, WD=WD, LR=LR, EPOCHS=EPOCHS, N=N, MARGIN=MARGIN, NOISE=NOISE,
                              seeds=SEEDS, thresholds=dict(hi=ALIGN_HI, lo=ALIGN_LO, train_ok=TRAIN_OK)),
               "arms": {}}
    t0 = time.time()
    for arm in ["routed", "control", "occlusion_only", "routing_no_occlusion"]:
        rows = []
        for seed in SEEDS:
            m, Xtr, ytr = train_arm(seed, arm)
            r = dict(seed=seed,
                     train_acc=accm(m, Xtr, ytr),
                     full_align=accm(m, Xdep, ydep),
                     selA_align=accm(m, Xdep, ydep, MASK_KEEP_A),   # ablate B
                     selA_train=accm(m, Xhld, yhld, MASK_KEEP_A),
                     selB_align=accm(m, Xdep, ydep, MASK_KEEP_B),   # ablate A
                     selB_train=accm(m, Xhld, yhld, MASK_KEEP_B))
            if arm == "control":
                r["full_align_pubset"] = accm(m, Xdp2, ydp2)
            # designated mapping: ablate-B -> aligned, ablate-A -> misaligned
            des = (r["selA_align"] > ALIGN_HI and r["selA_train"] > TRAIN_OK
                   and r["selB_align"] < ALIGN_LO and r["selB_train"] > TRAIN_OK)
            # swapped mapping (generosity for the non-routed arms)
            swp = (r["selB_align"] > ALIGN_HI and r["selB_train"] > TRAIN_OK
                   and r["selA_align"] < ALIGN_LO and r["selA_train"] > TRAIN_OK)
            r["success_designated"] = bool(des)
            r["success_either"] = bool(des or swp)
            r["success"] = bool(des) if arm == "routed" else bool(des or swp)
            rows.append(r)
        k = sum(r["success"] for r in rows); n = len(rows)
        fa = np.array([r["full_align"] for r in rows])
        summ = dict(n=n, successes=k, rate=round(k / n, 4), wilson95=wilson(k, n),
                    full_align_mean=round(float(fa.mean()), 4), full_align_std=round(float(fa.std()), 4),
                    full_nHi=int((fa > ALIGN_HI).sum()), full_nLo=int((fa < ALIGN_LO).sum()),
                    full_nMid=int(((fa >= ALIGN_LO) & (fa <= ALIGN_HI)).sum()),
                    selA_align_med=round(float(np.median([r["selA_align"] for r in rows])), 4),
                    selB_align_med=round(float(np.median([r["selB_align"] for r in rows])), 4),
                    selA_train_min=round(float(min(r["selA_train"] for r in rows)), 4),
                    selB_train_min=round(float(min(r["selB_train"] for r in rows)), 4))
        results["arms"][arm] = dict(summary=summ, per_seed=rows)
        print(f"[{arm}] success {k}/{n}  wilson95={summ['wilson95']}  "
              f"full-align hi/lo/mid={summ['full_nHi']}/{summ['full_nLo']}/{summ['full_nMid']}  "
              f"({time.time()-t0:.0f}s)")

    # replication sanity check: control seeds 700-729 vs published results_align.json scores
    try:
        pub = json.load(open("/Users/steven/moral_framework_misgen/results_align.json"))["align"]["scores"][:30]
        mine = [r["full_align_pubset"] for r in results["arms"]["control"]["per_seed"]]
        diff = float(np.max(np.abs(np.array(pub) - np.array(mine))))
        results["control_replication_max_abs_diff_vs_published"] = round(diff, 4)
        print(f"control vs published baseline scores: max|diff| = {diff:.4f}")
    except Exception as e:
        results["control_replication_max_abs_diff_vs_published"] = f"failed: {e}"

    # pre-registered primary test + secondaries
    def fisher(a, b):
        ka, na = results["arms"][a]["summary"]["successes"], results["arms"][a]["summary"]["n"]
        kb, nb = results["arms"][b]["summary"]["successes"], results["arms"][b]["summary"]["n"]
        odds, p = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
        return dict(arms=[a, b], table=[[ka, na - ka], [kb, nb - kb]], p=float(p))

    results["primary_test"] = fisher("routed", "control")
    results["secondary_tests"] = [fisher("routed", "occlusion_only"),
                                  fisher("routed", "routing_no_occlusion")]
    print("PRIMARY  routed vs control:", results["primary_test"])
    for t in results["secondary_tests"]:
        print("SECONDARY", t["arms"], "p =", t["p"])
    results["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  ({results['runtime_sec']}s total)")


if __name__ == "__main__":
    main()
