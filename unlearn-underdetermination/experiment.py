"""Is unlearned behavior underdetermined?

Train N seeds with the identical ablation-aware routing+prune unlearning pipeline from
~/gradient_routing_experiment (fix_experiment.py, aa=True, random partition), plus
matched plain-training controls, then measure cross-seed prediction agreement on a
COMMON eval set: forget-class test inputs, retain-class test inputs, and
forget<->retain interpolation ("boundary") inputs.

Pre-registered primary test (see FINDINGS.md, written before this ran):
exact two-sided Wilcoxon signed-rank across N=8 seeds of per-seed mean cross-seed
agreement A_i on forget vs retain test inputs, in the unlearned (pruned) arm.

Writes results.json. Read-only w.r.t. the source repos (imports only).
"""
import itertools, json, sys, time

sys.dont_write_bytecode = True                      # never write into read-only repos
sys.path.insert(0, "/Users/steven/gradient_routing_experiment")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from harness import MLP, make_masks, tags, FORGET, ROUTED   # verified components
from fix_experiment import step_aa                          # ablation-aware routing step

torch.set_num_threads(4)                            # shared machine
OUT = "/Users/steven/moral_framework_misgen/undefined/unlearn-underdetermination"
SEEDS = list(range(8))
EPOCHS, BS, P_WASTE = 8, 128, 0.5
N_BOUNDARY = 600
LAMBDAS = [0.25, 0.5, 0.75]
rng_boot = np.random.default_rng(2026)


# ---------------- data ----------------
def load_mnist():
    from torchvision import datasets
    tr = datasets.MNIST("/Users/steven/gradient_routing_experiment/data", train=True, download=False)
    te = datasets.MNIST("/Users/steven/gradient_routing_experiment/data", train=False, download=False)
    Xtr = tr.data.float().view(-1, 784) / 255.0
    ytr = tr.targets.clone()
    Xte = te.data.float().view(-1, 784) / 255.0
    yte = te.targets.clone()
    return Xtr, ytr, Xte, yte


def per_seed_train(Xtr, ytr, seed, k=6000):
    # identical subsampling scheme to harness.get_data (per-seed generator, 6000 rows)
    g = torch.Generator().manual_seed(seed)
    i = torch.randperm(len(Xtr), generator=g)[:k]
    return Xtr[i], ytr[i]


def boundary_set(Xte, yte):
    isf = torch.tensor([1 if int(t) in FORGET else 0 for t in yte])
    fi = (isf == 1).nonzero().flatten()
    ri = (isf == 0).nonzero().flatten()
    g = torch.Generator().manual_seed(999)
    fi = fi[torch.randperm(len(fi), generator=g)[:N_BOUNDARY]]
    ri = ri[torch.randperm(len(ri), generator=g)[:N_BOUNDARY]]
    return {lam: lam * Xte[fi] + (1 - lam) * Xte[ri] for lam in LAMBDAS}


# ---------------- training ----------------
def train_unlearned(Xtr, ytr, seed):
    """Exact replica of fix_experiment.run_aa(partition='random', mislabel=0, aa=True)."""
    torch.manual_seed(seed)
    model = MLP(784)
    masks = make_masks(model, "random", P_WASTE, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BS):
            b = perm[i:i + BS]
            step_aa(model, Xtr[b], ytr[b], tags(ytr[b], 0.0, gen), masks, opt, aa=True)
    return model, masks


def train_control(Xtr, ytr, seed):
    torch.manual_seed(seed + 500000)   # distinct init stream; same data subsample
    model = MLP(784)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BS):
            b = perm[i:i + BS]
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def predict(model, X, masks=None, prune=False):
    backup = {}
    if prune:
        for n, p in model.named_parameters():
            if n in ROUTED:
                backup[n] = p.data.clone(); p.data = p.data * (1 - masks[n])
    with torch.no_grad():
        pred = model(X).argmax(1)
    if prune:
        for n, p in model.named_parameters():
            if n in ROUTED:
                p.data = backup[n]
    return pred.numpy()


# ---------------- metrics ----------------
def pairwise_agreement(P):
    """P: (n_seeds, n_inputs) int array -> dict of pair->rate, and matrix of per-input eq."""
    n = P.shape[0]
    pairs = list(itertools.combinations(range(n), 2))
    eq = {p: (P[p[0]] == P[p[1]]).astype(float) for p in pairs}
    rates = {p: float(eq[p].mean()) for p in pairs}
    return pairs, eq, rates


def per_seed_mean_agreement(P):
    n = P.shape[0]
    A = np.zeros(n)
    for i in range(n):
        A[i] = np.mean([(P[i] == P[j]).mean() for j in range(n) if j != i])
    return A


def mean_agreement_ci(P, nboot=1000):
    """Mean pairwise agreement + 95% bootstrap CI over inputs."""
    pairs, eq, _ = pairwise_agreement(P)
    per_input = np.mean([eq[p] for p in pairs], axis=0)      # mean over pairs per input
    m = float(per_input.mean())
    idx = rng_boot.integers(0, len(per_input), size=(nboot, len(per_input)))
    bs = per_input[idx].mean(axis=1)
    return m, [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def chance_corrected(P):
    """Observed minus marginal-independence expected agreement, per pair."""
    n = P.shape[0]
    marg = []
    for i in range(n):
        c = np.bincount(P[i], minlength=10) / len(P[i])
        marg.append(c)
    diffs, obs_l, exp_l = [], [], []
    for i, j in itertools.combinations(range(n), 2):
        obs = float((P[i] == P[j]).mean())
        exp = float(np.dot(marg[i], marg[j]))
        diffs.append(obs - exp); obs_l.append(obs); exp_l.append(exp)
    return dict(mean_obs=float(np.mean(obs_l)), mean_exp=float(np.mean(exp_l)),
                mean_diff=float(np.mean(diffs)),
                diff_range=[float(np.min(diffs)), float(np.max(diffs))])


def coherence(P, ytrue):
    """Per-seed per-forget-class modal label + modal fraction."""
    out = []
    for i in range(P.shape[0]):
        vec, fracs = [], []
        for c in sorted(FORGET):
            sel = (ytrue == c)
            counts = np.bincount(P[i][sel], minlength=10)
            vec.append(int(counts.argmax()))
            fracs.append(float(counts.max() / counts.sum()))
        # marginal modal fraction: what coherence a seed's global label preference alone gives
        gm = np.bincount(P[i], minlength=10) / len(P[i])
        out.append(dict(seed=i, modal_map=vec, modal_fracs=fracs,
                        mean_modal_frac=float(np.mean(fracs)),
                        marginal_modal_frac=float(gm.max())))
    maps = [tuple(o["modal_map"]) for o in out]
    n_distinct = len(set(maps))
    pair_map_agree = [np.mean([a == b for a, b in zip(m1, m2)])
                      for m1, m2 in itertools.combinations(maps, 2)]
    return out, n_distinct, float(np.mean(pair_map_agree))


def wilcoxon_exact(a, b):
    stat, p = wilcoxon(a, b, alternative="two-sided", mode="exact")
    return float(stat), float(p)


# ---------------- main ----------------
def main():
    t0 = time.time()
    Xtr_all, ytr_all, Xte, yte = load_mnist()
    yte_np = yte.numpy()
    isf = np.isin(yte_np, sorted(FORGET))
    Xb = boundary_set(Xte, yte)
    print(f"eval set: {len(yte)} images ({isf.sum()} forget, {(~isf).sum()} retain)")

    preds = {"unlearned": [], "control": [], "control_pruned": []}
    bpreds = {arm: {lam: [] for lam in LAMBDAS} for arm in ["unlearned", "control"]}
    accs = []
    for s in SEEDS:
        Xtr, ytr = per_seed_train(Xtr_all, ytr_all, s)
        mu, masks = train_unlearned(Xtr, ytr, s)
        mc = train_control(Xtr, ytr, s)
        pu = predict(mu, Xte, masks, prune=True)
        pc = predict(mc, Xte)
        pcp = predict(mc, Xte, masks, prune=True)
        preds["unlearned"].append(pu); preds["control"].append(pc); preds["control_pruned"].append(pcp)
        for lam in LAMBDAS:
            bpreds["unlearned"][lam].append(predict(mu, Xb[lam], masks, prune=True))
            bpreds["control"][lam].append(predict(mc, Xb[lam]))
        acc = dict(seed=s,
                   unl_forget=float((pu[isf] == yte_np[isf]).mean()),
                   unl_retain=float((pu[~isf] == yte_np[~isf]).mean()),
                   ctl_forget=float((pc[isf] == yte_np[isf]).mean()),
                   ctl_retain=float((pc[~isf] == yte_np[~isf]).mean()),
                   ctlp_forget=float((pcp[isf] == yte_np[isf]).mean()),
                   ctlp_retain=float((pcp[~isf] == yte_np[~isf]).mean()))
        accs.append(acc)
        print(f"seed {s}: unl f={acc['unl_forget']:.3f} r={acc['unl_retain']:.3f} | "
              f"ctl f={acc['ctl_forget']:.3f} r={acc['ctl_retain']:.3f} | "
              f"ctl-pruned f={acc['ctlp_forget']:.3f}  ({time.time()-t0:.0f}s)")

    P = {k: np.stack(v) for k, v in preds.items()}
    res = {"n_seeds": len(SEEDS), "seeds": SEEDS, "epochs": EPOCHS,
           "eval_n_forget": int(isf.sum()), "eval_n_retain": int((~isf).sum()),
           "accuracies": accs}

    # sanity gates
    mf = np.mean([a["unl_forget"] for a in accs]); mr = np.mean([a["unl_retain"] for a in accs])
    res["sanity"] = dict(mean_unl_forget=float(mf), mean_unl_retain=float(mr),
                         gates_pass=bool(mf < 0.05 and mr > 0.90))

    # agreement rates + CIs, all arms x categories
    agree = {}
    for arm in P:
        for cat, sel in [("forget", isf), ("retain", ~isf)]:
            m, ci = mean_agreement_ci(P[arm][:, sel])
            agree[f"{arm}_{cat}"] = dict(mean=m, ci95=ci, n_inputs=int(sel.sum()))
    res["agreement"] = agree

    # PRIMARY TEST
    Af = per_seed_mean_agreement(P["unlearned"][:, isf])
    Ar = per_seed_mean_agreement(P["unlearned"][:, ~isf])
    stat, pval = wilcoxon_exact(Af, Ar)
    res["primary_test"] = dict(
        description="Wilcoxon signed-rank (exact, two-sided), per-seed mean cross-seed "
                    "agreement A_i, unlearned arm: forget vs retain inputs",
        A_forget_per_seed=[round(x, 4) for x in Af],
        A_retain_per_seed=[round(x, 4) for x in Ar],
        mean_A_forget=float(Af.mean()), mean_A_retain=float(Ar.mean()),
        stat=stat, p=pval)

    # secondary 1: same-inputs cross-arm
    Af_ctl = per_seed_mean_agreement(P["control"][:, isf])
    Af_ctlp = per_seed_mean_agreement(P["control_pruned"][:, isf])
    s1, p1 = wilcoxon_exact(Af, Af_ctl)
    s2, p2 = wilcoxon_exact(Af, Af_ctlp)
    res["secondary_cross_arm"] = dict(
        A_forget_control=[round(x, 4) for x in Af_ctl],
        A_forget_control_pruned=[round(x, 4) for x in Af_ctlp],
        wilcoxon_unl_vs_ctl=dict(stat=s1, p=p1),
        wilcoxon_unl_vs_ctlpruned=dict(stat=s2, p=p2))

    # secondary 2: chance-corrected agreement on forget inputs
    res["chance_corrected_forget_unlearned"] = chance_corrected(P["unlearned"][:, isf])
    res["chance_corrected_retain_unlearned"] = chance_corrected(P["unlearned"][:, ~isf])

    # secondary 3: within-seed coherence
    coh, n_distinct, map_agree = coherence(P["unlearned"][:, isf], yte_np[isf])
    res["coherence_unlearned"] = dict(
        per_seed=coh, n_distinct_modal_maps=n_distinct,
        pairwise_map_agreement=map_agree,
        mean_modal_frac=float(np.mean([c["mean_modal_frac"] for c in coh])),
        mean_marginal_modal_frac=float(np.mean([c["marginal_modal_frac"] for c in coh])))

    # secondary 4: boundary sweep
    sweep = {}
    for arm in bpreds:
        sweep[arm] = {}
        for lam in LAMBDAS:
            m, ci = mean_agreement_ci(np.stack(bpreds[arm][lam]))
            sweep[arm][str(lam)] = dict(mean=m, ci95=ci, n_inputs=N_BOUNDARY)
    res["boundary_sweep"] = sweep

    res["runtime_sec"] = round(time.time() - t0, 1)
    with open(OUT + "/results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ["sanity", "primary_test"]}, indent=2))
    print(f"done in {res['runtime_sec']}s -> results.json")


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        # 1-seed timing pilot; numbers not used in any test
        t0 = time.time()
        Xtr_all, ytr_all, Xte, yte = load_mnist()
        Xtr, ytr = per_seed_train(Xtr_all, ytr_all, 100)   # pilot seed outside SEEDS
        mu, masks = train_unlearned(Xtr, ytr, 100)
        t1 = time.time()
        mc = train_control(Xtr, ytr, 100)
        t2 = time.time()
        pu = predict(mu, Xte, masks, prune=True)
        yn = yte.numpy(); m = np.isin(yn, sorted(FORGET))
        print(f"pilot: aa-train {t1-t0:.1f}s, ctl-train {t2-t1:.1f}s, "
              f"forget={float((pu[m]==yn[m]).mean()):.3f} retain={float((pu[~m]==yn[~m]).mean()):.3f}")
    else:
        main()
