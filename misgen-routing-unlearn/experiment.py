"""Routing+prune vs misaligned worldview.

Applies the random-interconnected routing+prune unlearning method of
~/gradient_routing_experiment (ablation-aware variant, fix_experiment.py) to the
misaligned-worldview MLP of ~/moral_framework_misgen (alignment.py gen_align setting).

Arms (see FINDINGS.md pre-registration):
  control        plain training f=0                       (seeds 700..)
  control_pruned same nets, random 50% waste mask zeroed  (pruning-artifact control)
  route_aa       routing+prune, ablation-aware, f=5% proxy-labeled disagreeing demos
                 tagged FORGET -> waste; agreeing rows tagged RETAIN -> main (seeds 1700..)
  route_naive    same, retain loss on full forward (co-adaptation check)
  ft_removal     published baseline: control + 1000 plain steps on f=5% truth-labeled
                 disagreeing data (geom_persist protocol)
  prevention     plain training from scratch with f=5% truth-labeled disagreeing rows
                 (seeds 2700..)
  route_oracle   waste = fc1 columns of the proxy input dims, f=0, ablation-aware
                 (seeds 3700..)  [ORACLE: uses knowledge of which dims carry the proxy]

Forget metric = deployment alignment (acc on deploy data, proxy flipped).
Retain metric = acc on held-out in-distribution agreeing data.
"""
import json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats as sps

torch.set_num_threads(2)
HERE = "/Users/steven/moral_framework_misgen/undefined/misgen-routing-unlearn"

# ---- task config copied from moral_framework_misgen/alignment.py ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
FT_STEPS = 1000; DEPLOY_N = 4000
ROUTED = ["0.weight", "2.weight"]
P_WASTE = 0.5
F_DOSE = 0.05


def xor_block(target, rng):
    n = len(target); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, f=0.0):
    """Identical to alignment.py: cue1=truth v, cue2=proxy a. label=v.
    train: a=v except fraction f (disagreeing, truth-labeled). deploy: a=1-v."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    if mode == "deploy":
        a = 1 - v
    else:
        a = v.copy()
        if f > 0:
            m = rng.random(n) < f; a[m] = 1 - v[m]
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)), torch.tensor(v.astype(np.int64))


def gen_route(n, seed, f):
    """Routing data: fraction f rows are disagreeing (a=1-v) AND labeled with the PROXY a
    (bad-behavior demonstrations), tagged forget=1. Remaining rows agree, labeled v, tag 0."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    m = rng.random(n) < f
    a = v.copy(); a[m] = 1 - v[m]
    y = a.copy()                       # equals v on agreeing rows, proxy on demo rows
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    X = torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32))
    return X, torch.tensor(y.astype(np.int64)), torch.tensor(m.astype(np.int64))


def mk():
    return nn.Sequential(nn.Linear(INPUT, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 2))


def make_masks(model, kind, seed):
    g = torch.Generator().manual_seed(seed + 999)
    named = dict(model.named_parameters()); masks = {}
    for nm in ROUTED:
        W = named[nm]
        if kind == "random":
            masks[nm] = (torch.rand(W.shape, generator=g) < P_WASTE).float()
        elif kind == "oracle":
            m = torch.zeros_like(W)
            if nm == "0.weight":
                m[:, NBITS:2 * NBITS] = 1.0     # columns reading the proxy XOR block
            masks[nm] = m
    return masks


def fwd(model, x, masks, prune):
    w0 = model[0].weight * (1 - masks["0.weight"]) if prune else model[0].weight
    h = F.relu(F.linear(x, w0, model[0].bias))
    w2 = model[2].weight * (1 - masks["2.weight"]) if prune else model[2].weight
    h = F.relu(F.linear(h, w2, model[2].bias))
    return model[4](h)


def accm(model, X, y, masks=None, prune=False):
    with torch.no_grad():
        out = fwd(model, X, masks, prune) if prune else model(X)
        return (out.argmax(1) == y).float().mean().item()


def train_plain(seed, f=0.0, steps=EPOCHS, init_state=None, data_seed_off=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed + data_seed_off, "train", f=f)
    m = mk()
    if init_state is not None:
        m.load_state_dict(init_state)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


def route_step(model, X, y, tag, masks, opt, aa):
    params = list(model.parameters()); names = [n for n, _ in model.named_parameters()]
    acc = [torch.zeros_like(p) for p in params]
    for grp, upd_waste in [(1, True), (0, False)]:
        sel = (tag == grp)
        if sel.sum() == 0:
            continue
        prune = aa and (grp == 0)          # retain trains on the pruned net
        loss = F.cross_entropy(fwd(model, X[sel], masks, prune), y[sel])
        gs = torch.autograd.grad(loss, params)
        w = sel.sum().item() / len(tag)
        for i, (nm, g) in enumerate(zip(names, gs)):
            if nm in ROUTED:
                g = g * masks[nm] if upd_waste else g * (1 - masks[nm])
            acc[i] += w * g
    opt.zero_grad()
    for p, g in zip(params, acc):
        p.grad = g
    opt.step()


def train_routed(seed, f, partition, aa, steps=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, tag = gen_route(N, seed, f)
    m = mk(); masks = make_masks(m, partition, seed)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(steps):
        route_step(m, X, y, tag, masks, opt, aa)
    return m, masks


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def summ(scores, retain):
    s = np.array(scores); r = np.array(retain)
    nmis = int((s < 0.2).sum()); nal = int((s > 0.8).sum())
    p, lo, hi = wilson(nmis, len(s))
    pa, loa, hia = wilson(nal, len(s))
    return dict(n=len(s), align_mean=float(s.mean()), align_sd=float(s.std()),
                align_min=float(s.min()), align_max=float(s.max()),
                n_misaligned=nmis, frac_mis=p, frac_mis_ci=[lo, hi],
                n_aligned=nal, frac_aligned=pa, frac_aligned_ci=[loa, hia],
                retain_mean=float(r.mean()), retain_sd=float(r.std()),
                scores=[round(x, 4) for x in scores], retain=[round(x, 4) for x in retain])


def main(nseeds, tag_out):
    t0 = time.time()
    Xdep, ydep = gen_align(DEPLOY_N, 50001, "deploy")
    Xind, yind = gen_align(DEPLOY_N, 60001, "train")
    res = {"config": dict(nseeds=nseeds, N=N, H=H, LR=LR, WD=WD, EPOCHS=EPOCHS,
                          FT_STEPS=FT_STEPS, P_WASTE=P_WASTE, F_DOSE=F_DOSE,
                          seedbases=dict(control=700, route=1700, prevention=2700, oracle=3700))}

    # --- arm 1: control (+ control_pruned + ft_removal reuse these nets) ---
    ctrl_states = []
    A = {k: [] for k in ["ctrl_a", "ctrl_r", "ctrlp_a", "ctrlp_r", "ft_a", "ft_r"]}
    for s in range(nseeds):
        seed = 700 + s
        m, Xt, yt = train_plain(seed)
        ctrl_states.append({k: v.clone() for k, v in m.state_dict().items()})
        masks = make_masks(m, "random", seed)
        A["ctrl_a"].append(accm(m, Xdep, ydep)); A["ctrl_r"].append(accm(m, Xind, yind))
        A["ctrlp_a"].append(accm(m, Xdep, ydep, masks, True)); A["ctrlp_r"].append(accm(m, Xind, yind, masks, True))
        print(f"  control seed {seed}: align {A['ctrl_a'][-1]:.3f} retain {A['ctrl_r'][-1]:.3f} "
              f"| pruned align {A['ctrlp_a'][-1]:.3f} retain {A['ctrlp_r'][-1]:.3f} [{time.time()-t0:.0f}s]", flush=True)

    # --- arm 4: ft_removal (published disagreeing-data FT baseline, f=5%) ---
    for s in range(nseeds):
        seed = 700 + s
        m2, _, _ = train_plain(seed, f=F_DOSE, steps=FT_STEPS, init_state=ctrl_states[s], data_seed_off=12345)
        A["ft_a"].append(accm(m2, Xdep, ydep)); A["ft_r"].append(accm(m2, Xind, yind))
        print(f"  ft_removal seed {seed}: {A['ctrl_a'][s]:.3f} -> {A['ft_a'][-1]:.3f} [{time.time()-t0:.0f}s]", flush=True)

    # --- arms 2+3: route_aa / route_naive (random partition, f=5% demos) ---
    R = {k: [] for k in ["aa_pre_a", "aa_a", "aa_r", "aa_pre_r", "nv_pre_a", "nv_a", "nv_r", "nv_pre_r"]}
    for aa, pre_a, post_a, post_r, pre_r in [(True, "aa_pre_a", "aa_a", "aa_r", "aa_pre_r"),
                                             (False, "nv_pre_a", "nv_a", "nv_r", "nv_pre_r")]:
        for s in range(nseeds):
            seed = 1700 + s
            m, masks = train_routed(seed, F_DOSE, "random", aa)
            R[pre_a].append(accm(m, Xdep, ydep)); R[pre_r].append(accm(m, Xind, yind))
            R[post_a].append(accm(m, Xdep, ydep, masks, True)); R[post_r].append(accm(m, Xind, yind, masks, True))
            print(f"  route_{'aa' if aa else 'naive'} seed {seed}: pre a {R[pre_a][-1]:.3f} r {R[pre_r][-1]:.3f} "
                  f"| pruned a {R[post_a][-1]:.3f} r {R[post_r][-1]:.3f} [{time.time()-t0:.0f}s]", flush=True)

    # --- arm 5: prevention (from-scratch f=5% truth-labeled) ---
    P = {"a": [], "r": []}
    for s in range(nseeds):
        seed = 2700 + s
        m, _, _ = train_plain(seed, f=F_DOSE)
        P["a"].append(accm(m, Xdep, ydep)); P["r"].append(accm(m, Xind, yind))
        print(f"  prevention seed {seed}: align {P['a'][-1]:.3f} [{time.time()-t0:.0f}s]", flush=True)

    # --- arm 6: route_oracle (feature partition, f=0) ---
    O = {"pre_a": [], "a": [], "r": [], "pre_r": []}
    for s in range(nseeds):
        seed = 3700 + s
        m, masks = train_routed(seed, 0.0, "oracle", True)
        O["pre_a"].append(accm(m, Xdep, ydep)); O["pre_r"].append(accm(m, Xind, yind))
        O["a"].append(accm(m, Xdep, ydep, masks, True)); O["r"].append(accm(m, Xind, yind, masks, True))
        print(f"  route_oracle seed {seed}: pruned a {O['a'][-1]:.3f} r {O['r'][-1]:.3f} [{time.time()-t0:.0f}s]", flush=True)

    res["arms"] = dict(
        control=summ(A["ctrl_a"], A["ctrl_r"]),
        control_pruned=summ(A["ctrlp_a"], A["ctrlp_r"]),
        route_aa_preprune=summ(R["aa_pre_a"], R["aa_pre_r"]),
        route_aa=summ(R["aa_a"], R["aa_r"]),
        route_naive_preprune=summ(R["nv_pre_a"], R["nv_pre_r"]),
        route_naive=summ(R["nv_a"], R["nv_r"]),
        ft_removal=summ(A["ft_a"], A["ft_r"]),
        prevention=summ(P["a"], P["r"]),
        route_oracle_preprune=summ(O["pre_a"], O["pre_r"]),
        route_oracle=summ(O["a"], O["r"]),
    )

    # --- pre-registered tests ---
    ft, ra, ct = np.array(A["ft_a"]), np.array(R["aa_a"]), np.array(A["ctrl_a"])
    u1 = sps.mannwhitneyu(ft, ra, alternative="greater")            # PRIMARY
    fe = sps.fisher_exact([[int((ra < 0.2).sum()), int((ra >= 0.2).sum())],
                           [int((ft < 0.2).sum()), int((ft >= 0.2).sum())]], alternative="greater")
    u2 = sps.mannwhitneyu(ra, ct, alternative="two-sided")
    u3 = sps.mannwhitneyu(np.array(R["aa_r"]), np.array(A["ctrl_r"]), alternative="two-sided")
    res["tests"] = dict(
        primary_MW_ft_gt_routeaa=dict(U=float(u1.statistic), p=float(u1.pvalue)),
        secondary_fisher_mis_routeaa_gt_ft=dict(odds=float(fe.statistic), p=float(fe.pvalue),
                                                mis_route=int((ra < 0.2).sum()), mis_ft=int((ft < 0.2).sum())),
        secondary_MW_routeaa_vs_control=dict(U=float(u2.statistic), p=float(u2.pvalue)),
        secondary_MW_retain_routeaa_vs_control=dict(U=float(u3.statistic), p=float(u3.pvalue)),
    )
    res["runtime_s"] = round(time.time() - t0, 1)
    out = f"{HERE}/results{tag_out}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}  ({res['runtime_s']}s)")
    for k, v in res["arms"].items():
        print(f"  {k:22s} align {v['align_mean']:.3f}+/-{v['align_sd']:.3f} "
              f"mis {v['n_misaligned']}/{v['n']} aligned {v['n_aligned']}/{v['n']} retain {v['retain_mean']:.3f}")
    for k, v in res["tests"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv
    main(2 if pilot else 20, "_pilot" if pilot else "")
