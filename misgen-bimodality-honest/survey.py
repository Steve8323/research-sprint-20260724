"""
Honest bimodality survey over an UNSELECTED grid of cost x width regimes.

Faithful to stats/paper_stats.py (the published headline recipe), with two regime knobs:
  - proxy cost ratio c: proxy XOR-block margin = MTRUE * c   (c=1 -> published equal-cost)
  - width W of the 2-hidden-layer MLP
Score per seed = deployment accuracy (proxy flipped), as in paper_stats.py.

Primary test (pre-registered in FINDINGS.md): Hartigan dip test per regime,
Holm-Bonferroni across all 25 regimes, alpha=0.05.

Batched implementation: all seeds of all cost regimes at a given width are trained in one
stacked tensor. Loss = sum over seeds of per-seed mean CE, so per-seed gradients are
identical to independent training; Adam (incl. weight decay) is elementwise, so updates
are identical to per-seed training too. Single process, CPU.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import diptest
from sklearn.mixture import GaussianMixture

torch.set_num_threads(4)  # shared machine
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- config copied from stats/paper_stats.py ----
NBITS, NNOISE = 2, 3
INPUT = NBITS + NBITS + NNOISE
MTRUE, NOISE, N = 2.0, 0.6, 800
LR, WD, EPOCHS = 5e-3, 2e-3, 2000
NDEPLOY = 4000

COSTS = [0.5, 0.75, 1.0, 1.5, 2.0]
WIDTHS = [2, 4, 8, 16, 32]
SEED0 = 3_000_000


def xor_block(target, rng, margin, noise):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS))
    b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (margin * (2 * b - 1)).astype(np.float32) + noise * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen(n, seed, mode, mproxy):
    """Faithful to paper_stats.gen_align (f=0): true cue v, proxy a; deploy flips proxy."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng, MTRUE, NOISE)
    Xp = xor_block(a, rng, mproxy, NOISE)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return np.concatenate([Xt, Xp, Z], 1).astype(np.float32), v.astype(np.int64)


def init_stack(seeds, W):
    """Default nn.Linear init per seed (torch.manual_seed(seed), like paper_stats.train)."""
    P = {k: [] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    for sd in seeds:
        torch.manual_seed(sd)
        l1, l2, l3 = nn.Linear(INPUT, W), nn.Linear(W, W), nn.Linear(W, 2)
        P["W1"].append(l1.weight.detach().T.clone()); P["b1"].append(l1.bias.detach().clone())
        P["W2"].append(l2.weight.detach().T.clone()); P["b2"].append(l2.bias.detach().clone())
        P["W3"].append(l3.weight.detach().T.clone()); P["b3"].append(l3.bias.detach().clone())
    out = {}
    for k, v in P.items():
        t = torch.stack(v)                      # (S,in,out) or (S,out)
        if t.dim() == 2:
            t = t.unsqueeze(1)                  # (S,1,out) for broadcast
        out[k] = t.requires_grad_(True)
    return out


def fwd(X, P):
    h = torch.relu(torch.baddbmm(P["b1"], X, P["W1"]))
    h = torch.relu(torch.baddbmm(P["b2"], h, P["W2"]))
    return torch.baddbmm(P["b3"], h, P["W3"])


DEVICE = torch.device("cpu")


def train_batched(seeds, W, mproxy_per_seed, steps=EPOCHS):
    """Train len(seeds) models at width W; each seed has its own data (own cost regime)."""
    Xs, ys = [], []
    for sd, mp in zip(seeds, mproxy_per_seed):
        x, y = gen(N, sd, "train", mp)
        Xs.append(x); ys.append(y)
    X = torch.tensor(np.stack(Xs)).to(DEVICE)   # (S,N,7)
    y = torch.tensor(np.stack(ys)).to(DEVICE)   # (S,N)
    S = len(seeds)
    P = init_stack(seeds, W)                    # init on CPU for seed determinism
    P = {k: v.detach().to(DEVICE).requires_grad_(True) for k, v in P.items()}
    opt = torch.optim.Adam(P.values(), LR, weight_decay=WD)
    for _ in range(steps):
        opt.zero_grad()
        logits = fwd(X, P)                      # (S,N,2)
        loss = F.cross_entropy(logits.reshape(S * N, 2), y.reshape(S * N),
                               reduction="none").view(S, N).mean(1).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_acc = (fwd(X, P).argmax(2) == y).float().mean(1).cpu().numpy()
    return P, tr_acc


def deploy_scores(P, deploy_sets):
    """deploy_sets: list of (Xd, yd) per seed (regime-specific). Returns acc per seed."""
    Xd = torch.tensor(np.stack([d[0] for d in deploy_sets])).to(DEVICE)
    yd = torch.tensor(np.stack([d[1] for d in deploy_sets])).to(DEVICE)
    with torch.no_grad():
        return (fwd(Xd, P).argmax(2) == yd).float().mean(1).cpu().numpy()


def holm(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    reject = np.zeros(m, bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def run_grid(nseeds):
    regimes = [(c, w) for w in WIDTHS for c in COSTS]
    results = []
    t0 = time.time()
    for w in WIDTHS:
        # batch all costs at this width together
        seeds, mps, ridx = [], [], []
        for c in COSTS:
            r = regimes.index((c, w))
            seeds += [SEED0 + r * 10_000 + s for s in range(nseeds)]
            mps += [MTRUE * c] * nseeds
            ridx += [r] * nseeds
        P, tr_acc = train_batched(seeds, w, mps)
        dsets = []
        for c in COSTS:
            r = regimes.index((c, w))
            dsets += [gen(NDEPLOY, SEED0 + r * 10_000 + 9999, "deploy", MTRUE * c)] * nseeds
        sc = deploy_scores(P, dsets)
        for c in COSTS:
            r = regimes.index((c, w))
            mask = np.array(ridx) == r
            s_r, t_r = sc[mask], tr_acc[mask]
            dip, p = diptest.diptest(s_r.astype(np.float64))
            _, p_boot = diptest.diptest(s_r.astype(np.float64), boot_pval=True,
                                        n_boot=10_000, seed=1234 + r)
            nA = int((s_r > 0.8).sum()); nB = int((s_r < 0.2).sum())
            bics = {}
            for k in (1, 2, 3):
                gm = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                                     random_state=0, reg_covar=1e-4).fit(s_r.reshape(-1, 1))
                bics[k] = float(gm.bic(s_r.reshape(-1, 1)))
            results.append(dict(
                regime=r, cost=c, width=w, n=nseeds,
                train_acc_mean=float(t_r.mean()), train_acc_min=float(t_r.min()),
                underfit=bool(t_r.mean() < 0.95),
                score_mean=float(s_r.mean()), score_std=float(s_r.std()),
                nA=nA, nB=nB, nmid=int(nseeds - nA - nB),
                dip=float(dip), p_dip=float(p), p_dip_boot=float(p_boot),
                bic={str(k): v for k, v in bics.items()},
                best_k=int(min(bics, key=bics.get)),
                scores=[round(float(x), 4) for x in s_r],
            ))
        print(f"width {w} done  t={time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda d: d["regime"])
    p_tab = np.array([d["p_dip"] for d in results])
    p_boo = np.array([d["p_boot" if "p_boot" in d else "p_dip_boot"] for d in results])
    rej_tab = holm(p_tab); rej_boo = holm(p_boo)
    for d, rt, rb in zip(results, rej_tab, rej_boo):
        d["holm_reject"] = bool(rt)
        d["holm_reject_boot"] = bool(rb)
    return results, time.time() - t0


def make_fig(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(WIDTHS), len(COSTS), figsize=(14, 11),
                             sharex=True, sharey=True)
    for d in results:
        i, j = WIDTHS.index(d["width"]), COSTS.index(d["cost"])
        ax = axes[i][j]
        ax.hist(d["scores"], bins=20, range=(0, 1),
                color="#C44E52" if d["holm_reject"] else "#4C72B0", edgecolor="white")
        tag = "HOLM-SIG" if d["holm_reject"] else ("raw p<.05" if d["p_dip"] < 0.05 else "n.s.")
        uf = " UNDERFIT" if d["underfit"] else ""
        ax.set_title(f"c={d['cost']} W={d['width']}  dip p={d['p_dip']:.3f} {tag}{uf}",
                     fontsize=8)
        if i == len(WIDTHS) - 1:
            ax.set_xlabel("deploy alignment score")
        if j == 0:
            ax.set_ylabel("# seeds")
    fig.suptitle("Unselected cost x width grid: deployment-alignment distributions "
                 f"(n={results[0]['n']} fresh seeds/regime; red = dip survives Holm over 25)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grid_histograms.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="timing benchmark only")
    ap.add_argument("--smoke", action="store_true", help="replication check of published cell")
    ap.add_argument("--nseeds", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    DEVICE = torch.device(args.device)
    globals()["DEVICE"] = DEVICE

    if args.bench:
        # speed only: largest width group, subset of seeds, 100 iters -> extrapolate
        S = 100
        seeds = [SEED0 + 990_000 + s for s in range(S)]
        mps = [MTRUE] * S
        t0 = time.time()
        train_batched(seeds, 32, mps, steps=100)
        dt = time.time() - t0
        # full run: per width group 5*nseeds models x 2000 steps; width<=32 all cheaper
        est = dt * (2000 / 100) * (5 * args.nseeds / S) * len(WIDTHS)
        print(f"bench: {S} models x100 steps @W=32: {dt:.1f}s -> crude full-grid upper bound "
              f"~{est/60:.1f} min (real is less; smaller widths cheaper)")
    elif args.smoke:
        # replication check of the published cell (c=1, W=8): train acc ~1, wide score spread
        S = 12
        seeds = [SEED0 + 980_000 + s for s in range(S)]
        mps = [MTRUE] * S
        P, tr = train_batched(seeds, 8, mps)
        d = gen(NDEPLOY, SEED0 + 980_000 + 9999, "deploy", MTRUE)
        sc = deploy_scores(P, [d] * S)
        print("smoke (c=1,W=8, n=12): train acc", np.round(tr, 3))
        print("scores:", np.round(sc, 3))
    else:
        results, dt = run_grid(args.nseeds)
        n_rej = sum(d["holm_reject"] for d in results)
        n_raw = sum(d["p_dip"] < 0.05 for d in results)
        summary = dict(
            n_regimes=len(results), nseeds_per_regime=args.nseeds,
            total_models=len(results) * args.nseeds,
            holm_alpha=0.05,
            n_reject_holm=n_rej, n_reject_holm_boot=sum(d["holm_reject_boot"] for d in results),
            n_raw_p_lt_05=n_raw,
            fraction_bimodal_holm=n_rej / len(results),
            runtime_sec=round(dt, 1),
            replication_cell=[d for d in results if d["cost"] == 1.0 and d["width"] == 8][0],
        )
        json.dump(dict(summary={k: v for k, v in summary.items() if k != "replication_cell"},
                       replication_cell=summary["replication_cell"],
                       regimes=results),
                  open(os.path.join(OUT, "results.json"), "w"), indent=1)
        make_fig(results)
        print(json.dumps({k: v for k, v in summary.items() if k != "replication_cell"}, indent=1))
        print("\nper-regime table (cost, width, trainacc, nA/nB/mid, dip p, holm):")
        for d in results:
            print(f"  c={d['cost']:<4} W={d['width']:<2} tracc={d['train_acc_mean']:.3f} "
                  f"{d['nA']:>3}/{d['nB']:>3}/{d['nmid']:>3}  p={d['p_dip']:.4f} "
                  f"boot={d['p_dip_boot']:.4f} holm={'YES' if d['holm_reject'] else 'no'}"
                  f"{' UNDERFIT' if d['underfit'] else ''}")
