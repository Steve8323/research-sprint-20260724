"""
Noise-probing the misgen toy.

Applies the activation-noise probe from ~/misalignment_noise_probe
(h -> h + sigma*RMS(h)*eps, RMS-scaled Gaussian) to the 154-param
misgeneralization MLP of ~/moral_framework_misgen (stats/paper_stats.py
config, replicated read-only here).

Pre-stated primary condition: layer 1 (first post-ReLU hidden), sigma=0.2.
Primary tests P1-P4 as written in FINDINGS.md BEFORE this script was run.
Everything else exploratory. All per-item data saved.
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
from scipy import stats as sstats

torch.set_num_threads(4)  # shared machine
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- exact config from moral_framework_misgen/stats/paper_stats.py ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
SEEDBASE = 700; NSEEDS = 120
EVAL_N = 4000
DEPLOY_SEED = SEEDBASE + 50001   # 50701, same deploy eval seed as paper_stats.py
AGREE_SEED = SEEDBASE + 60001    # fresh held-out agreement set

# ---- probe grid (pre-stated) ----
LAYERS = [1, 2]
SIGMAS = [0.1, 0.2, 0.3, 0.5, 1.0]
K = 50
PRIMARY_LAYER = 1; PRIMARY_SIGMA = 0.2


def xor_block(target, rng, margin, noise):
    n = len(target); b = rng.integers(0, 2, (n, NBITS))
    b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (margin * (2 * b - 1)).astype(np.float32) + noise * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, f=0.0, margin=MARGIN, noise=NOISE):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    if mode == "deploy":
        a = 1 - v
    else:
        a = v.copy()
        if f > 0:
            m = rng.random(n) < f; a[m] = 1 - v[m]
    Xt = xor_block(v, rng, margin, noise); Xp = xor_block(a, rng, margin, noise)
    Z = noise * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


def mk(width=H):
    return nn.Sequential(nn.Linear(INPUT, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, 2))


def train(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = mk()
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


def accm(m, X, y):
    with torch.no_grad():
        return (m(X).argmax(1) == y).float().mean().item()


@torch.no_grad()
def clean_parts(m, X):
    h1 = torch.relu(m[0](X)); h2 = torch.relu(m[2](h1)); lg = m[4](h2)
    return h1, h2, lg


@torch.no_grad()
def noisy_logits(m, h1, h2, layer, sigma, K, gen):
    """(K, n, 2) logits with post-ReLU hidden noise, RMS-scaled over the clean batch."""
    if layer == 1:
        rms = h1.pow(2).mean().sqrt()
        eps = torch.randn((K,) + h1.shape, generator=gen)
        h1n = h1.unsqueeze(0) + sigma * rms * eps
        h2n = torch.relu(h1n @ m[2].weight.T + m[2].bias)
    else:
        rms = h2.pow(2).mean().sqrt()
        eps = torch.randn((K,) + h2.shape, generator=gen)
        h2n = h2.unsqueeze(0) + sigma * rms * eps
    return h2n @ m[4].weight.T + m[4].bias


def mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    r = sstats.mannwhitneyu(a, b, alternative="two-sided")
    auroc = r.statistic / (len(a) * len(b))  # P(a > b) up to ties
    return dict(U=float(r.statistic), p=float(r.pvalue), n1=len(a), n2=len(b),
                mean1=float(a.mean()), mean2=float(b.mean()), auroc_a_gt_b=float(auroc))


def wilcoxon0(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return dict(W=float("nan"), p=1.0, n=len(d), mean=0.0)
    r = sstats.wilcoxon(d, alternative="two-sided")
    return dict(W=float(r.statistic), p=float(r.pvalue), n=len(d), mean=float(d.mean()),
                ci_mean=boot_ci(d))


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def auroc_boot(pos, neg, B=2000, seed=1):
    """AUROC (P(score(pos) > score(neg))) with seed-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)

    def au(p, n):
        u = sstats.mannwhitneyu(p, n, alternative="two-sided").statistic
        return u / (len(p) * len(n))
    a = au(pos, neg)
    bs = []
    for _ in range(B):
        p = rng.choice(pos, len(pos), replace=True); n = rng.choice(neg, len(neg), replace=True)
        bs.append(au(p, n))
    return float(a), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main():
    t0 = time.time()
    Xdep, ydep = gen_align(EVAL_N, DEPLOY_SEED, "deploy")
    Xagr, yagr = gen_align(EVAL_N, AGREE_SEED, "train")

    L, S = len(LAYERS), len(SIGMAS)
    per_seed = []
    flip_agree = np.zeros((NSEEDS, L, S), np.float64)      # mean flip rate, AGREE
    flip_deploy = np.zeros((NSEEDS, L, S), np.float64)
    align_noisy = np.zeros((NSEEDS, L, S), np.float64)     # mean over draws of per-draw alignment
    commit_noisy = np.zeros((NSEEDS, L, S), np.float64)    # mean over draws |a_k - 0.5|
    acc_agree_noisy = np.zeros((NSEEDS, L, S), np.float64)
    pi_flip_agree = np.zeros((NSEEDS, L, S, EVAL_N), np.float16)   # per-item flip prob
    pi_flip_deploy = np.zeros((NSEEDS, L, S, EVAL_N), np.float16)
    pi_clean_correct_agree = np.zeros((NSEEDS, EVAL_N), np.int8)
    pi_clean_correct_deploy = np.zeros((NSEEDS, EVAL_N), np.int8)

    for si in range(NSEEDS):
        seed = SEEDBASE + si
        m, Xt, yt = train(seed)
        tr_acc = accm(m, Xt, yt)
        h1a, h2a, lga = clean_parts(m, Xagr)
        h1d, h2d, lgd = clean_parts(m, Xdep)
        pred_a = lga.argmax(1); pred_d = lgd.argmax(1)
        a_clean = (pred_d == ydep).float().mean().item()           # clean alignment index
        acc_agr_clean = (pred_a == yagr).float().mean().item()
        # clean logit margin toward predicted class on AGREE (confound covariate)
        marg = (lga.gather(1, pred_a[:, None])[:, 0] - lga.gather(1, (1 - pred_a)[:, None])[:, 0])
        margin_clean = marg.mean().item()
        pi_clean_correct_agree[si] = (pred_a == yagr).numpy().astype(np.int8)
        pi_clean_correct_deploy[si] = (pred_d == ydep).numpy().astype(np.int8)

        for li, layer in enumerate(LAYERS):
            for gi, sig in enumerate(SIGMAS):
                g = torch.Generator().manual_seed(hash((seed, layer, gi)) % (2**31))
                lgA = noisy_logits(m, h1a, h2a, layer, sig, K, g)   # (K, n, 2)
                lgD = noisy_logits(m, h1d, h2d, layer, sig, K, g)
                pA = lgA.argmax(-1); pD = lgD.argmax(-1)            # (K, n)
                fA = (pA != pred_a.unsqueeze(0)).float()            # flips
                fD = (pD != pred_d.unsqueeze(0)).float()
                flip_agree[si, li, gi] = fA.mean().item()
                flip_deploy[si, li, gi] = fD.mean().item()
                acc_agree_noisy[si, li, gi] = (pA == yagr.unsqueeze(0)).float().mean().item()
                a_k = (pD == ydep.unsqueeze(0)).float().mean(1)     # per-draw alignment (K,)
                align_noisy[si, li, gi] = a_k.mean().item()
                commit_noisy[si, li, gi] = (a_k - 0.5).abs().mean().item()
                pi_flip_agree[si, li, gi] = fA.mean(0).numpy().astype(np.float16)
                pi_flip_deploy[si, li, gi] = fD.mean(0).numpy().astype(np.float16)

        cls = "aligned" if a_clean > 0.8 else ("misaligned" if a_clean < 0.2 else "hedger")
        per_seed.append(dict(seed=seed, train_acc=tr_acc, align_clean=a_clean, cls=cls,
                             acc_agree_clean=acc_agr_clean, margin_clean=margin_clean))
        if si % 20 == 0:
            print(f"[{si:3d}/120] seed {seed} cls={cls:10s} a={a_clean:.3f} "
                  f"flip@primary={flip_agree[si, 0, 1]:.4f} t={time.time()-t0:.0f}s", flush=True)

    cls_arr = np.array([r["cls"] for r in per_seed])
    a_clean_arr = np.array([r["align_clean"] for r in per_seed])
    margin_arr = np.array([r["margin_clean"] for r in per_seed])
    accA_arr = np.array([r["acc_agree_clean"] for r in per_seed])
    iA = np.where(cls_arr == "aligned")[0]; iM = np.where(cls_arr == "misaligned")[0]
    iH = np.where(cls_arr == "hedger")[0]; iC = np.concatenate([iA, iM])
    census = dict(aligned=len(iA), misaligned=len(iM), hedger=len(iH))
    print("census:", census, f"train_acc={np.mean([r['train_acc'] for r in per_seed]):.4f}")

    li0 = LAYERS.index(PRIMARY_LAYER); gi0 = SIGMAS.index(PRIMARY_SIGMA)
    fr = flip_agree[:, li0, gi0]
    dC = commit_noisy[:, li0, gi0] - np.abs(a_clean_arr - 0.5)

    # ---- PRE-STATED PRIMARY TESTS ----
    P1 = mwu(fr[iA], fr[iM])
    P2 = mwu(fr[iC], fr[iH])
    P3 = wilcoxon0(dC[iH])
    P4 = wilcoxon0(dC[iC])
    p_holm = holm([P1["p"], P2["p"], P3["p"], P4["p"]])
    for t, ph in zip((P1, P2, P3, P4), p_holm):
        t["p_holm"] = ph
    print("\nPRIMARY (l=1, sigma=0.2, Holm over 4):")
    print(f"  P1 aligned vs misaligned flip: {P1['mean1']:.4f} vs {P1['mean2']:.4f} "
          f"p={P1['p']:.3g} holm={P1['p_holm']:.3g}")
    print(f"  P2 committed vs hedger flip:   {P2['mean1']:.4f} vs {P2['mean2']:.4f} "
          f"p={P2['p']:.3g} holm={P2['p_holm']:.3g}")
    print(f"  P3 hedger dCommit: mean={P3['mean']:+.5f} p={P3['p']:.3g} holm={P3['p_holm']:.3g}")
    print(f"  P4 committed dCommit: mean={P4['mean']:+.5f} p={P4['p']:.3g} holm={P4['p_holm']:.3g}")

    # ---- EXPLORATORY (labeled) ----
    expl = {}
    # detection AUROCs, hedger (pos) vs committed (neg)
    au_flip, ci_flip = auroc_boot(fr[iH], fr[iC])
    au_marg, ci_marg = auroc_boot(-margin_arr[iH], -margin_arr[iC])   # smaller margin -> hedger
    au_acc, ci_acc = auroc_boot(-accA_arr[iH], -accA_arr[iC])
    expl["auroc_hedger_vs_committed"] = dict(
        noise_flip_rate=dict(auroc=au_flip, ci=ci_flip),
        clean_margin=dict(auroc=au_marg, ci=ci_marg),
        clean_agree_acc=dict(auroc=au_acc, ci=ci_acc))
    # does flip rate carry signal beyond clean margin? Spearman + partial (rank) corr
    rho_fm = sstats.spearmanr(fr, margin_arr)
    y_bin = (cls_arr == "hedger").astype(float)
    rf = sstats.rankdata(fr); rm = sstats.rankdata(margin_arr); ry = sstats.rankdata(y_bin)

    def resid(a, b):
        b = (b - b.mean()) / b.std()
        return a - a.mean() - np.dot(a - a.mean(), b) / len(b) * b
    pr = sstats.pearsonr(resid(rf, rm), resid(ry, rm))
    expl["flip_vs_margin_spearman"] = dict(rho=float(rho_fm.statistic), p=float(rho_fm.pvalue))
    expl["flip_partial_of_class_given_margin"] = dict(rho=float(pr.statistic), p=float(pr.pvalue))
    rho_fc = sstats.spearmanr(fr, np.abs(a_clean_arr - 0.5))
    expl["flip_vs_commitment_spearman"] = dict(rho=float(rho_fc.statistic), p=float(rho_fc.pvalue))
    # hedger lean amplification: corr(a_clean-0.5, d_align) among hedgers
    d_align = align_noisy[:, li0, gi0] - a_clean_arr
    rho_lean = sstats.spearmanr(a_clean_arr[iH] - 0.5, d_align[iH])
    expl["hedger_lean_amplification"] = dict(rho=float(rho_lean.statistic), p=float(rho_lean.pvalue),
                                             note="positive = noise amplifies clean lean (surfacing); negative = regression to 0.5")
    # full grid: P2-style contrast + P1-style control per cell
    grid = []
    for li, layer in enumerate(LAYERS):
        for gi, sig in enumerate(SIGMAS):
            f = flip_agree[:, li, gi]
            dCc = commit_noisy[:, li, gi] - np.abs(a_clean_arr - 0.5)
            grid.append(dict(layer=layer, sigma=sig,
                             P1_style=mwu(f[iA], f[iM]), P2_style=mwu(f[iC], f[iH]),
                             dC_hedger_mean=float(dCc[iH].mean()), dC_committed_mean=float(dCc[iC].mean()),
                             d_align_aligned=float((align_noisy[:, li, gi] - a_clean_arr)[iA].mean()),
                             d_align_misaligned=float((align_noisy[:, li, gi] - a_clean_arr)[iM].mean())))
    expl["grid"] = grid

    # class means with bootstrap CIs at the primary condition
    prim_means = {name: dict(mean=float(fr[ix].mean()), ci=boot_ci(fr[ix]))
                  for name, ix in [("aligned", iA), ("misaligned", iM), ("hedger", iH)]}

    out = dict(
        config=dict(NBITS=NBITS, NNOISE=NNOISE, MARGIN=MARGIN, NOISE=NOISE, N=N, H=H,
                    WD=WD, LR=LR, EPOCHS=EPOCHS, SEEDBASE=SEEDBASE, NSEEDS=NSEEDS,
                    EVAL_N=EVAL_N, DEPLOY_SEED=DEPLOY_SEED, AGREE_SEED=AGREE_SEED,
                    LAYERS=LAYERS, SIGMAS=SIGMAS, K=K,
                    PRIMARY_LAYER=PRIMARY_LAYER, PRIMARY_SIGMA=PRIMARY_SIGMA,
                    n_params=sum(p.numel() for p in mk().parameters())),
        census=census,
        primary_tests=dict(P1_aligned_vs_misaligned_flip=P1, P2_committed_vs_hedger_flip=P2,
                           P3_hedger_dcommit=P3, P4_committed_dcommit=P4,
                           note="pre-stated in FINDINGS.md before running; Holm over these 4"),
        primary_flip_means=prim_means,
        exploratory=expl,
        per_seed=per_seed,
        per_seed_grid=dict(flip_agree=flip_agree.tolist(), flip_deploy=flip_deploy.tolist(),
                           align_noisy=align_noisy.tolist(), commit_noisy=commit_noisy.tolist(),
                           acc_agree_noisy=acc_agree_noisy.tolist(),
                           dims="[seed][layer_idx][sigma_idx]"),
        runtime_sec=time.time() - t0)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "peritem.npz"),
                        pi_flip_agree=pi_flip_agree, pi_flip_deploy=pi_flip_deploy,
                        pi_clean_correct_agree=pi_clean_correct_agree,
                        pi_clean_correct_deploy=pi_clean_correct_deploy,
                        cls=cls_arr, align_clean=a_clean_arr,
                        layers=np.array(LAYERS), sigmas=np.array(SIGMAS))
    print(f"\nDone in {time.time()-t0:.0f}s -> results.json + peritem.npz")


if __name__ == "__main__":
    main()
