"""
misgen-hedger-mode: what are the 55/120 "hedgers" in the stats/ 120-seed run of
~/moral_framework_misgen, and does hedger-ness predict flip susceptibility under
disagreeing-data fine-tuning?

Pre-registered tests: see PREREG.txt (written before this run). Reads the original repo
READ-ONLY; writes only into this directory.
"""
import json, time, itertools
import numpy as np
import torch, torch.nn as nn
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(6)
OUT = "/Users/steven/misgen-hedger-mode"
REPO = "/Users/steven/moral_framework_misgen"

# ---- exact stats/paper_stats.py config ----
NBITS = 2; NNOISE = 3; INPUT = 7; MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000; SEEDBASE = 700; NSEED = 120


def xor_block(t, rng):
    n = len(t); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], t)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, f=0.0):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    if mode == "deploy":
        a = 1 - v
    else:
        a = v.copy()
        if f > 0:
            m = rng.random(n) < f; a[m] = 1 - v[m]
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


def gen_indep(n, seed):
    """Both cues encode INDEPENDENT random bits (never seen in training); for probes."""
    rng = np.random.default_rng(seed)
    r1 = rng.integers(0, 2, n); r2 = rng.integers(0, 2, n)
    Xt = xor_block(r1, rng); Xp = xor_block(r2, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(r1.astype(np.int64)), torch.tensor(r2.astype(np.int64)))


def mk():
    return nn.Sequential(nn.Linear(INPUT, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 2))


def accm(m, X, y):
    with torch.no_grad():
        return (m(X).argmax(1) == y).float().mean().item()


def train(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train"); m = mk(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


def finetune(state, seed, f, ft_steps=1000):
    """geom_persist removal protocol: fresh train-style data (seed+12345) with anti-proxy frac f."""
    m = mk(); m.load_state_dict(state)
    Xft, yft = gen_align(N, seed + 12345, "train", f=f)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(ft_steps):
        opt.zero_grad(); lf(m(Xft), yft).backward(); opt.step()
    return m


def boot_ci_median(x, B=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def clopper_pearson(k, n, alpha=0.05):
    lo = sstats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = sstats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# ---- corner analysis ----
CORNERS = np.array(list(itertools.product([0, 1], repeat=4)))  # bA0,bA1,bB0,bB1
XA = CORNERS[:, 0] ^ CORNERS[:, 1]
XB = CORNERS[:, 2] ^ CORNERS[:, 3]
AGREE = XA == XB          # 8 train-type corners (label = XA there)
DISAGREE = ~AGREE         # 8 corners where the cues disagree
Xcorner = torch.tensor(np.concatenate(
    [MARGIN * (2 * CORNERS - 1), np.zeros((16, 3))], 1).astype(np.float32))


def corner_stats(m):
    with torch.no_grad():
        out = m(Xcorner)
        pred = out.argmax(1).numpy()
        marg = (out[:, 1] - out[:, 0]).abs().numpy()
    train_ok = int((pred[AGREE] == XA[AGREE]).sum())            # /8
    followA = (pred[DISAGREE] == XA[DISAGREE]).astype(int)      # 8-vector
    nA = int(followA.sum())
    return train_ok, nA, followA, float(marg[DISAGREE].mean()), float(marg[AGREE].mean()), pred


def gate_match(followA):
    """Does the 8-corner follow-A pattern equal 'follow A iff bit_k == c' for some bit/polarity?"""
    bits_dis = CORNERS[DISAGREE]  # 8 x 4
    for k in range(4):
        for c in (0, 1):
            if np.array_equal(followA, (bits_dis[:, k] == c).astype(int)):
                return f"bit{k}=={c}"
    return None


def saliency_reliance(m, X):
    Xg = X.clone().requires_grad_(True); out = m(Xg)
    g, = torch.autograd.grad((out[:, 1] - out[:, 0]).sum(), Xg)
    s = g.abs().mean(0).detach().numpy()
    t = s[0:2].sum(); p = s[2:4].sum()
    return float(t / (t + p + 1e-9))


def ablation_drops(m, Xev, yev, nrep=5, seed=0):
    base = accm(m, Xev, yev)
    rng = np.random.default_rng(seed)
    drops = []
    for blk in (slice(0, 2), slice(2, 4)):
        accs = []
        for r in range(nrep):
            Xp = Xev.clone()
            perm = torch.tensor(rng.permutation(len(Xev)))
            Xp[:, blk] = Xev[perm][:, blk]
            accs.append(accm(m, Xp, yev))
        drops.append(base - float(np.mean(accs)))
    return base, drops[0], drops[1]  # clean, dropA(true cue), dropB(proxy)


def probe_decode(m, Xi, r1, r2, ntr=2000):
    with torch.no_grad():
        h1 = torch.relu(m[0](Xi)); h2 = torch.relu(m[2](h1)).numpy()
    out = []
    for tgt in (r1.numpy(), r2.numpy()):
        clf = LogisticRegression(max_iter=1000).fit(h2[:ntr], tgt[:ntr])
        out.append(float(clf.score(h2[ntr:], tgt[ntr:])))
    return out  # decA, decB


def main():
    T0 = time.time()
    saved = json.load(open(f"{REPO}/stats/results.json"))["multimodality"]["scores"]
    saved = np.array(saved)

    # ---- 1. regenerate the 120 models ----
    Xdep, ydep = gen_align(4000, SEEDBASE + 50001, "deploy")
    models, scores, tacc = [], [], []
    for s in range(NSEED):
        m, Xt, yt = train(SEEDBASE + s)
        models.append(m); scores.append(accm(m, Xdep, ydep)); tacc.append(accm(m, Xt, yt))
        if (s + 1) % 30 == 0:
            print(f"  trained {s+1}/{NSEED}  [{time.time()-T0:.0f}s]", flush=True)
    scores = np.array(scores)
    max_dev = float(np.abs(scores - saved).max())
    grp = np.where(scores > 0.8, "aligned", np.where(scores < 0.2, "misaligned", "hedger"))
    nA_grp = {g: int((grp == g).sum()) for g in ("aligned", "misaligned", "hedger")}
    print(f"REPRO: max|score - saved| = {max_dev:.4f}; groups {nA_grp}; train acc {np.mean(tacc):.3f}")
    assert max_dev < 0.02, "reproduction failed"

    # ---- 2. characterization ----
    Xev, yev = gen_align(4000, SEEDBASE + 60001, "train")   # fresh train-mode eval for ablation
    Xi, r1, r2 = gen_indep(3000, SEEDBASE + 70001)          # independent-cue probe data
    rows = []
    for i, m in enumerate(models):
        train_ok, nA, followA, marg_dis, marg_agree, pred = corner_stats(m)
        gate = gate_match(followA)
        rel = saliency_reliance(m, Xdep)
        base, dropA, dropB = ablation_drops(m, Xev, yev, seed=i)
        decA, decB = probe_decode(m, Xi, r1, r2)
        rows.append(dict(seed=SEEDBASE + i, s=float(scores[i]), group=str(grp[i]),
                         train_corners_ok=train_ok, nA=nA, followA=followA.tolist(),
                         gate=gate, margin_disagree=marg_dis, margin_agree=marg_agree,
                         reliance=rel, acc_clean=base, dropA=dropA, dropB=dropB,
                         both_use=float(min(dropA, dropB)), decA=decA, decB=decB,
                         dec_other=float(min(decA, decB))))
    print(f"characterization done [{time.time()-T0:.0f}s]", flush=True)

    R = {g: [r for r in rows if r["group"] == g] for g in ("aligned", "misaligned", "hedger")}

    # T1: mixed tables among hedgers
    hed = R["hedger"]; nh = len(hed)
    mixed = [r for r in hed if 1 <= r["nA"] <= 7]
    k_mixed = len(mixed)
    t1_lo, t1_hi = clopper_pearson(k_mixed, nh)
    # committed-table counts for the committed groups too (context)
    def mixed_frac(rs):
        return sum(1 <= r["nA"] <= 7 for r in rs), len(rs)
    ctx = {g: mixed_frac(R[g]) for g in R}

    # T2: |s - nA/8|
    t2 = {}
    for g in R:
        res = [abs(r["s"] - r["nA"] / 8) for r in R[g]]
        lo, hi = boot_ci_median(res, seed=1)
        t2[g] = dict(median=float(np.median(res)), ci=[lo, hi], n=len(res))

    # T3: both-use hedger vs committed
    bu_h = [r["both_use"] for r in hed]
    bu_c = [r["both_use"] for r in R["aligned"] + R["misaligned"]]
    u3, p3 = sstats.mannwhitneyu(bu_h, bu_c, alternative="greater")

    # corner truth-table integrity
    frac_train_ok = float(np.mean([r["train_corners_ok"] == 8 for r in rows]))

    # exploratory: gates, nA distribution, margins, probes
    nA_hist_h = np.bincount([r["nA"] for r in hed], minlength=9).tolist()
    gates_h = [r["gate"] for r in hed if r["gate"]]
    marg = {g: float(np.median([r["margin_disagree"] for r in R[g]])) for g in R}
    marg_agree = {g: float(np.median([r["margin_agree"] for r in R[g]])) for g in R}
    dec = {g: dict(decA=float(np.median([r["decA"] for r in R[g]])),
                   decB=float(np.median([r["decB"] for r in R[g]])),
                   dec_other=float(np.median([r["dec_other"] for r in R[g]]))) for g in R}
    u_dec, p_dec = sstats.mannwhitneyu([r["dec_other"] for r in hed],
                                       [r["dec_other"] for r in R["aligned"] + R["misaligned"]],
                                       alternative="greater")
    rel_med = {g: float(np.median([r["reliance"] for r in R[g]])) for g in R}
    # within-hedger: reliance vs s
    rho_rel, p_rel = sstats.spearmanr([r["reliance"] for r in hed], [r["s"] for r in hed])
    rho_nA, p_nA = sstats.spearmanr([r["nA"] for r in hed], [r["s"] for r in hed])

    print(f"T1: hedgers mixed-table {k_mixed}/{nh} CI [{t1_lo:.2f},{t1_hi:.2f}] | context {ctx}")
    print(f"T2: median |s-nA/8| = {json.dumps(t2)}")
    print(f"T3: both_use hedger med {np.median(bu_h):.3f} vs committed {np.median(bu_c):.3f} p={p3:.2g}")
    print(f"nA hist (hedgers 0..8): {nA_hist_h} | gates matched: {len(gates_h)}/{k_mixed}")

    # ---- 3. flip susceptibility ----
    nonal = [i for i in range(NSEED) if scores[i] <= 0.8]
    ft = {}
    for f in (0.0, 0.0025, 0.005):
        s_after = {}
        for i in nonal:
            m2 = finetune(models[i].state_dict(), SEEDBASE + i, f)
            s_after[i] = accm(m2, Xdep, ydep)
        ft[f] = s_after
        print(f"  FT f={f}: done [{time.time()-T0:.0f}s]", flush=True)

    s_base = np.array([scores[i] for i in nonal])
    g_non = np.array([grp[i] for i in nonal])
    ftres = {}
    for f, s_after in ft.items():
        sa = np.array([s_after[i] for i in nonal])
        rho, prho = sstats.spearmanr(s_base, sa)
        hi_h = int((sa[g_non == "hedger"] > 0.8).sum()); n_h = int((g_non == "hedger").sum())
        hi_m = int((sa[g_non == "misaligned"] > 0.8).sum()); n_m = int((g_non == "misaligned").sum())
        odds, pfish = sstats.fisher_exact([[hi_h, n_h - hi_h], [hi_m, n_m - hi_m]])
        ftres[str(f)] = dict(
            spearman_rho=float(rho), spearman_p=float(prho),
            aligned_rate_hedger=[hi_h, n_h], aligned_rate_mis=[hi_m, n_m],
            fisher_odds=float(odds), fisher_p=float(pfish),
            mean_after_hedger=float(sa[g_non == "hedger"].mean()),
            mean_after_mis=float(sa[g_non == "misaligned"].mean()),
            delta_hedger=float((sa - s_base)[g_non == "hedger"].mean()),
            delta_mis=float((sa - s_base)[g_non == "misaligned"].mean()),
            s_after=[float(x) for x in sa])
        print(f"F={f}: rho(s,s')={rho:.3f} p={prho:.2g} | aligned-after: hedgers {hi_h}/{n_h} "
              f"vs mis {hi_m}/{n_m} Fisher p={pfish:.2g}")

    # exploratory: does nA predict s' beyond s (hedgers+mis, f=0.005)? partial Spearman via ranks
    sa5 = np.array(ftres["0.005"]["s_after"])
    nA_non = np.array([rows[i]["nA"] for i in nonal], float)
    marg_non = np.array([rows[i]["margin_disagree"] for i in nonal], float)
    def partial_spear(x, y, z):
        rx, ry, rz = (sstats.rankdata(v) for v in (x, y, z))
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return sstats.pearsonr(ex, ey)
    pr_nA = partial_spear(nA_non, sa5, s_base)
    pr_marg = partial_spear(marg_non, sa5, s_base)

    out = dict(
        config=dict(H=H, WD=WD, LR=LR, EPOCHS=EPOCHS, N=N, NSEED=NSEED, SEEDBASE=SEEDBASE,
                    ft_steps=1000, ft_fgrid=[0.0, 0.0025, 0.005], deploy_n=4000),
        repro=dict(max_score_dev=max_dev, groups=nA_grp, train_acc=float(np.mean(tacc)),
                   frac_all16_train_corners_correct=frac_train_ok),
        per_model=rows,
        T1=dict(k_mixed=k_mixed, n_hedgers=nh, frac=k_mixed / nh, ci=[t1_lo, t1_hi],
                mixed_by_group={g: list(ctx[g]) for g in ctx}),
        T2=t2,
        T3=dict(median_hedger=float(np.median(bu_h)), median_committed=float(np.median(bu_c)),
                U=float(u3), p=float(p3), n=[len(bu_h), len(bu_c)]),
        exploratory=dict(nA_hist_hedgers=nA_hist_h, gates_matched=len(gates_h),
                         gate_list=gates_h, margin_disagree_median=marg,
                         margin_agree_median=marg_agree, decode=dec,
                         dec_other_MW_p=float(p_dec), reliance_median=rel_med,
                         spearman_reliance_s_hedgers=[float(rho_rel), float(p_rel)],
                         spearman_nA_s_hedgers=[float(rho_nA), float(p_nA)],
                         partial_nA_after_ctrl_s=[float(pr_nA[0]), float(pr_nA[1])],
                         partial_margin_after_ctrl_s=[float(pr_marg[0]), float(pr_marg[1])]),
        flip=ftres,
        runtime_s=float(time.time() - T0))
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved results.json  [total {time.time()-T0:.0f}s]")


if __name__ == "__main__":
    main()
