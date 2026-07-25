"""
Non-circular removal dose-response.

Fine-tune already-trained misaligned-by-chance models on safety data containing an
EXACT fraction f of disagreeing (anti-proxy, i.e. disambiguating) rows,
f in {0, 0.5, 1, 2, 5, 10}%, and map the removal dose-response with CIs.

Config is identical to ../../geom_persist/persistence.py (mlp8 headline size), but:
  - fresh seedbase 21000 (independent replication, not the published seedbase 700)
  - exact-k nested disagreeing rows instead of iid Bernoulli(f) flips
  - n_target 40 (>= brief's 20), Wilson / bootstrap CIs, pre-registered McNemar primary.

Pre-registration: see FINDINGS.md (written before this ran).
Single process, 1 torch thread, CPU. Writes results.json next to this file.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "arch"))
import common as C                      # read-only import of the task
from common import gen_features, accuracy, _xor_block, NNOISE, NOISE, N
import models as M

torch.set_num_threads(1)

LR, WD, EPOCHS, FT_STEPS, DEPLOY_N = 5e-3, 2e-3, 2000, 1000, 4000
SEEDBASE = 21000
N_TARGET_MIS, N_TARGET_AL, MAX_SEEDS = 40, 12, 250
FGRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
TIME_GUARD_S = 480          # 8 min wall guard on the collection phase
RNG_BOOT = np.random.default_rng(0)


# ---------------------------------------------------------------- data
def gen_train_exact_f(n, seed, f):
    """Train-style rows; EXACTLY k=round(f*n) rows have the proxy flipped (a=1-v).
    Rows to flip = the k smallest of one shared uniform draw -> nested across f."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    u = rng.random(n)                      # consumed identically for every f
    a = v.copy()
    k = int(round(f * n))
    if k > 0:
        idx = np.argsort(u)[:k]
        a[idx] = 1 - v[idx]
    Xt = _xor_block(v, rng); Xp = _xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    X = np.concatenate([Xt, Xp, Z], 1).astype(np.float32)
    return torch.tensor(X), torch.tensor(v.astype(np.int64))


# ---------------------------------------------------------------- train / FT
def train_from_scratch(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_features(N, seed, "train")
    m = M.mlp_make(8, 1)()
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m


def finetune(state, seed, f):
    m = M.mlp_make(8, 1)(); m.load_state_dict(state)
    Xft, yft = gen_train_exact_f(N, seed + 12345, f)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(FT_STEPS):
        opt.zero_grad(); lf(m(Xft), yft).backward(); opt.step()
    return m


# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_mean_ci(x, nboot=10000):
    x = np.asarray(x)
    idx = RNG_BOOT.integers(0, len(x), (nboot, len(x)))
    ms = x[idx].mean(1)
    return (float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


def mcnemar_exact(a, b):
    """a, b: paired boolean success arrays. Two-sided exact binomial on discordants."""
    from scipy.stats import binomtest
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    nd = n01 + n10
    if nd == 0:
        return dict(n01=n01, n10=n10, p=1.0)
    p = binomtest(min(n01, n10), nd, 0.5, alternative="two-sided").pvalue
    return dict(n01=n01, n10=n10, p=float(p))


def logistic_f50(fs, success_mat):
    """MLE 2-param logistic in x=log10(f) (f>0 doses only); returns f50.
    success_mat: [n_seeds, n_doses] bool. Simple Newton/grid-free scipy fit."""
    from scipy.optimize import minimize
    x = np.log10(fs)
    Y = success_mat.astype(float)

    def nll(th):
        b0, b1 = th
        p = 1.0 / (1.0 + np.exp(-(b0 + b1 * x)))    # [n_doses]
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    r = minimize(nll, np.array([0.0, 3.0]), method="Nelder-Mead")
    b0, b1 = r.x
    if abs(b1) < 1e-9:
        return None
    return float(10 ** (-b0 / b1))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    Xdep, ydep = gen_features(DEPLOY_N, SEEDBASE + 50001, "deploy")

    # ---- collect misaligned + aligned models from one fresh sweep
    mis, ali, all_scores = [], [], []
    for s in range(MAX_SEEDS):
        seed = SEEDBASE + s
        m = train_from_scratch(seed)
        dep = accuracy(m, Xdep, ydep)
        Xt, yt = gen_features(N, seed, "train")
        tra = accuracy(m, Xt, yt)
        all_scores.append(dep)
        st = {k: v.clone() for k, v in m.state_dict().items()}
        if dep < 0.2 and len(mis) < N_TARGET_MIS:
            mis.append((seed, st, dep, tra))
        elif dep > 0.8 and len(ali) < N_TARGET_AL:
            ali.append((seed, st, dep, tra))
        if (s + 1) % 25 == 0:
            print(f"  sweep {s+1}/{MAX_SEEDS}: {len(mis)} misaligned, {len(ali)} aligned "
                  f"[t={time.time()-t0:.0f}s]", flush=True)
        if len(mis) >= N_TARGET_MIS and len(ali) >= N_TARGET_AL:
            break
        if time.time() - t0 > TIME_GUARD_S and len(mis) >= 20:
            print("  TIME GUARD hit; proceeding with what we have", flush=True)
            break
    all_scores = np.array(all_scores)
    print(f"sweep done: {len(all_scores)} seeds, {len(mis)} misaligned (<0.2), "
          f"{len(ali)} aligned (>0.8) kept | base rates: mis "
          f"{float((all_scores<0.2).mean()):.3f}, ali {float((all_scores>0.8).mean()):.3f} "
          f"[t={time.time()-t0:.0f}s]", flush=True)

    # ---- dose-response on misaligned models (paired across f)
    n_mis = len(mis)
    deploy_after = np.zeros((n_mis, len(FGRID)))
    train_after = np.zeros((n_mis, len(FGRID)))
    for j, f in enumerate(FGRID):
        for i, (seed, state, dep0, tra0) in enumerate(mis):
            m2 = finetune(state, seed, f)
            deploy_after[i, j] = accuracy(m2, Xdep, ydep)
            Xt, yt = gen_features(N, seed, "train")
            train_after[i, j] = accuracy(m2, Xt, yt)
        rem = deploy_after[:, j] > 0.8
        per = deploy_after[:, j] < 0.2
        lo, hi = wilson(int(rem.sum()), n_mis)
        print(f"  f={f*100:5.2f}% (k={int(round(f*N)):3d}/800) | mean deploy "
              f"{deploy_after[:,j].mean():.3f} | removed {int(rem.sum())}/{n_mis} "
              f"({rem.mean():.2f}, Wilson [{lo:.2f},{hi:.2f}]) | persist {per.mean():.2f} "
              f"| train {train_after[:,j].mean():.3f} [t={time.time()-t0:.0f}s]", flush=True)

    # ---- aligned control
    n_al = len(ali)
    al_after = np.zeros((n_al, len(FGRID)))
    for j, f in enumerate(FGRID):
        for i, (seed, state, dep0, tra0) in enumerate(ali):
            m2 = finetune(state, seed, f)
            al_after[i, j] = accuracy(m2, Xdep, ydep)
    print(f"aligned control done [t={time.time()-t0:.0f}s]", flush=True)

    # ---- stats
    removed = deploy_after > 0.8
    persist = deploy_after < 0.2
    curve = []
    for j, f in enumerate(FGRID):
        k_rem, k_per = int(removed[:, j].sum()), int(persist[:, j].sum())
        curve.append(dict(
            f=f, k_disagree=int(round(f * N)), n=n_mis,
            mean_deploy=float(deploy_after[:, j].mean()),
            mean_deploy_ci95=boot_mean_ci(deploy_after[:, j]),
            n_removed=k_rem, frac_removed=k_rem / n_mis,
            frac_removed_wilson95=wilson(k_rem, n_mis),
            n_persist=k_per, frac_persist=k_per / n_mis,
            frac_persist_wilson95=wilson(k_per, n_mis),
            train_after_mean=float(train_after[:, j].mean()),
            aligned_control_n=n_al,
            aligned_control_still_aligned=int((al_after[:, j] > 0.8).sum()),
            aligned_control_wilson95=wilson(int((al_after[:, j] > 0.8).sum()), n_al),
            aligned_control_mean_deploy=float(al_after[:, j].mean()),
        ))

    j05, j5 = FGRID.index(0.005), FGRID.index(0.05)
    primary = mcnemar_exact(removed[:, j05], removed[:, j5])
    primary.update(dict(
        test="exact McNemar, removal(f=0.5%) vs removal(f=5%), paired over seeds",
        removed_at_0p5=int(removed[:, j05].sum()), removed_at_5=int(removed[:, j5].sum()),
        n=n_mis))

    # secondary: f* and logistic f50 with seed bootstrap
    fstar = next((c["f"] for c in curve if c["frac_removed"] >= 0.9), None)
    fstar_conservative = next((c["f"] for c in curve
                               if c["frac_removed_wilson95"][0] >= 0.5), None)
    fpos = np.array([f for f in FGRID if f > 0])
    Spos = removed[:, [FGRID.index(f) for f in fpos]]
    f50 = logistic_f50(fpos, Spos.mean(0))
    f50_boot = []
    for _ in range(2000):
        idx = RNG_BOOT.integers(0, n_mis, n_mis)
        v = logistic_f50(fpos, Spos[idx].mean(0))
        if v is not None and 1e-5 < v < 1.0:
            f50_boot.append(v)
    f50_ci = (float(np.percentile(f50_boot, 2.5)),
              float(np.percentile(f50_boot, 97.5))) if len(f50_boot) > 100 else None

    out = dict(
        config=dict(LR=LR, WD=WD, EPOCHS=EPOCHS, ft_steps=FT_STEPS, N=N,
                    deploy_n=DEPLOY_N, seedbase=SEEDBASE, kind="mlp8", n_params=82,
                    fgrid=FGRID, exact_k_flips=True, nested_doses=True,
                    removed_def="deploy>0.8", persist_def="deploy<0.2"),
        sweep=dict(n_seeds_trained=len(all_scores),
                   base_misaligned_rate=float((all_scores < 0.2).mean()),
                   base_aligned_rate=float((all_scores > 0.8).mean()),
                   n_misaligned_used=n_mis, n_aligned_control=n_al,
                   deploy_before_mean=float(np.mean([d for (_, _, d, _) in mis]))),
        curve=curve,
        primary_mcnemar=primary,
        fstar_ge90=fstar, fstar_wilson_lb_ge50=fstar_conservative,
        f50_logistic=f50, f50_ci95_bootstrap=f50_ci,
        prevention_reference=dict(source="README / alignment.py",
                                  f_prevent=0.005, note="0.5% during training -> 90% "
                                  "of seeds aligned, 0% misaligned (30 seeds)"),
        per_seed=dict(seeds=[s for (s, _, _, _) in mis],
                      deploy_before=[d for (_, _, d, _) in mis],
                      deploy_after=deploy_after.tolist(),
                      aligned_control_after=al_after.tolist()),
        runtime_s=time.time() - t0,
    )
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nPRIMARY McNemar: removed {primary['removed_at_0p5']}/{n_mis} at f=0.5% vs "
          f"{primary['removed_at_5']}/{n_mis} at f=5%; discordant {primary['n01']}+{primary['n10']}, "
          f"p={primary['p']:.2e}")
    print(f"f*(>=90% removed) = {fstar}, f50(logistic) = {f50}, CI {f50_ci}")
    print(f"wrote results.json [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
