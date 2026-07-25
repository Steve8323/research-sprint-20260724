"""
r2-commitment-meter: a cheap per-model COMMITMENT METER (per-pattern determinism D and/or
logit margin M), measured PRE-fine-tune, and whether it predicts each seed's own CORRECTION
DOSE (continuous fine-tune fraction f needed to flip that seed to aligned).

Prereg: PREREG.md (written before this run). Task/model reimplemented from the READ-ONLY
misgen-hedger-mode/run.py (stats/paper_stats.py config). Writes only into this directory.
"""
import json, time, itertools
import numpy as np
import torch, torch.nn as nn
from scipy import stats as sstats

torch.set_num_threads(6)
OUT = "/Users/steven/research_sprint_20260724/r2-commitment-meter"

# ---- exact stats/paper_stats.py config ----
NBITS = 2; NNOISE = 3; INPUT = 7; MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
SEEDBASE = 40000                      # fresh independent seedbase
FT_STEPS = 1000
DEPLOY_N = 4000
K_NOISE = 500                         # noise draws per disagreement pattern for the meter
# per-seed dose search grid (percent) + bisection refinements
FGRID_PCT = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
N_REFINE = 3
CENSOR_PCT = 8.0
TARGET_NONAL = 60
MAX_SEEDS = 260
WALL_GUARD_S = 11 * 60


def xor_block(t, rng):
    n = len(t); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], t)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, k_flip=None, f=0.0):
    """train/deploy data. If k_flip is not None: exact-k nested anti-proxy flips (deterministic
    prefix of a fixed random mask); else Bernoulli(f)."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    if mode == "deploy":
        a = 1 - v
    else:
        a = v.copy()
        if k_flip is not None and k_flip > 0:
            idx = rng.permutation(n)[:k_flip]          # exact-k, nested (same rng order across k)
            a[idx] = 1 - v[idx]
        elif k_flip is None and f > 0:
            m = rng.random(n) < f; a[m] = 1 - v[m]
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


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
    return m


def finetune_k(state, seed, k_flip, ft_steps=FT_STEPS):
    m = mk(); m.load_state_dict(state)
    Xft, yft = gen_align(N, seed + 12345, "train", k_flip=k_flip)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(ft_steps):
        opt.zero_grad(); lf(m(Xft), yft).backward(); opt.step()
    return m, Xft, yft


# ---- disagreement / agreement pattern geometry ----
CORNERS = np.array(list(itertools.product([0, 1], repeat=4)))  # bA0,bA1,bB0,bB1
XA = CORNERS[:, 0] ^ CORNERS[:, 1]      # true-cue bit
XB = CORNERS[:, 2] ^ CORNERS[:, 3]      # proxy-cue bit
AGREE = XA == XB
DISAGREE = ~AGREE
DIS_CORNERS = CORNERS[DISAGREE]         # 8 x 4 cue-sign patterns (disagreement)
AGR_CORNERS = CORNERS[AGREE]            # 8 x 4 (agreement) -- for the negative control
XA_DIS = XA[DISAGREE]                    # true-cue label on each disagreement pattern
XA_AGR = XA[AGREE]


def _pattern_inputs(corners8, k, meter_seed):
    """Build (8*k, 7) input tensor: each of the 8 cue-sign patterns repeated k times with fresh
    noise on all 7 dims (cue dims get MARGIN sign + noise; 3 noise dims noise). Shared across
    all models via a fixed rng seed."""
    rng = np.random.default_rng(meter_seed)
    base = MARGIN * (2 * corners8 - 1)                        # 8 x 4
    Xs = []
    for j in range(8):
        cue = np.tile(base[j], (k, 1)) + NOISE * rng.standard_normal((k, 4)).astype(np.float32)
        z = NOISE * rng.standard_normal((k, NNOISE)).astype(np.float32)
        Xs.append(np.concatenate([cue, z], 1).astype(np.float32))
    return torch.tensor(np.concatenate(Xs, 0))               # (8k,7)


# Precompute shared meter inputs once (identical noise across all models)
_MSEED = SEEDBASE + 99001
XMET_DIS = _pattern_inputs(DIS_CORNERS, K_NOISE, _MSEED)      # (8*K,7)
XMET_AGR = _pattern_inputs(AGR_CORNERS, K_NOISE, _MSEED + 7)  # (8*K,7)


def meter(m, Xmet, xa_true):
    """Returns (D, M): per-pattern determinism and mean absolute logit margin.
    D = mean_k |2 p_k -1|, p_k = P(pred follows the TRUE cue) over K noise draws of pattern k.
    M = mean over all inputs of |logit1 - logit0|."""
    with torch.no_grad():
        out = m(Xmet)                                        # (8K,2)
        pred = out.argmax(1).numpy().reshape(8, K_NOISE)     # follows-true if pred==xa_true[k]
        margin = (out[:, 1] - out[:, 0]).abs().numpy()
    follow = (pred == xa_true[:, None]).mean(1)              # p_k per pattern (8,)
    D = float(np.mean(np.abs(2 * follow - 1)))
    M = float(np.mean(margin))
    return D, M


def per_seed_dose(state, seed, Xdep, ydep):
    """Titrate f to the flip crossing. Scan FGRID_PCT (exact-k), find first flip; bisect between
    last non-flip and first flip N_REFINE times. Returns (dose_pct, censored, n_ft_calls,
    train_acc_at_dose)."""
    calls = 0
    grid = FGRID_PCT
    flipped_at = None; last_noflip = 0.0
    ta_at = None
    for f in grid:
        k = int(round(f / 100.0 * N))
        m2, Xft, yft = finetune_k(state, seed, k); calls += 1
        s2 = accm(m2, Xdep, ydep)
        if s2 > 0.8:
            flipped_at = f; ta_at = accm(m2, Xft, yft); break
        last_noflip = f
    if flipped_at is None:
        return CENSOR_PCT, True, calls, None
    # bisection between last_noflip and flipped_at
    lo, hi = last_noflip, flipped_at
    dose = flipped_at
    for _ in range(N_REFINE):
        mid = 0.5 * (lo + hi)
        k = int(round(mid / 100.0 * N))
        m2, Xft, yft = finetune_k(state, seed, k); calls += 1
        s2 = accm(m2, Xdep, ydep)
        if s2 > 0.8:
            hi = mid; dose = mid; ta_at = accm(m2, Xft, yft)
        else:
            lo = mid
    return float(dose), False, calls, (float(ta_at) if ta_at is not None else None)


# ---- stats helpers ----
def boot_ci_spearman(x, y, B=5000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    rng = np.random.default_rng(seed); rs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        rs.append(sstats.spearmanr(x[idx], y[idx]).correlation)
    rs = np.array([r for r in rs if np.isfinite(r)])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def partial_spear(x, y, z):
    """Partial Spearman of x,y controlling z (rank-regress-residual; Pearson on residual ranks)."""
    rx, ry, rz = (sstats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = sstats.pearsonr(ex, ey)
    return float(r), float(p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    T0 = time.time()
    Xdep, ydep = gen_align(DEPLOY_N, SEEDBASE + 50001, "deploy")

    # ---- 1. sweep + train until >=60 non-aligned ----
    seeds = []; models = {}; scores = {}
    nonal = []; aligned = []
    s = 0
    while True:
        seed = SEEDBASE + s
        m = train(seed)
        sc = accm(m, Xdep, ydep)
        models[seed] = m; scores[seed] = sc; seeds.append(seed)
        grp = "aligned" if sc > 0.8 else ("misaligned" if sc < 0.2 else "hedger")
        if grp == "aligned":
            aligned.append(seed)
        else:
            nonal.append(seed)
        s += 1
        if (len(nonal) >= TARGET_NONAL and s >= 1) or s >= MAX_SEEDS or (time.time() - T0) > WALL_GUARD_S * 0.45:
            break
    groups = {seed: ("aligned" if scores[seed] > 0.8 else
                     ("misaligned" if scores[seed] < 0.2 else "hedger")) for seed in seeds}
    n_al = len(aligned); n_non = len(nonal)
    n_mis = sum(groups[s] == "misaligned" for s in nonal)
    n_hed = sum(groups[s] == "hedger" for s in nonal)
    print(f"SWEEP: {len(seeds)} seeds -> aligned {n_al}, misaligned {n_mis}, hedger {n_hed} "
          f"(non-aligned {n_non}) [{time.time()-T0:.0f}s]", flush=True)

    # ---- 2. commitment meter (PRE-FT) for every non-aligned seed ----
    rows = []
    for seed in nonal:
        D_dis, M_dis = meter(models[seed], XMET_DIS, XA_DIS)
        D_agr, M_agr = meter(models[seed], XMET_AGR, XA_AGR)   # negative control (agree patterns)
        rows.append(dict(seed=seed, group=groups[seed], deploy=float(scores[seed]),
                         D=D_dis, M=M_dis, D_agree=D_agr, M_agree=M_agr))
    D = np.array([r["D"] for r in rows]); M = np.array([r["M"] for r in rows])
    # standardized combined meter
    zD = (D - D.mean()) / (D.std() + 1e-9); zM = (M - M.mean()) / (M.std() + 1e-9)
    C = zD + zM
    for i, r in enumerate(rows):
        r["combined"] = float(C[i])
    print(f"METER: D range [{D.min():.3f},{D.max():.3f}] med {np.median(D):.3f}; "
          f"M med {np.median(M):.3f} [{time.time()-T0:.0f}s]", flush=True)

    # ---- 3. per-seed correction dose (titration) ----
    total_calls = 0
    for r in rows:
        dose, cens, calls, ta = per_seed_dose(models[r["seed"]].state_dict(), r["seed"], Xdep, ydep)
        r["dose_pct"] = dose; r["censored"] = cens; r["ft_calls"] = calls; r["train_acc_at_dose"] = ta
        total_calls += calls
        if (len([x for x in rows if "dose_pct" in x]) % 20) == 0:
            print(f"  dosed {sum('dose_pct' in x for x in rows)}/{n_non} "
                  f"(calls {total_calls}) [{time.time()-T0:.0f}s]", flush=True)
    dose = np.array([r["dose_pct"] for r in rows])
    cens = np.array([r["censored"] for r in rows])
    n_cens = int(cens.sum())
    print(f"DOSE: median {np.median(dose):.3f}% ; censored {n_cens}/{n_non} ; "
          f"total FT calls {total_calls} [{time.time()-T0:.0f}s]", flush=True)

    grp_bin = np.array([1 if r["group"] == "hedger" else 0 for r in rows])  # 1=hedger,0=misaligned
    init = np.array([r["deploy"] for r in rows])

    def spear(x, y):
        rr = sstats.spearmanr(x, y)
        return float(rr.correlation), float(rr.pvalue)

    # ---- PRIMARY: combined meter vs dose ----
    rho_c, p_c = spear(C, dose)
    ci_c = boot_ci_spearman(C, dose, seed=11)
    # meter individual
    rho_D, p_D = spear(D, dose); ci_D = boot_ci_spearman(D, dose, seed=12)
    rho_M, p_M = spear(M, dose); ci_M = boot_ci_spearman(M, dose, seed=13)
    # Kendall robustness (primary meter)
    tau_c, ptau_c = sstats.kendalltau(C, dose)
    # drop-censored robustness (primary meter)
    keep = ~cens
    rho_c_nc, p_c_nc = spear(C[keep], dose[keep]) if keep.sum() > 3 else (float("nan"), float("nan"))

    primary_pass = bool((ci_c[0] > 0) and (rho_c > 0))
    print(f"PRIMARY combined vs dose: rho={rho_c:.3f} p={p_c:.2g} CI{ci_c} -> "
          f"{'PASS' if primary_pass else 'FAIL'}", flush=True)
    print(f"  D vs dose rho={rho_D:.3f} CI{ci_D} | M vs dose rho={rho_M:.3f} CI{ci_M}", flush=True)

    # ---- SECONDARY: beat the worldview label? ----
    # 1. partial Spearman(meter, dose | group)
    pr_c, pp_c = partial_spear(C, dose, grp_bin)
    pr_D, pp_D = partial_spear(D, dose, grp_bin)
    pr_M, pp_M = partial_spear(M, dose, grp_bin)
    # label-alone: Spearman(group, dose) for reference
    rho_grp, p_grp = spear(grp_bin, dose)
    # 2. within-group
    within = {}
    for gname, gval in (("misaligned", 0), ("hedger", 1)):
        mask = grp_bin == gval
        if mask.sum() > 3:
            rho_g, p_g = spear(C[mask], dose[mask])
            within[gname] = dict(n=int(mask.sum()), rho=rho_g, p=p_g,
                                 dose_med=float(np.median(dose[mask])))
        else:
            within[gname] = dict(n=int(mask.sum()), rho=None, p=None)
    secondary_pass = bool(pr_c > 0 and pp_c < 0.05)
    print(f"SECONDARY partial(combined,dose|group): r={pr_c:.3f} p={pp_c:.2g} -> "
          f"{'beats label' if secondary_pass else 'no (label suffices/ null)'}", flush=True)
    print(f"  label-alone rho(group,dose)={rho_grp:.3f} p={p_grp:.2g}", flush=True)
    print(f"  within: {within}", flush=True)

    # ---- negative control: agree-pattern D vs dose (should be weaker) ----
    Dagr = np.array([r["D_agree"] for r in rows])
    rho_Dagr, p_Dagr = spear(Dagr, dose)
    neg_ctrl_ok = bool(abs(rho_D) > abs(rho_Dagr))
    print(f"NEG-CTRL: |rho(D_disagree)|={abs(rho_D):.3f} vs |rho(D_agree)|={abs(rho_Dagr):.3f} "
          f"-> {'specific' if neg_ctrl_ok else 'not specific'}", flush=True)

    # ---- positive control: aligned models stay aligned under FT ----
    pos = {}
    for f in (0.0, 1.0):
        k = int(round(f / 100.0 * N))
        stay = 0
        for seed in aligned:
            m2, _, _ = finetune_k(models[seed].state_dict(), seed, k)
            if accm(m2, Xdep, ydep) > 0.8:
                stay += 1
        lo, hi = wilson(stay, n_al)
        pos[str(f)] = dict(stay=stay, n=n_al, wilson=[lo, hi])
    print(f"POS-CTRL aligned stay-aligned: {pos} [{time.time()-T0:.0f}s]", flush=True)

    # ---- exploratory: meter vs initial alignment, dose vs init ----
    rho_Cinit, p_Cinit = spear(C, init)
    rho_doseinit, p_doseinit = spear(init, dose)
    # sanity train acc at dose (non-censored)
    tas = [r["train_acc_at_dose"] for r in rows if r["train_acc_at_dose"] is not None]
    dose_hist = np.histogram(dose, bins=[0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8.0001])[0].tolist()

    out = dict(
        config=dict(H=H, WD=WD, LR=LR, EPOCHS=EPOCHS, N=N, SEEDBASE=SEEDBASE,
                    FT_STEPS=FT_STEPS, DEPLOY_N=DEPLOY_N, K_NOISE=K_NOISE,
                    FGRID_PCT=FGRID_PCT, N_REFINE=N_REFINE, CENSOR_PCT=CENSOR_PCT),
        sweep=dict(n_seeds=len(seeds), aligned=n_al, misaligned=n_mis, hedger=n_hed,
                   n_nonaligned=n_non, base_rate_aligned=n_al / len(seeds),
                   base_rate_mis=n_mis / len(seeds), base_rate_hed=n_hed / len(seeds)),
        dose_summary=dict(median=float(np.median(dose)), mean=float(np.mean(dose)),
                          n_censored=n_cens, hist=dose_hist,
                          bins="[0,.25,.5,.75,1,1.5,2,3,5,8]"),
        PRIMARY=dict(meter="combined z(D)+z(M)", rho=rho_c, p=p_c, ci95=list(ci_c),
                     kendall_tau=float(tau_c), kendall_p=float(ptau_c),
                     rho_drop_censored=rho_c_nc, p_drop_censored=p_c_nc,
                     n=n_non, n_censored=n_cens, decision="PASS" if primary_pass else "FAIL"),
        meters=dict(D=dict(rho=rho_D, p=p_D, ci95=list(ci_D)),
                    M=dict(rho=rho_M, p=p_M, ci95=list(ci_M))),
        SECONDARY=dict(
            partial_combined=dict(r=pr_c, p=pp_c, decision="beats_label" if secondary_pass else "null"),
            partial_D=dict(r=pr_D, p=pp_D), partial_M=dict(r=pr_M, p=pp_M),
            label_alone_rho_group_dose=[rho_grp, p_grp],
            within_group=within),
        neg_control_agree_D=dict(rho=rho_Dagr, p=p_Dagr, rho_disagree_D=rho_D,
                                 specific=neg_ctrl_ok),
        pos_control_aligned=pos,
        exploratory=dict(rho_combined_init=[rho_Cinit, p_Cinit],
                         rho_dose_init=[rho_doseinit, p_doseinit],
                         train_acc_at_dose_min=float(min(tas)) if tas else None,
                         train_acc_at_dose_mean=float(np.mean(tas)) if tas else None,
                         total_ft_calls=total_calls),
        per_model=rows,
        runtime_s=float(time.time() - T0))
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved results.json  [total {time.time()-T0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
