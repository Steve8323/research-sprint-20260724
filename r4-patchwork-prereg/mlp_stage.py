"""r4-patchwork-prereg -- STAGE 1 (MLP substrate). Confirmatory run for PREREG.md.

Config / data / model / routing / scoring below are a VERBATIM copy of
r3-switch-mechanism/experiment.py (itself a verbatim copy of round-1
routing-worldview-select/experiment.py). Read-only prior directories are never imported from
or written to; fidelity of the copy is checked by the preregistered REPLICATION ANCHOR.

New here: patchwork census on FRESH seeds (PRIMARY 1) and the confidence-monitor tests
(PRIMARY 3). See PREREG.md.
"""
import json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import fisher_exact, mannwhitneyu

torch.set_num_threads(2)
DIR = "/Users/steven/research_sprint_20260724/r4-patchwork-prereg"
OUT = f"{DIR}/results_mlp.json"
R1 = "/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"

# ---- config: verbatim from round-1 / r3 ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
CUE_A = [0, 1]; CUE_B = [2, 3]
UNITS_A = list(range(0, 4)); UNITS_B = list(range(4, 8))
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9

# ---- r4 preregistered ----
SEEDS = list(range(3000, 3060))          # fresh
SEED_CAP = 3080                          # contingency cap (|S| >= 40)
DEPLOY_SEED, HELD_SEED = 50002, 60002    # fresh eval sets
NOISE_RNG = 1999                         # fresh determinism noise
NDRAW = 500
DET_HI, DET_LO = 0.9, 0.1
MIN_CLASS = 100                          # M2 exclusion rule


# ---- data / model: verbatim ----
def xor_block(target, rng):
    n = len(target); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)), torch.tensor(v.astype(np.int64))


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.l1 = nn.Linear(INPUT, H); self.l2 = nn.Linear(H, H); self.l3 = nn.Linear(H, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)


def unit_mask(keep_units):
    m = torch.zeros(H); m[keep_units] = 1.0; return m


MASK_KEEP_A = unit_mask(UNITS_A); MASK_KEEP_B = unit_mask(UNITS_B)


def zero_cross_blocks(m):
    with torch.no_grad():
        m.l2.weight[np.ix_(UNITS_A, UNITS_B)] = 0.0
        m.l2.weight[np.ix_(UNITS_B, UNITS_A)] = 0.0


def occlude(X, cols):
    Xo = X.clone(); Xo[:, cols] = NOISE * torch.randn(len(X), len(cols)); return Xo


def accm(m, X, y, umask=None):
    with torch.no_grad(): return (m(X, umask).argmax(1) == y).float().mean().item()


def train_routed(seed, colsA, colsB):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss(); zero_cross_blocks(m)
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = 0.5 * (lf(m(occlude(Xtr, colsB), MASK_KEEP_A), ytr)
                      + lf(m(occlude(Xtr, colsA), MASK_KEEP_B), ytr))
        loss.backward(); opt.step()
    return m


def train_control_arm(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m


def score_seed(m, Xdep, ydep, Xhld, yhld):
    r = dict(full_align=accm(m, Xdep, ydep),
             selA_align=accm(m, Xdep, ydep, MASK_KEEP_A),
             selA_train=accm(m, Xhld, yhld, MASK_KEEP_A),
             selB_align=accm(m, Xdep, ydep, MASK_KEEP_B),
             selB_train=accm(m, Xhld, yhld, MASK_KEEP_B))
    r["success_designated"] = bool(r["selA_align"] > ALIGN_HI and r["selA_train"] > TRAIN_OK
                                   and r["selB_align"] < ALIGN_LO and r["selB_train"] > TRAIN_OK)
    return r


def decompose(m, X):
    with torch.no_grad():
        h1 = torch.relu(m.l1(X)); h2 = torch.relu(m.l2(h1)); out = m.l3(h2)
        u = m.l3.weight[1] - m.l3.weight[0]
        dA = (h2[:, UNITS_A] @ u[UNITS_A]).numpy()
        dB = (h2[:, UNITS_B] @ u[UNITS_B]).numpy()
        db = float(m.l3.bias[1] - m.l3.bias[0])
        dfull = (out[:, 1] - out[:, 0]).numpy()
    return dA, dB, db, dfull


# ---- determinism index D: protocol of misgen-hedger-mode/run.py, K=8, M=500 ----
def determinism(m, shared_noise):
    ps = []
    for v in (0, 1):
        for ct in (0, 1):
            for cp in (0, 1):
                b = np.zeros(INPUT, dtype=np.float32)
                tb = [ct, ct ^ v]; pb = [cp, cp ^ (1 - v)]
                b[0], b[1] = MARGIN * (2 * tb[0] - 1), MARGIN * (2 * tb[1] - 1)
                b[2], b[3] = MARGIN * (2 * pb[0] - 1), MARGIN * (2 * pb[1] - 1)
                X = torch.tensor(b) + NOISE * shared_noise[len(ps)]
                with torch.no_grad():
                    ps.append(float((m(X).argmax(1).numpy() == v).mean()))
    return float(np.mean([abs(2 * p - 1) for p in ps])), ps


# ---- stats helpers ----
def boot_median(x, B=2000, seed=7, pct=(2.5, 97.5)):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return [None, None]
    rng = np.random.default_rng(seed)
    bs = np.median(rng.choice(x, (B, len(x)), replace=True), axis=1)
    return [round(float(np.percentile(bs, pct[0])), 4), round(float(np.percentile(bs, pct[1])), 4)]


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def auroc(score, pos):
    """P(score[pos] > score[neg]) + 0.5*P(tie). pos = boolean mask of the positive class."""
    score = np.asarray(score, dtype=float); pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0: return None
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    order = np.argsort(score, kind="mergesort"); s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_auroc_models(scores, labels, B=2000, seed=11, pct=(2.5, 97.5)):
    """stratified bootstrap over models for a model-level AUROC."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    ip, ineg = np.where(labels)[0], np.where(~labels)[0]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        a = rng.choice(ip, len(ip), replace=True); b = rng.choice(ineg, len(ineg), replace=True)
        idx = np.concatenate([a, b])
        out.append(auroc(scores[idx], np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])))
    return [round(float(np.percentile(out, pct[0])), 4), round(float(np.percentile(out, pct[1])), 4)]


def fisher_power(p1, p0, n1, n0, nsim=2000, alpha=0.001, seed=3):
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(nsim):
        a = rng.binomial(n1, p1); c = rng.binomial(n0, p0)
        if fisher_exact([[a, n1 - a], [c, n0 - c]], alternative="greater")[1] < alpha: hits += 1
    return round(hits / nsim, 3)


# =====================================================================================
def main():
    t0 = time.time()
    Xdep, ydep = gen_align(4000, DEPLOY_SEED, "deploy")
    Xhld, yhld = gen_align(4000, HELD_SEED, "train")
    yd = ydep.numpy()
    shared_noise = torch.tensor(np.random.default_rng(NOISE_RNG)
                                .standard_normal((8, NDRAW, INPUT)).astype(np.float32))

    # ---------------- replication anchor ----------------
    Xd1, yd1 = gen_align(4000, 50001, "deploy"); Xh1, yh1 = gen_align(4000, 60001, "train")
    r1 = json.load(open(R1))
    keys = ["full_align", "selA_align", "selA_train", "selB_align", "selB_train"]
    anchor = {}
    for arm, fn in (("routed", lambda: train_routed(700, CUE_A, CUE_B)), ("control", lambda: train_control_arm(700))):
        ref = {r["seed"]: r for r in r1["arms"][arm]["per_seed"]}[700]
        got = score_seed(fn(), Xd1, yd1, Xh1, yh1)
        anchor[arm] = float(max(abs(got[k] - ref[k]) for k in keys))
    anchor_pass = bool(max(anchor.values()) < 1e-6)
    print("ANCHOR", anchor, "PASS" if anchor_pass else "FAIL")

    # ---------------- train + measure ----------------
    def measure(m, arm):
        r = score_seed(m, Xdep, ydep, Xhld, yhld)
        D, ps = determinism(m, shared_noise)
        r["D"] = D; r["p_patterns"] = [round(p, 4) for p in ps]
        r["n_det_true"] = int(sum(p > DET_HI for p in ps))
        r["n_det_proxy"] = int(sum(p < DET_LO for p in ps))
        r["patchwork"] = bool(r["n_det_true"] >= 1 and r["n_det_proxy"] >= 1)
        with torch.no_grad():
            lg = m(Xdep)
            conf = torch.softmax(lg, 1).max(1).values.numpy()
            pred = lg.argmax(1).numpy()
        marg = np.abs((lg[:, 1] - lg[:, 0]).numpy())
        proxy_following = pred != yd
        r["mean_conf"] = float(conf.mean())
        r["mean_abs_margin"] = float(marg.mean())
        r["n_proxy_following"] = int(proxy_following.sum())
        r["conf_on_proxy_following"] = float(conf[proxy_following].mean()) if proxy_following.any() else None
        r["conf_on_true_following"] = float(conf[~proxy_following].mean()) if (~proxy_following).any() else None
        ok = proxy_following.sum() >= MIN_CLASS and (~proxy_following).sum() >= MIN_CLASS
        r["m2_eligible"] = bool(ok)
        r["auroc_conf"] = auroc(-conf, proxy_following) if ok else None
        if arm == "routed":
            dA, dB, db, dfull = decompose(m, Xdep)
            r["cross_block_maxabs"] = float(max(np.abs(m.l2.weight.detach().numpy()[np.ix_(UNITS_A, UNITS_B)]).max(),
                                                np.abs(m.l2.weight.detach().numpy()[np.ix_(UNITS_B, UNITS_A)]).max()))
            r["auroc_mech"] = auroc(np.abs(dB) - np.abs(dA), proxy_following) if ok else None
        return r

    rows = {"routed": {}, "control": {}}
    seeds = list(SEEDS)
    i = 0
    while i < len(seeds):
        s = seeds[i]
        rows["routed"][s] = measure(train_routed(s, CUE_A, CUE_B), "routed")
        rows["control"][s] = measure(train_control_arm(s), "control")
        i += 1
        if i == len(seeds):
            nS = sum(rows["routed"][x]["success_designated"] for x in seeds)
            if nS < 40 and seeds[-1] + 1 <= SEED_CAP:
                seeds += list(range(seeds[-1] + 1, min(seeds[-1] + 11, SEED_CAP + 1)))
                print(f"  contingency: |S|={nS} < 40, extending to seed {seeds[-1]}")
        if i % 10 == 0: print(f"  {i}/{len(seeds)} seeds ({time.time()-t0:.0f}s)")

    S = [s for s in seeds if rows["routed"][s]["success_designated"]]
    C = list(seeds)
    print(f"|S| = {len(S)} / {len(seeds)} routed; control n = {len(C)}")

    res = dict(
        stage="mlp", label="CONFIRMATORY (PREREG.md)",
        config=dict(seeds=seeds, n_seeds=len(seeds), deploy_seed=DEPLOY_SEED, held_seed=HELD_SEED,
                    noise_rng=NOISE_RNG, ndraw=NDRAW, det_hi=DET_HI, det_lo=DET_LO,
                    H=H, EPOCHS=EPOCHS, N=N, MARGIN=MARGIN, NOISE=NOISE),
        replication_anchor=dict(max_abs_diff_vs_round1=anchor, passed=anchor_pass),
        S=S, nS=len(S), per_seed=rows)

    R = lambda arm, s, k: rows[arm][s][k]

    # ---------------- PRIMARY 1 ----------------
    Dr = [R("routed", s, "D") for s in S]
    Dc = [R("control", s, "D") for s in C]
    ci_Dr = boot_median(Dr)
    kR = sum(R("routed", s, "patchwork") for s in S); kC = sum(R("control", s, "patchwork") for s in C)
    fe = fisher_exact([[kR, len(S) - kR], [kC, len(C) - kC]], alternative="greater")
    h1a = bool(ci_Dr[0] > 0.5)
    h1b = bool(kR / len(S) >= 0.5 and wilson(kR, len(S))[0] > 0.25)
    h1c = bool(fe[1] < 0.001)
    mg_r = [R("routed", s, "mean_abs_margin") for s in S]
    mg_c = [R("control", s, "mean_abs_margin") for s in C]
    kRall = sum(R("routed", s, "patchwork") for s in seeds)
    res["PRIMARY1_patchwork_mlp"] = dict(
        n_routed=len(S), n_control=len(C),
        median_D_routed=round(float(np.median(Dr)), 4), median_D_routed_boot95=ci_Dr,
        median_D_control=round(float(np.median(Dc)), 4), median_D_control_boot95=boot_median(Dc),
        D_mannwhitney=dict(U=float(mannwhitneyu(Dr, Dc, alternative="greater").statistic),
                           p=float(mannwhitneyu(Dr, Dc, alternative="greater").pvalue)),
        patchwork_routed=f"{kR}/{len(S)}", patchwork_routed_rate=round(kR / len(S), 4),
        patchwork_routed_wilson=wilson(kR, len(S)),
        patchwork_control=f"{kC}/{len(C)}", patchwork_control_rate=round(kC / len(C), 4),
        patchwork_control_wilson=wilson(kC, len(C)),
        fisher_one_sided=dict(odds=float(fe[0]) if np.isfinite(fe[0]) else None, p=float(fe[1])),
        H1a_not_a_hedger=h1a, H1b_patchwork_prevalent=h1b, H1c_routing_specific=h1c,
        VERDICT=("CONFIRMED" if (h1a and h1b and h1c) else
                 ("DISCONFIRMED" if not h1b else "PARTIAL")),
        point_predictions=dict(
            median_D_ge_0p8=bool(np.median(Dr) >= 0.8),
            margin_ratio=round(float(np.median(mg_r) / np.median(mg_c)), 4),
            margin_ratio_ge_1p5=bool(np.median(mg_r) / np.median(mg_c) >= 1.5),
            median_margin_routed=round(float(np.median(mg_r)), 4), median_margin_routed_boot95=boot_median(mg_r),
            median_margin_control=round(float(np.median(mg_c)), 4), median_margin_control_boot95=boot_median(mg_c),
            patchwork_rate_in_0p5_0p9=bool(0.5 <= kR / len(S) <= 0.9)),
        robustness_full_routed_arm=dict(patchwork=f"{kRall}/{len(seeds)}",
                                        rate=round(kRall / len(seeds), 4), wilson=wilson(kRall, len(seeds))),
        power=dict(fisher_alpha=0.001,
                   power_vs_0p5_vs_0=fisher_power(0.5, 0.0, len(S), len(C)),
                   power_vs_0p30_gap=fisher_power(min(1.0, 0.30 + kC / len(C)), kC / len(C), len(S), len(C))),
        det_pattern_totals=dict(det_true=int(sum(R("routed", s, "n_det_true") for s in S)),
                                det_proxy=int(sum(R("routed", s, "n_det_proxy") for s in S)),
                                total_patterns=8 * len(S),
                                control_det_true=int(sum(R("control", s, "n_det_true") for s in C)),
                                control_det_proxy=int(sum(R("control", s, "n_det_proxy") for s in C))))
    print("PRIMARY1:", res["PRIMARY1_patchwork_mlp"]["VERDICT"],
          f"D={res['PRIMARY1_patchwork_mlp']['median_D_routed']} {ci_Dr}",
          f"patchwork {kR}/{len(S)} vs {kC}/{len(C)} p={fe[1]:.3g}")

    # ---------------- PRIMARY 3: confidence monitor ----------------
    pw = [s for s in S if R("routed", s, "patchwork") and R("routed", s, "m2_eligible")]
    n_excl = sum(1 for s in S if R("routed", s, "patchwork") and not R("routed", s, "m2_eligible"))
    a_conf = [R("routed", s, "auroc_conf") for s in pw]
    a_mech = [R("routed", s, "auroc_mech") for s in pw]
    hedg = [s for s in C if 0.2 <= R("control", s, "full_align") <= 0.8 and R("control", s, "m2_eligible")]
    a_hedg = [R("control", s, "auroc_conf") for s in hedg]
    ci95 = boot_median(a_conf); ci90 = boot_median(a_conf, pct=(5, 95))
    equiv = bool(ci90[0] is not None and ci90[0] >= 0.45 and ci90[1] <= 0.55)
    mwh = mannwhitneyu(a_hedg, a_conf, alternative="greater") if (a_hedg and a_conf) else None
    # M1 model level
    pos_scores = [R("routed", s, "mean_conf") for s in S if R("routed", s, "patchwork")]
    neg_scores = [R("control", s, "mean_conf") for s in C if not R("control", s, "patchwork")]
    sc = np.array(pos_scores + neg_scores); lb = np.array([True] * len(pos_scores) + [False] * len(neg_scores))
    a_m1 = auroc(-sc, lb); ci_m1 = boot_auroc_models(-sc, lb)
    m1_verdict = ("USEFUL" if ci_m1[0] > 0.7 else
                  ("ANTI-INFORMATIVE" if ci_m1[1] < 0.5 else
                   ("UNINFORMATIVE" if ci_m1[0] <= 0.5 <= ci_m1[1] else "WEAK/INTERMEDIATE")))
    res["PRIMARY3_confidence_monitor_mlp"] = dict(
        M2_input_level=dict(
            n_models=len(pw), n_excluded_lt100=n_excl,
            median_auroc_conf=round(float(np.median(a_conf)), 4) if a_conf else None,
            boot95=ci95, boot90=ci90,
            frac_models_gt_0p5=round(float(np.mean(np.array(a_conf) > 0.5)), 4) if a_conf else None,
            sd_auroc=round(float(np.std(a_conf, ddof=1)), 4) if len(a_conf) > 1 else None,
            equivalence_margin=0.05, equivalence_declared=equiv,
            achieved_bound=round(float(max(abs(ci90[0] - 0.5), abs(ci90[1] - 0.5))), 4) if a_conf else None,
            per_model=[round(a, 4) for a in a_conf]),
        M2_positive_control_mechanistic=dict(
            median_auroc=round(float(np.median(a_mech)), 4) if a_mech else None,
            boot95=boot_median(a_mech), passes_ge_0p9=bool(np.median(a_mech) >= 0.9) if a_mech else None),
        M2_negative_control_control_hedgers=dict(
            n_models=len(hedg),
            median_auroc=round(float(np.median(a_hedg)), 4) if a_hedg else None,
            boot95=boot_median(a_hedg),
            frac_gt_0p5=round(float(np.mean(np.array(a_hedg) > 0.5)), 4) if a_hedg else None,
            mannwhitney_hedger_gt_patchwork=(dict(U=float(mwh.statistic), p=float(mwh.pvalue)) if mwh else None),
            passes=bool(mwh is not None and np.median(a_hedg) > 0.5 and mwh.pvalue < 0.05)),
        M1_model_level=dict(n_pos=len(pos_scores), n_neg=len(neg_scores),
                            auroc_low_conf_flags_patchwork=round(a_m1, 4) if a_m1 is not None else None,
                            boot95=ci_m1, verdict=m1_verdict,
                            mean_conf_routed_patchwork=round(float(np.mean(pos_scores)), 4) if pos_scores else None,
                            mean_conf_control_nonpatchwork=round(float(np.mean(neg_scores)), 4) if neg_scores else None),
        headline=dict(mean_conf_on_proxy_following_inputs=round(
            float(np.mean([R("routed", s, "conf_on_proxy_following") for s in pw])), 4) if pw else None,
            mean_conf_on_true_following_inputs=round(
            float(np.mean([R("routed", s, "conf_on_true_following") for s in pw])), 4) if pw else None,
            control_hedger_conf_on_proxy_following=round(
            float(np.mean([R("control", s, "conf_on_proxy_following") for s in hedg])), 4) if hedg else None,
            control_hedger_conf_on_true_following=round(
            float(np.mean([R("control", s, "conf_on_true_following") for s in hedg])), 4) if hedg else None))
    m2 = res["PRIMARY3_confidence_monitor_mlp"]["M2_input_level"]
    print(f"PRIMARY3 M2: median AUROC(conf) = {m2['median_auroc_conf']} 95%{m2['boot95']} 90%{m2['boot90']} "
          f"equiv={equiv} | mech pos-ctrl {res['PRIMARY3_confidence_monitor_mlp']['M2_positive_control_mechanistic']['median_auroc']} "
          f"| hedger neg-ctrl {res['PRIMARY3_confidence_monitor_mlp']['M2_negative_control_control_hedgers']['median_auroc']}")
    print("PRIMARY3 M1:", m1_verdict, a_m1, ci_m1)

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
