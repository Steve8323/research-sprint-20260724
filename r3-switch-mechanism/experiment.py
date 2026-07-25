"""
Round-3 mechanism experiment: WHAT IS THE ROUTED WORLDVIEW SWITCH COMPUTING?

Everything about data / model / routing / scoring below is a VERBATIM copy of
routing-worldview-select/experiment.py (round 1) == r2-routing-switch-learned/experiment.py
(round 2). The read-only round-1/2 directories are never imported or written to; the copy is
validated by the preregistered REPLICATION ANCHOR (per-seed metrics must match round-1's
results.json to <1e-6, which would fail on any transcription error).

New here: alpha-gated sweeps (PRIMARY), per-column closed-form rule extraction (SECONDARY 1),
exact additive decomposition + determinism index (SECONDARY 2). See PREREG.md.
"""
import json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr

torch.set_num_threads(2)
DIR = "/Users/steven/research_sprint_20260724/r3-switch-mechanism"
OUT = f"{DIR}/results.json"
R1 = "/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"

# ---- config: verbatim from round-1 ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
CUE_A = [0, 1]; CUE_B = [2, 3]
UNITS_A = list(range(0, 4)); UNITS_B = list(range(4, 8))
SEEDS = list(range(700, 730))
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9
ALPHAS = [round(a, 2) for a in np.linspace(0, 1, 21)]


# ---- data: verbatim ----
def xor_block(target, rng):
    n = len(target); b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)), torch.tensor(v.astype(np.int64))


# ---- model: verbatim ----
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
    return m, Xtr, ytr


def train_control_arm(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


def score_seed(m, Xdep, ydep, Xhld, yhld):
    r = dict(full_align=accm(m, Xdep, ydep),
             selA_align=accm(m, Xdep, ydep, MASK_KEEP_A),
             selA_train=accm(m, Xhld, yhld, MASK_KEEP_A),
             selB_align=accm(m, Xdep, ydep, MASK_KEEP_B),
             selB_train=accm(m, Xhld, yhld, MASK_KEEP_B))
    r["success_designated"] = bool(r["selA_align"] > ALIGN_HI and r["selA_train"] > TRAIN_OK
                                   and r["selB_align"] < ALIGN_LO and r["selB_train"] > TRAIN_OK)
    return r


# =====================================================================================
# NEW: alpha gate, decomposition, closed-form readout
# =====================================================================================
def gate(keep, alpha):
    """1.0 on the kept column's units, alpha on the other column's units. alpha=0 == round-1
    ablation mask; alpha=1 == un-ablated model (h*1 == h)."""
    m = torch.ones(H)
    m[UNITS_B if keep == "A" else UNITS_A] = alpha
    return m


def sweep(m, X, y):
    """deploy alignment vs alpha, both directions."""
    out = {}
    for keep in ("A", "B"):
        out[keep] = [accm(m, X, y, gate(keep, a)) for a in ALPHAS]
    return out


def decompose(m, X):
    """EXACT additive split of the logit difference into the two columns (valid because the l2
    cross-blocks are identically zero after routed training; residual is reported)."""
    with torch.no_grad():
        h1 = torch.relu(m.l1(X)); h2 = torch.relu(m.l2(h1)); out = m.l3(h2)
        u = m.l3.weight[1] - m.l3.weight[0]
        dA = (h2[:, UNITS_A] @ u[UNITS_A]).numpy()
        dB = (h2[:, UNITS_B] @ u[UNITS_B]).numpy()
        db = float(m.l3.bias[1] - m.l3.bias[0])
        dfull = (out[:, 1] - out[:, 0]).numpy()
    return dA, dB, db, dfull


def contribution_sweep(m, X, y, keep="A"):
    """secondary gate: scale only the OTHER column's additive logit contribution."""
    dA, dB, db, _ = decompose(m, X)
    keepd, otherd = (dA, dB) if keep == "A" else (dB, dA)
    yn = y.numpy()
    return [float(((keepd + a * otherd + db > 0).astype(np.int64) == yn).mean()) for a in ALPHAS]


def first_cross(alphas_desc, ys, level):
    """first crossing of `level` scanning alpha from 1 downward; linear interpolation."""
    for k in range(1, len(ys)):
        lo, hi = min(ys[k - 1], ys[k]), max(ys[k - 1], ys[k])
        if lo <= level <= hi:
            if ys[k] == ys[k - 1]: return float(alphas_desc[k])
            f = (level - ys[k - 1]) / (ys[k] - ys[k - 1])
            return float(alphas_desc[k - 1] + f * (alphas_desc[k] - alphas_desc[k - 1]))
    return float(alphas_desc[-1])


def transition(curve_asc):
    """curve_asc = alignment at ALPHAS (ascending alpha). Returns dict with R, alpha_10,
    alpha_90, W per PREREG (scan from alpha=1 downward)."""
    ad = list(reversed(ALPHAS)); ys = list(reversed(curve_asc))
    y1, y0 = ys[0], ys[-1]; R = y0 - y1
    a10 = first_cross(ad, ys, y1 + 0.1 * R)
    a90 = first_cross(ad, ys, y1 + 0.9 * R)
    inside = sum(1 for v in curve_asc if min(y1 + 0.1 * R, y1 + 0.9 * R) < v < max(y1 + 0.1 * R, y1 + 0.9 * R))
    mono = all(curve_asc[k + 1] <= curve_asc[k] + 0.02 for k in range(len(curve_asc) - 1))
    rho = float(spearmanr(ALPHAS, curve_asc).statistic)
    return dict(y_full=y1, y_sel=y0, R=R, alpha_10=a10, alpha_90=a90, W=a10 - a90,
                n_inside=inside, monotone=bool(mono), spearman=rho)


CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def closed_form(m, col, Xdep):
    """(a) weight-derived noiseless truth table on the column's own cue pair, (b) agreement with
    the closed-form sign-product rule on deploy data, (c) weight mass shares."""
    own = CUE_A if col == "A" else CUE_B
    other = CUE_B if col == "A" else CUE_A
    msk = gate(col, 0.0)
    # (a) truth table at the 4 noiseless corners, everything else 0
    Xc = torch.zeros(4, INPUT)
    for k, (s1, s2) in enumerate(CORNERS):
        Xc[k, own[0]] = s1 * MARGIN; Xc[k, own[1]] = s2 * MARGIN
    with torch.no_grad():
        lg = m(Xc, msk)
        pred_corner = lg.argmax(1).numpy().tolist()
        marg_corner = (lg[:, 1] - lg[:, 0]).numpy().tolist()
    want = [1 if s1 * s2 < 0 else 0 for (s1, s2) in CORNERS]
    tt_match = bool(pred_corner == want)
    # (b) agreement with c(x) = 1[x_i*x_j < 0] on deploy data
    cf = (Xdep[:, own[0]] * Xdep[:, own[1]] < 0).long()
    with torch.no_grad():
        agree = float((m(Xdep, msk).argmax(1) == cf).float().mean())
    # (c) weight mass
    W1 = m.l1.weight.detach().abs().numpy()
    units = UNITS_A if col == "A" else UNITS_B
    tot = W1[units].sum()
    return dict(truth_table=pred_corner, truth_table_target=want, tt_match=tt_match,
                corner_margins=[round(v, 3) for v in marg_corner],
                cf_agreement=agree,
                mass_own=float(W1[np.ix_(units, own)].sum() / tot),
                mass_other=float(W1[np.ix_(units, other)].sum() / tot),
                mass_noise=float(W1[np.ix_(units, [4, 5, 6])].sum() / tot))


def other_cue_invariance(m, col, Xdep, nbase=200, nres=200, seed=1234):
    """behavioural: flip rate of the column-only prediction when the OTHER cue block is resampled
    (fresh XOR block + fresh noise on those 2 dims), own block and noise dims held fixed."""
    other = CUE_B if col == "A" else CUE_A
    msk = gate(col, 0.0)
    rng = np.random.default_rng(seed)
    base = Xdep[:nbase]
    with torch.no_grad():
        p0 = m(base, msk).argmax(1).numpy()
    flips = 0.0
    tgt = rng.integers(0, 2, nbase * nres)
    blk = xor_block(tgt, rng)                     # (nbase*nres, 2)
    Xr = base.repeat_interleave(nres, 0).clone()
    Xr[:, other] = torch.tensor(blk)
    with torch.no_grad():
        pr = m(Xr, msk).argmax(1).numpy()
    flips = float((pr != np.repeat(p0, nres)).mean())
    return flips


# ---- determinism index D: protocol copied from misgen-hedger-mode/run.py ----
def determinism(m, shared_noise):
    """8 cue-disagreement sign patterns (v, c_t, c_p); p_k = P(pred = v); D = mean|2p_k - 1|."""
    nd = shared_noise.shape[1]
    ps = []
    for v in (0, 1):
        for ct in (0, 1):
            for cp in (0, 1):
                b = np.zeros(INPUT, dtype=np.float32)
                tb = [ct, ct ^ v]; pb = [cp, cp ^ (1 - v)]
                b[0], b[1] = MARGIN * (2 * tb[0] - 1), MARGIN * (2 * tb[1] - 1)
                b[2], b[3] = MARGIN * (2 * pb[0] - 1), MARGIN * (2 * pb[1] - 1)
                idx = len(ps)
                X = torch.tensor(b) + NOISE * shared_noise[idx]
                with torch.no_grad():
                    pk = float((m(X).argmax(1).numpy() == v).mean())
                ps.append(pk)
    D = float(np.mean([abs(2 * p - 1) for p in ps]))
    return D, ps


# ---- stats ----
def boot_median(x, B=2000, seed=7):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    if len(x) == 0: return [None, None]
    bs = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def main():
    t0 = time.time()
    Xdep, ydep = gen_align(4000, 50001, "deploy")
    Xhld, yhld = gen_align(4000, 60001, "train")
    shared_noise = torch.tensor(np.random.default_rng(999).standard_normal((8, 500, INPUT)).astype(np.float32))

    res = {"config": dict(H=H, WD=WD, LR=LR, EPOCHS=EPOCHS, N=N, MARGIN=MARGIN, NOISE=NOISE,
                          seeds=SEEDS, alphas=ALPHAS,
                          thresholds=dict(hi=ALIGN_HI, lo=ALIGN_LO, train_ok=TRAIN_OK)),
           "routed": {}, "control": {}}

    for arm in ("routed", "control"):
        rows = {}
        for seed in SEEDS:
            m = (train_routed(seed, CUE_A, CUE_B) if arm == "routed" else train_control_arm(seed))[0]
            r = score_seed(m, Xdep, ydep, Xhld, yhld)
            # structural
            W2 = m.l2.weight.detach().numpy()
            r["cross_block_maxabs"] = float(max(np.abs(W2[np.ix_(UNITS_A, UNITS_B)]).max(),
                                                np.abs(W2[np.ix_(UNITS_B, UNITS_A)]).max()))
            dA, dB, db, dfull = decompose(m, Xdep)
            r["additivity_resid_max"] = float(np.abs(dfull - (dA + dB + db)).max())
            r["opp_sign_rate"] = float((np.sign(dA) != np.sign(dB)).mean())
            mx = np.maximum(np.abs(dA), np.abs(dB)); mn = np.minimum(np.abs(dA), np.abs(dB))
            r["balance_ratio"] = float(np.median(mn / np.maximum(mx, 1e-12)))
            r["mean_absdA"] = float(np.abs(dA).mean()); r["mean_absdB"] = float(np.abs(dB).mean())
            r["mean_abs_margin"] = float(np.abs(dfull).mean())
            r["logit_bias"] = db
            D, ps = determinism(m, shared_noise)
            r["D"] = D; r["p_patterns"] = [round(p, 4) for p in ps]
            # sweeps
            sw = sweep(m, Xdep, ydep)
            r["sweep_keepA"] = sw["A"]; r["sweep_keepB"] = sw["B"]
            r["trans_keepA"] = transition(sw["A"]); r["trans_keepB"] = transition(sw["B"])
            cs = contribution_sweep(m, Xdep, ydep, "A")
            r["sweep_contribA"] = cs; r["trans_contribA"] = transition(cs)
            # closed form per column
            for col in ("A", "B"):
                cf = closed_form(m, col, Xdep)
                cf["flip_rate_other_cue"] = other_cue_invariance(m, col, Xdep)
                r[f"col{col}"] = cf
            rows[seed] = r
        res[arm]["per_seed"] = rows
        print(f"[{arm}] trained+analysed {len(rows)} seeds ({time.time()-t0:.0f}s)")

    # ---------------- replication anchor ----------------
    keys = ["full_align", "selA_align", "selA_train", "selB_align", "selB_train"]
    anchor = {}
    r1 = json.load(open(R1))
    for arm in ("routed", "control"):
        ref = {r["seed"]: r for r in r1["arms"][arm]["per_seed"]}
        d = max(abs(res[arm]["per_seed"][s][k] - ref[s][k]) for s in SEEDS for k in keys)
        anchor[arm] = float(d)
    res["replication_anchor_max_abs_diff_vs_round1"] = anchor
    res["replication_anchor_pass"] = bool(max(anchor.values()) < 1e-6)
    print("replication anchor:", anchor, "PASS" if res["replication_anchor_pass"] else "FAIL")

    # ---------------- switchable set S ----------------
    S = [s for s in SEEDS if res["routed"]["per_seed"][s]["success_designated"]]
    res["S"] = S; res["nS"] = len(S)
    print(f"switchable set |S| = {len(S)} / {len(SEEDS)}")

    def R_(arm, s, key): return res[arm]["per_seed"][s][key]

    # ---------------- PRIMARY: graded control ----------------
    W = [R_("routed", s, "trans_keepA")["W"] for s in S]
    Rr = [R_("routed", s, "trans_keepA")["R"] for s in S]
    medW = float(np.median(W))
    verdict = "GRADED" if medW >= 0.25 else ("TOGGLE/THRESHOLD-LIKE" if medW <= 0.10 else "INTERMEDIATE")
    Rc = [R_("control", s, "trans_keepA")["R"] for s in SEEDS]
    mw = mannwhitneyu(Rr, Rc, alternative="two-sided")
    nmono = sum(R_("routed", s, "trans_keepA")["monotone"] for s in S)
    ndial = sum(R_("routed", s, "trans_keepA")["n_inside"] >= 3 for s in S)
    res["PRIMARY_graded_control"] = dict(
        n=len(S), median_W=round(medW, 4), boot95=boot_median(W),
        per_seed_W=[round(w, 4) for w in W], verdict=verdict,
        decision_rule="GRADED if median W>=0.25; TOGGLE if <=0.10; else INTERMEDIATE",
        alpha_10_median=round(float(np.median([R_("routed", s, "trans_keepA")["alpha_10"] for s in S])), 4),
        alpha_90_median=round(float(np.median([R_("routed", s, "trans_keepA")["alpha_90"] for s in S])), 4),
        range_R_median=round(float(np.median(Rr)), 4), range_R_boot95=boot_median(Rr),
        monotone_frac=round(nmono / len(S), 4), monotone_wilson=wilson(nmono, len(S)),
        spearman_median=round(float(np.median([R_("routed", s, "trans_keepA")["spearman"] for s in S])), 4),
        usable_dial_frac=round(ndial / len(S), 4), usable_dial_wilson=wilson(ndial, len(S)),
        negcontrol_range_R_median=round(float(np.median(Rc)), 4), negcontrol_range_R_boot95=boot_median(Rc),
        negcontrol_mannwhitney=dict(U=float(mw.statistic), p=float(mw.pvalue)))
    print("PRIMARY:", verdict, "median W =", round(medW, 4), res["PRIMARY_graded_control"]["boot95"])

    # secondary sweep readouts
    Wb = [R_("routed", s, "trans_keepB")["W"] for s in S]
    Wc = [R_("routed", s, "trans_contribA")["W"] for s in S]
    res["secondary_sweeps"] = dict(
        misselect_keepB=dict(median_W=round(float(np.median(Wb)), 4), boot95=boot_median(Wb),
                             median_R=round(float(np.median([R_("routed", s, "trans_keepB")["R"] for s in S])), 4)),
        contribution_gate=dict(median_W=round(float(np.median(Wc)), 4), boot95=boot_median(Wc),
                               median_R=round(float(np.median([R_("routed", s, "trans_contribA")["R"] for s in S])), 4)))
    print("secondary sweeps:", json.dumps(res["secondary_sweeps"]))

    # ---------------- SECONDARY 1: closed-form column rule ----------------
    def col_stats(arm, seeds):
        tt = [res[arm]["per_seed"][s][f"col{c}"]["tt_match"] for s in seeds for c in ("A", "B")]
        ag = [res[arm]["per_seed"][s][f"col{c}"]["cf_agreement"] for s in seeds for c in ("A", "B")]
        fl = [res[arm]["per_seed"][s][f"col{c}"]["flip_rate_other_cue"] for s in seeds for c in ("A", "B")]
        mo = [res[arm]["per_seed"][s][f"col{c}"]["mass_own"] for s in seeds for c in ("A", "B")]
        ot = [res[arm]["per_seed"][s][f"col{c}"]["mass_other"] for s in seeds for c in ("A", "B")]
        return dict(n_columns=len(tt), tt_match=int(sum(tt)), tt_frac=round(sum(tt) / len(tt), 4),
                    tt_wilson=wilson(sum(tt), len(tt)),
                    cf_agreement_median=round(float(np.median(ag)), 4), cf_agreement_boot95=boot_median(ag),
                    cf_agreement_min=round(float(np.min(ag)), 4),
                    flip_rate_median=round(float(np.median(fl)), 4), flip_rate_boot95=boot_median(fl),
                    flip_rate_max=round(float(np.max(fl)), 4),
                    mass_own_median=round(float(np.median(mo)), 4),
                    mass_other_median=round(float(np.median(ot)), 4))
    cs_r = col_stats("routed", S); cs_c = col_stats("control", SEEDS)
    cs_r["verdict_tt"] = "PASS (>=90% columns implement own-cue XOR)" if cs_r["tt_frac"] >= 0.9 else "FAIL"
    cs_r["verdict_agreement"] = "PASS (median>=0.95)" if cs_r["cf_agreement_median"] >= 0.95 else "FAIL"
    cs_r["verdict_invariance"] = "PASS (median flip<=0.05)" if cs_r["flip_rate_median"] <= 0.05 else "FAIL"
    res["SECONDARY1_closed_form"] = dict(routed_S=cs_r, control_negcontrol=cs_c)
    print("SECONDARY1 routed:", json.dumps({k: cs_r[k] for k in
          ("tt_frac", "cf_agreement_median", "flip_rate_median", "mass_own_median", "mass_other_median")}))
    print("SECONDARY1 control:", json.dumps({k: cs_c[k] for k in
          ("tt_frac", "cf_agreement_median", "flip_rate_median", "mass_own_median", "mass_other_median")}))

    # ---------------- SECONDARY 2: un-ablated model ----------------
    def s2(arm, seeds):
        b = [res[arm]["per_seed"][s]["balance_ratio"] for s in seeds]
        o = [res[arm]["per_seed"][s]["opp_sign_rate"] for s in seeds]
        d = [res[arm]["per_seed"][s]["D"] for s in seeds]
        mg = [res[arm]["per_seed"][s]["mean_abs_margin"] for s in seeds]
        return dict(n=len(seeds),
                    balance_median=round(float(np.median(b)), 4), balance_boot95=boot_median(b),
                    oppsign_median=round(float(np.median(o)), 4), oppsign_boot95=boot_median(o),
                    D_median=round(float(np.median(d)), 4), D_boot95=boot_median(d),
                    D_frac_le_0p5=round(float(np.mean(np.array(d) <= 0.5)), 4),
                    mean_abs_margin_median=round(float(np.median(mg)), 4), mean_abs_margin_boot95=boot_median(mg),
                    additivity_resid_max=float(max(res[arm]["per_seed"][s]["additivity_resid_max"] for s in seeds)),
                    cross_block_maxabs=float(max(res[arm]["per_seed"][s]["cross_block_maxabs"] for s in seeds)))
    a2 = s2("routed", S); c2 = s2("control", SEEDS)
    a2["verdict_mixture"] = ("MIXTURE/vote-cancellation" if (a2["balance_median"] >= 0.5 and a2["oppsign_median"] >= 0.9)
                             else ("DOMINANT+SUPPRESSED" if a2["balance_median"] <= 0.2 else "INTERMEDIATE"))
    a2["verdict_D"] = ("uncommitted/noise-driven (D<=0.5)" if a2["D_median"] <= 0.5
                       else ("deterministic (D>=0.8)" if a2["D_median"] >= 0.8 else "intermediate"))
    mwD = mannwhitneyu([res["routed"]["per_seed"][s]["D"] for s in S],
                       [res["control"]["per_seed"][s]["D"] for s in SEEDS], alternative="two-sided")
    res["SECONDARY2_unablated"] = dict(routed_S=a2, control=c2,
                                       D_mannwhitney_routed_vs_control=dict(U=float(mwD.statistic), p=float(mwD.pvalue)),
                                       hedger_paper_reference=dict(hedger_median_D=0.315, committed_median_D=0.868,
                                                                   source="misgen-hedger-mode/FINDINGS.md"))
    print("SECONDARY2 routed:", a2["verdict_mixture"], "| D:", a2["D_median"], a2["verdict_D"],
          "| balance", a2["balance_median"], "| oppsign", a2["oppsign_median"])
    print("SECONDARY2 control: D:", c2["D_median"], "| balance", c2["balance_median"], "| oppsign", c2["oppsign_median"])

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
