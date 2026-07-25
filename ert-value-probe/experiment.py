#!/usr/bin/env python3
"""Is step-value more readable from legible intermediates?

ERT (words at layer boundaries) vs matched continuous transformer (noisy
continuous residual at layer boundaries), both trained on the multi-step
Task M (one permutation hop per layer). Step-value V*(boundary state) =
P(final answer correct | resume from this boundary state), estimated by K=16
Monte-Carlo resumptions. Small probes predict V* from
(input token one-hot ++ boundary representation).

Pre-registered protocol: see FINDINGS.md (written before this ran).
Writes results.json. CPU only, seeded, single process.
"""
import os, sys, json, time, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EBL = "/Users/steven/english_bottleneck_lm"
sys.path.insert(0, EBL)
OUT = os.path.dirname(os.path.abspath(__file__))

from vocab import build_embeddings, V, N_CLASS, CLASS_OF  # noqa: E402
import tasks                                              # noqa: E402
from tasks import make_batch_M, RPOS, SEQ_LEN             # noqa: E402
from model import (EnglishResidualTransformer, ContinuousBaseline,
                   _tok_to_class_logits, gumbel_st)       # noqa: E402
from paraphrase import ParaphraseChannel                  # noqa: E402

torch.set_num_threads(4)
DEV = "cpu"

# ---------------- fixed config (pre-registered) ----------------
D, LAYERS, EPS, RHO = 64, 4, 0.5, 1.0
B_TRAIN, LR, ANCHOR = 256, 3e-3, 700
# Amendment 1 (pre-main-run, after smoke pilot showed both models cross from
# chance to ~1.0 accuracy before step 100): finer/earlier checkpoint grid and
# larger sigma grid. Selection rule (argmin |acc-0.80|) unchanged.
CKPTS = [30, 50, 75, 100, 150, 250, 400, 700]
SIGMAS = [1.0, 2.0, 4.0, 6.0]
TARGET_ACC = 0.80
K_ROLL = 16
TAU_EVAL = 0.3
N_POOL, N_VAL, N_TEST = 1024, 256, 2048
NS = [32, 64, 128, 256, 512, 1024]
N_PRIMARY = 256
SEEDS = [0, 1, 2, 3, 4]
MLP_EPOCHS = 300
TIME_GUARD_S = 460

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def _tau(step, total, hi=1.0, lo=0.3):
    return hi + (lo - hi) * (step / max(1, total))


# ---------------- models ----------------
class NoisyBaseline(ContinuousBaseline):
    """ContinuousBaseline + additive Gaussian noise on the residual stream at
    every layer boundary (the continuous analog of the ERT's commit+paraphrase
    stochasticity). Trained AND evaluated with the noise."""

    def __init__(self, E, sigma):
        super().__init__(E, LAYERS, seq_len=SEQ_LEN)
        self.sigma = sigma

    def forward_layers(self, ids, **kw):
        x = F.one_hot(ids, V).float() @ self.emb + self.pos.unsqueeze(0)
        outs = []
        L = len(self.blocks)
        for li, (blk, hd) in enumerate(zip(self.blocks, self.heads)):
            x = blk(x)
            outs.append(hd(self.lnf(x)))
            if li < L - 1:
                x = x + self.sigma * torch.randn_like(x)
        return outs


def new_ert(E, seed):
    torch.manual_seed(20_000 + seed)
    para = ParaphraseChannel(rho=RHO, device=DEV)
    return EnglishResidualTransformer(E, LAYERS, paraphrase=para, seq_len=SEQ_LEN)


def new_base(E, sigma, seed):
    torch.manual_seed(30_000 + seed)
    return NoisyBaseline(E, sigma)


# ---------------- training (recipe copied from train.py, tau anchored) ------
def train_M_steps(model, steps, ckpt_eval=None):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    accs = {}
    for s in range(steps):
        ids, lt, _ = make_batch_M(B_TRAIN, LAYERS, DEV)
        outs = model.forward_layers(ids, tau=_tau(s, ANCHOR))
        loss = sum(F.cross_entropy(_tok_to_class_logits(o[:, RPOS, :]), lt[:, l])
                   for l, o in enumerate(outs)) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
        if ckpt_eval and (s + 1) in ckpt_eval:
            accs[s + 1] = eval_acc(model)
            model.train()
    return accs


@torch.no_grad()
def eval_acc(model, B=1500, reps=2):
    model.eval()
    acc = 0.0
    for _ in range(reps):
        ids, _, final = make_batch_M(B, LAYERS, DEV)
        pred = _tok_to_class_logits(model.forward(ids, tau=TAU_EVAL)[:, RPOS, :]).argmax(-1)
        acc += (pred == final).float().mean().item()
    return acc / reps


# ---------------- episode collection + MC value ----------------
@torch.no_grad()
def collect_ert(m, ids):
    m.eval()
    y = F.one_hot(ids, V).float()
    states = []
    logits = None
    for li in range(m.n_layers):
        x = y @ m.E + m.pos.unsqueeze(0)
        x = m.blocks[li](x)
        z = m.lnf(x)
        logits = m.unemb[li](z)
        if li < m.n_layers - 1:
            y = m.para(gumbel_st(logits, TAU_EVAL))
            states.append(dict(tok=y.argmax(-1).clone(),        # (B,T) committed words
                               h=z[:, RPOS, :].clone()))        # (B,d) pre-commit
    pred = _tok_to_class_logits(logits[:, RPOS, :]).argmax(-1)
    return states, pred


@torch.no_grad()
def vstar_ert(m, tok, start_layer, target, K=K_ROLL, chunk=8192):
    B, T = tok.shape
    rep = tok.repeat_interleave(K, 0)
    tgt = target.repeat_interleave(K, 0)
    corr = torch.empty(B * K)
    for i in range(0, B * K, chunk):
        y = F.one_hot(rep[i:i + chunk], V).float()
        logits = None
        for li in range(start_layer, m.n_layers):
            x = y @ m.E + m.pos.unsqueeze(0)
            x = m.blocks[li](x)
            logits = m.unemb[li](m.lnf(x))
            if li < m.n_layers - 1:
                y = m.para(gumbel_st(logits, TAU_EVAL))
        pred = _tok_to_class_logits(logits[:, RPOS, :]).argmax(-1)
        corr[i:i + chunk] = (pred == tgt[i:i + chunk]).float()
    c = corr.view(B, K)
    return (c.mean(1).numpy(), c[:, :K // 2].mean(1).numpy(),
            c[:, K // 2:].mean(1).numpy())


@torch.no_grad()
def collect_base(m, ids):
    m.eval()
    x = F.one_hot(ids, V).float() @ m.emb + m.pos.unsqueeze(0)
    states = []
    logits = None
    for li in range(LAYERS):
        x = m.blocks[li](x)
        logits = m.heads[li](m.lnf(x))
        if li < LAYERS - 1:
            x = x + m.sigma * torch.randn_like(x)          # post-noise = Markov state
            states.append(dict(x=x.clone(), h=x[:, RPOS, :].clone()))
    pred = _tok_to_class_logits(logits[:, RPOS, :]).argmax(-1)
    return states, pred


@torch.no_grad()
def vstar_base(m, xstate, start_layer, target, K=K_ROLL, chunk=8192):
    B = xstate.shape[0]
    rep = xstate.repeat_interleave(K, 0)
    tgt = target.repeat_interleave(K, 0)
    corr = torch.empty(B * K)
    for i in range(0, B * K, chunk):
        x = rep[i:i + chunk].clone()
        logits = None
        for li in range(start_layer, LAYERS):
            x = m.blocks[li](x)
            logits = m.heads[li](m.lnf(x))
            if li < LAYERS - 1:
                x = x + m.sigma * torch.randn_like(x)
        pred = _tok_to_class_logits(logits[:, RPOS, :]).argmax(-1)
        corr[i:i + chunk] = (pred == tgt[i:i + chunk]).float()
    c = corr.view(B, K)
    return (c.mean(1).numpy(), c[:, :K // 2].mean(1).numpy(),
            c[:, K // 2:].mean(1).numpy())


# ---------------- probes ----------------
def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def standardize(Xtr, Xva, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def fit_mlp(Xtr, ytr, Xva, yva, Xte, seed, hid=64, epochs=MLP_EPOCHS, lr=1e-2, wd=1e-4):
    torch.manual_seed(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr, dtype=torch.float32)
    Xva = torch.tensor(Xva, dtype=torch.float32); yva = torch.tensor(yva, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hid), nn.ReLU(), nn.Linear(hid, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best_v, best_state = float("inf"), None
    for e in range(epochs):
        loss = F.mse_loss(net(Xtr).squeeze(-1), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (e + 1) % 10 == 0:
            with torch.no_grad():
                vl = F.mse_loss(net(Xva).squeeze(-1), yva).item()
            if vl < best_v:
                best_v, best_state = vl, copy.deepcopy(net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        return net(Xte).squeeze(-1).numpy()


def fit_ridge(Xtr, ytr, Xva, yva, Xte, alphas=(0.1, 1, 10, 100, 1000)):
    def aug(X):
        return np.hstack([X, np.ones((len(X), 1), X.dtype)])
    Xtr1, Xva1, Xte1 = aug(Xtr), aug(Xva), aug(Xte)
    A = Xtr1.T @ Xtr1
    b = Xtr1.T @ ytr
    best = None
    for al in alphas:
        w = np.linalg.solve(A + al * np.eye(A.shape[0]), b)
        vmse = float(((Xva1 @ w - yva) ** 2).mean())
        if best is None or vmse < best[0]:
            best = (vmse, w)
    return Xte1 @ best[1]


def probe_arm(X, v, seed_tag, n, probe="mlp"):
    """X: (N,Df) full feature matrix; v: (N,) targets.
    Split: [0:N_POOL) train pool, [N_POOL:N_POOL+N_VAL) val, tail = test."""
    Xtr, ytr = X[:n], v[:n]
    Xva, yva = X[N_POOL:N_POOL + N_VAL], v[N_POOL:N_POOL + N_VAL]
    Xte = X[N_POOL + N_VAL:]
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    if probe == "mlp":
        return fit_mlp(Xtr, ytr, Xva, yva, Xte, seed=abs(hash(seed_tag)) % (2 ** 31))
    return fit_ridge(Xtr, ytr, Xva, yva, Xte)


def partial_r_within_input(pred, v, inp):
    """Center pred and v by input-token group means on the test set, then r."""
    pred = np.asarray(pred, float).copy(); v = np.asarray(v, float).copy()
    for t in np.unique(inp):
        m = inp == t
        if m.sum() >= 2:
            pred[m] -= pred[m].mean(); v[m] -= v[m].mean()
        else:
            pred[m] = 0.0; v[m] = 0.0
    return pearson(pred, v)


def onehot_np(idx, n):
    out = np.zeros((len(idx), n), np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


# ---------------- per-seed dataset build ----------------
SUCC = tasks.SUCC  # fixed class permutation
CLASS_NP = CLASS_OF.numpy()


def class_pow(c, k):
    c = np.asarray(c).copy()
    for _ in range(k):
        c = SUCC[c]
    return c


def build_dataset(model, kind, N):
    """Returns dict with ids, per-boundary states, V* targets."""
    ids, _, final = make_batch_M(N, LAYERS, DEV)
    if kind == "ert":
        states, _ = collect_ert(model, ids)
        vs = [vstar_ert(model, st["tok"], l + 1, final) for l, st in enumerate(states)]
    else:
        states, _ = collect_base(model, ids)
        vs = [vstar_base(model, st["x"], l + 1, final) for l, st in enumerate(states)]
    return dict(ids=ids.numpy(), final=final.numpy(), states=states, vs=vs)


def arm_features(ds, kind, arm, l, Epub):
    inp = ds["ids"][:, RPOS]
    inp_oh = onehot_np(inp, V)
    st = ds["states"][l]
    if kind == "ert":
        tok0 = st["tok"][:, RPOS].numpy()
        if arm == "words":
            return np.hstack([inp_oh, onehot_np(tok0, V)])
        if arm == "hidden":                       # ERT pre-commit continuous
            return np.hstack([inp_oh, st["h"].numpy()])
        if arm == "embfeat":
            return np.hstack([Epub[inp], Epub[tok0]])
        if arm == "classfeat":
            return np.hstack([onehot_np(CLASS_NP[inp] + 1, N_CLASS + 1),
                              onehot_np(CLASS_NP[tok0] + 1, N_CLASS + 1)])
        if arm == "words_allpos":
            toks = st["tok"].numpy()
            return np.hstack([inp_oh] + [onehot_np(toks[:, p], V) for p in range(SEQ_LEN)])
    else:
        if arm == "hidden":
            return np.hstack([inp_oh, st["h"].numpy()])
        if arm == "hidden_allpos":
            return np.hstack([inp_oh, st["x"].numpy().reshape(len(inp), -1)])
    if arm == "input_only":
        return inp_oh
    raise ValueError(arm)


# ---------------- stats helpers ----------------
def t_ci(xs):
    xs = np.asarray(xs, float)
    m, s, n = xs.mean(), xs.std(ddof=1), len(xs)
    from scipy.stats import t
    h = t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return [float(m - h), float(m + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    global N_POOL, N_VAL, N_TEST, NS, SEEDS, CKPTS, MLP_EPOCHS, K_ROLL, N_PRIMARY
    if args.smoke:
        N_POOL, N_VAL, N_TEST = 128, 64, 256
        NS = [32, 128]; N_PRIMARY = 128
        SEEDS = [0]; CKPTS = [30, 60]; MLP_EPOCHS = 60
        globals().update(N_POOL=N_POOL, N_VAL=N_VAL, N_TEST=N_TEST)

    R = {"config": dict(d=D, layers=LAYERS, eps=EPS, rho=RHO, B=B_TRAIN, lr=LR,
                        anchor=ANCHOR, ckpts=CKPTS, sigmas=SIGMAS, target_acc=TARGET_ACC,
                        K=K_ROLL, tau_eval=TAU_EVAL, n_pool=N_POOL, n_val=N_VAL,
                        n_test=N_TEST, ns=NS, n_primary=N_PRIMARY, seeds=SEEDS,
                        mlp_epochs=MLP_EPOCHS, smoke=bool(args.smoke)),
         "pilot": {}, "per_seed": [], "summary": {}}
    Epub = build_embeddings(D, EPS)  # public frozen dictionary, shared everywhere
    N_EP = N_POOL + N_VAL + N_TEST

    # ---------- pilot calibration (pre-registered rule: argmin |acc-0.80|) ----------
    log("PILOT: ERT")
    tasks._rng = np.random.default_rng(999)
    ert_p = new_ert(Epub, 99)
    accs_e = train_M_steps(ert_p, max(CKPTS), ckpt_eval=set(CKPTS))
    steps_ert = min(accs_e, key=lambda s: abs(accs_e[s] - TARGET_ACC))
    R["pilot"]["ert_accs"] = {str(k): v for k, v in accs_e.items()}
    R["pilot"]["ert_steps"] = int(steps_ert)
    log(f"  ERT pilot accs {accs_e} -> steps={steps_ert}")

    log("PILOT: baseline (sigma grid)")
    best = None
    R["pilot"]["base_accs"] = {}
    for sg in SIGMAS:
        tasks._rng = np.random.default_rng(998)
        bm = new_base(Epub, sg, 98)
        accs_b = train_M_steps(bm, max(CKPTS), ckpt_eval=set(CKPTS))
        R["pilot"]["base_accs"][str(sg)] = {str(k): v for k, v in accs_b.items()}
        for st, a in accs_b.items():
            if best is None or abs(a - TARGET_ACC) < abs(best[2] - TARGET_ACC):
                best = (sg, st, a)
        log(f"  sigma={sg}: {accs_b}")
    sigma_b, steps_base, _ = best
    R["pilot"]["base_sigma"] = float(sigma_b)
    R["pilot"]["base_steps"] = int(steps_base)
    log(f"  baseline -> sigma={sigma_b}, steps={steps_base}")

    # ---------- main loop over model seeds ----------
    main_arms = [("ert", "words"), ("base", "hidden")]
    extra_arms = [("ert", "input_only"), ("base", "input_only"),
                  ("ert", "hidden"), ("ert", "embfeat"), ("ert", "classfeat"),
                  ("ert", "words_allpos"), ("base", "hidden_allpos")]
    n_bound = LAYERS - 1

    for seed in SEEDS:
        srec = {"seed": seed, "arms": {}}
        log(f"SEED {seed}: train ERT ({steps_ert} steps)")
        tasks._rng = np.random.default_rng(500 + seed)
        torch.manual_seed(10_000 + seed)
        ert = new_ert(Epub, seed)
        train_M_steps(ert, steps_ert)
        srec["ert_acc"] = eval_acc(ert)
        log(f"  ERT acc={srec['ert_acc']:.3f}; collecting {N_EP} episodes + V*")
        ds_e = build_dataset(ert, "ert", N_EP)

        log(f"SEED {seed}: train baseline (sigma={sigma_b}, {steps_base} steps)")
        base = new_base(Epub, sigma_b, seed)
        train_M_steps(base, steps_base)
        srec["base_acc"] = eval_acc(base)
        log(f"  base acc={srec['base_acc']:.3f}; collecting episodes + V*")
        ds_b = build_dataset(base, "base", N_EP)

        dsets = {"ert": ds_e, "base": ds_b}
        # V* target stats + split-half reliability per boundary
        for kind, ds in dsets.items():
            vt = []
            for l in range(n_bound):
                v, h1, h2 = ds["vs"][l]
                vte = v[N_POOL + N_VAL:]
                rel = pearson(h1[N_POOL + N_VAL:], h2[N_POOL + N_VAL:])
                sb = 2 * rel / (1 + rel) if np.isfinite(rel) else float("nan")
                vt.append(dict(mean=float(vte.mean()), std=float(vte.std()),
                               split_half_r=rel, sb_reliability=sb,
                               degenerate=bool(vte.std() < 0.05)))
            srec[f"{kind}_vstats"] = vt

        # descriptive on-track oracle
        for kind, ds in dsets.items():
            oracle = []
            inp_cls = CLASS_NP[ds["ids"][:, RPOS]]
            for l in range(n_bound):
                v = ds["vs"][l][0][N_POOL + N_VAL:]
                tgt_cls = class_pow(inp_cls, l + 1)[N_POOL + N_VAL:]
                if kind == "ert":
                    got = CLASS_NP[ds["states"][l]["tok"][:, RPOS].numpy()][N_POOL + N_VAL:]
                    ontrack = (got == tgt_cls).astype(float)
                else:
                    with torch.no_grad():
                        z = base.lnf(ds["states"][l]["x"])
                        cl = _tok_to_class_logits(base.heads[l](z)[:, RPOS, :]).argmax(-1).numpy()
                    ontrack = (cl[N_POOL + N_VAL:] == tgt_cls).astype(float)
                oracle.append(pearson(ontrack, v))
            srec[f"{kind}_oracle_ontrack_r"] = oracle

        # ---------- probes ----------
        def run_probe(kind, arm, l, n, probe):
            ds = dsets[kind]
            X = arm_features(ds, kind, arm, l, Epub.numpy())
            v = ds["vs"][l][0]
            pred = probe_arm(X, v, f"{seed}-{kind}-{arm}-{l}-{n}-{probe}", n, probe)
            vte = v[N_POOL + N_VAL:]
            inp_te = ds["ids"][N_POOL + N_VAL:, RPOS]
            return dict(r=pearson(pred, vte),
                        r_within_input=partial_r_within_input(pred, vte, inp_te))

        for kind, arm in main_arms:
            rec = {}
            for probe in ["mlp", "ridge"]:
                for n in NS:
                    per_l = [run_probe(kind, arm, l, n, probe) for l in range(n_bound)]
                    rec[f"{probe}_n{n}"] = per_l
            srec["arms"][f"{kind}_{arm}"] = rec
            log(f"  probes done: {kind}_{arm}")

        if time.time() - T0 < TIME_GUARD_S or args.smoke:
            for kind, arm in extra_arms:
                per_l = [run_probe(kind, arm, l, N_PRIMARY, "mlp") for l in range(n_bound)]
                srec["arms"][f"{kind}_{arm}_extra"] = per_l
            srec["extras_done"] = True
            log("  extra arms done")
        else:
            srec["extras_done"] = False
            log("  TIME GUARD: extras skipped")

        R["per_seed"].append(srec)
        json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    # ---------- summary + primary test ----------
    def rbar(srec, key, probe, n):
        per_l = srec["arms"][key][f"{probe}_n{n}"]
        return float(np.mean([d["r"] for d in per_l]))

    prim_e = [rbar(s, "ert_words", "mlp", N_PRIMARY) for s in R["per_seed"]]
    prim_b = [rbar(s, "base_hidden", "mlp", N_PRIMARY) for s in R["per_seed"]]
    S = {"primary": {"ert_words_rbar_per_seed": prim_e,
                     "base_hidden_rbar_per_seed": prim_b,
                     "ert_mean": float(np.mean(prim_e)),
                     "base_mean": float(np.mean(prim_b))}}
    if len(SEEDS) >= 3:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(prim_e, prim_b, alternative="two-sided", method="exact")
        S["primary"]["mannwhitney_U"] = float(u.statistic)
        S["primary"]["mannwhitney_p_two_sided"] = float(u.pvalue)
        S["primary"]["ert_ci95"] = t_ci(prim_e)
        S["primary"]["base_ci95"] = t_ci(prim_b)

    curves = {}
    for key in ["ert_words", "base_hidden"]:
        for probe in ["mlp", "ridge"]:
            curves[f"{key}_{probe}"] = {
                str(n): {"mean": float(np.mean([rbar(s, key, probe, n) for s in R["per_seed"]])),
                         "per_seed": [rbar(s, key, probe, n) for s in R["per_seed"]]}
                for n in NS}
    S["learning_curves_rbar"] = curves

    # within-input partial r at n_primary (mlp), and per-boundary detail
    def rbar_w(srec, key):
        per_l = srec["arms"][key][f"mlp_n{N_PRIMARY}"]
        return float(np.mean([d["r_within_input"] for d in per_l]))
    S["within_input_r"] = {k: [rbar_w(s, k) for s in R["per_seed"]]
                           for k in ["ert_words", "base_hidden"]}
    S["per_boundary_primary"] = {
        k: [[d["r"] for d in s["arms"][k][f"mlp_n{N_PRIMARY}"]] for s in R["per_seed"]]
        for k in ["ert_words", "base_hidden"]}

    if all(s.get("extras_done") for s in R["per_seed"]):
        ex = {}
        for kind, arm in [("ert", "input_only"), ("base", "input_only"), ("ert", "hidden"),
                          ("ert", "embfeat"), ("ert", "classfeat"),
                          ("ert", "words_allpos"), ("base", "hidden_allpos")]:
            key = f"{kind}_{arm}_extra"
            ex[key] = [float(np.mean([d["r"] for d in s["arms"][key]])) for s in R["per_seed"]]
        S["extras_rbar_per_seed"] = ex
        # paired deltas over input-only control
        S["delta_over_input_only"] = {
            "ert_words_minus_inputonly": [a - b for a, b in zip(prim_e, ex["ert_input_only_extra"])],
            "base_hidden_minus_inputonly": [a - b for a, b in zip(prim_b, ex["base_input_only_extra"])]}

    S["vstar_reliability_sb"] = {
        kind: [[s[f"{kind}_vstats"][l]["sb_reliability"] for l in range(n_bound)]
               for s in R["per_seed"]] for kind in ["ert", "base"]}
    S["model_accs"] = {"ert": [s["ert_acc"] for s in R["per_seed"]],
                       "base": [s["base_acc"] for s in R["per_seed"]]}
    R["summary"] = S
    R["wall_time_s"] = time.time() - T0
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log(f"DONE in {R['wall_time_s']:.0f}s -> results.json")
    log(f"PRIMARY: ERT-words rbar {prim_e} vs BASE-hidden rbar {prim_b}")
    if "mannwhitney_p_two_sided" in S["primary"]:
        log(f"  MW exact p={S['primary']['mannwhitney_p_two_sided']:.4f}")


if __name__ == "__main__":
    main()
