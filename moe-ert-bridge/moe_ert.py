"""moe_ert.py -- MoE legibility inside the ERT (moe-ert-bridge).

Adds a Switch-style top-1 MoE MLP to every block of the English-Residual
Transformer toy (~/english_bottleneck_lm, read-only import) and tests, with
ground truth known:
  P1  do experts align with meaning classes? (AMI + permutation test)
  P2  do plain-language (class-level) profiles predict routing on held-out
      synonyms never used to build the profile? (vs majority baseline, McNemar)

Pre-registered tests are in FINDINGS.md (written before this was run).
Run:  python3 moe_ert.py [--smoke]
"""
import sys, os, json, time, argparse

sys.path.insert(0, "/Users/steven/english_bottleneck_lm")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from scipy.stats import binomtest

import tasks  # noqa: E402  (module-level rng reseeded per seed below)
from tasks import make_batch_M, RPOS, SEQ_LEN
from vocab import (build_embeddings, V, N_CLASS, CLASS_OF, CLASSES, MEMBERS,
                   ITOS)
from paraphrase import ParaphraseChannel
from model import gumbel_st, _tok_to_class_logits

torch.set_num_threads(2)  # shared machine, be polite

OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cpu"
D, LAYERS, EPS, RHO = 64, 4, 0.5, 1.0
N_EXP, MULT = 8, 2
AUX_W = 0.01
CLASS_NAME = [c[0] for c in CLASSES]  # class rendered by its first word


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class MoEMLP(nn.Module):
    """Switch-style top-1 MoE: softmax gate, hard top-1 dispatch, output scaled
    by the chosen gate prob (gradient path to the router), load-balance aux."""

    def __init__(self, d, n_exp=8, mult=2):
        super().__init__()
        self.n_exp = n_exp
        self.gate = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                          nn.Linear(mult * d, d)) for _ in range(n_exp))

    def forward(self, x):                       # x: (B,T,d)
        gp = torch.softmax(self.gate(x), -1)    # (B,T,E)
        top1 = gp.argmax(-1)                    # (B,T)
        sel = F.one_hot(top1, self.n_exp).float()
        outs = torch.stack([e(x) for e in self.experts], dim=2)  # (B,T,E,d)
        p1 = (gp * sel).sum(-1, keepdim=True)   # prob of chosen expert
        y = (outs * sel.unsqueeze(-1)).sum(2) * p1
        f = sel.mean(dim=(0, 1))                # dispatch fraction per expert
        P = gp.mean(dim=(0, 1))                 # mean gate prob per expert
        aux = self.n_exp * (f * P).sum()        # Switch aux (min 1.0 balanced)
        return y, top1, aux


class MoEBlock(nn.Module):
    def __init__(self, d, heads=4, n_exp=8, mult=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.moe = MoEMLP(d, n_exp, mult)

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        y, top1, aux = self.moe(self.ln2(x))
        return x + y, top1, aux


class MoEERT(nn.Module):
    """EnglishResidualTransformer with MoE MLPs; records per-layer incoming
    words (the legible stream) and top-1 expert assignments."""

    def __init__(self, E, n_layers=4, heads=4, n_exp=8, mult=2,
                 paraphrase=None, seq_len=SEQ_LEN):
        super().__init__()
        self.register_buffer("E", E)
        self.d = E.shape[1]
        self.pos = nn.Parameter(0.02 * torch.randn(seq_len, self.d))
        self.blocks = nn.ModuleList(
            MoEBlock(self.d, heads, n_exp, mult) for _ in range(n_layers))
        self.unemb = nn.ModuleList(
            nn.Linear(self.d, V) for _ in range(n_layers))
        self.lnf = nn.LayerNorm(self.d)
        self.para = paraphrase
        self.n_layers = n_layers

    def forward_layers(self, ids, tau=0.5, record=False):
        y = F.one_hot(ids, V).float()
        outs, recs, aux_tot = [], [], 0.0
        for li, (blk, un) in enumerate(zip(self.blocks, self.unemb)):
            in_tok = y.argmax(-1)               # actual incoming words, (B,T)
            x = y @ self.E + self.pos.unsqueeze(0)
            x, top1, aux = blk(x)
            aux_tot = aux_tot + aux
            logits = un(self.lnf(x))
            outs.append(logits)
            if record:
                recs.append((in_tok.detach().cpu(), top1.detach().cpu()))
            if li < self.n_layers - 1:
                y = gumbel_st(logits, tau)
                if self.para is not None:
                    y = self.para(y)
        return outs, aux_tot / self.n_layers, recs


# --------------------------------------------------------------------------- #
# train / eval (mirrors english_bottleneck_lm/train.py deep supervision)
# --------------------------------------------------------------------------- #
def _tau(step, total, hi=1.0, lo=0.3):
    return hi + (lo - hi) * (step / max(1, total))


def train_M_moe(model, steps, B=256, lr=3e-3, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        ids, lt, _ = make_batch_M(B, LAYERS, DEV)
        outs, aux, _ = model.forward_layers(ids, tau=_tau(s, steps))
        loss = sum(F.cross_entropy(_tok_to_class_logits(o[:, RPOS, :]), lt[:, l])
                   for l, o in enumerate(outs)) / len(outs)
        loss = loss + AUX_W * aux
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, steps // 5) == 0:
            log(f"    step {s:4d}  loss {loss.item():.3f}  aux {aux.item():.3f}")


@torch.no_grad()
def eval_M_moe(model, B=2000, reps=3):
    model.eval()
    acc = 0.0
    for _ in range(reps):
        ids, _, final = make_batch_M(B, LAYERS, DEV)
        outs, _, _ = model.forward_layers(ids, tau=0.3)
        pred = _tok_to_class_logits(outs[-1][:, RPOS, :]).argmax(-1)
        acc += (pred == final).float().mean().item()
    return acc / reps


@torch.no_grad()
def collect_routing(model, n_seq, batch=500):
    """Per layer: incoming word id and top-1 expert at RPOS. Returns
    words (L, n_seq), experts (L, n_seq) int arrays."""
    model.eval()
    W = [[] for _ in range(LAYERS)]
    X = [[] for _ in range(LAYERS)]
    done = 0
    while done < n_seq:
        b = min(batch, n_seq - done)
        ids, _, _ = make_batch_M(b, LAYERS, DEV)
        _, _, recs = model.forward_layers(ids, tau=0.3, record=True)
        for l, (in_tok, top1) in enumerate(recs):
            W[l].append(in_tok[:, RPOS].numpy())
            X[l].append(top1[:, RPOS].numpy())
        done += b
    return (np.stack([np.concatenate(w) for w in W]),
            np.stack([np.concatenate(x) for x in X]))


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, c - hw, c + hw)


def perm_test_ami(experts, classes, n_perm, rng):
    obs = ami_score(classes, experts)
    ge = 0
    lab = classes.copy()
    for _ in range(n_perm):
        rng.shuffle(lab)
        if ami_score(lab, experts) >= obs:
            ge += 1
    return obs, (1 + ge) / (1 + n_perm)


def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar on paired binary outcomes."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0, b, c
    return binomtest(min(b, c), b + c, 0.5).pvalue, b, c


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #
def run_seed(seed, steps, n_prof, n_eval, n_perm, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tasks._rng = np.random.default_rng(10_000 + seed)  # fresh data stream
    hold_rng = np.random.default_rng(500 + seed)
    heldout_syn = hold_rng.integers(0, 4, N_CLASS)      # one synonym/class
    heldout_tok = set(int(MEMBERS[c][heldout_syn[c]]) for c in range(N_CLASS))

    E = build_embeddings(D, EPS).to(DEV)  # deterministic (seed=0 inside)
    para = ParaphraseChannel(rho=RHO, device=DEV).to(DEV)
    model = MoEERT(E, LAYERS, n_exp=N_EXP, mult=MULT, paraphrase=para).to(DEV)

    t0 = time.time()
    train_M_moe(model, steps, log=log)
    m_acc = eval_M_moe(model)
    log(f"  seed {seed}: task-M acc {m_acc:.3f}  (train {time.time()-t0:.0f}s)")

    pw, pe = collect_routing(model, n_prof)   # profile split
    ew, ee = collect_routing(model, n_eval)   # eval split
    cls_of = CLASS_OF.numpy()

    res = {"seed": seed, "task_M_acc": m_acc,
           "heldout_words": [ITOS[MEMBERS[c][heldout_syn[c]]]
                             for c in range(N_CLASS)]}

    # ---- profiles from PROFILE split (non-heldout content words only) ----- #
    counts = np.zeros((LAYERS, N_EXP, N_CLASS), dtype=np.int64)
    for l in range(LAYERS):
        for w, e in zip(pw[l], pe[l]):
            c = cls_of[w]
            if c >= 0 and int(w) not in heldout_tok:
                counts[l, e, c] += 1
    expert_of_class = counts.argmax(axis=1)   # (LAYERS, N_CLASS)
    class_seen = counts.sum(axis=1) > 0       # (LAYERS, N_CLASS)
    majority_expert = counts.sum(axis=2).argmax(axis=1)  # (LAYERS,)

    profiles = []
    for l in range(LAYERS):
        for e in range(N_EXP):
            cls = [CLASS_NAME[c] for c in range(N_CLASS)
                   if class_seen[l, c] and expert_of_class[l, c] == e]
            n_tok = int(counts[l, e].sum())
            if n_tok > 0 or cls:
                profiles.append({
                    "layer": l, "expert": e, "n_tokens_profile": n_tok,
                    "profile_en": ("expert handles words meaning: "
                                   + (", ".join(cls) if cls else "(no class assigned)"))})
    res["profiles"] = profiles

    # ---- P1: AMI(expert, class) on EVAL split, per layer + permutation ---- #
    perm_rng = np.random.default_rng(9000 + seed)
    p1 = []
    for l in range(LAYERS):
        c = cls_of[ew[l]]
        keep = c >= 0
        obs, p = perm_test_ami(ee[l][keep], c[keep], n_perm, perm_rng)
        p1.append({"layer": l, "n": int(keep.sum()), "ami": float(obs),
                   "perm_p": float(p),
                   "content_frac": float(keep.mean())})
        log(f"    P1 L{l}: AMI={obs:.3f} p={p:.4g} n={int(keep.sum())}")
    res["P1_per_layer"] = p1
    res["P1_mean_ami"] = float(np.mean([r["ami"] for r in p1]))

    # ---- P2: profile predictor on heldout-synonym EVAL instances ---------- #
    prof_ok, maj_ok, n_inst = [], [], 0
    per_layer_p2 = []
    for l in range(LAYERS):
        c = cls_of[ew[l]]
        mask = (c >= 0) & np.isin(ew[l], list(heldout_tok))
        cc, xx = c[mask], ee[l][mask]
        pred_prof = expert_of_class[l, cc]
        pred_maj = np.full_like(xx, majority_expert[l])
        prof_ok.append(pred_prof == xx)
        maj_ok.append(pred_maj == xx)
        per_layer_p2.append({"layer": l, "n": int(mask.sum()),
                             "acc_profile": float((pred_prof == xx).mean()) if mask.sum() else None,
                             "acc_majority": float((pred_maj == xx).mean()) if mask.sum() else None})
        n_inst += int(mask.sum())
    prof_ok = np.concatenate(prof_ok); maj_ok = np.concatenate(maj_ok)
    pval, b, cdisc = mcnemar_exact(prof_ok, maj_ok)
    acc, lo, hi = wilson(int(prof_ok.sum()), len(prof_ok))
    macc, mlo, mhi = wilson(int(maj_ok.sum()), len(maj_ok))
    res["P2"] = {"n_heldout_instances": int(len(prof_ok)),
                 "acc_profile": acc, "acc_profile_ci": [lo, hi],
                 "acc_majority": macc, "acc_majority_ci": [mlo, mhi],
                 "mcnemar_p": float(pval), "discordant": [b, cdisc],
                 "chance": 1.0 / N_EXP, "per_layer": per_layer_p2}
    log(f"    P2: profile acc {acc:.3f} [{lo:.3f},{hi:.3f}] vs majority "
        f"{macc:.3f}, McNemar p={pval:.3g}, n={len(prof_ok)}")

    # ---- secondary descriptives ------------------------------------------ #
    # spec score in the moe_legibility convention (per layer,expert on EVAL)
    ec = np.zeros((LAYERS, N_EXP, N_CLASS), dtype=np.int64)
    for l in range(LAYERS):
        for w, e in zip(ew[l], ee[l]):
            c = cls_of[w]
            if c >= 0:
                ec[l, e, c] += 1
    qn = ec / np.maximum(ec.sum(axis=2, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(qn * np.log(np.maximum(qn, 1e-12))).sum(axis=2)
    spec = 1 - ent / np.log(N_CLASS)
    used = ec.sum(axis=2) > 0
    load = ec.sum(axis=2) / np.maximum(ec.sum(axis=(1, 2), keepdims=False)[:, None], 1)
    res["secondary"] = {
        "spec_mean_used": float(spec[used].mean()),
        "spec_max": float(spec[used].max()),
        "experts_used_per_layer": [int(u.sum()) for u in used],
        "load_per_layer": [[round(float(x), 3) for x in row] for row in load],
    }
    # word-level lookup skyline on NON-heldout eval instances (descriptive):
    # how deterministic is routing given the exact word?
    sky_ok, sky_n = 0, 0
    for l in range(LAYERS):
        wt = {}
        for w, e in zip(pw[l], pe[l]):
            wt.setdefault(int(w), []).append(int(e))
        wt = {w: np.bincount(v).argmax() for w, v in wt.items()}
        c = cls_of[ew[l]]
        mask = (c >= 0) & ~np.isin(ew[l], list(heldout_tok))
        for w, e in zip(ew[l][mask], ee[l][mask]):
            if int(w) in wt:
                sky_ok += int(wt[int(w)] == e); sky_n += 1
    res["secondary"]["word_lookup_acc_nonheldout"] = sky_ok / max(1, sky_n)
    res["secondary"]["word_lookup_n"] = sky_n
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    # NOTE: pre-registration said 1200 steps; reduced to 800 BEFORE the full
    # run purely for the shared-machine runtime budget (smoke: acc 0.889 at
    # 100 steps, so 800 is ample). Recorded as a deviation in FINDINGS.md.
    steps = 100 if args.smoke else 800
    n_prof = 500 if args.smoke else 3000
    n_eval = 500 if args.smoke else 3000
    n_perm = 100 if args.smoke else 1000
    seeds = [0] if args.smoke else [0, 1, 2, 3]

    t0 = time.time()
    all_res = []
    for s in seeds:
        print(f"=== seed {s} ===", flush=True)
        all_res.append(run_seed(s, steps, n_prof, n_eval, n_perm))

    # aggregate primary numbers across seeds
    amis = np.array([r["P1_mean_ami"] for r in all_res])
    accs = np.array([r["P2"]["acc_profile"] for r in all_res])
    majs = np.array([r["P2"]["acc_majority"] for r in all_res])
    n = len(all_res)
    def tci(x):
        if len(x) < 2:
            return [float(x.mean()), None, None]
        from scipy.stats import t
        h = t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]
    agg = {
        "config": dict(d=D, layers=LAYERS, eps=EPS, rho=RHO, n_exp=N_EXP,
                       expert_mult=MULT, aux_w=AUX_W, steps=steps, B=256,
                       lr=3e-3, n_prof=n_prof, n_eval=n_eval, n_perm=n_perm,
                       seeds=seeds, smoke=args.smoke),
        "P1_mean_ami_across_seeds": tci(amis),
        "P1_all_cells_p_lt_01": bool(all(c["perm_p"] < 0.01
                                         for r in all_res for c in r["P1_per_layer"])),
        "P2_acc_profile_across_seeds": tci(accs),
        "P2_acc_majority_across_seeds": tci(majs),
        "P2_all_seeds_sig": bool(all(r["P2"]["mcnemar_p"] < 0.01 and
                                     r["P2"]["acc_profile"] > r["P2"]["acc_majority"]
                                     for r in all_res)),
        "task_M_acc": [r["task_M_acc"] for r in all_res],
        "wall_time_s": time.time() - t0,
        "seeds_detail": all_res,
    }
    out = os.path.join(OUT, "results_smoke.json" if args.smoke else "results.json")
    json.dump(agg, open(out, "w"), indent=2)
    print(f"\nP1 mean AMI over seeds: {agg['P1_mean_ami_across_seeds']}")
    print(f"P1 all (seed,layer) perm p<0.01: {agg['P1_all_cells_p_lt_01']}")
    print(f"P2 profile acc over seeds: {agg['P2_acc_profile_across_seeds']} "
          f"vs majority {agg['P2_acc_majority_across_seeds']}")
    print(f"P2 significant in all seeds: {agg['P2_all_seeds_sig']}")
    print(f"task-M acc per seed: {agg['task_M_acc']}")
    print(f"DONE in {agg['wall_time_s']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
