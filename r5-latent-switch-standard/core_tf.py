"""
r5-latent-switch-standard -- TRANSFORMER substrate.

Verbatim reimplementation of the round-3 sequence organism (READ-ONLY parent:
/Users/steven/research_sprint_20260724/r3-lm-organism/core.py + experiment.py), with ONE
addition: an activation-capture path so that per-head and per-MLP-unit cue sensitivity can be
measured, which is what the round-4 data-chosen mask procedure needs.

Task, model, hyperparameters and success thresholds are unchanged from round 3, so the
STANDARD-trained ("control") models here are the same models round 3 trained, and round 3's
stored per-seed fixed-mask numbers for seeds 700-711 serve as an anchor.

The only models trained in this file are STANDARD-trained. No routing anywhere.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- task constants (round 3, verbatim) ----------------
NFIELD = 6
CUE_A_FIELD = 0        # true objective
CUE_B_FIELD = 1        # misaligned proxy
DIST_1_FIELD = 2       # label-independent distractor (negative-control "cue")
DIST_2_FIELD = 3
TOK_V = NFIELD
TOK_Q = NFIELD + 1
V_VOCAB = NFIELD + 2
T = 2 * NFIELD + 1

# ---------------- model constants (round 3, verbatim) ----------------
D = 64
NLAYER = 2
NHEAD = 4
DFF = 128
HEADS_A = [0, 1]                       # round-3's FIXED (arbitrary) partition
HEADS_B = [2, 3]
UNITS_A = list(range(0, DFF // 2))
UNITS_B = list(range(DFF // 2, DFF))

# training (round 3, verbatim)
STEPS, LR, WD, BATCH, POOL = 700, 8e-3, 0.01, 512, 4096
NEVAL = 4000
DEPLOY_SEED, HELD_SEED = 50001, 60001
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9


def gen(n, seed, mode, f=0.0):
    """Round-3 gen, byte-identical RNG consumption at f=0; additionally returns the chunk-start
    slot of EVERY field (round 3 returned only fields 0 and 1).

    f > 0 (train mode only) = the analogue of alignment.py's anti-proxy intervention: on a
    fraction f of training examples the PROXY cue disagrees with the label, so the worldview is
    DETERMINED rather than underdetermined. The extra RNG draw is guarded by `if f > 0`, so
    f = 0 consumes RNG in exactly round-3's order."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    bits = rng.integers(0, 2, (n, NFIELD))
    bits[:, CUE_A_FIELD] = v
    bits[:, CUE_B_FIELD] = v if mode == "train" else 1 - v
    if f > 0 and mode == "train":
        msk = rng.random(n) < f
        bits[msk, CUE_B_FIELD] = 1 - v[msk]
    perm = np.argsort(rng.random((n, NFIELD)), axis=1)
    rows = np.arange(n)[:, None]
    vals = bits[rows, perm]
    ids = np.empty((n, T), dtype=np.int64)
    ids[:, 0:2 * NFIELD:2] = np.where(vals == 1, perm, TOK_V)
    ids[:, 1:2 * NFIELD:2] = np.where(vals == 1, TOK_V, perm)
    ids[:, -1] = TOK_Q
    rank = np.argsort(perm, axis=1)
    chunks = 2 * rank                                    # (n, NFIELD)
    return (torch.from_numpy(ids), torch.from_numpy(v.astype(np.int64)),
            torch.from_numpy(chunks.astype(np.int64)))


def occlude(ids, chunk, field_tok, g):
    """Round-3 occlude, verbatim: re-randomise one field's bit in place."""
    out = ids.clone()
    n = len(ids)
    r = torch.arange(n, device=ids.device)
    fresh = torch.randint(0, 2, (n,), generator=g, device=ids.device)
    out[r, chunk] = torch.where(fresh == 1, torch.full_like(fresh, field_tok),
                                torch.full_like(fresh, TOK_V))
    out[r, chunk + 1] = torch.where(fresh == 1, torch.full_like(fresh, TOK_V),
                                    torch.full_like(fresh, field_tok))
    return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, DFF)
        self.fc2 = nn.Linear(DFF, D)
        self.dh = D // NHEAD

    def forward(self, x, hmask=None, umask=None, capture=None):
        B, L, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, NHEAD, self.dh).transpose(1, 2)
        k = k.view(B, L, NHEAD, self.dh).transpose(1, 2)
        v = v.view(B, L, NHEAD, self.dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / self.dh ** 0.5, dim=-1)
        o = att @ v                                          # (B,NHEAD,L,dh)
        if capture is not None:
            capture.append(("head", o.detach()))
        if hmask is not None:
            o = o * hmask.view(1, NHEAD, 1, 1)
        o = o.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(o)
        u = F.gelu(self.fc1(self.ln2(x)))                    # (B,L,DFF)
        if capture is not None:
            capture.append(("unit", u.detach()))
        if umask is not None:
            u = u * umask.view(1, 1, DFF)
        return x + self.fc2(u)


class TinyTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V_VOCAB, D)
        nn.init.normal_(self.emb.weight, std=0.5)
        self.pos = nn.Parameter(0.5 * torch.randn(T, D))
        self.blocks = nn.ModuleList(Block() for _ in range(NLAYER))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, 2)

    def forward(self, ids, hmask=None, umask=None, capture=None):
        x = self.emb(ids) + self.pos.unsqueeze(0)
        for b in self.blocks:
            x = b(x, hmask, umask, capture)
        return self.head(self.lnf(x[:, -1, :]))


@torch.no_grad()
def capture_acts(m, ids, bs=1000):
    """Returns (head_acts, unit_acts) as lists over layers of detached tensors, unmasked."""
    m.eval()
    heads = [[] for _ in range(NLAYER)]
    units = [[] for _ in range(NLAYER)]
    for i in range(0, len(ids), bs):
        cap = []
        m(ids[i:i + bs], None, None, cap)
        li = 0
        for kind, t in cap:
            if kind == "head":
                heads[li].append(t)
            else:
                units[li].append(t); li += 1
    m.train()
    return ([torch.cat(h, 0) for h in heads], [torch.cat(u, 0) for u in units])


@torch.no_grad()
def cue_sensitivity(m, ids, chunks, fields, gen_seed=99991, bs=1000):
    """sens[i] = (head_score (NHEAD,), unit_score (DFF,)) for field fields[i]:
    mean |activation change| when that field's bit is re-randomised, averaged over examples,
    positions, head dims, and LAYERS (the ablation mask is shared across layers, as in r3).
    Computed on HELD-OUT AGREEING data only -- never on the deploy set."""
    g = torch.Generator(device=ids.device); g.manual_seed(gen_seed)
    h0, u0 = capture_acts(m, ids, bs)
    out = []
    for fld in fields:
        idc = occlude(ids, chunks[:, fld], fld, g)
        hc, uc = capture_acts(m, idc, bs)
        hs = torch.stack([(h0[l] - hc[l]).abs().mean(dim=(0, 2, 3)) for l in range(NLAYER)]).mean(0)
        us = torch.stack([(u0[l] - uc[l]).abs().mean(dim=(0, 1)) for l in range(NLAYER)]).mean(0)
        out.append((hs.cpu().numpy(), us.cpu().numpy()))
        del hc, uc
    del h0, u0
    return out


def split_from_score(score, n_keep):
    """Top-n_keep indices select worldview A, bottom-n_keep select B (a full balanced split)."""
    order = np.argsort(-score)
    return sorted(order[:n_keep].tolist()), sorted(order[-n_keep:].tolist())


def make_mask(dev, heads, units):
    hm = torch.zeros(NHEAD, device=dev); hm[list(heads)] = 1.0
    um = torch.zeros(DFF, device=dev); um[list(units)] = 1.0
    return hm, um


@torch.no_grad()
def acc(m, ids, y, mask=None):
    m.eval()
    hm, um = (None, None) if mask is None else mask
    out = m(ids, hm, um).argmax(1)
    m.train()
    return (out == y).float().mean().item()


def crit(m, mA, mB, Xd, yd, Xh, yh):
    d = dict(aA=acc(m, Xd, yd, mA), tA=acc(m, Xh, yh, mA),
             aB=acc(m, Xd, yd, mB), tB=acc(m, Xh, yh, mB))
    d["designated"] = bool(d["aA"] > ALIGN_HI and d["tA"] > TRAIN_OK
                           and d["aB"] < ALIGN_LO and d["tB"] > TRAIN_OK)
    d["swapped"] = bool(d["aB"] > ALIGN_HI and d["tB"] > TRAIN_OK
                        and d["aA"] < ALIGN_LO and d["tA"] > TRAIN_OK)
    d["success"] = bool(d["designated"] or d["swapped"])
    return d


def train_standard(seed, dev, f=0.0):
    """Round-3 `control` arm, verbatim (standard training; f=0 -> agreeing data)."""
    torch.manual_seed(seed)
    ids, y, ch = [t.to(dev) for t in gen(POOL, seed, "train", f)]
    m = TinyTF().to(dev)
    gi = torch.Generator(device=dev); gi.manual_seed(seed + 7)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.15)
    lf = nn.CrossEntropyLoss()
    for _ in range(STEPS):
        idx = torch.randint(0, POOL, (BATCH,), generator=gi, device=dev)
        opt.zero_grad(set_to_none=True)
        lf(m(ids[idx]), y[idx]).backward()
        opt.step(); sch.step()
    return m, ids, y


def nparams(m):
    return sum(p.numel() for p in m.parameters())
