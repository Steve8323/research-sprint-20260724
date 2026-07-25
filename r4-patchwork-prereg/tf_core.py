"""core.py -- tiny-transformer sequence version of the moral-framework misgeneralization organism.

TASK ("two-rule word-order language").
Each example is a sequence of 6 two-token chunks followed by a query token. Chunk r holds one
field marker F_i and one value token V; **the bit of that field is encoded by WORD ORDER**:

    bit = 1  ->  [F_i, V]        bit = 0  ->  [V, F_i]

    vocab: F0..F5 (6 field markers), V (value), Q (query)  -> V_VOCAB = 8
    seq  : chunk(p0) chunk(p1) ... chunk(p5) Q             -> T = 13
    p is a fresh random permutation of the 6 fields per example (random chunk positions).

Why this encoding: **every example contains exactly the same multiset of tokens** (one of each
F0..F5, six V, one Q) in both the training and the deployment distribution. A bag-of-tokens /
position-blind / count-based model is therefore *provably* at chance (0.5). All label information
lives in order + position, so the model must run an attention circuit (locate marker F_i, read the
parity of its slot). An earlier pilot design (field-then-value with 0/1 value tokens) leaked
81.25% of the label into the token count; that shortcut is eliminated here by construction.

    cue A (TRUE objective)   = bit of field F0
    cue B (misaligned PROXY) = bit of field F1
    fields F2..F5            = label-independent distractor bits

    TRAIN  : b_F0 = b_F1 = v = label            (the two cues agree on every training example)
    DEPLOY : b_F0 = v = label, b_F1 = 1 - v     (the proxy flips; the true objective does not)

Sequence-model analogue of the ALIGN toy in ~/moral_framework_misgen/alignment.py: two equal-cost
cues that agree on every training example, so the training loss is indifferent between the two
"worldviews" (read F0 vs read F1); only deployment separates them.

MODEL: ~68k-param pre-LN encoder transformer (2 layers, 4 heads, d=64, d_ff=128), 2-way head at
the query position. Attention written by hand so individual heads / MLP units can be ablated.

ROUTING GROUPS: group A = heads {0,1} + MLP units [0:64) in every layer
                group B = heads {2,3} + MLP units [64:128) in every layer
Ablating a group zeroes those heads' per-head outputs (before the attention out-projection) and
those MLP hidden units, in every layer. Embedding / positions / LayerNorms / readout head are
SHARED -- unlike the round-1 MLP toy there is a shared residual stream (stated disanalogy).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- task constants ----------------
NFIELD = 6
CUE_A_FIELD = 0        # true objective
CUE_B_FIELD = 1        # misaligned proxy
TOK_V = NFIELD         # 6  (the single value token)
TOK_Q = NFIELD + 1     # 7
V_VOCAB = NFIELD + 2   # 8
T = 2 * NFIELD + 1     # 13

# ---------------- model constants ----------------
D = 64
NLAYER = 2
NHEAD = 4
DFF = 128
HEADS_A = [0, 1]
HEADS_B = [2, 3]
UNITS_A = list(range(0, DFF // 2))
UNITS_B = list(range(DFF // 2, DFF))


def gen(n, seed, mode):
    """Returns ids (n,T) int64, y (n,), cA (n,), cB (n,).

    cA/cB are the *chunk start slot* of cue field A / B, used by occlude() to re-randomise that
    cue's bit in place (by rewriting the 2-token chunk)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    bits = rng.integers(0, 2, (n, NFIELD))
    bits[:, CUE_A_FIELD] = v
    bits[:, CUE_B_FIELD] = v if mode == "train" else 1 - v
    perm = np.argsort(rng.random((n, NFIELD)), axis=1)        # (n,6): field occupying chunk r
    rows = np.arange(n)[:, None]
    vals = bits[rows, perm]                                   # (n,6): its bit
    ids = np.empty((n, T), dtype=np.int64)
    # bit=1 -> [F, V]; bit=0 -> [V, F]
    ids[:, 0:2 * NFIELD:2] = np.where(vals == 1, perm, TOK_V)
    ids[:, 1:2 * NFIELD:2] = np.where(vals == 1, TOK_V, perm)
    ids[:, -1] = TOK_Q
    rank = np.argsort(perm, axis=1)                           # rank[f] = chunk holding field f
    cA = 2 * rank[:, CUE_A_FIELD]
    cB = 2 * rank[:, CUE_B_FIELD]
    return (torch.from_numpy(ids), torch.from_numpy(v.astype(np.int64)),
            torch.from_numpy(cA.astype(np.int64)), torch.from_numpy(cB.astype(np.int64)))


def occlude(ids, chunk, field_tok, g):
    """Worldview-consistent view: re-randomise the OTHER cue's bit (rewrite its 2-token chunk with
    a fresh uniform bit) so that cue carries no label information. This is the token-sequence
    analogue of round-1's 'replace the other cue's input columns with fresh noise'. Sequence
    length, token multiset and marker positions are unchanged -- only the within-chunk order."""
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

    def forward(self, x, hmask=None, umask=None):
        B, L, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, NHEAD, self.dh).transpose(1, 2)
        k = k.view(B, L, NHEAD, self.dh).transpose(1, 2)
        v = v.view(B, L, NHEAD, self.dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / self.dh ** 0.5, dim=-1)
        o = att @ v                                          # (B,H,L,dh)
        if hmask is not None:
            o = o * hmask.view(1, NHEAD, 1, 1)
        o = o.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(o)
        u = F.gelu(self.fc1(self.ln2(x)))
        if umask is not None:
            u = u * umask.view(1, 1, DFF)
        return x + self.fc2(u)


class TinyTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V_VOCAB, D)
        nn.init.normal_(self.emb.weight, std=0.5)
        # positional embedding at the SAME scale as the token embedding: the cue circuit is
        # positional, so a 0.02-scale positional signal is drowned out (pilot-verified: with
        # 0.02 the model plateaus at the position-blind ceiling).
        self.pos = nn.Parameter(0.5 * torch.randn(T, D))
        self.blocks = nn.ModuleList(Block() for _ in range(NLAYER))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, 2)

    def forward(self, ids, hmask=None, umask=None):
        x = self.emb(ids) + self.pos.unsqueeze(0)
        for b in self.blocks:
            x = b(x, hmask, umask)
        return self.head(self.lnf(x[:, -1, :]))


def masks(device):
    hA = torch.zeros(NHEAD, device=device); hA[HEADS_A] = 1.0
    hB = torch.zeros(NHEAD, device=device); hB[HEADS_B] = 1.0
    uA = torch.zeros(DFF, device=device); uA[UNITS_A] = 1.0
    uB = torch.zeros(DFF, device=device); uB[UNITS_B] = 1.0
    return (hA, uA), (hB, uB)


@torch.no_grad()
def acc(m, ids, y, mask=None):
    m.eval()
    hm, um = (None, None) if mask is None else mask
    out = m(ids, hm, um).argmax(1)
    m.train()
    return (out == y).float().mean().item()


def nparams(m):
    return sum(p.numel() for p in m.parameters())
