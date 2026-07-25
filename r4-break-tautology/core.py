"""
r4-break-tautology -- shared core.

Reimplements the round-1 routing-worldview-select setup (READ-ONLY parent:
/Users/steven/research_sprint_20260724/routing-worldview-select/experiment.py) and generalizes
it along three axes that break the "structurally disjoint columns + train-mask == eval-mask"
tautology flagged by both round-3 audits:

  (1) OVERLAP     -- the two routed unit groups share a pre-stated fraction of units.
  (2) SHARED TRUNK-- hidden layer 1 is fully shared; routing masks act only on hidden layer 2.
  (3) SOFT TRAIN  -- the training mask leaks (other column kept at eps>0); eval mask is hard 0/1.
                     Every parameter then receives gradient from BOTH worldview views.

Data / hyperparameters are byte-identical to round 1 for the anchor arm.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)  # shared machine, 5 concurrent agents

# ---- config: verbatim from round-1 / alignment.py ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
WD = 2e-3; LR = 5e-3; EPOCHS = 2000
CUE_A = [0, 1]            # true-objective cue columns
CUE_B = [2, 3]            # misaligned-proxy cue columns
ALIGN_HI, ALIGN_LO, TRAIN_OK = 0.8, 0.2, 0.9   # pre-stated thresholds, inherited
SEEDS = list(range(700, 730))


# ---- data: verbatim from alignment.py (f=0) ----
def xor_block(target, rng):
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS)); b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode):
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = xor_block(v, rng); Xp = xor_block(a, rng)
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


# ---- model ----
class Net(nn.Module):
    """Same shape as round 1. `mask_l1` controls whether the unit mask is applied after
    hidden layer 1 (round-1 behaviour) or not (shared-trunk variant)."""

    def __init__(self, H, mask_l1=True):
        super().__init__()
        self.H = H; self.mask_l1 = mask_l1
        self.l1 = nn.Linear(INPUT, H); self.l2 = nn.Linear(H, H); self.l3 = nn.Linear(H, 2)

    def forward(self, x, umask=None):
        h = F.relu(self.l1(x))
        if umask is not None and self.mask_l1: h = h * umask
        h = F.relu(self.l2(h))
        if umask is not None: h = h * umask
        return self.l3(h)

    def acts(self, x, layer):
        """Pre-mask activations, unmasked forward, at hidden layer 1 or 2."""
        with torch.no_grad():
            h1 = F.relu(self.l1(x))
            if layer == 1: return h1
            return F.relu(self.l2(h1))


def unit_mask(H, keep_units, other_units=None, eps=0.0):
    """Hard mask (eps=0) or soft/leaky mask (eps>0 on the *other* group's exclusive units)."""
    m = torch.zeros(H)
    if eps > 0 and other_units is not None:
        m[list(other_units)] = eps
    m[list(keep_units)] = 1.0
    return m


def allowed_l2(H, groups):
    """Boolean [H,H]: l2[i,j] may be nonzero iff some group contains both i and j.
    Generalizes round-1's `zero_cross_blocks` to overlapping groups; entries outside the
    union of the within-group blocks receive exactly zero gradient under hard masking."""
    A = torch.zeros(H, H, dtype=torch.bool)
    for g in groups:
        idx = torch.tensor(sorted(g))
        A[idx[:, None], idx[None, :]] = True
    return A


def occlude(X, cols):
    """Worldview-consistent view: the OTHER cue's columns replaced by fresh noise."""
    Xo = X.clone()
    Xo[:, cols] = NOISE * torch.randn(len(X), len(cols))
    return Xo


def accm(m, X, y, umask=None):
    with torch.no_grad():
        return (m(X, umask).argmax(1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# Arm specifications.  Every routed arm keeps COLUMN WIDTH w = 8 (except the
# round-1 replication anchor at w = 4), so column-trainability is matched to the
# ceiling and any drop is attributable to the tautology-breaking manipulation.
# ---------------------------------------------------------------------------
def make_spec(name):
    """-> dict(H, SA, SB, mask_l1, eps, zero_cross, routed, score_layer)"""
    if name == "anchor_disjoint_w4":
        return dict(H=8, SA=list(range(0, 4)), SB=list(range(4, 8)), mask_l1=True,
                    eps=0.0, zero_cross=True, routed=True, score_layer=1)
    if name == "ceiling_disjoint_w8":
        return dict(H=16, SA=list(range(0, 8)), SB=list(range(8, 16)), mask_l1=True,
                    eps=0.0, zero_cross=True, routed=True, score_layer=1)
    if name == "floor_control":
        return dict(H=16, SA=list(range(0, 8)), SB=list(range(8, 16)), mask_l1=True,
                    eps=0.0, zero_cross=False, routed=False, score_layer=1)
    if name == "overlap25":     # |SA ^ SB| = 2 of 8
        return dict(H=14, SA=list(range(0, 8)), SB=list(range(6, 14)), mask_l1=True,
                    eps=0.0, zero_cross=True, routed=True, score_layer=1)
    if name == "overlap50":     # |SA ^ SB| = 4 of 8
        return dict(H=12, SA=list(range(0, 8)), SB=list(range(4, 12)), mask_l1=True,
                    eps=0.0, zero_cross=True, routed=True, score_layer=1)
    if name == "overlap75":     # |SA ^ SB| = 6 of 8
        return dict(H=10, SA=list(range(0, 8)), SB=list(range(2, 10)), mask_l1=True,
                    eps=0.0, zero_cross=True, routed=True, score_layer=1)
    if name == "shared_trunk":  # hidden layer 1 fully shared; routing only in layer 2
        return dict(H=16, SA=list(range(0, 8)), SB=list(range(8, 16)), mask_l1=False,
                    eps=0.0, zero_cross=False, routed=True, score_layer=2)
    if name == "soft25":        # train mask leaks at 0.25; eval mask hard
        return dict(H=16, SA=list(range(0, 8)), SB=list(range(8, 16)), mask_l1=True,
                    eps=0.25, zero_cross=True, routed=True, score_layer=1)
    if name == "soft50":
        return dict(H=16, SA=list(range(0, 8)), SB=list(range(8, 16)), mask_l1=True,
                    eps=0.50, zero_cross=True, routed=True, score_layer=1)
    raise ValueError(name)


def train_arm(seed, name):
    sp = make_spec(name)
    H, SA, SB = sp["H"], sp["SA"], sp["SB"]
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train")
    m = Net(H, mask_l1=sp["mask_l1"]); lf = nn.CrossEntropyLoss()
    allow = allowed_l2(H, [SA, SB])
    if sp["zero_cross"]:
        with torch.no_grad():
            m.l2.weight[~allow] = 0.0
    # training masks (may be soft); eval masks are always hard
    trA = unit_mask(H, SA, SB, sp["eps"]); trB = unit_mask(H, SB, SA, sp["eps"])
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        if not sp["routed"]:
            loss = lf(m(Xtr), ytr)
        else:
            loss = 0.5 * (lf(m(occlude(Xtr, CUE_B), trA), ytr)
                          + lf(m(occlude(Xtr, CUE_A), trB), ytr))
        loss.backward(); opt.step()
    return m, sp, Xtr, ytr, allow


# ---- learned (data-derived) eval mask: does selectability need the training partition? ----
def learned_masks(m, sp, X, gen_seed=99991):
    """Score each unit by how much its (unmasked) activation moves when cue A vs cue B is
    occluded; keep the top-w by (sensA - sensB) to select worldview A, and the top-w by
    (sensB - sensA) for B.  Uses the same cue-column oracle the whole line already assumes,
    but NOT the training unit partition -- so train mask != eval mask by construction."""
    g = torch.Generator().manual_seed(gen_seed)
    XA = X.clone(); XA[:, CUE_A] = NOISE * torch.randn(len(X), len(CUE_A), generator=g)
    XB = X.clone(); XB[:, CUE_B] = NOISE * torch.randn(len(X), len(CUE_B), generator=g)
    L = sp["score_layer"]
    a0, aA, aB = m.acts(X, L), m.acts(XA, L), m.acts(XB, L)
    sensA = (a0 - aA).abs().mean(0)   # sensitivity to cue A
    sensB = (a0 - aB).abs().mean(0)
    score = (sensA - sensB).numpy()
    w = len(sp["SA"])
    keepA = sorted(np.argsort(-score)[:w].tolist())
    keepB = sorted(np.argsort(score)[:w].tolist())
    return keepA, keepB, score.tolist()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))
