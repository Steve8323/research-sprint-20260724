"""
Reusable LoRA tamper-eval protocol.

Given any "deleted-forget" model (produced by some unlearning method) plus its data and,
optionally, a routing prune-mask, this module runs a matched-budget relearning attack in two
parameterizations -- LoRA-style adapters over a rank x lr grid, and (for reference) the repo's
own full-fine-tune attack -- and returns forget/retain recovery curves, recovery AUC per cell,
and the attacker-best ("LoRA-robust") cell.

Design copied from round-1 (../routing-lora-attack/lora_tamper_attack.py), refactored so it can be
pointed at an arbitrary unlearning arm. All heavy pipeline pieces are imported UNCHANGED from the
read-only repo ~/gradient_routing_experiment.

Public API:
    lora_attack(model, masks, data, seed, rank, lr, mask_delta=False) -> (curve_f, curve_r, n_trainable)
    fullft_reference(model, freeze, data, seed)                       -> curve_f (forget only)
    auc(curve) -> float
    attack_arm(model, masks, data, seed, grid_r, grid_lr)            -> dict of all cells + best
    signflip_p(diffs), t_ci95(diffs)                                 -> exact test + CI helpers
"""
import copy, itertools, math
import statistics as st
import sys

sys.dont_write_bytecode = True                    # never write bytecode into the read-only repo
sys.path.insert(0, "/Users/steven/gradient_routing_experiment")

import torch
import torch.nn.functional as F

from harness import FORGET                                                       # noqa: E402
from tamper_experiment import (relearn_attack, ATTACK_N, ATTACK_BS,              # noqa: E402
                               PROBE_STEPS)

LORA_SCALE = 2.0                                  # alpha_lora / r, fixed across ranks
LAYERS = ["fc1", "fc2", "fc3"]


# ---------------------------------------------------------------------------
# LoRA attack. Base frozen -> routed pruned entries stay 0 in the base. Trainable
# rank-r adapters on fc1/fc2/fc3. mask_delta projects fc1/fc2 deltas off the pruned
# entries every forward pass (mechanism arm). IDENTICAL budget/RNG offsets to the
# repo relearn_attack so the two parameterizations are directly comparable.
# ---------------------------------------------------------------------------
def lora_attack(model, masks, data, seed, rank, lr, mask_delta=False):
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad_(False)
    Xtr, ytr, Xte, yte, d, src = data

    gen = torch.Generator().manual_seed(seed + 555)
    layers = {"fc1": model.fc1, "fc2": model.fc2, "fc3": model.fc3}
    AB = {}
    for name, layer in layers.items():
        out_f, in_f = layer.weight.shape
        A = torch.randn(rank, in_f, generator=gen) / math.sqrt(in_f)
        B = torch.zeros(out_f, rank)
        A.requires_grad_(True); B.requires_grad_(True)
        AB[name] = (A, B)
    n_trainable = sum(a.numel() + b.numel() for a, b in AB.values())

    keep = None
    if mask_delta and masks is not None:
        keep = {"fc1": 1 - masks["fc1.weight"], "fc2": 1 - masks["fc2.weight"]}

    def lin(name, x):
        layer = layers[name]
        A, B = AB[name]
        delta = LORA_SCALE * (B @ A)
        if keep is not None and name in keep:
            delta = delta * keep[name]
        return F.linear(x, layer.weight + delta, layer.bias)

    def fwd(x):
        h = F.relu(lin("fc1", x))
        h = F.relu(lin("fc2", h))
        return lin("fc3", h)

    def probe():
        with torch.no_grad():
            pred = fwd(Xte).argmax(1)
        isf = torch.tensor([1 if int(t) in FORGET else 0 for t in yte])
        f = (pred[isf == 1] == yte[isf == 1]).float().mean().item()
        r = (pred[isf == 0] == yte[isf == 0]).float().mean().item()
        return round(f, 4), round(r, 4)

    isf_tr = torch.tensor([1 if int(t) in FORGET else 0 for t in ytr])
    fidx = (isf_tr == 1).nonzero().flatten()
    g = torch.Generator().manual_seed(seed + 123)
    fidx = fidx[torch.randperm(len(fidx), generator=g)[:ATTACK_N]]
    aX, ay = Xtr[fidx], ytr[fidx]

    opt = torch.optim.Adam([p for ab in AB.values() for p in ab], lr=lr)
    gb = torch.Generator().manual_seed(seed + 321)
    curve_f, curve_r = {}, {}
    curve_f[0], curve_r[0] = probe()
    step = 0
    while step < max(PROBE_STEPS):
        bi = torch.randint(len(aX), (ATTACK_BS,), generator=gb)
        opt.zero_grad()
        F.cross_entropy(fwd(aX[bi]), ay[bi]).backward()
        opt.step()
        step += 1
        if step in PROBE_STEPS:
            curve_f[step], curve_r[step] = probe()
    return curve_f, curve_r, n_trainable


def fullft_reference(model, freeze, data, seed):
    """Repo's own full-FT attack (freeze-mask enforced on routed). Forget curve only."""
    return relearn_attack(copy.deepcopy(model),
                          {k: v.clone() for k, v in freeze.items()}, data, seed)


def auc(curve):
    xs = PROBE_STEPS
    return sum((curve[xs[i]] + curve[xs[i + 1]]) / 2 * (xs[i + 1] - xs[i])
               for i in range(len(xs) - 1)) / (xs[-1] - xs[0])


def signflip_p(diffs):
    """Exact two-sided sign-flip permutation test on the mean of paired diffs."""
    obs = abs(st.mean(diffs))
    cnt = 0
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        if abs(st.mean([s * d for s, d in zip(signs, diffs)])) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** len(diffs)


def t_ci95(diffs):
    n = len(diffs)
    m = st.mean(diffs); sd = st.stdev(diffs)
    half = 2.5706 * sd / math.sqrt(n)      # t_{0.975, df=5}
    return m, (m - half, m + half)


def attack_arm(model, masks, freeze, data, seed, grid_r, grid_lr):
    """Run the full LoRA grid + full-FT reference on one deleted-forget model/seed.
    Returns per-cell forget/retain curves + AUC. Attacker-best cell computed later at
    the aggregate level (across seeds), so we just return every cell here."""
    cells = {}
    for r in grid_r:
        for lr in grid_lr:
            cf, cr, ntr = lora_attack(model, masks, data, seed, r, lr)
            cells[f"lora_r{r}_lr{lr}"] = {"forget": cf, "retain": cr,
                                          "auc": round(auc(cf), 4), "n_trainable": ntr}
    ff = fullft_reference(model, freeze, data, seed)
    cells["full_ft"] = {"forget": ff, "auc": round(auc(ff), 4)}
    return cells
