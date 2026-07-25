"""pilot.py -- PILOT ONLY (not confirmatory). Purpose: (a) timing on this shared box,
(b) check the task is learnable to train acc >= 0.95 by the tiny transformer, for both the
standard arm and the routed (occlude+ablate) arm.

DELIBERATELY prints NO deployment-probe / alignment numbers, so that preregistration of the
primary outcomes is not contaminated by having seen them. Pilot seeds (0,1) are disjoint from
the confirmatory seeds (700..).
"""
import sys, time
import torch, torch.nn as nn
from core import gen, occlude, TinyTF, masks, acc, nparams

DEV = sys.argv[1] if len(sys.argv) > 1 else "mps"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-3
WD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
NTR = 1536
torch.set_num_threads(2)


def run(seed, arm):
    dev = torch.device(DEV)
    torch.manual_seed(seed)
    ids, y, pA, pB = gen(NTR, seed, "train")
    ids, y, pA, pB = ids.to(dev), y.to(dev), pA.to(dev), pB.to(dev)
    hid, hy, _, _ = gen(1500, 60001, "train")
    hid, hy = hid.to(dev), hy.to(dev)
    m = TinyTF().to(dev)
    (hA, uA), (hB, uB) = masks(dev)
    g = torch.Generator(device=dev); g.manual_seed(seed + 99)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=WD)
    lf = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        if arm == "control":
            loss = lf(m(ids), y)
        else:  # routed
            loss = 0.5 * (lf(m(occlude(ids, pB, g), hA, uA), y)
                          + lf(m(occlude(ids, pA, g), hB, uB), y))
        loss.backward(); opt.step()
    dt = time.time() - t0
    if arm == "control":
        tr, hl = acc(m, ids, y), acc(m, hid, hy)
    else:
        tr = 0.5 * (acc(m, ids, y, (hA, uA)) + acc(m, ids, y, (hB, uB)))
        hl = 0.5 * (acc(m, hid, hy, (hA, uA)) + acc(m, hid, hy, (hB, uB)))
    print(f"  {arm:8s} seed{seed} train={tr:.3f} heldout={hl:.3f}  {dt:.1f}s")
    return dt


if __name__ == "__main__":
    print(f"device={DEV} steps={STEPS} lr={LR} wd={WD} params={nparams(TinyTF())}")
    tot = 0
    for arm in ["control", "routed"]:
        for s in [0, 1]:
            tot += run(s, arm)
    print(f"total {tot:.1f}s -> est per (control+routed) seed pair = {tot/2:.1f}s")
