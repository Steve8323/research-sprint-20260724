"""WALL-CLOCK-ONLY pilot (disclosed in PREREG.md).

Trains ONE control and ONE routed transformer on a fresh seed and prints ONLY elapsed
seconds. No alignment / deploy / determinism / patchwork / confidence quantity is computed
or printed, so no outcome measure is peeked at. Purpose: set n for the transformer stage
under the shared-machine contention present today.
"""
import time, sys
import torch, torch.nn as nn
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r4-patchwork-prereg")
from tf_core import gen, occlude, TinyTF, masks, CUE_A_FIELD, CUE_B_FIELD, nparams

DEV = torch.device("mps")
STEPS, LR, WD, BATCH, POOL = 700, 8e-3, 0.01, 512, 4096
torch.set_num_threads(2)


def train_arm(seed, arm, mA, mB):
    torch.manual_seed(seed)
    ids, y, cA, cB = [t.to(DEV) for t in gen(POOL, seed, "train")]
    m = TinyTF().to(DEV)
    g = torch.Generator(device=DEV); g.manual_seed(seed + 99)
    gi = torch.Generator(device=DEV); gi.manual_seed(seed + 7)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.15)
    lf = nn.CrossEntropyLoss()
    for _ in range(STEPS):
        idx = torch.randint(0, POOL, (BATCH,), generator=gi, device=DEV)
        bi, by, bcA, bcB = ids[idx], y[idx], cA[idx], cB[idx]
        opt.zero_grad(set_to_none=True)
        if arm == "control":
            loss = lf(m(bi), by)
        else:
            loss = 0.5 * (lf(m(occlude(bi, bcB, CUE_B_FIELD, g), *mA), by)
                          + lf(m(occlude(bi, bcA, CUE_A_FIELD, g), *mB), by))
        loss.backward(); opt.step(); sch.step()
    return m


if __name__ == "__main__":
    mA, mB = masks(DEV)
    print("params", nparams(TinyTF()))
    for arm in ("control", "routed"):
        t = time.time(); train_arm(9999, arm, mA, mB)
        torch.mps.synchronize()
        print(f"{arm}: {time.time()-t:.1f} s/seed")
