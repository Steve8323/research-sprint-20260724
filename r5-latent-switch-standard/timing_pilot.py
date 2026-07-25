"""TIMING PILOT -- prints WALL-CLOCK SECONDS ONLY.

No accuracy, no success flag, no mask, no weight statistic is computed or printed. Its sole
purpose is to fix n and the arm list in PREREG.md under the sprint's ~15-minute budget.
Run BEFORE the prereg is stamped; disclosed in the prereg.
"""
import time, sys
import torch

sys.path.insert(0, "/Users/steven/research_sprint_20260724/r5-latent-switch-standard")
import core_tf as CT
import core_mlp as CM

dev = torch.device("mps")

# --- transformer: one standard-trained seed + one full mask-fit + one scoring pass ---
t0 = time.time()
m, ids, y = CT.train_standard(700, dev)
t_train = time.time() - t0

Xh, yh, chh = [t.to(dev) for t in CT.gen(CT.NEVAL, CT.HELD_SEED, "train")]
Xd, yd, chd = [t.to(dev) for t in CT.gen(CT.NEVAL, CT.DEPLOY_SEED, "deploy")]
t0 = time.time()
_ = CT.cue_sensitivity(m, Xh[:2000], chh[:2000], [CT.CUE_A_FIELD, CT.CUE_B_FIELD])
t_sens = time.time() - t0

hm, um = CT.make_mask(dev, [0, 1], list(range(64)))
t0 = time.time()
for _ in range(12):                       # ~ 1 data-chosen + 10 random + 1 fixed, x4 accs each
    _ = CT.acc(m, Xd, yd, (hm, um))
t_eval12 = time.time() - t0

print(f"TF train/seed      {t_train:8.2f} s")
print(f"TF sensitivity     {t_sens:8.2f} s")
print(f"TF 12 acc calls    {t_eval12:8.2f} s")

# --- MLP: one seed at each candidate width ---
for H in [8, 16, 32, 64]:
    Xtr, ytr = CM.gen_align(CM.N, 700, "train")
    t0 = time.time()
    _ = CM.train_standard(700, H, CM.inp_dim(2), Xtr, ytr)
    print(f"MLP H={H:<3d} train   {time.time()-t0:8.2f} s")

Xtr3, ytr3 = CM.gen_k(CM.N, 700, 3, "train")
t0 = time.time()
_ = CM.train_standard(700, 24, CM.inp_dim(3), Xtr3, ytr3)
print(f"MLP k=3 H=24 train {time.time()-t0:8.2f} s")
