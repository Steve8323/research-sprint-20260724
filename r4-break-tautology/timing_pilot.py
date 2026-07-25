"""Timing-only pilot. Prints wall-clock seconds per seed-run. NO outcome metric is computed
or printed, so nothing here can inform the preregistered hypotheses."""
import time, core
for arm in ["anchor_disjoint_w4", "ceiling_disjoint_w8", "overlap50", "shared_trunk", "soft25", "floor_control"]:
    t0 = time.time(); core.train_arm(700, arm); dt = time.time() - t0
    print(f"{arm:22s} {dt:6.2f} s/seed  -> 30 seeds = {dt*30:6.1f} s")
