"""Timing pilot ONLY (pre-stated in PREREG.md): 1 seed of the smallest and largest cells,
plus a single-seed check that the k=2/w=4 routed path reproduces round-1 exactly.
No scientific result is read off this file."""
import time, json, sys
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r3-multiway-switch")
from experiment import train_run, gen, keep_mask, hid_cols, accm, NEVAL, HELD_SEED, PROBE_SEED0

for (k, w, arm) in [(2, 4, "routed"), (4, 8, "routed"), (4, 8, "control")]:
    t = time.time(); train_run(700, k, w, arm); print(f"k={k} w={w} {arm}: {time.time()-t:.1f}s", flush=True)

# round-1 exactness spot-check, seed 700, k=2 w=4 routed
k, w = 2, 4
m, _, _ = train_run(700, k, w, "routed")
cols = hid_cols(k, w); H = k * w
Xh, yh, _ = gen(NEVAL, HELD_SEED, k, "train")
Xp, yp, _ = gen(NEVAL, PROBE_SEED0, k, ("probe", 0))
mine = dict(selA_align=accm(m, Xp, yp, keep_mask(cols, 0, H)),
            selB_align=accm(m, Xp, yp, keep_mask(cols, 1, H)),
            selA_train=accm(m, Xh, yh, keep_mask(cols, 0, H)),
            selB_train=accm(m, Xh, yh, keep_mask(cols, 1, H)))
r1 = json.load(open("/Users/steven/research_sprint_20260724/routing-worldview-select/results.json"))
old = [r for r in r1["arms"]["routed"]["per_seed"] if r["seed"] == 700][0]
print("mine:", {kk: round(v, 6) for kk, v in mine.items()})
print("r1  :", {kk: round(old[kk], 6) for kk in mine})
print("maxdiff:", max(abs(old[kk] - mine[kk]) for kk in mine))
