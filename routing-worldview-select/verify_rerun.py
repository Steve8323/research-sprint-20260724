"""
Audit spot-check (second pass, 2026-07-24 afternoon): re-run a subset of seeds with the
exact experiment.py code and compare per-seed numbers against the stored results.json.
Training is deterministic given the seed, so numbers must match exactly.
"""
import json, sys, time
sys.dont_write_bytecode = False
sys.path.insert(0, "/Users/steven/moral_framework_misgen/undefined/routing-worldview-select")
import experiment as E

STORED = json.load(open("/Users/steven/moral_framework_misgen/undefined/routing-worldview-select/results.json"))
CHECK_SEEDS = [700, 701, 702]
ARMS = ["routed", "control"]

Xdep, ydep = E.gen_align(4000, 50001, "deploy")
Xhld, yhld = E.gen_align(4000, 60001, "train")

t0 = time.time()
report = {"checked": [], "max_abs_diff": 0.0}
for arm in ARMS:
    stored_rows = {r["seed"]: r for r in STORED["arms"][arm]["per_seed"]}
    for seed in CHECK_SEEDS:
        m, Xtr, ytr = E.train_arm(seed, arm)
        new = dict(train_acc=E.accm(m, Xtr, ytr),
                   full_align=E.accm(m, Xdep, ydep),
                   selA_align=E.accm(m, Xdep, ydep, E.MASK_KEEP_A),
                   selA_train=E.accm(m, Xhld, yhld, E.MASK_KEEP_A),
                   selB_align=E.accm(m, Xdep, ydep, E.MASK_KEEP_B),
                   selB_train=E.accm(m, Xhld, yhld, E.MASK_KEEP_B))
        old = stored_rows[seed]
        diffs = {k: abs(new[k] - old[k]) for k in new}
        md = max(diffs.values())
        report["max_abs_diff"] = max(report["max_abs_diff"], md)
        report["checked"].append(dict(arm=arm, seed=seed, max_abs_diff=md))
        print(f"{arm} seed {seed}: max|diff| = {md:.6f}")

report["verdict"] = "REPRODUCES EXACTLY" if report["max_abs_diff"] == 0.0 else \
                    ("reproduces within 1e-3" if report["max_abs_diff"] < 1e-3 else "MISMATCH")
report["runtime_sec"] = round(time.time() - t0, 1)
json.dump(report, open("/Users/steven/moral_framework_misgen/undefined/routing-worldview-select/verify_rerun.json", "w"), indent=1)
print("verdict:", report["verdict"], f"({report['runtime_sec']}s)")
