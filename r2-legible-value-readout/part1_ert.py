#!/usr/bin/env python3
"""PART 1 — ERT toy, variance-matched & calibration-fixed rerun.

Round-1's primary was confounded: (a) the pooled contrast is dominated by
input-difficulty memorization, and (b) ERT was trained to acc~0.99 leaving 6/15
value targets near-degenerate (no variance to read). This rerun:
  1. calibrates BOTH models by matching pooled V* std to TARGET_VSTD (not by acc),
  2. makes the DIFFICULTY-REMOVED (within-input) r the pre-registered primary,
  3. uses tuned ridge as the matched-capacity primary probe (MLP secondary).

Reuses the round-1 machinery from ert-value-probe/experiment.py (read-only import).
Pre-registration: see PREREG.md.  CPU only, seeded, single process.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

EBL = "/Users/steven/english_bottleneck_lm"
R1 = "/Users/steven/research_sprint_20260724/ert-value-probe"
sys.path.insert(0, EBL)
sys.path.insert(0, R1)
OUT = os.path.dirname(os.path.abspath(__file__))

from vocab import build_embeddings, V, N_CLASS, CLASS_OF  # noqa: E402
import tasks                                              # noqa: E402
from tasks import make_batch_M, RPOS, SEQ_LEN             # noqa: E402

# import round-1 experiment machinery (models, training, V*, probes, features)
import experiment as E1  # noqa: E402
from experiment import (new_ert, new_base, train_M_steps, eval_acc,
                        build_dataset, arm_features, probe_arm,
                        partial_r_within_input, pearson, t_ci,
                        N_POOL, N_VAL, N_TEST, RHO, EPS, D, LAYERS)  # noqa: E402

torch.set_num_threads(4)
DEV = "cpu"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

# -------- pre-registered config --------
CKPTS_E = [20, 30, 40, 60, 90, 140, 220]           # ERT step grid
BASE_GRID = [(s, st) for s in (1.0, 2.0, 4.0) for st in (15, 25, 40, 70)]
TARGET_VSTD = 0.28
ACC_BAND = (0.80, 0.98)   # Amendment 1: both models must be working (see PREREG)
SEEDS = [0, 1, 2, 3, 4]
N_PRIMARY = 256
NS = [64, 128, 256, 512, 1024]
K_ROLL = E1.K_ROLL  # 16
N_BOUND = LAYERS - 1
N_EP = N_POOL + N_VAL + N_TEST
N_PILOT = 1024  # pilot episodes for variance calibration
TIME_DROP_SECONDARY = 6 * 60  # if seed-0 done later than this, drop MLP + curve tails

CLASS_NP = CLASS_OF.numpy()


def pooled_vstd(vs):
    """Mean over boundaries of the V* std over ALL pilot episodes (calibration
    uses the whole pilot set, not the train/val/test split)."""
    stds = [float(vs[l][0].std()) for l in range(N_BOUND)]
    return float(np.mean(stds)), stds


# ============ pilot calibration (variance-matched) ============
def calibrate():
    Epub = build_embeddings(D, EPS)
    cal = {"ert": {}, "base": {}}

    # ERT: train once to max ckpt on a pilot seed, snapshot V* std at each ckpt.
    log("PILOT ERT: variance calibration")
    best_e = None
    for steps in CKPTS_E:
        tasks._rng = np.random.default_rng(999)
        torch.manual_seed(20_099)
        m = new_ert(Epub, 99)
        train_M_steps(m, steps)
        acc = eval_acc(m)
        ds = build_dataset(m, "ert", N_PILOT)
        vstd, stds = pooled_vstd(ds["vs"])
        in_band = ACC_BAND[0] <= acc <= ACC_BAND[1]
        cal["ert"][str(steps)] = dict(acc=acc, vstd=vstd, per_boundary=stds, in_band=in_band)
        log(f"  ERT steps={steps}: acc={acc:.3f} pooledV*std={vstd:.3f} band={in_band} "
            f"{[round(s,2) for s in stds]}")
        if in_band and (best_e is None or abs(vstd - TARGET_VSTD) < abs(best_e[1] - TARGET_VSTD)):
            best_e = (steps, vstd)
    assert best_e is not None, "no ERT config in accuracy band"
    ert_steps = best_e[0]

    # baseline: grid over (sigma, steps)
    log("PILOT baseline: variance calibration")
    best_b = None
    for (sigma, steps) in BASE_GRID:
        tasks._rng = np.random.default_rng(998)
        torch.manual_seed(30_098)
        m = new_base(Epub, sigma, 98)
        train_M_steps(m, steps)
        acc = eval_acc(m)
        ds = build_dataset(m, "base", N_PILOT)
        vstd, stds = pooled_vstd(ds["vs"])
        in_band = ACC_BAND[0] <= acc <= ACC_BAND[1] and vstd > 0.05
        cal["base"][f"s{sigma}_st{steps}"] = dict(sigma=sigma, steps=steps, acc=acc,
                                                   vstd=vstd, per_boundary=stds, in_band=in_band)
        if in_band and (best_b is None or abs(vstd - TARGET_VSTD) < abs(best_b[2] - TARGET_VSTD)):
            best_b = (sigma, steps, vstd)
        log(f"  base sigma={sigma} steps={steps}: acc={acc:.3f} pooledV*std={vstd:.3f} band={in_band}")
    assert best_b is not None, "CALIBRATION FAILED: no baseline config in accuracy band with V*std>0.05"
    base_sigma, base_steps = best_b[0], best_b[1]
    log(f"  CHOSEN ERT steps={ert_steps} (V*std {best_e[1]:.3f}); "
        f"baseline sigma={base_sigma} steps={base_steps} (V*std {best_b[2]:.3f})")
    return Epub, ert_steps, base_sigma, base_steps, cal


def main():
    Epub, ert_steps, base_sigma, base_steps, cal = calibrate()
    Epub_np = Epub.numpy()
    R = dict(config=dict(target_vstd=TARGET_VSTD, ckpts_e=CKPTS_E, base_grid=BASE_GRID,
                         seeds=SEEDS, n_primary=N_PRIMARY, ns=NS, K=K_ROLL, rho=RHO, eps=EPS,
                         d=D, layers=LAYERS, ert_steps=ert_steps,
                         base_sigma=base_sigma, base_steps=base_steps),
             calibration=cal, per_seed=[])

    main_arms = [("ert", "words"), ("base", "hidden")]
    secondary_arms = [("ert", "input_only"), ("base", "input_only"),
                      ("ert", "classfeat"), ("base", "hidden_allpos")]

    drop_secondary = False
    for seed in SEEDS:
        srec = dict(seed=seed, arms={}, vstats={})
        log(f"SEED {seed}: train ERT ({ert_steps} steps)")
        tasks._rng = np.random.default_rng(500 + seed)
        torch.manual_seed(10_000 + seed)
        ert = new_ert(Epub, seed)
        train_M_steps(ert, ert_steps)
        srec["ert_acc"] = eval_acc(ert)
        ds_e = build_dataset(ert, "ert", N_EP)

        log(f"SEED {seed}: train baseline (sigma={base_sigma}, {base_steps} steps)")
        base = new_base(Epub, base_sigma, seed)
        train_M_steps(base, base_steps)
        srec["base_acc"] = eval_acc(base)
        ds_b = build_dataset(base, "base", N_EP)
        dsets = {"ert": ds_e, "base": ds_b}

        # V* stats + reliability + degeneracy flag
        for kind, ds in dsets.items():
            vt = []
            for l in range(N_BOUND):
                v, h1, h2 = ds["vs"][l]
                vte = v[N_POOL + N_VAL:]
                rel = pearson(h1[N_POOL + N_VAL:], h2[N_POOL + N_VAL:])
                sb = 2 * rel / (1 + rel) if np.isfinite(rel) else float("nan")
                vt.append(dict(mean=float(vte.mean()), std=float(vte.std()),
                               split_half_r=rel, sb_reliability=sb,
                               degenerate=bool(vte.std() < 0.05)))
            srec["vstats"][kind] = vt
        log(f"  ERT acc={srec['ert_acc']:.3f} V*std={[round(v['std'],2) for v in srec['vstats']['ert']]}"
            f"  base acc={srec['base_acc']:.3f} V*std={[round(v['std'],2) for v in srec['vstats']['base']]}")

        def run(kind, arm, l, n, probe):
            ds = dsets[kind]
            X = arm_features(ds, kind, arm, l, Epub_np)
            v = ds["vs"][l][0]
            pred = probe_arm(X, v, f"{seed}-{kind}-{arm}-{l}-{n}-{probe}", n, probe)
            vte = v[N_POOL + N_VAL:]
            inp_te = ds["ids"][N_POOL + N_VAL:, RPOS]
            return dict(r=pearson(pred, vte),
                        r_within_input=partial_r_within_input(pred, vte, inp_te))

        probes = ["ridge"] if drop_secondary else ["ridge", "mlp"]
        ns = [N_PRIMARY] if drop_secondary else NS
        for kind, arm in main_arms:
            rec = {}
            for probe in probes:
                for n in ns:
                    rec[f"{probe}_n{n}"] = [run(kind, arm, l, n, probe) for l in range(N_BOUND)]
            srec["arms"][f"{kind}_{arm}"] = rec

        if not drop_secondary:
            for kind, arm in secondary_arms:
                key = f"{kind}_{arm}"
                srec["arms"][key] = {f"ridge_n{N_PRIMARY}":
                                     [run(kind, arm, l, N_PRIMARY, "ridge") for l in range(N_BOUND)]}
        R["per_seed"].append(srec)
        json.dump(R, open(os.path.join(OUT, "results_part1.json"), "w"), indent=1)
        log(f"  seed {seed} done")
        if seed == 0 and time.time() - T0 > TIME_DROP_SECONDARY:
            drop_secondary = True
            log("  TIME GUARD: dropping MLP + learning-curve tails for remaining seeds")

    # ---------- summary + tests ----------
    def rbar(s, key, probe, n, metric):
        per_l = s["arms"][key][f"{probe}_n{n}"]
        return float(np.mean([d[metric] for d in per_l]))

    def gather(key, probe, n, metric):
        return [rbar(s, key, probe, n, metric) for s in R["per_seed"]]

    from scipy.stats import mannwhitneyu

    # PRIMARY: difficulty-removed (within-input) ridge r at n=256
    prim_e = gather("ert_words", "ridge", N_PRIMARY, "r_within_input")
    prim_b = gather("base_hidden", "ridge", N_PRIMARY, "r_within_input")
    u = mannwhitneyu(prim_e, prim_b, alternative="two-sided", method="exact")
    primary = dict(metric="within_input_r__ridge__n256",
                   ert_words_per_seed=prim_e, base_hidden_per_seed=prim_b,
                   ert_mean=float(np.mean(prim_e)), base_mean=float(np.mean(prim_b)),
                   ert_ci95=t_ci(prim_e), base_ci95=t_ci(prim_b),
                   mannwhitney_U=float(u.statistic),
                   mannwhitney_p_two_sided=float(u.pvalue))

    # secondary confirmatory: pooled r ridge n256
    pe = gather("ert_words", "ridge", N_PRIMARY, "r")
    pb = gather("base_hidden", "ridge", N_PRIMARY, "r")
    up = mannwhitneyu(pe, pb, alternative="two-sided", method="exact")
    pooled = dict(ert_words_per_seed=pe, base_hidden_per_seed=pb,
                  ert_mean=float(np.mean(pe)), base_mean=float(np.mean(pb)),
                  mannwhitney_U=float(up.statistic), mannwhitney_p_two_sided=float(up.pvalue))

    # variance-match audit
    def deg_count(kind):
        return int(sum(v["degenerate"] for s in R["per_seed"] for v in s["vstats"][kind]))
    def mean_vstd(kind):
        return float(np.mean([v["std"] for s in R["per_seed"] for v in s["vstats"][kind]]))
    vm = dict(ert_degenerate=deg_count("ert"), base_degenerate=deg_count("base"),
              ert_mean_vstd=mean_vstd("ert"), base_mean_vstd=mean_vstd("base"),
              ert_acc=[s["ert_acc"] for s in R["per_seed"]],
              base_acc=[s["base_acc"] for s in R["per_seed"]])
    vm["variance_match_ok"] = bool(vm["ert_degenerate"] <= 2 and vm["base_degenerate"] <= 2
                                   and (max(vm["ert_mean_vstd"], vm["base_mean_vstd"]) /
                                        max(1e-6, min(vm["ert_mean_vstd"], vm["base_mean_vstd"])) <= 1.5))

    # secondary arms (classfeat, input-only) within-input r, if run
    def maybe(key, metric):
        try:
            return gather(key, "ridge", N_PRIMARY, metric)
        except Exception:
            return None
    secondary = dict(
        classfeat_within_input=maybe("ert_classfeat", "r_within_input"),
        classfeat_pooled=maybe("ert_classfeat", "r"),
        ert_words_within_input=prim_e, base_hidden_within_input=prim_b,
        ert_input_only_pooled=maybe("ert_input_only", "r"),
        base_input_only_pooled=maybe("base_input_only", "r"),
    )

    # learning curves (within-input), ridge, if run
    curves = {}
    for key in ["ert_words", "base_hidden"]:
        try:
            curves[key] = {str(n): dict(
                within=gather(key, "ridge", n, "r_within_input"),
                pooled=gather(key, "ridge", n, "r")) for n in NS
                if f"ridge_n{n}" in R["per_seed"][0]["arms"][key]}
        except Exception:
            pass

    R["summary"] = dict(primary=primary, pooled_secondary=pooled,
                        variance_match=vm, secondary=secondary, learning_curves=curves)
    json.dump(R, open(os.path.join(OUT, "results_part1.json"), "w"), indent=1)

    log("=== PART 1 SUMMARY ===")
    log(f"variance-match: ERT deg {vm['ert_degenerate']}/15 (meanV*std {vm['ert_mean_vstd']:.3f}) "
        f"base deg {vm['base_degenerate']}/15 (meanV*std {vm['base_mean_vstd']:.3f}) "
        f"-> ok={vm['variance_match_ok']}")
    log(f"PRIMARY within-input r ridge n256: ERT-words {primary['ert_mean']:+.3f} "
        f"{primary['ert_ci95']} vs base-hidden {primary['base_mean']:+.3f} {primary['base_ci95']} "
        f"| MW U={primary['mannwhitney_U']:.0f} p={primary['mannwhitney_p_two_sided']:.4f}")
    log(f"  per-seed ERT-words: {[round(x,3) for x in prim_e]}")
    log(f"  per-seed base-hid : {[round(x,3) for x in prim_b]}")
    log(f"pooled secondary ridge n256: ERT {pooled['ert_mean']:+.3f} vs base {pooled['base_mean']:+.3f} "
        f"p={pooled['mannwhitney_p_two_sided']:.4f}")
    if secondary["classfeat_within_input"]:
        log(f"classfeat within-input r: {[round(x,3) for x in secondary['classfeat_within_input']]} "
            f"(mean {np.mean(secondary['classfeat_within_input']):+.3f})")
    log(f"done in {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
