"""
APPLICATION (a): class-matched null for the moe-ert-bridge profile->routing claim.

Re-runs the MoE-ERT (reduced budget, see PREREG) to collect per-instance routing on
held-out-synonym eval instances, rebuilds the same class->expert profile the published
run used, and asks: does the TRUE class->expert map predict held-out routing above a
CLASS-MATCHED null that permutes the class->expert assignment within each layer
(preserving class marginals, expert set, and per-layer majority structure)?

Uses the reusable topic_null.topic_null_effect with items=24 classes, group=layer.

Pre-registration: PREREG.md, application (a). Reduced budget recorded there pre-run.
"""
import sys, os, json, time

sys.path.insert(0, "/Users/steven/english_bottleneck_lm")
sys.path.insert(0, "/Users/steven/research_sprint_20260724/moe-ert-bridge")

import numpy as np

import moe_ert as M                # published module, imported read-only
from vocab import N_CLASS, MEMBERS, CLASS_OF, ITOS
from topic_null import topic_null_effect

from tasks import RPOS  # noqa

STEPS = 400
N_PROF = 2000
N_EVAL = 2000
SEEDS = [0, 1, 2, 3]
K = 5000
LAYERS = M.LAYERS
N_EXP = M.N_EXP
CHANCE = 1.0 / N_EXP

import torch
from vocab import build_embeddings
from paraphrase import ParaphraseChannel
import tasks


def collect_seed(seed):
    """Train one MoE-ERT and return, per layer: the profile map (expert_of_class),
    the held-out-synonym eval instances (class array, observed-expert array)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tasks._rng = np.random.default_rng(10_000 + seed)
    hold_rng = np.random.default_rng(500 + seed)
    heldout_syn = hold_rng.integers(0, 4, N_CLASS)
    heldout_tok = set(int(MEMBERS[c][heldout_syn[c]]) for c in range(N_CLASS))

    E = build_embeddings(M.D, M.EPS).to(M.DEV)
    para = ParaphraseChannel(rho=M.RHO, device=M.DEV).to(M.DEV)
    model = M.MoEERT(E, LAYERS, n_exp=N_EXP, mult=M.MULT, paraphrase=para).to(M.DEV)

    M.train_M_moe(model, STEPS, log=lambda *a: None)
    m_acc = M.eval_M_moe(model)

    pw, pe = M.collect_routing(model, N_PROF)   # profile split
    ew, ee = M.collect_routing(model, N_EVAL)   # eval split
    cls_of = CLASS_OF.numpy()

    # profile map from profile split, non-heldout content words only (as published)
    counts = np.zeros((LAYERS, N_EXP, N_CLASS), dtype=np.int64)
    for l in range(LAYERS):
        for w, e in zip(pw[l], pe[l]):
            c = cls_of[w]
            if c >= 0 and int(w) not in heldout_tok:
                counts[l, e, c] += 1
    expert_of_class = counts.argmax(axis=1)         # (LAYERS, N_CLASS)  TRUE map
    majority_expert = counts.sum(axis=2).argmax(1)  # (LAYERS,)

    # held-out-synonym eval instances, per layer
    per_layer = []
    for l in range(LAYERS):
        c = cls_of[ew[l]]
        mask = (c >= 0) & np.isin(ew[l], list(heldout_tok))
        per_layer.append({"cls": c[mask].astype(int), "obs_expert": ee[l][mask].astype(int)})
    return m_acc, expert_of_class, majority_expert, per_layer


def score_map(class_to_expert, per_layer):
    """Pooled held-out accuracy of a (LAYERS, N_CLASS) class->expert map."""
    ok, n = 0, 0
    for l in range(LAYERS):
        cls = per_layer[l]["cls"]; obs = per_layer[l]["obs_expert"]
        if len(cls):
            pred = class_to_expert[l, cls]
            ok += int((pred == obs).sum()); n += len(cls)
    return ok / n if n else float("nan"), n


def main():
    t0 = time.time()
    seeds_out = []
    for seed in SEEDS:
        m_acc, eoc, maj, per_layer = collect_seed(seed)

        # observed accuracy of the TRUE map
        obs_acc, n_inst = score_map(eoc, per_layer)
        # majority baseline accuracy (published comparison)
        maj_map = np.broadcast_to(maj[:, None], (LAYERS, N_CLASS)).copy()
        maj_acc, _ = score_map(maj_map, per_layer)

        # ---- CLASS-MATCHED NULL via reusable topic_null ----
        # items = 24 classes, group = layer; permuting classes within a layer =
        # permuting the class->expert assignment (each layer has all 24 classes).
        labels = np.tile(np.arange(LAYERS)[:, None], (1, N_CLASS)).ravel()  # group=layer
        # flat class->expert vector, ordered (layer, class)
        base_flat = eoc.ravel().copy()  # length LAYERS*N_CLASS, expert id per (l,c)
        LC = LAYERS * N_CLASS

        def effect_fn(perm):
            permuted = base_flat[perm].reshape(LAYERS, N_CLASS)
            acc, _ = score_map(permuted, per_layer)
            return acc

        res = topic_null_effect(effect_fn, labels, K=K, seed=7000 + seed)

        residual_frac = (res.observed - res.null_mean) / (res.observed - CHANCE) \
            if (res.observed - CHANCE) != 0 else float("nan")

        seeds_out.append({
            "seed": seed, "task_M_acc": float(m_acc), "n_heldout_instances": int(n_inst),
            "obs_acc_true_map": res.observed,
            "class_matched_null": res.to_dict(),
            "majority_baseline_acc": maj_acc,
            "chance": CHANCE,
            "class_explained_share": res.group_explained_share,
            "residual_over_above_chance": residual_frac,
        })
        print(f"seed {seed}: task {m_acc:.3f}  obs {res.observed:.3f}  "
              f"class-null {res.null_mean:.3f} [{res.null_q025:.3f},{res.null_q975:.3f}]  "
              f"maj {maj_acc:.3f}  chance {CHANCE:.3f}  share {res.group_explained_share:.3f}"
              f"  p={res.p_one_sided:.2g}  n={n_inst}", flush=True)

    # aggregate / decision
    obs = np.array([s["obs_acc_true_map"] for s in seeds_out])
    nm = np.array([s["class_matched_null"]["null_mean"] for s in seeds_out])
    ps = np.array([s["class_matched_null"]["p_one_sided"] for s in seeds_out])
    resf = np.array([s["residual_over_above_chance"] for s in seeds_out])
    all_sig = bool(np.all(ps < 0.01) and np.all(obs > nm))
    all_resid = bool(np.all(resf > 0.5))

    def tci(x):
        from scipy.stats import t
        m = float(x.mean())
        if len(x) < 2:
            return [m, None, None]
        h = t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return [m, m - h, m + h]

    out = {
        "application": "a_moe_profile_routing_class_matched_null",
        "config": {"steps": STEPS, "n_prof": N_PROF, "n_eval": N_EVAL,
                   "seeds": SEEDS, "K": K, "n_exp": N_EXP, "n_class": N_CLASS,
                   "chance": CHANCE, "layers": LAYERS,
                   "note": "reduced budget 400 steps / 2000 seqs (PREREG); "
                           "published run was 800 / 3000"},
        "per_seed": seeds_out,
        "aggregate": {
            "obs_acc_across_seeds_tci": tci(obs),
            "class_null_mean_across_seeds_tci": tci(nm),
            "majority_baseline_across_seeds_tci":
                tci(np.array([s["majority_baseline_acc"] for s in seeds_out])),
            "class_explained_share_across_seeds_tci":
                tci(np.array([s["class_explained_share"] for s in seeds_out])),
            "residual_over_above_chance_tci": tci(resf),
        },
        "decision_all_seeds_p_lt01_and_obs_gt_null": all_sig,
        "decision_residual_gt_half_above_chance_all_seeds": all_resid,
        "decision_correspondence_load_bearing": bool(all_sig and all_resid),
        "runtime_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("results_app_a.json", "w"), indent=2)
    print("\n=== DECISION (a) ===")
    print(f"obs acc (seeds): {tci(obs)}")
    print(f"class-matched null mean (seeds): {tci(nm)}")
    print(f"majority baseline (seeds): {out['aggregate']['majority_baseline_across_seeds_tci']}")
    print(f"class-explained share (seeds): {out['aggregate']['class_explained_share_across_seeds_tci']}")
    print(f"all seeds p<0.01 & obs>null: {all_sig}")
    print(f"residual > half of above-chance in all seeds: {all_resid}")
    print(f"=> correspondence load-bearing: {out['decision_correspondence_load_bearing']}")
    print(f"wrote results_app_a.json  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
