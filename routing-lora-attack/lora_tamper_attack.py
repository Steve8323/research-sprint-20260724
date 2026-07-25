"""
Low-rank (LoRA-style) tamper attack on routed+pruned unlearning vs GA-unlearning.

Pre-registered in FINDINGS.md (same directory) BEFORE this script was run.

Reuses the published pipeline unchanged (read-only imports):
  - train_routed("random", seed)  from tamper_experiment.py  (ablation-aware clean scalpel)
  - train_baseline(seed)          from tamper_experiment.py  (KL-to-uniform + retain-CE unlearning)
  - apply_prune / relearn_attack  from tamper_experiment.py  (original full-FT attack, reference arm)

New: lora_attack() -- base weights frozen (pruned entries stay exactly 0 in the base),
trainable rank-r adapters W_eff = W + (alpha/r) B A on fc1/fc2/fc3 weights, attacked with
the IDENTICAL budget (200 forget examples, bs 64, 100 steps, same probe schedule).

Arms per seed (seeds 0-5):
  {routed, baseline} x lora {r1,r4,r16} x lr {1e-3, 1e-2}     (r4 lr1e-3 = PRIMARY)
  routed  x masked-lora r4 x lr {1e-3, 1e-2}                  (mechanism check)
  {routed, baseline} x full-FT (original attack, freeze-mask)  (reference/anchor)

Primary test: paired diff in recovery AUC (routed - baseline) under lora r4 lr1e-3,
exact two-sided sign-flip permutation test on the mean, n=6; plus mean diff w/ 95% t-CI.
"""
import copy, itertools, json, math, os, sys, time
import statistics as st

sys.path.insert(0, "/Users/steven/gradient_routing_experiment")
import torch
import torch.nn.functional as F

from harness import FORGET                                        # noqa: E402
from tamper_experiment import (train_routed, train_baseline,      # noqa: E402
                               apply_prune, relearn_attack,
                               ATTACK_LR, ATTACK_N, ATTACK_BS, PROBE_STEPS, SEEDS)

torch.set_num_threads(3)   # shared machine; harness sets 8, override

OUT = "/Users/steven/moral_framework_misgen/undefined/routing-lora-attack"
LORA_SCALE = 2.0           # alpha/r, fixed across ranks (no rank/scale confound)
GRID_R = [1, 4, 16]
GRID_LR = [1e-3, 1e-2]
PRIMARY = (4, 1e-3)        # pre-registered: rank 4, the original attack's lr


# ---------------------------------------------------------------------------
# LoRA attack: freeze the whole base model, train only rank-r adapters on the
# three weight matrices. mask_delta (mechanism arm) projects the fc1/fc2 deltas
# off the pruned entries every forward pass.
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
            delta = delta * keep[name]          # delta may not touch pruned entries
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

    # identical attack-set construction to relearn_attack (same generator offsets)
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


def main():
    t0 = time.time()
    per_seed = []
    for seed in SEEDS:
        rec = {"seed": seed, "attacks": {}}
        # ---- defended models (published pipeline, unchanged) ----
        rmodel, rmasks, rdata, (rf0, rr0, rf1, rr1) = train_routed("random", seed)
        rec["routed_setup"] = {"forget_pre": rf0, "retain_pre": rr0,
                               "forget_pruned": rf1, "retain_pruned": rr1}
        bmodel, _, bdata, (_, _, bf1, br1) = train_baseline(seed)
        rec["baseline_setup"] = {"forget_post": bf1, "retain_post": br1}

        pruned = copy.deepcopy(rmodel)
        freeze = apply_prune(pruned, rmasks)   # permanent zero + freeze masks

        def add(key, cf, cr=None, ntr=None):
            e = {"forget": cf, "auc": round(auc(cf), 4)}
            if cr is not None:
                e["retain"] = cr
            if ntr is not None:
                e["n_trainable"] = ntr
            rec["attacks"][key] = e

        for rank in GRID_R:
            for lr in GRID_LR:
                cf, cr, ntr = lora_attack(pruned, rmasks, rdata, seed, rank, lr)
                add(f"routed_lora_r{rank}_lr{lr}", cf, cr, ntr)
                cf, cr, ntr = lora_attack(bmodel, None, bdata, seed, rank, lr)
                add(f"baseline_lora_r{rank}_lr{lr}", cf, cr, ntr)
        for lr in GRID_LR:   # mechanism: masked LoRA on routed, r=4
            cf, cr, ntr = lora_attack(pruned, rmasks, rdata, seed, 4, lr, mask_delta=True)
            add(f"routed_loramasked_r4_lr{lr}", cf, cr, ntr)
        # reference: ORIGINAL full-FT attack (freeze-mask enforced on routed)
        m = copy.deepcopy(pruned)
        add("routed_full_ft", relearn_attack(m, {k: v.clone() for k, v in freeze.items()},
                                             rdata, seed))
        m = copy.deepcopy(bmodel)
        add("baseline_full_ft", relearn_attack(m, {}, bdata, seed))

        per_seed.append(rec)
        print(f"seed {seed} done  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- aggregate ----------------
    arms = list(per_seed[0]["attacks"].keys())
    agg = {}
    for arm in arms:
        aucs = [r["attacks"][arm]["auc"] for r in per_seed]
        finals = [r["attacks"][arm]["forget"][max(PROBE_STEPS)] for r in per_seed]
        entry = {"auc_per_seed": aucs,
                 "auc_mean": round(st.mean(aucs), 4),
                 "auc_sd": round(st.stdev(aucs), 4),
                 "final_forget_mean": round(st.mean(finals), 4),
                 "mean_forget_curve": {s: round(st.mean(
                     r["attacks"][arm]["forget"][s] for r in per_seed), 4)
                     for s in PROBE_STEPS}}
        if "retain" in per_seed[0]["attacks"][arm]:
            entry["final_retain_mean"] = round(st.mean(
                r["attacks"][arm]["retain"][max(PROBE_STEPS)] for r in per_seed), 4)
        agg[arm] = entry

    def paired(arm_a, arm_b):
        d = [r["attacks"][arm_a]["auc"] - r["attacks"][arm_b]["auc"] for r in per_seed]
        m, ci = t_ci95(d)
        return {"per_seed_diffs": [round(x, 4) for x in d],
                "mean_diff": round(m, 4),
                "ci95": [round(ci[0], 4), round(ci[1], 4)],
                "p_exact_signflip": round(signflip_p(d), 5)}

    pr, plr = PRIMARY
    pk_r, pk_b = f"routed_lora_r{pr}_lr{plr}", f"baseline_lora_r{pr}_lr{plr}"
    primary = paired(pk_r, pk_b)
    primary["arms"] = [pk_r, pk_b]

    secondary = {}
    for rank in GRID_R:
        for lr in GRID_LR:
            if (rank, lr) == PRIMARY:
                continue
            secondary[f"routed_vs_baseline_lora_r{rank}_lr{lr}"] = paired(
                f"routed_lora_r{rank}_lr{lr}", f"baseline_lora_r{rank}_lr{lr}")
    secondary["routed_vs_baseline_full_ft"] = paired("routed_full_ft", "baseline_full_ft")
    for lr in GRID_LR:
        secondary[f"masked_vs_unmasked_routed_lora_r4_lr{lr}"] = paired(
            f"routed_loramasked_r4_lr{lr}", f"routed_lora_r4_lr{lr}")

    # POST-HOC (labeled): attacker's best grid cell per model, then compare
    def best_cell(model):
        return max(((r, l) for r in GRID_R for l in GRID_LR),
                   key=lambda k: agg[f"{model}_lora_r{k[0]}_lr{k[1]}"]["auc_mean"])
    br_, bl_ = best_cell("routed"), best_cell("baseline")
    posthoc_best = paired(f"routed_lora_r{br_[0]}_lr{br_[1]}",
                          f"baseline_lora_r{bl_[0]}_lr{bl_[1]}")
    posthoc_best["best_routed_cell"] = list(br_)
    posthoc_best["best_baseline_cell"] = list(bl_)
    posthoc_best["NOTE"] = "post-hoc selection of attacker-best cell per model"

    out = {
        "meta": {"n_seeds": len(SEEDS), "seeds": SEEDS, "probe_steps": PROBE_STEPS,
                 "attack": {"n_examples": ATTACK_N, "bs": ATTACK_BS,
                            "steps": max(PROBE_STEPS), "fullft_lr": ATTACK_LR},
                 "lora": {"scale_alpha_over_r": LORA_SCALE, "grid_r": GRID_R,
                          "grid_lr": GRID_LR, "primary": list(PRIMARY),
                          "layers": ["fc1", "fc2", "fc3"],
                          "init": "A~N(0,1/in_f), B=0", "biases": "frozen"},
                 "runtime_s": None},
        "setup_means": {
            "routed_forget_pruned": round(st.mean(
                r["routed_setup"]["forget_pruned"] for r in per_seed), 4),
            "routed_retain_pruned": round(st.mean(
                r["routed_setup"]["retain_pruned"] for r in per_seed), 4),
            "baseline_forget_post": round(st.mean(
                r["baseline_setup"]["forget_post"] for r in per_seed), 4),
            "baseline_retain_post": round(st.mean(
                r["baseline_setup"]["retain_post"] for r in per_seed), 4)},
        "aggregate": agg,
        "primary_test": primary,
        "secondary_tests": secondary,
        "posthoc_attacker_best": posthoc_best,
        "per_seed": per_seed,
    }
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\nSETUP:", json.dumps(out["setup_means"]))
    print("\nmean forget-recovery curves (steps %s):" % PROBE_STEPS)
    for arm in arms:
        print(f"  {arm:32s} AUC={agg[arm]['auc_mean']:.4f}+-{agg[arm]['auc_sd']:.4f}  "
              + " ".join(f"{agg[arm]['mean_forget_curve'][s]:.3f}" for s in PROBE_STEPS))
    print("\nPRIMARY (routed - baseline, lora r4 lr1e-3):", json.dumps(primary))
    for k, v in secondary.items():
        print(f"  {k}: mean_diff={v['mean_diff']} ci={v['ci95']} p={v['p_exact_signflip']}")
    print("POSTHOC attacker-best:", json.dumps(posthoc_best))
    print(f"\ntotal {out['meta']['runtime_s']}s")


if __name__ == "__main__":
    main()
