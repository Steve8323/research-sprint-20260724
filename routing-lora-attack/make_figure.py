"""Regenerate recovery_curves.png from results.json (real run data, mean +- sd over 6 seeds)."""
import json
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/steven/moral_framework_misgen/undefined/routing-lora-attack"
d = json.load(open(OUT + "/results.json"))
steps = d["meta"]["probe_steps"]


def band(arm):
    per = [[r["attacks"][arm]["forget"][str(s)] if isinstance(list(r["attacks"][arm]["forget"])[0], str)
            else r["attacks"][arm]["forget"][s] for s in steps] for r in d["per_seed"]]
    mean = [st.mean(c[i] for c in per) for i in range(len(steps))]
    sd = [st.stdev(c[i] for c in per) for i in range(len(steps))]
    return mean, sd


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

panels = [
    ("Matched budget (lr 1e-3): PRIMARY",
     [("routed_lora_r4_lr0.001", "tab:blue", "-", "routed+pruned, LoRA r4"),
      ("baseline_lora_r4_lr0.001", "tab:red", "-", "GA-baseline, LoRA r4"),
      ("routed_full_ft", "tab:blue", "--", "routed+pruned, full-FT (orig attack)"),
      ("baseline_full_ft", "tab:red", "--", "GA-baseline, full-FT (orig attack)")]),
    ("Attacker-tuned LoRA (post-hoc best cell)",
     [("routed_lora_r16_lr0.01", "tab:blue", "-", "routed+pruned, LoRA r16 lr1e-2"),
      ("baseline_lora_r4_lr0.01", "tab:red", "-", "GA-baseline, LoRA r4 lr1e-2"),
      ("routed_loramasked_r4_lr0.01", "tab:green", ":", "routed, MASKED LoRA r4 lr1e-2"),
      ("routed_full_ft", "tab:blue", "--", "routed+pruned, full-FT (orig attack)")]),
]

for ax, (title, arms) in zip(axes, panels):
    for arm, c, ls, lab in arms:
        m, s = band(arm)
        ax.plot(steps, m, ls, color=c, label=lab, lw=2)
        ax.fill_between(steps, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)], color=c, alpha=0.12)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("attack steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
axes[0].set_ylabel("forget-class test accuracy")
fig.suptitle("LoRA relearning attack: routed+pruned vs GA-unlearned (mean ± sd, 6 seeds)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "/recovery_curves.png", dpi=150)
print("wrote recovery_curves.png")
