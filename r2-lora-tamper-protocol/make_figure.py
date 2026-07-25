"""Provisional-vs-LoRA-robust recovery-AUC bar chart across all arms."""
import json, sys
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/Users/steven/research_sprint_20260724/r2-lora-tamper-protocol/results.json"))
agg = d["aggregate"]
arms = d["meta"]["arms"]
PROV = d["meta"]["provisional_cell"]

prov = [agg[a]["cells"][PROV]["auc_mean"] for a in arms]
prov_sd = [agg[a]["cells"][PROV]["auc_sd"] for a in arms]
rob_cells = [agg[a]["lora_robust_cell"] for a in arms]
rob = [agg[a]["cells"][rob_cells[i]]["auc_mean"] for i, a in enumerate(arms)]
rob_sd = [agg[a]["cells"][rob_cells[i]]["auc_sd"] for i, a in enumerate(arms)]
ff = [agg[a]["cells"]["full_ft"]["auc_mean"] for a in arms]

x = np.arange(len(arms)); w = 0.27
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - w, prov, w, yerr=prov_sd, capsize=3, label="provisional (LoRA r4 lr1e-3)", color="#7fb3d5")
ax.bar(x, rob, w, yerr=rob_sd, capsize=3, label="LoRA-robust (attacker-best cell)", color="#c0392b")
ax.bar(x + w, ff, w, label="repo full-FT (freeze-mask)", color="#95a5a6", alpha=0.8)
ax.axhline(agg["GA_baseline"]["cells"][agg["GA_baseline"]["lora_robust_cell"]]["auc_mean"],
           ls="--", color="k", lw=1, label="GA-baseline robust AUC")
ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
ax.set_ylabel("forget-recovery AUC (higher = attack recovers faster)")
ax.set_ylim(0, 1)
ax.set_title("LoRA tamper protocol: every unlearning arm, weak vs tuned attack (n=6 seeds)")
for i, a in enumerate(arms):
    ax.text(x[i], rob[i] + 0.02, rob_cells[i].replace("lora_", ""), ha="center", fontsize=7, rotation=90, color="#c0392b")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("/Users/steven/research_sprint_20260724/r2-lora-tamper-protocol/protocol_table.png", dpi=130)
print("wrote protocol_table.png")
