"""Fill FINDINGS.md Result/Caveats from results.json (run after survey.py)."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json")))
S = R["summary"]; regs = R["regimes"]; rep = R["replication_cell"]

lines = []
lines.append(f"Grid: 25 regimes x n={S['nseeds_per_regime']} fresh seeds = "
             f"{S['total_models']} models, runtime {S['runtime_sec']}s (single process, MPS).")
lines.append("")
nrej = S["n_reject_holm"]; nraw = S["n_raw_p_lt_05"]
lines.append(f"PRIMARY RESULT: {nrej}/25 regimes reject unimodality after Holm-Bonferroni "
             f"(family alpha=0.05); {nraw}/25 at raw p<0.05. Bootstrap-p sensitivity check: "
             f"{S['n_reject_holm_boot']}/25 after Holm.")
sig = [d for d in regs if d["holm_reject"]]
if sig:
    lines.append("Holm-significant regimes: " +
                 ", ".join(f"(c={d['cost']}, W={d['width']}, p={d['p_dip']:.4f})" for d in sig))
lines.append("")
lines.append(f"Replication cell (c=1.0, W=8, the published regime, fresh seeds): "
             f"dip p={rep['p_dip']:.4f} (boot {rep['p_dip_boot']:.4f}), "
             f"nA/nB/mid = {rep['nA']}/{rep['nB']}/{rep['nmid']}, "
             f"best GMM k={rep['best_k']}, train acc {rep['train_acc_mean']:.3f}. "
             f"Published: p=0.0156 at n=120 (verified by recomputation on the stored scores).")
lines.append("")
lines.append("Per-regime table (cost, width, mean train acc, nA/nB/mid, dip p, boot p, Holm):")
lines.append("")
lines.append("| c | W | train acc | nA/nB/mid | dip p | boot p | Holm | note |")
lines.append("|---|---|-----------|-----------|-------|--------|------|------|")
for d in regs:
    note = "UNDERFIT" if d["underfit"] else ""
    if d["cost"] == 1.0 and d["width"] == 8:
        note = (note + " replication cell").strip()
    lines.append(f"| {d['cost']} | {d['width']} | {d['train_acc_mean']:.3f} | "
                 f"{d['nA']}/{d['nB']}/{d['nmid']} | {d['p_dip']:.4f} | "
                 f"{d['p_dip_boot']:.4f} | {'YES' if d['holm_reject'] else 'no'} | {note} |")
lines.append("")
lines.append("Figure: grid_histograms.png (5x5 score histograms, Holm-significant in red).")
result_txt = "\n".join(lines)

p = os.path.join(OUT, "FINDINGS.md")
t = open(p).read()
t = t.replace("## Result\n\n[filled in after the run]", "## Result\n\n" + result_txt)
open(p, "w").write(t)
print(result_txt)
