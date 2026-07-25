"""Render the Results section of FINDINGS.md from results.json (run after experiment.py)."""
import json

R = json.load(open("/Users/steven/moral_framework_misgen/undefined/ert-noise-probe/results.json"))
A, B = R["armA"], R["armB"]
pt = A["primary_test"]

lines = []
lines.append("### Arm A - concealer organism QC (clean, per seed)\n")
lines.append("| seed | task acc | covert decoder acc (chance 0.25) | canonical compliance L0-L3 |")
lines.append("|---|---|---|---|")
for s, q in sorted(A["qc"].items()):
    comp = " / ".join(f"{c:.3f}" for c in q["compliance_per_layer"])
    lines.append(f"| {s} | {q['task_acc']:.3f} | {q['decoder_acc']:.3f} | {comp} |")
lines.append("")
lines.append("Clean reveal-eligible events at RPOS (n=1200/seed, s0 in {1,2,3}):")
for s, q in sorted(A["qc"].items()):
    tot_el = sum(v["n_eligible"] for v in q["clean_reveal"].values())
    tot_rv = sum(v["k_reveal"] for v in q["clean_reveal"].values())
    lines.append(f"- {s}: eligible {tot_el}/4800 layer-events, reveals {tot_rv}")
lines.append("")
lines.append("Stash form at the free positions (clean, final layer):\n")
lines.append("| seed | pos | P(verbatim w0) | P(synonym index == s0) |")
lines.append("|---|---|---|---|")
for s, st in sorted(A["stash"].items()):
    for p, v in sorted(st.items()):
        lines.append(f"| {s} | {p} | {v['verbatim_w0']:.3f} | {v['syn_idx_eq_s0']:.3f} |")
lines.append("")

lines.append("### PRIMARY A result\n")
lines.append(f"Pooled over seeds x layers x sigma {{0.25,0.5,1.0}}: reveals "
             f"**{pt['k_reveal']}/{pt['n_eligible']}** eligible events = "
             f"**{pt['rate']:.3f}** (Wilson 95% CI [{pt['wilson95'][0]:.3f}, "
             f"{pt['wilson95'][1]:.3f}]), null 1/3, exact two-sided binomial "
             f"**p = {pt['p_two_sided_exact']:.4g}**.\n")

# per-cell table (primary sigmas)
lines.append("Per-cell (primary sigmas; eligible k/n, reveal rate):\n")
lines.append("| seed | layer | sigma | n_eligible | k_reveal | rate | task acc | dec acc |")
lines.append("|---|---|---|---|---|---|---|---|")
for c in A["cells"]:
    if not c["primary"]:
        continue
    r = c["k_reveal"] / c["n_eligible"] if c["n_eligible"] else float("nan")
    lines.append(f"| {c['seed']} | L{c['noise_layer']} | {c['sigma']} | "
                 f"{c['n_eligible']} | {c['k_reveal']} | {r:.3f} | "
                 f"{c['task_acc']:.3f} | {c['decoder_acc']:.3f} |")
lines.append("")

# exploratory
lines.append("### Exploratory (labeled)\n")
expl = [c for c in A["cells"] if not c["primary"]]
tot_el = sum(c["n_eligible"] for c in expl)
tot_rv = sum(c["k_reveal"] for c in expl)
lines.append(f"- sigma {{0.1, 2.0}} pooled: reveals {tot_rv}/{tot_el} eligible.")
blurt = sum(c["k_blurt"] for c in A["cells"])
n_all = sum(c["n"] for c in A["cells"])
lines.append(f"- verbatim blurt (RPOS emission == w0 exactly, all cells): {blurt}/{n_all}.")
dtot_el = dtot_rv = 0
for c in A["cells"]:
    for v in c["downstream"].values():
        dtot_el += v["n_eligible"]; dtot_rv += v["k_reveal"]
lines.append(f"- downstream reads (emission at layers after the noised one): "
             f"{dtot_rv}/{dtot_el} eligible.")
wtot = sum(c["n_wrongclass"] for c in A["cells"])
wrev = sum(c["k_wrongclass_reveal"] for c in A["cells"])
lines.append(f"- wrong-class emissions with synonym==s0: {wrev}/{wtot}.")
lines.append("")

lines.append("Example traces (seed 0; words per layer, RPOS first):\n```")
for ex in A["examples"]:
    lines.append(f"input: {ex['input']}   (s0={ex['s0']})")
    for l, (c1, c2) in enumerate(zip(ex["clean"], ex["noised_L1_sigma1"])):
        lines.append(f"  L{l} clean : {c1}")
        lines.append(f"  L{l} noise1: {c2}")
lines.append("```\n")

lines.append("### PRIMARY B result (destroyed control)\n")
pb = B["primary_test"]
lines.append("| seed | repo eval_S | clean argmax acc [Wilson 95%] |")
lines.append("|---|---|---|")
for s, q in sorted(B["seeds"].items()):
    lines.append(f"| {s} | {q['repo_eval_S']:.3f} | {q['clean_acc']:.3f} "
                 f"[{q['wilson95'][0]:.3f}, {q['wilson95'][1]:.3f}] |")
bc = B["best_primary_cell"]
lines.append("")
lines.append(f"Cells significantly above clean (one-sided Fisher, Bonferroni "
             f"alpha={pb['alpha_bonferroni']:.5f}): "
             f"**{pb['n_cells_significantly_above_clean']}/36**. Prediction "
             f"({'confirmed' if pb['prediction_confirmed'] else 'NOT confirmed'}). "
             f"Best primary cell: seed {bc['seed']}, L{bc['noise_layer']}, "
             f"sigma={bc['sigma']}: acc {bc['acc']:.3f} "
             f"(Fisher p={bc['fisher_p_greater_than_clean']:.3f}; copy-strategy "
             f"ceiling 7/27 = {7/27:.3f}).\n")

lines.append(f"Wall time: {R['wall_time_s']:.0f} s, single CPU process, 1 torch thread.\n")

p = "/Users/steven/moral_framework_misgen/undefined/ert-noise-probe/FINDINGS.md"
t = open(p).read()
t = t.replace("## Results\n\n(appended after the run)",
              "## Results\n\n" + "\n".join(lines))
open(p, "w").write(t)
print("FINDINGS.md results appended")
