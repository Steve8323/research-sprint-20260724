"""Summarize results.json -> stdout + append Results section to FINDINGS.md."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
S = R.get("summary", {})


def fmt(a, t):
    k = f"{a}_{t}"
    if k not in S:
        return f"{a} {t}: (not run)"
    d = S[k]
    per = ", ".join(f"{x:.3f}" for x in d["per_seed"])
    return (f"{a:8s} {t}: mean={d['mean']:.3f}  per-seed=[{per}]  "
            f"(n_seeds={d['n_seeds']}, each acc on n=6000, Wilson95 ~ +/-0.011 at p=0.25)")


lines = []
lines.append("### Numbers (rho=1.0, 600 steps, 3 seeds, eval n=6000/model)")
lines.append("")
lines.append("```")
for a in ["oracle", "distill", "geom", "random"]:
    for t in ["M", "S"]:
        if f"{a}_{t}" in S:
            lines.append(fmt(a, t))
lines.append("```")
lines.append("")
pt = S.get("primary_test", {})
lines.append(f"**Pre-stated primary test** ({pt.get('criterion','')}): "
             f"**{'PASSED' if pt.get('passed') else 'FAILED'}**")
lines.append(f"- distill M per seed: {[round(x,3) for x in pt.get('distill_M_per_seed',[])]}")
lines.append(f"- distill S per seed: {[round(x,3) for x in pt.get('distill_S_per_seed',[])]}")
dmo = S.get("distill_minus_oracle", {})
if dmo:
    lines.append(f"- distill minus oracle, M per seed: {[round(x,3) for x in dmo['M_per_seed']]}, "
                 f"S per seed: {[round(x,3) for x in dmo['S_per_seed']]}")
lines.append("")
lines.append("### Learned-partition quality")
lines.append("")
lines.append("```")
for a in ["distill", "geom", "random"]:
    for p in R["arms"][a]["partitions"]:
        extra = (f" probe_model_M={p['probe_model_M_acc']:.3f}"
                 if "probe_model_M_acc" in p else "")
        lines.append(f"{a:8s} seed{p['seed']}: ARI={p['ari']:.3f} purity={p['purity']:.3f} "
                     f"cross_class_swap={p['cross_class_swap_frac']:.3f} "
                     f"pair_coverage={p['within_class_pair_coverage']:.3f} "
                     f"groups={p['n_groups']} singletons={p['n_singletons']}{extra}")
lines.append("```")
if "skipped" in R:
    lines.append("")
    lines.append(f"Skipped for budget: {R['skipped']}")
lines.append("")
lines.append(f"Wall time: {R.get('wall_time_s', float('nan')):.0f}s")

text = "\n".join(lines)
print(text)
with open(os.path.join(HERE, "FINDINGS.md")) as f:
    md = f.read()
md = md.replace("(to be appended after the run)", text)
open(os.path.join(HERE, "FINDINGS.md"), "w").write(md)
print("\n[FINDINGS.md updated]")
