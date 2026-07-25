"""Render the corrected tamper table (markdown) from results.json. No new statistics."""
import json, sys

R = json.load(open("/Users/steven/research_sprint_20260724/r4-tamper-power-fix/results.json"))
C, ctrl, desc = R["claims"], R["controls"], R["descriptive"]

R3 = {  # quoted from r3-switch-tamper/results.json, for the "what round 3 said" column
    "C1_missing_cell_auc_switch_vs_h4_FUNC": "NEVER RUN (the missing cell)",
    "C2a_flipped100_switch_vs_h4_FUNC": "NEVER RUN (the missing cell)",
    "C2b_capflip_switch_vs_h4_FUNC": "NEVER RUN (the missing cell)",
    "C3_auc_switch_vs_h4_PREREG": "p=0.400, n=23v12, quoted as 'matches the switch'",
    "C4_auc_switch_vs_corrected_PREREG": "PRIMARY A1 p=0.133, n=23v12, reported as a null",
    "C5_auc_switch_vs_corrected_FUNC": "p=0.019, n=23v12, 'the live advantage'",
    "C6_fullftjit_switch_vs_h4_FUNC": "NEVER RUN (the missing cell)",
}

lines = []
lines.append("| claim | arms / metric / attacker | n | round 3 | round 4 result | verdict |")
lines.append("|---|---|---|---|---|---|")
for k, v in C.items():
    n = f"{v['n1']} vs {v['n2']}"
    if "fisher_p" in v:
        res = (f"p1={v['p1']} vs p2={v['p2']}, diff={v['diff']:+.3f}, "
               f"90% CI [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}], Fisher p={v['fisher_p']:.3g}")
    else:
        res = (f"means {v['mean1']:.3f} vs {v['mean2']:.3f}, diff={v['diff']:+.4f} "
               f"95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], MWU p={v['p_perm']:.4g}, "
               f"HL={v['hl']:+.4f}")
    lines.append(f"| **{k.split('_')[0]}** | {v['claim']} — {v['metric']} | {n} | "
                 f"{R3.get(k,'')} | {res} | **{v['verdict']}** (margin {v['delta_prereg']}, "
                 f"achieved bound {v['delta_star']}) |")
print("\n".join(lines))
print()
print("### descriptive arm table")
print("| attacker : arm | n | best cell | mean AUC9 [95% CI] | flipped@100 | median flip | "
      "mean collateral | fullft AUC9 | fullft+jitter AUC9 |")
print("|---|---|---|---|---|---|---|---|---|")
for k, d in desc.items():
    print(f"| {k} | {d['n']} | {d['best_cell']} | {d['mean_auc9']} {d['ci_auc9']} | {d['flipped']} | "
          f"{d['median_flip']} | {d['mean_final_agree']} | {d['mean_fullft_auc9']} | "
          f"{d['mean_fullftjit_auc9']} |")
print()
print("### controls")
for k, v in ctrl.items():
    if isinstance(v, dict) and "verdict" in v:
        print(f"- **{k}**: diff={v['diff']:+.4f} 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
              f"p={v.get('p_perm', v.get('fisher_p')):.4g}, achieved bound {v['delta_star']} -> {v['verdict']}")
    else:
        print(f"- **{k}**: {v}")
print()
print("### Holm over C1-C6")
for k, v in R["holm"].items():
    print(f"- {k}: p={v['p']:.4g}, threshold {v['holm_threshold']}, "
          f"adj {v['p_holm_adj']}, rejected={v['rejected']}")
print()
print("### pools / anchors")
print(json.dumps({k: R[k] for k in ["routed", "std_pool", "h4_pool", "corrected_f5",
                                    "anchor_batched_vs_sequential", "attack_gate", "structural",
                                    "dropped", "runtime_s"]}, indent=1))
