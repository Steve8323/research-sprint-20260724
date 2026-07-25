"""Renders the confirmatory results.json into the markdown tables used in FINDINGS.md.
Pure formatting -- computes no new statistic except the two clearly-labelled extra Fisher
tests comparing routed arms to the LEARNED-mask floor, which are S4 as preregistered."""
import json
import numpy as np
from scipy.stats import fisher_exact

D = "/Users/steven/research_sprint_20260724/r4-break-tautology/"
R = json.load(open(D + "results.json"))
A = R["arms"]; GT = R["grad_trace"]
order = ["anchor_disjoint_w4", "ceiling_disjoint_w8", "overlap25", "overlap50", "overlap75",
         "shared_trunk", "soft25", "soft50", "floor_control"]
order = [a for a in order if a in A]

print("### Main table\n")
print("| arm | H | w | shared units | shared params (grad trace) | fixed-mask k/n | rate [Wilson 95%] | p vs floor | verdict | learned-mask k/n |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in order:
    s = A[a]["summary"]; g = GT[a]
    pv = R.get("primary", {}).get(a)
    p = f"{pv['p_vs_floor']:.3g}" if pv else "-"
    vd = pv["verdict"] if pv else ("ceiling" if "ceiling" in a else ("floor" if "floor" in a else "anchor"))
    print(f"| `{a}` | {s['H']} | {s['w']} | {s['overlap_units']} | {g['shared_params']}/{g['total_params']} "
          f"| {s['successes']}/{s['n']} | {s['rate']:.3f} {s['wilson95']} | {p} | {vd} "
          f"| {s['learned_successes']}/{s['n']} |")

print("\n### S1 — distance from the disjoint ceiling\n")
print("| variant | ceiling | variant | p vs ceiling | Newcombe 95% CI (ceiling − variant) | equivalent at δ=0.20? |")
print("|---|---|---|---|---|---|")
for v, d in R["S1_vs_ceiling"].items():
    print(f"| `{v}` | {d['ceiling'][0]}/{d['ceiling'][1]} | {d['variant'][0]}/{d['variant'][1]} "
          f"| {d['p_vs_ceiling']:.3g} | {d['newcombe95_ceiling_minus_variant']} | {d['equivalent_to_ceiling']} |")

print("\n### S2 — overlap dose-response\n")
print(json.dumps(R["S2_trend"], indent=1))

print("\n### S4 — learned eval mask (train mask != eval mask)\n")
s4 = R["S4_learned_mask"]
print(f"floor under learned mask: {s4['floor_learned'][0]}/{s4['floor_learned'][1]}\n")
print("| arm | learned k/n | rate [Wilson 95%] | exact mask match | mean Jaccard vs true S_A | p raw | p Holm | passes |")
print("|---|---|---|---|---|---|---|---|")
for a, d in s4["arms"].items():
    print(f"| `{a}` | {d['k']}/{d['n']} | {d['rate']:.3f} {d['wilson95']} | {d['exact_mask_match']}/{d['n']} "
          f"| {d['jaccard_mean']:.3f} | {d['p_raw']:.3g} | {d['p_holm']:.3g} | {d['passes']} |")

print("\n### Anchor A0\n")
print(json.dumps(R["anchor_A0"], indent=1))

print("\n### Failure composition (pre-stated capacity diagnostic)\n")
print("| arm | misses | of which competence misses | median selA_train | median selB_train |")
print("|---|---|---|---|---|")
for a in order:
    s = A[a]["summary"]
    print(f"| `{a}` | {s['n_misses']} | {s['n_misses_competence']} | {s['selA_train_med']:.3f} | {s['selB_train_med']:.3f} |")

print("\n### Cross-block mass (median max|w| over the disallowed L2 region)\n")
for a in order:
    print(f"  {a:22s} {A[a]['summary']['cross_block_maxabs_med']}")

print(f"\nruntime_sec = {R.get('runtime_sec')}")
