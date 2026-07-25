#!/usr/bin/env python3
"""Merge Part 1 (ERT) and Part 2 (vineppo) into a single results.json."""
import json
from pathlib import Path

D = Path(__file__).parent
p1 = json.load((D / "results_part1.json").open())
p2 = json.load((D / "results_part2.json").open())

# trim bulky per-prediction vectors from p2 into a compact form for top-level results
p2_compact = {k: v for k, v in p2.items()
              if k not in ("predictions", "v_gold_vector", "pid_vector", "k_vector")}

out = dict(
    study="r2-legible-value-readout",
    part1_ert=dict(
        config=p1["config"],
        calibration_choice=dict(ert_steps=p1["config"]["ert_steps"],
                                base_sigma=p1["config"]["base_sigma"],
                                base_steps=p1["config"]["base_steps"]),
        summary=p1["summary"],
    ),
    part2_vineppo=p2_compact,
)
json.dump(out, (D / "results.json").open("w"), indent=1)
print("wrote results.json")
print("Part1 primary:", p1["summary"]["primary"]["metric"],
      "MW p=", p1["summary"]["primary"]["mannwhitney_p_two_sided"])
print("Part2 honest primary text-hidden diff:",
      p2["HONEST_within_model"]["paired_diffs_text_minus_hidden"])
