"""
POST-HOC judge-validity check (labeled post-hoc; not part of the pre-registered analysis).

The primary run returned a null with a visibly coarse judge (scores snap to 0/50/100).
To interpret the null we need to know whether the Qwen2.5-3B judge measures anything:
rescore the 32 noise-free sample texts saved in align_judge_results.json — which carry
alignment scores from the repo's original Llama-3.1-8B judge (CUDA run) — with the SAME
3B judge + prompt used in judge_rescore.py, and compute (a) Spearman correlation with the
8B judge, (b) group means base/EM/SFT/noise+SFT vs the 8B judge's ordering.
"""
import json, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hl
from scipy import stats

hl.set_verbosity_error()
REPO = "/Users/steven/misalignment_noise_probe"
OUT = "/Users/steven/moral_framework_misgen/undefined/noiseprobe-behavioral-judge"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

import judge_rescore as J  # reuse ALIGN_SYS / ALIGN_USR / judge_score

def main():
    t0 = time.time()
    d = json.load(open(f"{REPO}/align_judge_results.json"))
    name = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                             attn_implementation="eager").to(DEV).eval()
    rows = []
    for mn, samps in d["samples"].items():
        for s in samps:
            v = J.judge_score(tok, m, J.ALIGN_SYS, J.ALIGN_USR, s["q"], s["resp"])
            rows.append({"group": mn, "axis": s["axis"], "score_8b": s["score"], "score_3b": v})
            print(f"  {mn:>9s} {s['axis']:<13s} 8B={s['score']:>3d} 3B={v}", flush=True)
    ok = [r for r in rows if r["score_3b"] is not None]
    sp = stats.spearmanr([r["score_8b"] for r in ok], [r["score_3b"] for r in ok])
    groups = {}
    for g in ("base", "EM", "SFT", "noise+SFT"):
        gr = [r for r in ok if r["group"] == g]
        groups[g] = {"n": len(gr),
                     "mean_8b": float(np.mean([r["score_8b"] for r in gr])),
                     "mean_3b": float(np.mean([r["score_3b"] for r in gr]))}
    out = {"label": "POST-HOC judge validity check (not pre-registered)",
           "n_texts": len(ok), "n_parse_fail": len(rows) - len(ok),
           "spearman_3b_vs_8b": float(sp.statistic), "p": float(sp.pvalue),
           "group_means": groups, "per_text": rows, "runtime_s": round(time.time() - t0, 1)}
    res = json.load(open(f"{OUT}/results.json"))
    res["posthoc_judge_validity"] = out
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    print(f"\nSpearman(3B,8B) = {sp.statistic:.2f} (p={sp.pvalue:.1e}, n={len(ok)})")
    for g, v in groups.items():
        print(f"  {g:>9s}: 8B {v['mean_8b']:5.1f} | 3B {v['mean_3b']:5.1f}")
    print(f"done {out['runtime_s']}s")

if __name__ == "__main__":
    main()
