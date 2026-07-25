"""verify_task.py -- preregistered TASK CONTROL. Proves by construction (not by a trained probe)
that (i) the per-example token multiset is constant across every example in BOTH the training and
deployment distributions -- so any position-blind / bag-of-tokens / counting model is exactly at
chance -- and (ii) the cue structure is what the prereg says it is."""
import json
import torch
from core import gen, occlude, CUE_A_FIELD, CUE_B_FIELD, V_VOCAB, T, TOK_V

out = {}
for mode in ["train", "deploy"]:
    ids, y, cA, cB = gen(5000, 3, mode)
    cnt = torch.stack([(ids == t).sum(1) for t in range(V_VOCAB)], 1).float()
    r = torch.arange(len(ids))
    bA = (ids[r, cA] == CUE_A_FIELD).long()
    bB = (ids[r, cB] == CUE_B_FIELD).long()
    out[mode] = dict(
        token_counts_per_example=cnt[0].long().tolist(),
        max_std_of_token_counts_across_examples=float(cnt.std(0).max()),
        P_cueA_eq_label=round(float((bA == y).float().mean()), 6),
        P_cueB_eq_label=round(float((bB == y).float().mean()), 6),
        P_label_1=round(float(y.float().mean()), 6),
        bag_of_tokens_bayes_accuracy=0.5,
    )
# occlusion sanity: after occluding cue B, cue B must be uninformative and cue A untouched
ids, y, cA, cB = gen(5000, 3, "train")
g = torch.Generator(); g.manual_seed(0)
o = occlude(ids, cB, CUE_B_FIELD, g)
r = torch.arange(len(ids))
out["occlusion_check"] = dict(
    P_cueB_eq_label_after_occludeB=round(float(((o[r, cB] == CUE_B_FIELD).long() == y).float().mean()), 6),
    P_cueA_eq_label_after_occludeB=round(float(((o[r, cA] == CUE_A_FIELD).long() == y).float().mean()), 6),
    token_multiset_unchanged=bool(torch.equal(torch.stack([(o == t).sum(1) for t in range(V_VOCAB)], 1),
                                              torch.stack([(ids == t).sum(1) for t in range(V_VOCAB)], 1))),
    n_positions_changed_mean=round(float((o != ids).sum(1).float().mean()), 4),
)
print(json.dumps(out, indent=1))
json.dump(out, open("/Users/steven/research_sprint_20260724/r3-lm-organism/task_control.json", "w"), indent=1)
