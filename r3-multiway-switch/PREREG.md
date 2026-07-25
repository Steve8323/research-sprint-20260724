# PREREG — From a 2-way worldview switch to k-way selection (round 3)

Stamped: `date` output at time of writing, before any experiment code was run:

```
Sat Jul 25 10:10:25 EDT 2026
```

## Background (read-only prior rounds)
In the `moral_framework_misgen` ALIGN toy, two equal-cost XOR cues agree on every training
example (cue A = true objective, input cols [0,1]; cue B = misaligned proxy, cols [2,3]; cols
[4,5,6] noise; label = v). Standard training leaves the adopted worldview to the seed. Round-1
(`routing-worldview-select/`) showed that routing the two worldview-consistent gradient
components (built by occluding the *other* cue's input columns with fresh noise and
forward-ablating the other hidden column) into disjoint hidden columns makes the worldview
selectable post-hoc by ablation: **22/30** seeds at column width 4, **30/30** at width 8
(post-hoc), vs **0/30** for standard training. Round-2 (`r2-routing-switch-learned/`) showed the
cue-localization oracle is removable (learned localizer 23/30 ≈ oracle 22/30).

Both rounds are **k = 2**. This round asks whether the switch is a *switch* or only a *toggle*.

## Hypothesis
H1 (k-way selectability). With k ≥ 3 mutually-agreeing cues and routing into k disjoint hidden
columns of width w, the worldview becomes **k-way** post-hoc selectable: ablating down to column
i yields a model that follows cue i and *only* cue i, for every i, in the same run.

H2 (capacity scaling). Required column width grows with k only through per-column trainability,
not through any k-specific interference: i.e. the k-way clean-selection rate is predicted by
`q_w^k`, where `q_w` = per-column success probability estimated from the k = 2 cell at the same
width. (Round-1: width-4 gave 22/30 at k=2 → q_4 ≈ 0.86; width-8 gave 30/30 → q_8 ≈ 1.0.)

H3 (no cross-worldview interference). Selecting column i leaves negligible behavioral influence
from cue j ≠ i.

## Setup (everything not listed is copied verbatim from round-1 `experiment.py`)
- Input = k XOR cue blocks of 2 dims each + 3 noise dims (`INPUT = 2k+3`); k=2 reproduces the
  original 7-dim input exactly (same RNG consumption order → byte-identical data and initial weights).
- MLP: 2 hidden layers of `H = k*w` units, `MARGIN=2.0, NOISE=0.6, N=800, LR=5e-3, WD=2e-3,
  EPOCHS=2000` full-batch Adam, CrossEntropy. Hidden columns `cols[i] = [i*w, ..., i*w+w-1]`;
  all off-diagonal hidden→hidden blocks zeroed at init (they also receive exactly zero gradient).
- Routing loss per step = mean over i of CE( model(view_i, keep-column-i mask), y ), where
  `view_i` = training batch with **all cue blocks except i** replaced by fresh N(0, 0.6) noise.
  No example ever shows a cue disagreeing with the label. (Implemented as one stacked forward
  over the k views with per-row masks; algebraically identical to the round-1 loss, and for k=2
  it consumes the torch RNG stream in the same order → exact round-1 reproduction.)
- Selection = keep column s, zero all other hidden units (both hidden layers).

## Evaluation sets (all fixed, n = 4000)
- `HELD` — held-out *agreeing* set (train distribution), gen seed 60001. Competence check.
- `PROBE_j` (j = 0..k-1) — cue j encodes the label, **all other cues are flipped**, gen seed
  50001+j. Accuracy ≈ 1 ⇔ the model follows cue j; ≈ 0 ⇔ it follows some other cue. For k=2,
  `PROBE_0` is *exactly* round-1's deploy set (gen seed 50001).
- `DECOUP` — every cue's bit drawn independently, gen seed 70001. Used only for interference.

## Per-seed k-way clean-selection success (pre-stated thresholds, inherited 0.8 / 0.2 / 0.9)
For a run to count as a success, **every** column s ∈ {0..k−1} must satisfy
1. `acc(HELD | keep s) > 0.9` (the column is competent on the training distribution),
2. `acc(PROBE_s | keep s) > 0.8` (it follows its own cue), and
3. `acc(PROBE_j | keep s) < 0.2` for all j ≠ s (it does not follow any other cue).

- **Routed arms are scored under the DESIGNATED mapping** (column i ↔ cue i), because routing
  builds that correspondence by construction — this is the strong k-way claim.
- **The control (standard-training) arm is scored generously**, as the best over all k!
  column→cue permutations, since its column labels are arbitrary.
- Also reported for routed arms (secondary): best-permutation score.

## Grid (pre-stated, small) and n
k ∈ {2, 3, 4} × column width w ∈ {4, 8} × arm ∈ {routed, control} × **n = 20 seeds** (700–719,
the prior rounds' seeds) = 240 training runs. Nothing else will be run confirmatorily.

## PRIMARY tests and decision criterion
- **P1 (k = 3, w = 8):** two-sided Fisher exact, routed vs control, 20 vs 20.
- **P2 (k = 4, w = 8):** two-sided Fisher exact, routed vs control, 20 vs 20.
- Bonferroni over the 2 primaries: each is judged at **α = 0.025**.
- **DECISION:** H1 is SUPPORTED at a given k iff (Fisher p < 0.025) **AND** (routed successes
  ≥ 14/20, i.e. ≥ 70%, the round-1 k=2 width-4 ballpark). If either fails, H1 is REJECTED at
  that k and reported as a negative result. No post-hoc rescue will be folded into the primary.
- Wilson 95% CIs on every rate.

## SECONDARY (labeled secondary)
- **S1 capacity contrast:** within each k, Fisher exact w=8 vs w=4 (routed). Pre-stated
  direction: w=8 ≥ w=4.
- **S2 independent-column prediction (H2):** `q_w = sqrt(rate(k=2, w))`; predicted rate at k is
  `q_w^k`. Report observed vs predicted with Wilson 95% CI on observed; declare "excess,
  k-specific failure" iff the observed Wilson interval excludes the predicted value.
- **S3 interference (H3):** on `DECOUP`, for a model with column s kept, define
  `leak_j = 2·|acc(pred == cue_j bit) − 0.5|` and `self_s` likewise for j = s.
  Per-seed worst-case leak = `max_s max_{j≠s} leak_j`. Pre-stated bar: selection is "clean" iff
  the median per-seed worst-case leak < 0.10. Compare k=3 and k=4 vs k=2 at the same width with
  a two-sided Mann–Whitney U test (does interference grow with k?).
- **S4 hedging cost (descriptive):** un-ablated (full) model accuracy on HELD and on each
  PROBE_j, per cell.

## Controls
- **Positive control / replication anchor:** the k=2, w=4 routed and control cells must reproduce
  round-1 `routing-worldview-select/results.json` per-seed metrics for seeds 700–719 exactly
  (`selA_align`, `selB_align`, `train_acc`, `full_align` on PROBE_0/HELD); max |diff| reported,
  expected 0.000000. If this anchor fails, the k>2 numbers are reported as a *new* experiment,
  not as an extension.
- **Negative control:** the control arm at every (k, w) — standard training on the same agreeing
  data; expected ≈ 0 successes, and it defines the floor for the primaries.

## Compute budget / downscope rule
Single process, torch 2 threads, CPU only, no downloads. Target ≤ 15 min wall clock. If a timing
pilot (2 seeds of the largest cell, k=4/w=8 — timing only, no results reported) projects > 12
min, downscope in this pre-stated order and say so explicitly in FINDINGS.md:
(1) drop the k=2, w=8 routed+control cells (already known from round-1's post-hoc 30/30),
(2) reduce EPOCHS from 2000 to 1200 for **all** cells (keeping the grid balanced), which forfeits
the exact round-1 anchor and would be stated as such.

## Anything not listed above is POST-HOC and will be labeled as such in FINDINGS.md.
