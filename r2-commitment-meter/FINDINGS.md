# A commitment meter that predicts correction cost

## Question
Round 1 showed hedgers (low per-pattern determinism) flip aligned at a fine-tune dose ~10x
smaller than committed misaligned models, but its controlled test (does the determinism index
predict flip *beyond starting alignment*, at a single fixed dose) was NULL at the marginal
dose. Round 2 asks the clean version: **build a cheap per-model COMMITMENT METER measured
before any fine-tuning, and test whether it predicts each seed's own continuous CORRECTION
DOSE** (the fraction f of disagreeing data needed to flip that seed to aligned). Secondary:
does the continuous meter beat just knowing the worldview label (misaligned vs hedger)?

Pre-registration in `PREREG.md` (stamped Fri Jul 24 23:09:27 EDT 2026, before any run).

## Method (see PREREG.md for the full plan)
- Task/model: exact `stats/paper_stats.py` config -- 154-param 2-hidden-layer width-8 MLP,
  7-dim equal-cost XOR-cue input, Adam lr=5e-3 wd=2e-3, 2000 epochs from scratch. Reimplemented
  from the READ-ONLY `misgen-hedger-mode/run.py`. **Fresh independent seedbase 40000.**
- Population: swept 79 seeds -> 19 aligned (24%), 20 misaligned (25%), 40 hedger (51%);
  **n = 60 non-aligned** models are the correction targets (target was >=60, hit exactly).
- **Commitment meter (pre-FT, frozen weights):** on the 8 cue-*disagreement* input patterns,
  over K=500 shared noise draws/pattern: **D** = per-pattern determinism mean_k|2p_k-1|;
  **M** = mean |logit margin|; **combined** = z(D)+z(M). D range [0.10, 0.97], median 0.478.
- **Per-seed correction dose (DV):** titrated per seed by scanning exact-k nested doses
  f in {0,.25,.5,.75,1,1.5,2,3,5,8}% (1000 FT steps each) then 3 bisection refinements at the
  flip crossing (deploy alignment > 0.8). 0/60 censored; every seed flipped by 8%.
- Runtime 376 s (6.3 min), single CPU process, 373 fine-tune calls. Train acc at dose = 1.000
  for every seed (correction never broke the task).

## Results (plan above unedited)

### PRIMARY -- PASS: the meter predicts correction dose
Combined meter z(D)+z(M) vs per-seed dose: **Spearman rho = 0.570, p = 2.0e-06,
95% bootstrap CI [0.337, 0.740]** (n=60, 0 censored). The CI excludes 0 in the pre-registered
(+) direction -> the pre-stated decision rule fires **PASS**. Robustness (pre-stated):
Kendall tau = 0.429 (p=4.6e-06); drop-censored is identical (no censored seeds).
- Which meter: M (margin) rho=0.581 [0.363,0.745]; D (determinism) rho=0.551 [0.332,0.715];
  combined 0.570. All three predict; margin is marginally the best single meter, and the two
  are near-interchangeable ("how hard the model leans on disagreement inputs").

### SECONDARY -- NULL: the continuous meter does NOT beat the worldview label
- **Partial Spearman(combined, dose | group) = 0.106, p = 0.42** (group = misaligned/hedger
  binary). The continuous meter adds no dose information beyond the label. (D partial 0.060
  p=0.65; M partial 0.154 p=0.24 -- all null.)
- The label alone predicts dose strongly: rho(group, dose) = -0.651, p = 1.7e-08 (hedgers,
  group 1, need the smaller dose).
- Within-group Spearman(combined, dose): **misaligned rho = 0.588, p = 0.006 (n=20)** but
  **hedger rho = -0.033, p = 0.84 (n=40)**.

**Why the split.** Hedger doses are compressed at the floor (26/40 at f <= 0.25%, whole range
0-0.69%), so there is almost no graded dose variance for the meter to explain inside the hedger
group. The misaligned group has real spread (0.19-3.25%) and there the meter genuinely works.
So the headline rho=0.57 is a mix of (a) a strong between-group effect the label already
captures, plus (b) a real, label-independent within-misaligned effect. Honest reading:
**the meter is a good graded cost predictor among committed (misaligned) models, but for
non-aligned models as a whole a one-bit worldview label is as good as the continuous meter.**

### Negative control -- meter is geometry-specific (passes)
Determinism computed on the cue-*agree* (train-type) patterns predicts dose far worse:
|rho(D_agree)| = 0.152 (p=0.25) vs |rho(D_disagree)| = 0.551. The meter's power comes from
the correction-relevant disagreement geometry, not generic confidence.

### Positive control -- high-dose FT is signal, not damage (passes)
Aligned models fine-tuned at f=1% stay aligned 19/19 (Wilson [0.83, 1.00]). At the f=0
circular baseline only 12/19 stay (7 drift) -- consistent with round-1's finding that f=0
carries no worldview signal and just measures drift; not a removal failure.

### Exploratory (post-hoc, labeled as such)
- Initial deploy alignment predicts dose even better than the meter (rho(init, dose) = -0.69,
  p=8e-10). The meter and initial alignment are correlated (rho(combined, init) = -0.57):
  committed misaligned models sit near deploy 0 AND score high on the meter. So "how far the
  model already is from aligned" is the dominant single predictor of cost; the meter's extra
  value is the within-misaligned graded prediction, not a new axis.
- Dose histogram (bins [0,.25,.5,.75,1,1.5,2,3,5,8]%): 27,19,4,1,3,4,1,1,0 -- a heavy floor
  plus a committed tail out to 3.25%.

## Headline
**A pre-fine-tune commitment meter (per-pattern determinism or logit margin on
cue-disagreement inputs) predicts a model's per-seed correction dose (Spearman 0.57,
CI [0.34, 0.74]); but for the non-aligned population as a whole it does not beat a one-bit
worldview label -- its added value is a genuine graded cost prediction *within* the committed
(misaligned) models (rho 0.59, p=0.006).**

## Caveats
- Toy scale: one 154-param MLP on the constructed equal-cost XOR-cue task, one FT budget
  (1000 steps at the training LR), seedbase 40000 only. Absolute doses (median 0.34%, tail
  3.25%) are task/config-specific; the claim is the *ranking* relationship, not the numbers.
- SECONDARY is NULL and that is the honest result: knowing worldview label ~= knowing the meter,
  for the whole non-aligned set. The within-misaligned effect (n=20) is the surviving
  label-independent signal and rests on a modest n.
- The primary rho is partly carried by the between-group (misaligned-vs-hedger) contrast, which
  is exactly what the label also captures; the partial test isolates that overlap and finds no
  residual across the whole set. Reported plainly.
- "Flip" threshold 0.8 and the 1000-step FT budget are inherited from the round-1 repo, not
  tuned. Dose from one shared FT dataset per (seed, f); dataset-draw variance unsampled.
- Meter uses K=500 noise draws/pattern (SE(p_k) ~ 0.022). Committed models' D is partly
  mechanically high (extreme alignment forces one-sided p_k), an inherited property.

## Files
- `PREREG.md` -- pre-registration (stamped before the run)
- `run.py` -- full experiment, single process, 376 s
- `results.json` -- config, sweep, per-model rows (meter D/M, dose, censor, train acc), all tests
- `run.log` -- console output
