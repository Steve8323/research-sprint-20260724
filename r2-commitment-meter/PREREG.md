# PREREG — A commitment meter that predicts correction cost

Stamped (shell `date`): **Fri Jul 24 23:09:27 EDT 2026**
Nothing above the "Results" line in FINDINGS is edited after the run starts.

## Motivation / relation to round 1
Round-1 `misgen-hedger-mode` computed a per-pattern determinism index D and tested whether
it predicts *flip-or-not at a single fixed fine-tune dose* (f=0.5%), controlling for initial
deploy alignment: **NULL at the marginal dose** (partial Spearman −0.043, p=0.69), positive
only at f=2% (−0.317, p=2.6e-3). Round-1 `misgen-removal-dose` showed the *population* dose
curve (f50≈0.94%, f*≈5%) but did not attach a dose to each model.

Round 2 asks the cleaner question the brief specifies: **build a cheap per-model COMMITMENT
METER measured BEFORE any fine-tuning, and test whether it predicts each seed's own
correction dose** (the continuous fine-tune fraction f needed to flip that seed to aligned).
The dependent variable is now a graded per-seed dose, not a binary flip, which is the variable
round 1 lacked.

## Task / model (exact replication of stats/paper_stats.py config)
2-hidden-layer width-8 MLP (154 params); 7-dim input (true XOR cue dims 0–1, proxy XOR cue
dims 2–3, 3 noise dims); MARGIN=2.0, NOISE=0.6, N=800 train rows; Adam lr=5e-3, wd=2e-3,
2000 full-batch epochs from scratch. Deploy eval = 4000 fresh rows with proxy anti-correlated
(the round-1 harness). Code imported/reimplemented from
`misgen-hedger-mode/run.py` (READ-ONLY source).

**Fresh independent seedbase = 40000** (NOT the published 700, NOT removal-dose's 21000).
Report base rates.

## Groups (round-1 thresholds, inherited)
- aligned  : deploy alignment > 0.8
- misaligned: deploy alignment < 0.2
- hedger   : in between
Correction target = the non-aligned population (misaligned + hedger). Aligned models are a
control (already correct; nominal dose 0).

## Commitment meter (measured PRE-fine-tune, on the trained model)
On the 8 cue-disagreement input patterns (the 4 cue dims take 8 sign patterns where true and
proxy cues disagree), over K=500 shared fresh noise draws per pattern:
- **D (determinism)** = mean_k |2 p_k − 1|, p_k = P(prediction follows the TRUE cue). D∈[0,1];
  1 = per-pattern deterministic, 0 = coin-flip. (round-1 definition, unchanged.)
- **M (logit margin)** = mean over disagreement inputs of |logit_1 − logit_0| (absolute margin).
- **combined** = z(D)+z(M) (standardized sum), and a 2-predictor rank model.
All three are computed from the frozen trained weights with NO fine-tuning.

## Per-seed correction DOSE (the dependent variable)
For each non-aligned seed, fine-tune from its trained weights for 1000 Adam steps
(same lr/wd) on N=800 fresh train-style rows (seed+12345) containing exactly k = round(f·800)
anti-proxy rows (proxy disagrees, label = truth) — the round-1 removal protocol, with
**exact-k nested dosing** (removes binomial dose noise; matches misgen-removal-dose).
Flip = post-FT deploy alignment > 0.8.
**dose_i = smallest f on the search that flips seed i.** Found by scanning a fixed nested
grid f ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8}% then, if it flips somewhere in
(f_lo, f_hi], **3 bisection refinements** between the last non-flip and first flip to sharpen
the crossing. dose recorded as the refined crossing f (in %). Seeds that never flip by f=8%
are **right-censored at 8%** (recorded dose = 8, flagged censored). k is exact per f.

## PRIMARY test (pre-registered, confirmatory)
Spearman rank correlation **ρ(commitment_meter, dose)** over all non-aligned seeds, for the
combined meter z(D)+z(M) as the primary meter (D-alone and M-alone reported alongside).
- **Direction:** more committed → larger dose, so predicted ρ > 0.
- **CI:** 5000-resample seed bootstrap 95% CI on ρ. Also report the exact Spearman p.
- **DECISION RULE (fixed in advance):** the meter "predicts correction cost" if the combined
  meter's bootstrap 95% CI for ρ excludes 0 in the predicted (+) direction. Censored seeds
  are included at dose=8 (a conservative lower bound on their true dose). Robustness:
  re-run the primary (a) dropping censored seeds, (b) Kendall τ instead of Spearman.

## SECONDARY test (pre-registered): does the meter beat the worldview label alone?
The trivial predictor is the binary group label (misaligned vs hedger). Two pre-stated tests:
1. **Partial Spearman** ρ(meter, dose | group), group coded as the binary misaligned/hedger
   indicator (rank-regress-residual method, Pearson on residual ranks, df=n−3). If the
   partial ρ is still positive with p<0.05, the *continuous* meter carries dose information
   beyond the label. NULL is an acceptable, reportable outcome.
2. **Within-group** Spearman ρ(meter, dose) computed inside the misaligned group and inside
   the hedger group separately (label held constant by construction). Report both with n.

## Positive & negative controls (pre-stated)
- **Positive control:** aligned models fine-tuned at f=0 and f=1% must stay aligned
  (frac aligned, Wilson CI). Confirms high-dose removal is signal, not generic damage, and
  that aligned = trivially-corrected end of the commitment axis.
- **Negative control (meter specificity):** a *scrambled* meter — D computed on the 8
  cue-AGREE patterns (train-type; not the correction-relevant disagreement geometry) —
  should predict dose MORE WEAKLY than the disagreement-pattern D. Report ρ for both;
  pre-stated expectation: |ρ_disagree| > |ρ_agree|.
- **Sanity:** post-FT train accuracy ≥ 0.99 at the recorded dose (correction did not break
  the task); reproduction check — regenerated group counts reported.

## n / sample size
Sweep fresh seeds from seedbase 40000 until **n_nonaligned ≥ 60** non-aligned models (target
60–90; hard cap 260 seeds or an 11-minute wall guard, whichever first). Report actual n and
base rates. If the wall guard trips before 60, use whatever n≥40 is reached and flag it.

## Statistics conventions
- Spearman/Kendall for rank tests; exact p from scipy. 5000-resample seed bootstrap for ρ CIs.
- Wilson intervals for proportions. All n reported. Two-sided unless a direction is pre-stated
  (primary direction is +).
- Nulls are acceptable outcomes and will be reported plainly.

## Runtime budget
Single process, CPU, ≤1 torch thread group is 6 threads (round-1 setting). Target < 12 min
wall. If titration overruns, the pre-stated downscope is: shrink the fine-fraction grid to
{0,0.5,1,2,3,5,8}% and drop bisection to 2 refinements (still per-seed continuous dose).

## What is confirmatory vs exploratory
Confirmatory: the PRIMARY combined-meter Spearman CI test, and the two SECONDARY tests.
Everything else (D-alone vs M-alone comparison as the "which meter" question, Kendall
robustness, dose histograms, dose-vs-initial-alignment, meter-vs-initial-alignment, any
regression R², logistic dose model) is EXPLORATORY / post-hoc and labeled as such.
