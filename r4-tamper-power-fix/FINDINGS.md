# FINDINGS — The missing tamper cell, with equivalence bounds (round 4)

PREREG: `PREREG.md`, stamped **Sat Jul 25 11:11:13 EDT 2026**, Amendment 1 (n 80 -> 60) stamped
**11:18:44**, both before the confirmatory run. Code: `experiment.py`. Data: `results.json`.
Post-hoc, physically separate: `posthoc.py` / `posthoc.json`. Confirmatory runtime **840.6 s**
(14.0 min) on a box carrying 3 other agents' jobs; probes + smoke added ~105 s.

---

## Headline

**Round 3's conclusion survives, but the reason it gave is wrong.** Round 3 argued the worldview
switch is not a safety property because the tamper-resistance of hard deletion "is not separable
from parameter count" — an H=4 capacity control "matches the switch" (p=0.40, n=23 vs 12). At
n=60 vs 60 the two are **not** equivalent: they differ, significantly, in *both directions*
depending on how strong the attacker is. The switch beats the capacity control under the weak
preregistered LoRA attacker (−0.079 AUC, p=0.0014), is indistinguishable from it under the realistic
functional attacker (bounded at 0.065 AUC — **the missing cell**), and **loses to it** under the
strongest honest attacker (+0.093 AUC, p=0.030). The one regime where routing wins is also the
regime where the attack produces a broken model (26.7% of attacked switches still competent) and
where the exactly-zero dead weights make `relu'(0)=0` block revival; adding 1% weight jitter flips
the sign. So: not a safety property — but because the advantage is attacker-dependent and
sign-unstable, not because deletion and capacity are the same thing.

---

## The corrected tamper table

Verdicts follow the preregistered rule: MWU p < 0.05 -> REAL DIFFERENCE with a CI; otherwise the
**achieved equivalence bound** `delta*` (the 90% CI bound on |mean difference|; TOST at margin
`delta` rejects exactly when `delta* <= delta`). "No difference" is never a verdict. Scale reference:
the **total defense effect** measured in this run is 0.178 AUC (no-defense 0.944 vs switch 0.767,
functional attacker).

| # | claim (arms / metric / attacker) | n | round 3 said | round 4 | verdict |
|---|---|---|---|---|---|
| **C1** | **THE MISSING CELL** — switch vs H=4 capacity control, best-cell AUC, **functional** attacker | 60v60 | never run | 0.767 vs 0.786, d = **−0.020**, 95% CI [−0.074, +0.034], MWU p = 0.35, HL −0.017 | **BOUNDED AT delta\* = 0.065** (margin 0.05). Any routing-specific advantage here is at most **36% of the total defense effect**, and zero is well inside the interval. |
| **C2a** | switch vs H=4, **flipped@100** (uncensorable), functional | 60v60 | never run | 0.667 vs 0.650, d = **+0.017**, 90% CI [−0.124, +0.156], Fisher p = 1.0 | **BOUNDED AT delta\* = 0.156** (margin 0.15) |
| **C2b** | switch vs H=4, capped flip time min(flip,101), functional | 60v60 | never run | 50.1 vs 48.9 steps, d = **+1.1**, 95% CI [−13.1, +15.3], MWU p = 0.62 | **BOUNDED AT delta\* = 13.0 steps** (margin 10) |
| **C3** | switch vs H=4, best-cell AUC, **prereg** attacker (r3's load-bearing null) | 60v60 | "matches the switch", p = 0.40, n=23v12 | 0.755 vs 0.834, d = **−0.079**, 95% CI [−0.124, −0.033], MWU p = **0.0014**, HL −0.050 | **REAL DIFFERENCE — round 3's equivalence claim is overturned.** Survives Holm. |
| **C4** | switch vs corrected(f=5%), best-cell AUC, **prereg** attacker (r3 PRIMARY A1) | 60v60 | null, p = 0.133, n=23v12 | 0.755 vs 0.835, d = **−0.080**, 95% CI [−0.124, −0.037], MWU p = **0.0034**, HL −0.044 | **REAL DIFFERENCE — r3's A1 was a false negative.** Survives Holm. |
| **C5** | switch vs corrected, best-cell AUC, **functional** attacker | 60v60 | "the live advantage", p = 0.019, n=23v12 | means 0.767 vs 0.842 (Welch 95% CI [−0.117, −0.033], **excludes 0**) but ranks overlap: MWU p = 0.21, HL −0.026 [−0.098, +0.011] | **BOUNDED AT delta\* = 0.110** by the preregistered rule. The two preregistered summaries disagree: a **tail effect, not a location shift** (see below). |
| **C6** | switch vs H=4, **full-FT + 1% jitter** (strongest honest attacker), functional | 60v60 | never run | 0.560 vs 0.467, d = **+0.093**, 95% CI [+0.034, +0.153], MWU p = **0.031** | **REAL DIFFERENCE, AGAINST the switch** — the capacity control is *more* tamper-resistant. **Does not survive Holm** (adj. p = 0.16). |

Holm–Bonferroni over C1–C6 (7 tests): C3 (adj 0.0095) and C4 (adj 0.020) are rejected; C6
(adj 0.16), C5, C1, C2a, C2b are not.

## Repair (A): the missing cell, in one line

`h4_aligned` under the Amendment-2 functional attacker — never run in round 3 — behaves **just like
the switch**: mean AUC 0.786 vs 0.767, flipped 39/60 vs 40/60, median flip 30 vs 31 steps. This is
the condition round 3 called "the only one where the switch shows a live, non-degenerate advantage",
and the capacity control shows the same advantage. The advantage over the *corrected* baseline is
real; the advantage over *the same number of live parameters* is bounded at 0.065 AUC.

## Repair (B): the degenerate flip test, fixed

Round 3's flip-cost comparison against the capacity control (post-hoc P1, p = 0.20) had both arms
pinned at the censoring bound 101. The **preregistered dynamic-range check** confirms it formally:
under the extreme configuration "perfect switch (0/60 flipped) vs capacity control behaving like no
defense (60/60 flipped)", that test returns **p = 1.0** — it cannot reject even when the effect is
total. The replacement, `flipped@100` (binary, uncensorable), returns **p = 2.1e-35** on the same
extreme configuration. Round 3's flip null was not weak evidence of equivalence; it was **no
evidence at all**, and is withdrawn rather than re-interpreted.

## What the switch actually does (POST-HOC — `posthoc.json`, not preregistered)

C5's internal disagreement has a clean explanation. The switch arm is **bimodal**: 11/60 models
resist strongly (AUC < 0.6, min 0.484) while the median switch (0.815) sits close to the corrected
baseline (0.859). SD 0.153 vs 0.060 for corrected (Levene p = 1.3e-8, SD ratio 2.57). Routing does not
make the typical model harder to tamper with; it makes a **minority** of models much harder.

And that tail is **not routing-specific either**: the H=4 capacity control has 9/60 strong
resisters vs the switch's 11/60 (Fisher p = 0.81) with an identical spread (SD ratio 1.05, Levene
p = 0.77), while the corrected baseline has 0/60 (p = 5.9e-4). The resistant tail is a property of
*having few live units*, not of routing.

The attacker-strength ladder (switch minus H=4 capacity control, negative = switch more resistant):

| attacker | switch | H=4 | diff | p |
|---|---|---|---|---|
| prereg LoRA, best cell | 0.755 | 0.834 | **−0.079** | 0.0014 |
| functional LoRA, best cell | 0.767 | 0.786 | −0.020 | 0.35 |
| prereg full-FT *(relu'(0)=0 artifact)* | 0.376 | 0.535 | −0.159 | 1e-6 |
| functional full-FT *(artifact)* | 0.216 | 0.467 | −0.251 | <1e-6 |
| prereg full-FT **+ 1% jitter** | 0.657 | 0.541 | **+0.116** | 9.3e-5 |
| functional full-FT **+ 1% jitter** | 0.560 | 0.467 | **+0.093** | 0.030 |

The sign flips exactly when the jitter defeats the exactly-zero-weight artifact. Collateral tells
the same story: under the preregistered attacker only 26.7% (switch) / 16.7% (H=4) of attacked
models are still competent at step 100, versus 68.3% for the corrected baseline — the regime where
routing "wins" is the regime where the attacker ends up with a broken model. Under the functional
attacker those figures are 56.7% / 48.3% / 98.3%.

## Controls

- **Positive (equivalence machinery)**: switch vs no-defense under the functional attacker returns
  REAL DIFFERENCE, d = −0.178, 95% CI [−0.218, −0.137], p = 5e-5. The procedure can see a known effect.
- **Calibration (truth exactly 0)**: random split-halves of each arm (n=30v30) bound at 0.118
  (switch) and 0.104 (H=4). At n=60v60 the achieved bound is 0.065 — i.e. the C1 bound is genuinely
  tighter than what noise alone produces at half the n, but 0.05 was out of reach as preregistered.
- **Replication anchor**: routed selectability 70/90 (0.778, Wilson [0.68, 0.85]); on round 3's own
  seed block 700–729 it reproduces **24/30** vs round 3's 23/30. Localizer matched the oracle 90/90.
  Cross-block weight mass 0.0/60; masked-switch 0-step flip below threshold 60/60 (round 3's
  structural result re-verified). Gates dropped 0 models from all five arms.
- **Vectorised trainer**: bitwise-identical to the sequential round-3 trainer,
  max|diff| = **0.0** on both a fresh H=4 and a fresh H=8 seed (and 0.0 on three seeds pre-stamp).
- **Base rates** drifted within sampling error: misaligned 0.253 (r3 0.333, n=51), aligned 0.230
  (r3 0.235), H=4 aligned 0.279 (r3 0.182, n=66 — r3's Wilson CI [0.107, 0.293] contains 0.279).

## Preregistered drops (budget guard, as written in PREREG §6)

The run crossed the 660 s guard during the attack phase, so **two preregistered drops fired**:
`aligned_chance` under the *prereg* attacker, and the `switch_misaligned` **readout positive
control**. C1–C6 and the functional-attacker positive control were never at risk and all ran.
The readout control is therefore not re-verified at n=60 in this round; round 3 established it
(mean AUC 0.995, flip cost 0/23) and this round's smoke run reproduced it (mean AUC 0.9953, flip
0/6). Treat it as carried over, not re-run.

## Honest limits

1. **The C1 verdict is a bound, not equivalence.** delta* = 0.065 misses the preregistered 0.05
   margin. As stated in advance, n=60 gives only ~36% power for a 0.05 TOST verdict; the bound is
   the deliverable. A routing-specific advantage of up to ~1/3 of the total defense effect is not
   excluded.
2. **C6 does not survive Holm** over the six claims. The sign reversal is supported by the
   uncorrected test (p=0.030) and by its prereg-attacker twin (p=9.3e-5, which was not one of the
   six preregistered claims and is reported in the post-hoc ladder). Treat the reversal as
   established in direction, provisional in significance.
3. **C5's two preregistered summaries disagree.** The prereg rule makes MWU the arbiter, so the
   verdict is a bound; the mean-difference CI excluding zero is reported and not suppressed. Both
   were preregistered for reporting; the bimodality *explanation* is post-hoc.
4. C3/C4 overturn round-3 nulls, but under an attacker that mostly destroys the model. They say
   round 3's *statistics* were wrong, not that the switch is safe.
5. Still a 154-parameter toy, one task, one routing recipe, one correction baseline (f=5%). DRIFT
   and STABILITY were explicitly out of scope; round 3's D1 (benign fine-tuning erases selectability,
   p = 3e-5) is untouched and remains the strongest argument against the switch.
6. The functional attacker is stronger than the preregistered one but still budget-matched at 100
   steps. Nothing here bounds what an attacker with more compute does.

## What should change in the write-up

Replace "a routing-free H=4 model with the identical 62 live parameters **matches** the switch
(0.834 vs 0.840, p=0.40)" with: *at n=60 per arm the two differ under both LoRA attackers in
opposite directions depending on attacker strength; under the realistic functional attacker the
difference is bounded at 0.065 AUC (36% of the total defense effect), and the resistant tail that
drives the switch's apparent advantage is present in the capacity control at the same rate
(11/60 vs 9/60, p=0.81).* And withdraw the flip-cost equivalence claim entirely: that test had no
dynamic range.
