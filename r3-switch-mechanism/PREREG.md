# PREREG — What is the routed worldview switch actually computing?

STAMP: (see `date` output at the bottom of this file; nothing above the "RESULTS" line is edited after that stamp)

Round-3 adversarial follow-up to `routing-worldview-select/` (round 1, oracle localizer) and
`r2-routing-switch-learned/` (round 2, learned localizer). Those two established that routing
cue-localized gradients into disjoint hidden columns makes the learned moral worldview selectable
post-hoc by ablation (22/30 oracle, 23/30 learned, 0/30 standard). This round asks what the switch
IS, mechanistically, and whether the control it gives is a knob or a toggle.

## Setup (inherited verbatim, read-only sources)

Data / model / routing / scoring are imported unchanged from
`/Users/steven/research_sprint_20260724/r2-routing-switch-learned/experiment.py`
(which itself copies round-1 byte-for-byte): 7-dim inputs (true-cue XOR block = dims 0-1,
proxy-cue XOR block = dims 2-3, 3 noise dims), MARGIN 2.0, NOISE 0.6, N=800, 2-hidden-layer
width-8 MLP (154 params), Adam lr 5e-3, wd 2e-3, 2000 full-batch steps.
Routing: view A occludes cue-B columns and is forced through hidden UNITS_A=[0,1,2,3];
view B symmetric through UNITS_B=[4,5,6,7]; l2 cross-blocks zeroed at init.
Arms trained here: `routed` (oracle column sets, = round-1's headline arm) and `control`
(standard unrouted training) — the latter is the negative control for every test below.
Seeds 700-729 (n=30/arm), identical to rounds 1-2. Deploy set = `gen_align(4000, 50001, "deploy")`
(cues anti-correlated), held-out agreeing set = `gen_align(4000, 60001, "train")`, both fixed.
Round-2 showed the learned localizer recovers the oracle column pairs on 30/30 seeds, so the
oracle routing analysed here is the same object as the learned-localizer switch.

REPLICATION ANCHOR (must pass or the whole run is void): per-seed `full_align / selA_align /
selA_train / selB_align / selB_train` of my `routed` and `control` arms must match round-1's
`results.json` to max |diff| < 1e-6.

SWITCHABLE SET S = routed seeds meeting round-1's designated-success rule
(selA_align > 0.8 AND selA_train > 0.9 AND selB_align < 0.2 AND selB_train > 0.9).
Round-1 reports 22/30, so |S| is expected ~22. **Contingency (preregistered):** if |S| < 20,
extend seeds 730, 731, ... in order (training both arms) until |S| = 20.

## Declared pilot (peeking disclosure)

Before writing this prereg I ran ONE routed model (seed 700) to check a structural claim. It
established: (i) after routed training the l2 cross-blocks are **exactly 0.0** (weight decay from
a zero init with identically-zero gradient), so the network's logit difference is
**exactly additive**: Δ_full(x) = Δ_A(x) + Δ_B(x) + Δ_bias, where
Δ_c(x) = h2[UNITS_c] · (l3.w[1]-l3.w[0])[UNITS_c]; measured additivity residual 1.9e-6;
(ii) on 500 deploy rows, mean|Δ_A| = 6.91, mean|Δ_B| = 4.33, opposite-sign rate 0.946.
Consequence for honesty: the SECONDARY-2 thresholds below are **pilot-informed** (one seed) and
SECONDARY-2 is therefore confirmatory-of-a-peek, not a blind test. **The PRIMARY (graded-control
alpha sweep) and SECONDARY-1 (closed-form column rule) have not been run, peeked at, or piloted
in any form.**

## PRIMARY (blind) — is the control GRADED or a TOGGLE?

Gate: `g(alpha, keep=A)` = a hidden-unit mask with 1.0 on UNITS_A and `alpha` on UNITS_B, applied
at BOTH hidden layers exactly where round-1's ablation mask is applied. By construction
alpha = 0 reproduces round-1's selection ablation and alpha = 1 reproduces the un-ablated model
bit-for-bit. Sweep alpha over the 21-point grid {0.00, 0.05, ..., 1.00}; y(alpha) = deploy
alignment on the fixed 4000-row deploy set.

For every seed in S, on the align-selecting direction (keep A, gate B down):
let y1 = y(1), y0 = y(0), range R = y0 - y1. Define alpha_90 and alpha_10 as the alpha at which
the curve, scanned from alpha = 1 downward, first crosses y1 + 0.9R and y1 + 0.1R respectively
(linear interpolation between grid points). **Transition width W = alpha_10 - alpha_90.**

**Pre-stated decision rule (fixed now, not revisable):**
- **GRADED** if median_S(W) >= 0.25
- **TOGGLE / THRESHOLD-LIKE** if median_S(W) <= 0.10
- **INTERMEDIATE** otherwise.
Report median W with a 2000-resample bootstrap 95% CI, plus every per-seed curve. n = |S| >= 20.

Pre-stated secondary readouts on the same sweep (all decided now):
1. **Monotonicity:** per-seed Spearman rho(alpha, y) (expected negative); fraction of seeds whose
   curve is monotone non-increasing in alpha within tolerance 0.02.
2. **Usable dial:** fraction of seeds for which >= 3 of the 21 grid alphas land strictly inside
   (y1 + 0.1R, y1 + 0.9R) — i.e. intermediate worldview behaviour is actually reachable by
   choosing alpha, not merely interpolated between two plateaus.
3. **Negative control:** identical sweep on the `control` (unrouted) models using the same
   arbitrary unit split. Pre-stated test: Mann-Whitney U on R (routed S vs control, n=30),
   expectation R_control ~ 0.
4. **Mis-selecting direction** (keep B, gate A down; alignment expected to fall toward 0): same
   W computed on |R|, reported alongside.
5. **Secondary gate variant (contribution gate):** scale only column B's additive logit
   contribution, Δ = Δ_A + alpha·Δ_B + Δ_bias (legitimate because of exact additivity). Same W
   rule, reported separately. This isolates "is the *vote* graded" from "is the *unit gate*
   graded"; it is a secondary, not the primary.

## SECONDARY 1 (blind) — does each column implement a closed-form worldview rule alone?

Ground truth: with block bits (b0, b1) encoded as ±MARGIN, the block's XOR label is 1 iff the two
signs differ, i.e. the closed-form rule is the **sign-product detector** c(x) = 1[x_i · x_j < 0]
on that column's own cue pair. For each seed in S and each column c in {A (own dims 0-1, TRUE
cue), B (own dims 2-3, PROXY cue)}:

- **(a) Weight-derived truth table (no fitting).** Evaluate the column-only network (other column
  gated to 0) at the four noiseless corners x[own pair] = (±M, ±M), every other input = 0. The
  column "implements XOR in closed form" iff argmax logit = 1 exactly at the two corners with
  x_i·x_j < 0 and = 0 at the other two. Report the fraction of columns matching; pre-stated
  threshold for the claim "each column implements its own worldview rule in isolation":
  **>= 90% of columns in S match**.
- **(b) Closed-form agreement on data.** Agreement between the column-only prediction and c(x)
  over the 4000-row deploy set. Pre-stated threshold: **median agreement >= 0.95**.
- **(c) Other-cue invariance.** (weights) share of the column's |W1| row-mass on its own cue dims
  vs the other cue's dims; (behaviour) prediction flip rate when the OTHER cue block is resampled
  (fresh random XOR block + fresh noise on those 2 dims) with own block and noise dims held fixed,
  over 200 base rows x 200 resamples. Pre-stated threshold: **median flip rate <= 0.05**.
- **Negative control:** the same (a)-(c) on `control` models' arbitrary halves (units 0-3 / 4-7),
  expected to fail all three.

## SECONDARY 2 (pilot-informed, declared above) — what is the UN-ablated model doing?

- Verify exact additivity on all seeds: max |Δ_full - (Δ_A + Δ_B + Δ_bias)| over the deploy set,
  and max |l2 cross-block weight|. (Expected ~1e-6 and exactly 0.)
- On the (all-disagreement) deploy set: **opposite-sign rate** = fraction of rows with
  sign(Δ_A) != sign(Δ_B); **balance ratio B** = median row-wise
  min(|Δ_A|,|Δ_B|) / max(|Δ_A|,|Δ_B|).
  Pre-stated rule: **"genuine two-column mixture / vote-cancellation"** if median_S(B) >= 0.5 AND
  median_S(opposite-sign) >= 0.9; **"dominant column, suppressed column"** if median_S(B) <= 0.2;
  else intermediate.
- **Determinism index D** (protocol copied from `misgen-hedger-mode/run.py`): 8 cue-disagreement
  sign patterns indexed by (v, c_t, c_p), true block (c_t, c_t^v), proxy block (c_p, c_p^(1-v)),
  500 fresh noise draws per pattern (noise tensor shared across models); p_k = P(prediction = v);
  D = mean_k |2 p_k - 1|. That paper's pre-stated rule: D <= 0.5 = "uncommitted / noise-driven",
  D >= 0.8 = "deterministic patchwork". Pre-stated here: routed un-ablated models fall in the
  **uncommitted regime (median D <= 0.5)**, i.e. the same characterization it found for hedgers
  (median 0.315) and not the committed one (0.868). Comparison to those numbers is descriptive
  (different training procedure, same architecture/data/seeds).
- Also report mean |logit margin| on the disagreement set, routed vs control, as in that paper.

## Statistics

All n reported. Proportions: Wilson 95%. Medians/means: 2000-resample bootstrap 95% CI.
Group comparisons: Mann-Whitney U (two-sided) or Fisher exact, stated per test. No p-hacking
detours: the three decision rules above are the only ones that determine the verdicts.

## Cost / scope

Single process, `torch.set_num_threads(2)`, CPU only, no downloads. 60 model trainings at ~1.5 s
= ~90 s plus analysis; budget < 15 min. Everything is written under
`/Users/steven/research_sprint_20260724/r3-switch-mechanism/`; all round-1/2 dirs are read-only.

Anything not specified above and reported later is labeled **post-hoc / exploratory**.

---
STAMP (`date` at time of writing, before any confirmatory run):
Sat Jul 25 10:11:23 EDT 2026
