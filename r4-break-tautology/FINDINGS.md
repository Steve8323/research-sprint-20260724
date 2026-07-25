# Selectability without structurally disjoint columns — round 4

**Slug:** `r4-break-tautology` · **Status:** completed · **Date:** 2026-07-25
**PREREG.md** stamped `Sat Jul 25 11:10:15 EDT 2026`, before `experiment.py` was executed.
Confirmatory runtime 1285.7 s wall, single process, 2 torch threads.

---

## Headline

**The structural-disjointness objection is answered; the effect size is not what the paper thought it was.**

Five of the six preregistered tautology-breaking variants leave post-hoc worldview selectability
*at the disjoint ceiling*: sharing the entire first hidden layer (`shared_trunk`, 130/434
parameters receiving gradient from both worldview views) gives **60/60**; training under a leaky
mask and evaluating under a hard one, with 396 of 434 parameters shared (`soft50`), gives
**59/60**; 25% and 50% overlapping unit groups give **60/60** and **58/60**. All five are
statistically **equivalent to the disjoint ceiling** at the prestated margin δ = 0.20 (Newcombe
95% CIs on ceiling − variant all upper-bounded below 0.114). Only 75% overlap breaks it
(**2/60**). So selectability is not an artifact of zero shared weights and it is not an artifact
of train-mask == eval-mask.

**But the same run destroys the denominator.** Every prior round measured routing against a
control scored under an *arbitrary fixed* unit partition — which for a standard-trained network
is a coin toss over 12,870 possible splits. Score the identical standard-trained models under a
mask chosen from data instead, and the floor moves from **1/60 to 51/60 (85%, Wilson
[0.739, 0.919])**. Routing's honest contribution shrinks from ~+98 points to **+15 points**
(60/60 vs 51/60, Fisher p = 0.0028, Newcombe 95% CI on the difference [0.058, 0.261]).

One sentence for the paper: *given the cue oracle, ordinary training already produces two
individually-competent, oppositely-committed subnetworks that ablation can select between;
routing does not create the switch, it makes the switch reliable (85% → 100%) and makes its
labelling free.*

---

## 1. Anchor (A0) — the reimplementation is the round-1 experiment

| check | result |
|---|---|
| `anchor_disjoint_w4` vs round-1 `routing-worldview-select/results.json`, seeds 700–729 | **max abs diff = 0.000000000** on all four selection metrics |
| per-seed success booleans identical | **True** |
| successes | 22/30 both |

Anchored. Everything below is the round-1 setup with one manipulation at a time.

## 2. Primary results (n = 60 seeds/arm, seeds 700–759; floor and ceiling likewise n = 60)

Bonferroni α = 0.05/6 = 0.008333. Verdicts are exactly the prestated rule.

| arm | H | col width w | shared units | **shared params (grad trace)** | k/n | rate [Wilson 95%] | p vs floor | **verdict** |
|---|---|---|---|---|---|---|---|---|
| `anchor_disjoint_w4` | 8 | 4 | 0 | 2/154 | 22/30 | 0.733 [0.556, 0.858] | — | anchor |
| `ceiling_disjoint_w8` | 16 | 8 | 0 | **2**/434 | 60/60 | 1.000 [0.940, 1.0] | — | ceiling |
| `overlap25` | 14 | 8 | 2 | 28/352 | 60/60 | 1.000 [0.940, 1.0] | 1.3e-33 | **SURVIVES** |
| `overlap50` | 12 | 8 | 4 | 53/278 | 58/60 | 0.967 [0.886, 0.991] | 2.2e-30 | **SURVIVES** |
| `overlap75` | 10 | 8 | 6 | 93/212 | 2/60 | 0.033 [0.009, 0.114] | 1.0 | **BROKEN** |
| `shared_trunk` | 16 | 8 | 0 (whole L1 shared) | 130/434 | 60/60 | 1.000 [0.940, 1.0] | 1.3e-33 | **SURVIVES** |
| `soft25` (train≠eval) | 16 | 8 | 0 | 328/434 | 60/60 | 1.000 [0.940, 1.0] | 1.3e-33 | **SURVIVES** |
| `soft50` (train≠eval) | 16 | 8 | 0 | 396/434 | 59/60 | 0.983 [0.911, 0.997] | 7.5e-32 | **SURVIVES** |
| `floor_control` | 16 | 8 | — | 433/434 | 1/60 | 0.017 [0.003, 0.089] | — | floor |

**S3 (did the manipulation actually break disjointness?) passes for every variant.** The
gradient trace at initialization reproduces round 3's finding on the ceiling — exactly **2**
shared parameters, the output bias — and every variant is strictly above it, up to 396/434.
No variant was voided. (Note: the soft arms' counts are a *lower bound*; 106 of `soft25`'s
tensors have zero gradient at init only because those ReLU units are dead at init, not because
anything blocks them. All 112 first-layer weights are shared in both soft arms.)

**S1 (distance from the ceiling).** Equivalence, not a bare null:

| variant | k/n | p vs ceiling | Newcombe 95% CI (ceiling − variant) | equivalent at δ=0.20? |
|---|---|---|---|---|
| `overlap25` | 60/60 | 1.0 | [−0.060, 0.060] | **yes** |
| `overlap50` | 58/60 | 0.50 | [−0.032, 0.114] | **yes** |
| `shared_trunk` | 60/60 | 1.0 | [−0.060, 0.060] | **yes** |
| `soft25` | 60/60 | 1.0 | [−0.060, 0.060] | **yes** |
| `soft50` | 59/60 | 1.0 | [−0.045, 0.089] | **yes** |
| `overlap75` | 2/60 | 3.9e-32 | [0.866, 0.991] | no |

**S2 (overlap dose-response).** Cochran–Armitage across 0/25/50/75% overlap:
successes 60, 60, 58, 2 of 60; **z = −11.73, one-sided p < 1e-30**. Monotone decreasing, as
predicted.

**The `overlap75` failure is not a capacity failure.** The prestated capacity diagnostic: of its
58 misses, **0 are competence misses** — median held-out accuracy under both selection masks is
**1.000**. The failure is 57 seeds with keep-A deploy alignment ≤ 0.8 and 58 with keep-B ≥ 0.2:
both selections land in the middle. Contrast `floor_control`, whose 59 misses are 46 competence
misses, and `anchor_disjoint_w4`, whose 8 misses are 8/8 competence misses (round 3's
width-4 trainability diagnosis, reconfirmed).

## 3. The result that changes the paper: the floor was an artifact of the mask

S4 was preregistered as "train mask != eval mask": score every arm again under an ablation mask
*derived from data* (rank units by cue-A minus cue-B activation sensitivity, keep the top w),
never read off the training partition. The intended target was the routed arms. The
consequential number turned out to be the **control**.

| | fixed arbitrary mask | data-chosen mask |
|---|---|---|
| `floor_control` (standard training) | **1/60** | **51/60** (0.85, Wilson [0.739, 0.919]) |
| `ceiling_disjoint_w8` | 60/60 | 60/60 (mask recovered exactly 60/60, Jaccard 1.000) |
| `shared_trunk` | 60/60 | 60/60 |
| `soft25` / `soft50` | 60/60 / 59/60 | 60/60 / 59/60 |
| `overlap25` / `overlap50` / `overlap75` | 60/60 / 58/60 / 2/60 | 50/60 / 27/60 / 0/60 |

Continuous version (post-hoc): the control's mean selection margin (keep-A deploy alignment
minus keep-B) is **−0.018 [−0.097, +0.061]** under the fixed mask — no direction at all — and
**+0.910** under the data-chosen mask.

**Verification of this number** (`verify_floor_learned.py`, 10 control seeds retrained from
scratch, independent re-derivation):

- reproduces `results.json` exactly, max abs diff **0.0**;
- 8/10 succeed, and **all 8 under the designated mapping** — none rescued by the control's
  generous either-mapping scoring;
- every learned mask is a genuine full 8/8 split of the 16 units (not degenerate);
- **negative control 1:** random 8/8 splits succeed **1/50** (vs 8/10, Fisher p = 1.5e-7);
- **negative control 2:** masks built from sensitivity to the *noise* dims rather than the cues
  succeed **0/10** (Fisher p = 7.1e-4).

So the standard-trained network really does contain two unit-disjoint subnetworks, each
individually competent on the training distribution (>0.9), that commit to opposite worldviews —
and the procedure that finds them is measuring something specific, not fitting noise.

**Effect size, restated with both denominators:**

| comparison | difference | Fisher p | Newcombe 95% CI |
|---|---|---|---|
| ceiling 60/60 vs round-1-style floor 1/60 | +0.983 | 1.3e-33 | [0.890, 0.997] |
| ceiling 60/60 vs **fair** floor 51/60 | **+0.150** | **0.0028** | **[0.058, 0.261]** |
| `shared_trunk` 60/60 vs fair floor | +0.150 | 0.0028 | [0.058, 0.261] |
| `soft50` 59/60 vs fair floor | +0.133 | 0.017 | [0.034, 0.245] |

## 4. Post-hoc (labeled; in `posthoc.json` and `extra_stats.txt`, never in `results.json`)

- **The binary criterion turns graded degradation into a cliff.** Mean selection margin across
  the overlap dose: 0.996 → 0.977 → 0.804 → 0.387 (bootstrap CIs [0.996,0.997], [0.969,0.983],
  [0.781,0.827], [0.359,0.417]). The 60/60 → 2/60 collapse at 75% is the 0.8/0.2 thresholds
  crossing a smoothly falling curve, not a phase transition in the mechanism.
- **Why 75% overlap fails is plausibly *private* capacity, not sharing per se.** At w=8 and 75%
  overlap each worldview has only **2 exclusive units**, which is the bare minimum for a 2-bit
  XOR; the shared units carry competence (both masks reach 1.000 held-out) but not
  differentiation. This is an interpretation, not a test — the prestated capacity diagnostic
  (total competence) says "not capacity", and distinguishing them needs a w-sweep at fixed
  overlap, which was not run.
- **Shared units are less cue-committed than exclusive ones.** Mean |sensitivity score| of shared
  vs exclusive units: 0.61× (overlap25), 0.34× (overlap50), 0.38× (overlap75). Shared units sit
  between the two worldviews rather than picking one.

## 5. What this does to the line

**Restores:** the k-way / basis-free / mechanism results are no longer vulnerable to "your
columns are disjoint subnets, so of course you can select them". Selectability tolerates a fully
shared first layer, a train mask that differs from the eval mask, and up to 91% of parameters
receiving gradient from both worldviews.

**Narrows, hard:** the number that has anchored the whole line — "routed 22/30 vs control 0/30" —
is 100 points wide only because the control was denied a mask. The defensible claim is
**+15 points over a fairly-scored control**, and the fair control's 85% is itself a new,
paper-worthy finding about standard training: underdetermination leaves a *usable* dual structure
in the weights, not a mush. Round 3 said "report it as a demonstration, never as an effect size";
this run says the effect size exists but is an order of magnitude smaller than advertised.

**Reinforces round 3's bottleneck.** Both the fair floor and every learned-mask number here still
consume the input-space cue oracle. This experiment moves *nothing* on localization; it just
shows that once you have the oracle, routing is close to unnecessary.

## 6. Caveats

- Toy scale throughout: 7-dim synthetic inputs, ≤1.2k-parameter MLPs, searched-for equal-cost XOR
  cues. One substrate (MLP); the round-3 transformer was not re-run.
- The learned mask uses the cue-column oracle. It is oracle-free only with respect to the
  *training unit partition*, which is what "train mask != eval mask" required.
- The S4 rates across arms are confounded by H: the overlap arms have H = 14/12/10, so their
  top-w and bottom-w learned masks necessarily overlap, while the H=16 arms get a clean 8/8 split.
  The `overlap50` learned-mask number (27/60) should not be read as a routing failure.
- `floor_control` is the only non-routed control; round 1's `occlusion_only` and
  `routing_no_occlusion` arms were not re-run.
- No per-variant matched non-routed control was run (each variant is compared to one H=16
  standard-training floor).
- Confirmatory wall-clock 1285.7 s on a machine running five other agents; at 2 threads that is
  ~43 core-minutes, over the sprint's ~15-minute guidance. Reported rather than hidden.
- The soft-arm gradient-trace counts are lower bounds (dead ReLUs at init), see §2.

## 7. Files

- `PREREG.md` — stamped 11:10:15, before any confirmatory run
- `core.py` — data/model/arm specs; `experiment.py` — confirmatory run + all prestated tests
- `results.json` — per-seed rows, primaries, S1/S2/S3/S4, anchor
- `verify_floor_learned.py` / `.json` — independent verification + two negative controls
- `posthoc.py` / `posthoc.json`, `extra_stats.txt`, `make_tables.py` — post-hoc, segregated
- `timing_pilot.py` (runtime only), `smoke.py` (crash-only), `run.log`
