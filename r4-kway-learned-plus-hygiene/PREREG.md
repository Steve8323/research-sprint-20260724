# PREREG — k-way selection with a DATA-ONLY localizer, plus portfolio hygiene (round 4)

Stamped with `date` at time of writing, **before any code in this directory was written or run**:

```
Sat Jul 25 11:09:33 EDT 2026
```

All writes go only to `/Users/steven/research_sprint_20260724/r4-kway-learned-plus-hygiene/`.
Prior-round directories and the `~/` repos are READ-ONLY inputs.

---

## PART 1 — Join the k-way result to the round-2 data-only localizer (CONFIRMATORY)

### Background and the exact gap being closed

- `r3-multiway-switch/` showed k-way post-hoc worldview selection at k=3 and k=4 (routed 20/20 vs
  control 0/20, Fisher p = 1.45e-11) — but it used the **round-1 ORACLE** cue localization
  (`cue_cols(k) = [[2i, 2i+1]]`). So "k-way" and "localizer-free" cannot appear in the same sentence.
- `r2-routing-switch-learned/` replaced the oracle at k=2 with a **data-only pair probe (L1)**:
  for every pair of input columns (i,j) fit a logistic probe on `[x_i, x_j, x_i*x_j]` against the
  label, greedily take the top-2 *disjoint* pairs. 30/30 exact recovery, selectability 23/30.
- `r3-entangled-localization/` **RETRACTED** the framing of that result: L1 works only because it is
  handed a coordinate system in which the two concepts already occupy disjoint input columns. That
  coordinate privilege *is* oracle information. Basis-agnostic localizers coin-flip.

**Therefore this experiment does NOT claim to restore oracle-free localization.** The claim under
test is narrower and stated in advance: *given the same coordinate privilege round 2 had (concepts
occupy disjoint, axis-aligned input-column groups), does the data-only probe still find the right
groups when there are k>2 of them, and does the end-to-end pipeline
`probe -> route -> post-hoc select` still deliver k-way selection?* Any claim beyond that (e.g.
"oracle-free k-way selection") is out of scope and will be refused in FINDINGS.md.

### Hypotheses

- **H1 (localization scales in k).** The L1 pair probe recovers **exactly** the k true cue pairs at
  k=3 and k=4, well above the combinatorial chance rate for choosing k disjoint pairs at random.
  Risk that makes this non-trivial: the candidate-pair set grows as C(2k+3, 2) (36 at k=3, 55 at
  k=4 vs 21 at k=2), so more chances for a spurious pair to beat a true one; and greedy disjoint
  selection can be poisoned by one bad early pick.
- **H2 (end-to-end k-way selectability with a learned localizer).** Routing into k columns using the
  **discovered** groups yields clean k-way post-hoc selection at rates far above the standard-training
  floor.

### Setup — everything not listed here is copied VERBATIM from `r3-multiway-switch/experiment.py`

(which is itself verbatim from `routing-worldview-select/experiment.py` and
`~/moral_framework_misgen/alignment.py`). The functions are copied into this directory's
`experiment.py` rather than imported, so that no bytecode is written into a read-only directory.

- Input = k XOR cue blocks of 2 dims + 3 noise dims (`INPUT = 2k+3`). Label = v; **every** cue
  encodes v in training (the underdetermined, equal-cost regime).
- MLP, 2 hidden layers of `H = k*w`, `MARGIN=2.0, NOISE=0.6, N=800, LR=5e-3, WD=2e-3, EPOCHS=2000`,
  full-batch Adam, CrossEntropy. Hidden columns `cols[i] = [i*w .. i*w+w-1]`; all off-diagonal
  hidden->hidden blocks zeroed at init (they also receive exactly zero gradient).
- Routing loss per step = mean over i of CE(model(view_i, keep-column-i mask), y), where `view_i` =
  training batch with **all dims in the other discovered groups** replaced by fresh N(0, 0.6) noise.
  (With the oracle groups this is byte-identical to `r3-multiway-switch`.)
- Fixed at **w = 8** (r3's clean-capacity setting: oracle 20/20 at both k). w=4 is NOT run.
- **n = 20 seeds per k**, seeds 700-719 (the prior rounds' seeds), k in {3, 4}.

### The localizer (data-only, no cue knowledge), verbatim from `r2-routing-switch-learned/localize_L1`

For every pair (i,j) of the `2k+3` input columns, fit a 3-parameter logistic probe on
`[x_i, x_j, x_i*x_j]` to the training labels (300 full-batch GD steps, lr 0.5, zero init); score =
train accuracy. Sort pairs by descending accuracy (ties broken by `itertools.combinations` order),
greedily accept a pair iff both its columns are unused, stop at k pairs. Uses the training inputs
and labels only; no knowledge of which columns are cues, of k's block structure, or of the noise dims.
The probe is fit on the same n=800 training set the model is trained on (as in round 2).

### Arms (per k, all at w = 8, all n = 20 seeds)

| arm | column groups used for routing | scoring |
|---|---|---|
| `learned` | the k groups returned by L1 on that seed's training data | localizer-induced mapping (below) |
| `control` | none (standard training, no routing) | generous: best over all k! permutations |
| `broken` (SECONDARY) | a pre-specified **cross-cue** partition, group i = {2i, (2i+3) mod 2k} | generous: best over all k! permutations |

`broken` is disjoint and contains no true cue pair at either k (k=3: {0,3},{2,5},{4,1}; k=4:
{0,3},{2,5},{4,7},{6,1}). It is the *informative* floor demanded by r3's audit ("the honest floor
for a routed arm is routing-with-a-broken-split, not standard training").

### Evaluation sets (fixed, n = 4000 each, identical generation to r3-multiway-switch)

`HELD` (agreeing, seed 60001); `PROBE_j` for j=0..k-1 (cue j agrees, all other cues flipped, seed
50001+j). Same as r3, so r3's stored oracle numbers are directly comparable.

### Per-seed k-way clean-selection success (thresholds inherited: 0.8 / 0.2 / 0.9)

Let `g(s)` = the index of the true cue whose input dims are exactly the discovered group placed in
slot s. A run counts as a success iff **for every** column s in {0..k-1}:
1. `acc(HELD | keep s) > 0.9`,
2. `acc(PROBE_{g(s)} | keep s) > 0.8`, and
3. `acc(PROBE_j | keep s) < 0.2` for all j != g(s).

If any discovered group is not exactly one true cue pair, `g` is undefined and the seed is scored a
**failure** (conservative). Ground truth is used only to *evaluate*, never to route.
`control` and `broken` have no induced mapping and are scored generously (best over all k!
permutations), exactly as r3 scored its control arm.

### PRIMARY tests and decision criterion

- **P1 (k=3, w=8):** two-sided Fisher exact, `learned` vs `control`, 20 vs 20.
- **P2 (k=4, w=8):** two-sided Fisher exact, `learned` vs `control`, 20 vs 20.
- Bonferroni over the 2 primaries: each judged at **alpha = 0.025**.
- **DECISION:** H2 is SUPPORTED at a given k iff (Fisher p < 0.025) AND (`learned` successes
  >= 14/20, i.e. >= 70%, r3's pre-stated bar). Otherwise REJECTED at that k and reported as a
  negative result. No post-hoc rescue folded into the primary.
- Wilson 95% CI on every rate.

### PRIMARY (localization family), Bonferroni over 2, alpha = 0.025 each

- **L3 (k=3), L4 (k=4):** exact one-sided binomial test of the number of seeds with **exact**
  recovery of the k true pairs, against the combinatorial chance rate
  `p0 = 1 / M`, `M = D! / ((D-2k)! * 2^k * k!)`, `D = 2k+3`
  (M = 1260 at k=3 so p0 = 7.94e-4; M = 17325 at k=4 so p0 = 5.77e-5).
  DECISION: H1 SUPPORTED at that k iff p < 0.025 AND recovery >= 18/20.

### SECONDARY (labeled secondary; no claim rests on them alone)

- **S1 probe margin.** Per seed: `min(accuracy of the k accepted true pairs) - max(accuracy over all
  pairs that are not a true cue pair)`. Pre-stated bar for "clean separation": margin > 0.2 on every
  seed. Report median, min, and how it changes from k=3 to k=4.
- **S2 partial recovery.** Mean number of true cue pairs among the k accepted groups (0..k), for
  seeds that fail exact recovery.
- **S3 broken-split floor.** `broken` successes/20 with Wilson CI, and Fisher `learned` vs `broken`
  (two-sided, reported at alpha = 0.05, secondary). Pre-stated interpretation: if `broken` is near
  `learned`, then routing works with *any* disjoint partition and the localizer is not load-bearing,
  which would deflate H2 to near-tautology; if `broken` is near 0, the localizer is load-bearing.
- **S4 permuted-label localizer null (empirical).** Run L1 on the same training inputs with the label
  vector randomly permuted (seed 900+seed), 20 seeds per k, no training. Report exact-recovery rate
  (expected 0/20) and the top accepted pair accuracy (expected ~0.5).
- **S5 oracle anchor / identity check.** For every seed where the discovered groups equal the oracle
  groups **in the same slot order**, the routed model is trained on numerically identical arguments
  to `r3-multiway-switch`'s oracle run, so its per-seed metrics must reproduce
  `r3-multiway-switch/results.json` cells `k3_w8_routed` / `k4_w8_routed`. Report max |diff| over
  `held[]` and `probe[][]`; pre-stated tolerance **1e-3** (r3's own rerun determinism was 5e-5).
  If the discovered order permutes the oracle order, that seed is excluded from the anchor and the
  permutation is reported.

### Power discipline (round-3's stated failure mode)

- No "no difference" statement will be made from a non-significant Fisher. If `learned` vs `broken`
  or `learned` vs the r3 oracle cell is non-significant, I will report a two-sided 95% Newcombe
  interval on the rate difference and state the equivalence margin it actually excludes, not "no
  difference".
- With n = 20/arm, Fisher has 80% power at alpha = 0.025 only for rate gaps >= ~0.45 (Wilson-based
  approximation, stated here in advance). Any smaller gap is uninformative at this n and will be
  reported as such.
- The `learned` vs oracle comparison is expected to be an **identity, not a statistical test**: if
  localization is exact and order-preserving, the two arms train the same model from the same seed.
  This will be reported as an identity check (S5), and explicitly NOT as evidence of equivalence.

### Pre-stated deflations that will appear in FINDINGS.md regardless of outcome

1. The k routed columns are structurally disjoint subnets sharing only the 2-element output bias,
   and each group is trained under exactly the ablation mask it is later evaluated with — so a high
   selectability rate is close to guaranteed by construction. It is a demonstration, not an effect size
   (r3's audit; carried forward verbatim).
2. The coordinate privilege above: this does not restore "oracle-free localization".
3. Toy scale: <=11-dim synthetic inputs, <=1.2k-param MLPs, searched-for equal-cost XOR cues.

### Compute budget and pre-stated downscope order

Single process, torch 2 threads, CPU only, no downloads, target <= 15 min wall clock for everything
in this directory. A timing pilot (1 seed of k=4/w=8 `learned` and 1 seed of k=4/w=8 `control`,
**timing only — no scientific quantity read off it**) gates the run. If the projection exceeds
12 min, downscope in this order and say so explicitly in FINDINGS.md:
(1) drop the `broken` arm at k=4; (2) drop the `broken` arm at k=3;
(3) reduce EPOCHS 2000 -> 1200 for all cells, which forfeits the S5 oracle anchor and would be
stated as such. n stays at 20 seeds per k in all cases.

---

## PART 2 — Portfolio hygiene (ANALYSIS-ONLY on saved artifacts, no new training)

Fixed now, before any recomputation. Deliverable: one markdown table file `SELECTION_AUDIT.md` in
this directory. Nothing in prior directories is modified.

1. **Two-sided constant.** `r3-selection-inflation-curve/run.py` computes the selection-inflation
   ratio for its "max adjacent gap" statistic as `R_gap = |gm[jg]| / (a_kg * gap_SE)` with
   `jg = argmax |gm|` (a **two-sided** selection over `k_gaps = 4` gaps) but `a_kg = blom_a(4) = 1.029`,
   the constant for a one-sided max of k normals. The correct constant for a max of |N| is
   `b_k = E[max_j |N_j|]`, `b_4 ~= 1.47`. I will (a) compute `b_k` by exact numerical integration
   plus MC for k = 2..25 and check the two agree, (b) recompute all four `R_gap` values
   (2 p_waste x 2 metrics) with `b_4`, (c) restate verdicts under the pre-stated scale
   (SUSPECT < 1.5 <= BORDERLINE < 3 <= ROBUST), and (d) state explicitly which Part-C R values are
   NOT affected because their selection direction was pre-specified one-sided (`argmin` for recovery
   AUC, `argmax` for retain, `argmax` for the dip statistic).
2. **Connectivity modularity restated as family-dependent.** Recompute the max-over-theta modularity
   permutation p under three nested theta families from the saved 60x60 barrier matrix
   (`~/moral_framework_misgen/stats/results.json`, read-only): (i) all 1770 thetas, (ii) the published
   window theta in [0.0015, 0.090], (iii) thetas with at least 60 edges. 1000 permutations, seed 0,
   same incremental Q computation r3 used (sanity-checked against
   `misgen-connectivity-graph/results.json` curves). Report all three p-values in one row and state
   that the threshold-free primary (AUC = 0.725) is IMMUNE.
3. **routing-hybrid-knob T3 demotion.** From `routing-hybrid-knob/results.json`, recompute at
   p_waste = 0.5 the retain (quality) axis reliability `rho_hat = max(0, 1 - s^2/S^2)` and R, and
   report the paired per-seed evidence for the specific interior-vs-endpoint contrasts the T3
   criterion requires. Pre-stated demotion rule: if `rho_hat = 0` on the quality axis, T3's
   "interior alphas are Pareto-useful" is **unsupported**, because the criterion requires an interior
   alpha to be *strictly better than an endpoint on the quality axis* and that axis carries no
   reliable between-cell variance. Also report the equivalence bound (paired 95% CI) rather than
   claiming "no difference".
4. **Two-line selection audit for every headline number in REPORT.md / REPORT2.md / REPORT3.md.**
   Columns: `k` (number of cells the number was selected from; k=1 = pre-specified), `n`, `s`
   (per-cell SE), `S` (SD across cells), `rho_hat`, `R`, verdict. Where s/S are not recoverable from
   saved artifacts I will write `n/a (not recoverable)` rather than inventing them. Every k=1 row is
   marked IMMUNE with the reason. Rows are taken from the reports' own headline/key-number columns.

## What counts as success

Negative results are fine and will be reported plainly. Specifically: if L1 fails to recover the
groups at k=4, that is a real limit of the round-2 localizer and will be the headline. If the
`broken` arm scores near `learned`, the k-way selectability number is near-tautological and I will
say so. Anything not listed above is POST-HOC and will be labeled POST-HOC in FINDINGS.md.
