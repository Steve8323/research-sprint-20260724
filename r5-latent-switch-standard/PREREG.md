# PREREG — Is the worldview switch LATENT in standard-trained models? (round 5)

Stamped, before any confirmatory code was run:

```
Sat Jul 25 13:58:30 EDT 2026
```

## 0. Full disclosure of what has already been run at stamp time

1. `core_mlp.py` and `core_tf.py` (harnesses only — no experiment executed through them).
2. `timing_pilot.py`, which prints **wall-clock seconds only**. Its complete output:
   `TF train/seed 8.05 s · TF sensitivity 0.60 s · TF 12 acc calls 0.60 s · MLP H=8 0.44 s ·
   H=16 0.45 s · H=32 0.50 s · H=64 0.55 s · MLP k=3 H=24 0.50 s`. No accuracy, mask, success
   flag or weight statistic was computed or printed. Used only to fix `n` and the arm list.
3. A power simulation on synthetic Bernoulli draws (§6) — no model, no data.
4. **Prior published numbers I have read and that are therefore not blind:** round-4
   `r4-break-tautology` (fixed-mask floor 1/60, data-chosen-mask floor 51/60 at H=16, random-split
   negative control 1/50, noise-sensitivity negative control 0/10) and round-3 `r3-lm-organism`
   (transformer **control** arm under the arbitrary fixed head/unit partition: **0/12** successes;
   routed arm 12/12; control `deploy_full` spread 0.00–0.93, n_hi=2 n_lo=2 n_mid=8). Round 3 never
   ran a data-chosen mask on the transformer, so every primary outcome below is unseen.

---

## 1. The claim under test

Round 4 found that scoring **standard-trained** (unrouted, un-occluded) MLPs under a unit mask
*chosen from data* — cue-A-minus-cue-B activation sensitivity measured on held-out **agreeing**
data, never on the deploy set — moves the "selectability floor" from 1/60 to **51/60**. Read
literally: ordinary training under cue underdetermination does not produce an entangled
compromise; it leaves a **latent switch** — two unit-disjoint halves, each individually competent
on the training distribution, committed to *opposite* worldviews — that can be set post-hoc with
no training intervention at all.

That was one architecture (2-hidden-layer MLP, 7 input dims), one width (H = 16), k = 2, and one
data condition (fully agreeing). This round asks whether the latent switch is a general property
of underdetermined training or an artifact of that cell.

## 2. Hypotheses

- **H1 (substrate generality).** The latent switch exists in the round-3 **transformer**
  organism, where the two candidate groups share a residual stream, embeddings, LayerNorms and
  the readout head, and where the cue is an attention circuit rather than an input coordinate.
- **H2 (multi-way).** With k = 3 mutually-agreeing cues, a data-chosen 3-way partition of a
  standard-trained network selects among 3 worldviews.
- **H3 (width).** Latent selectability increases with width / overparameterization.
- **H4 (specificity — the negative control).** When the training data **disagree** (a fraction f
  of examples where the proxy cue is wrong, so the worldview is *determined*), the identical
  mask-finding procedure finds nothing to select.

**Every model trained in this round is STANDARD-trained.** No routing, no occlusion in any loss,
no mask at training time, anywhere. The manipulation is entirely at scoring time.

## 3. Substrates, arms and n

### TF-A — PRIMARY. Transformer, agreeing data.
Round-3 organism verbatim (68,546 params; d=64, 2 layers, 4 heads, d_ff=128; word-order
two-rule language, T=13, vocab=8; AdamW max_lr 8e-3 OneCycle, wd 0.01, batch 512 from a 4096
pool, 700 steps, MPS). Cue A = field F0 (true objective), cue B = field F1 (proxy, flips at
deploy). Seeds **700–729 (n = 30)**.

### TF-N — negative control. Transformer, DISAGREEING data, f = 0.25.
Identical except that on 25% of training examples the proxy cue's bit is flipped, so cue A is the
uniquely loss-minimising worldview. Seeds **700–715 (n = 16)**.

### MLP-W — width sweep. Round-1/round-4 ALIGN toy, standard training.
H ∈ **{8, 16, 32, 64}**, k = 2, f = 0, seeds **700–729 (n = 30)** per width. Everything else
verbatim round 4 (INPUT=7, MARGIN=2.0, NOISE=0.6, N=800, LR=5e-3, WD=2e-3, EPOCHS=2000, full-batch
Adam). The H = 16 cell is an **anchor** (§7).

### MLP-N — negative control. ALIGN toy, DISAGREEING data.
f ∈ {0.05, 0.25} × H ∈ {16, 64}, seeds 700–729 (n = 30) per cell.

### MLP-K3 — multi-way. 3 mutually-agreeing cues (round-3 multiway toy), standard training.
H ∈ {24, 48}, seeds 700–729 (n = 30) per cell.

Priority order is TF-A, MLP-W, TF-N, MLP-N, MLP-K3. **Downscope rules, fixed now:** TF-A's seed
loop stops early if elapsed > 420 s (analysis then uses the seeds that completed; if fewer than 20
completed the primary is reported as NOT RUN, never as a null). Any arm whose start time is past
780 s of total elapsed is skipped and the skip is stated in FINDINGS.md. Primaries never depend on
a skipped arm.

## 4. The mask-finding procedure (identical in spirit to round 4's S4, no deploy leakage)

For each cue c, resample that cue's evidence with fresh noise on the **held-out agreeing** set
(MLP: replace cue c's two input columns with N(0, 0.6²); transformer: rewrite cue c's 2-token
chunk with a fresh uniform bit — round-3's `occlude`, sequence length, token multiset and marker
positions unchanged). `sens_c(u)` = mean |change in unit u's pre-mask activation|.

- **MLP (k=2):** `score(u) = sens_A(u) − sens_B(u)`; keep-A = top H/2, keep-B = bottom H/2
  (scored at hidden layer 1, mask applied at both hidden layers — verbatim round 4).
- **Transformer (k=2):** the ablatable units are 4 attention heads and 128 MLP units, masked
  identically in both layers (round-3's group definition). Sensitivity is averaged over batch,
  sequence positions, head dims and **layers**. keep-A = top-2 heads + top-64 units by
  `sens_A − sens_B`; keep-B = the complementary 2 heads + 64 units. Same group *sizes* as round
  3's arbitrary partition; only the membership is data-chosen.
- **MLP (k=3):** each unit is assigned to `argmax_c sens_c(u)` — a genuine partition, sizes
  unconstrained and reported.

**Deploy leakage:** the deploy set is used only to *score* a mask, never to choose one. The
sensitivity statistic sees held-out agreeing data only.
**Oracle used:** which input columns / which field carries which cue. This is the same oracle the
whole line already consumes and is NOT removed here; stated as a limitation up front.

## 5. Success criterion (verbatim rounds 1/3/4, unchanged)

Deploy set n = 4000 seed 50001; held-out agreeing set n = 4000 seed 60001.
```
designated = accA_deploy > 0.8 and accA_held > 0.9 and accB_deploy < 0.2 and accB_held > 0.9
swapped    = accB_deploy > 0.8 and accB_held > 0.9 and accA_deploy < 0.2 and accA_held > 0.9
success    = designated or swapped
```
k = 3 uses round-3's multiway criterion: per-group held-out > 0.9, own-probe > 0.8, every
off-probe < 0.2, under the designated labelling or **any** permutation.

**Applied identically to the data-chosen mask and to every control.** No control is scored under a
stricter rule than the method.

## 6. FAIR CONTROL (the explicit lesson of round 4)

The control is a **random balanced mask given ten times the search freedom of the method**:

- For each trained model, draw **K = 10** random masks of the *same shape and group sizes* as the
  data-chosen one (TF: random 2/2 head split and random 64/64 unit split; MLP: random H/2 split;
  k=3: random partition with the *same group sizes* the data-chosen partition produced).
- A seed counts as a random-control success if **any** of its 10 draws succeeds. The method gets
  exactly **one** mask. The control is therefore strictly advantaged.
- Also reported: the per-draw random rate (out of 10n draws), which is the unadvantaged floor.

**PRIMARY TEST (TF-A).** Exact McNemar (two-sided binomial on discordant pairs), data-chosen vs
random-best-of-10, paired within seed (same trained model).

**PRIMARY DECISION, fixed now:**
- **LATENT SWITCH PRESENT** iff McNemar p < 0.01 **and** data-chosen successes ≥ n/2.
- **PARTIAL** iff p < 0.01 but successes < n/2.
- **ABSENT** otherwise — reported together with the Wilson 95% CI on the data-chosen rate and the
  power table below, never as a bare "no difference".

**Power (simulated, 4000 reps, before the run), for the joint criterion at α = 0.01 with a random
control whose per-seed best-of-10 rate is 0.10:**

| true data-chosen rate | 0.40 | 0.50 | 0.60 | 0.70 | 0.85 | 1.00 |
|---|---|---|---|---|---|---|
| power, n = 30 | 0.154 | 0.524 | 0.872 | 0.982 | 1.000 | 1.000 |
| power, n = 24 | 0.142 | 0.426 | 0.755 | 0.937 | 0.998 | 1.000 |

So a non-significant TF-A is informative against true rates ≥ 0.7 and *not* informative against
rates ≤ 0.5; that limit is stated in advance and will be repeated in FINDINGS.md.

## 7. Secondary tests (Holm-corrected within this family, α = 0.05)

- **S1 (width trend, H3).** Cochran–Armitage trend on MLP-W data-chosen successes across
  H ∈ {8,16,32,64} with scores log2 H, one-sided increasing. Plus per-width McNemar vs that
  width's random-best-of-10 control.
- **S2 (specificity, H4).** Fisher exact, data-chosen success at f = 0.25 vs f = 0, matched
  substrate and H (TF-N vs TF-A; MLP-N vs MLP-W). **Negative control passes** iff p < 0.05 and the
  f = 0.25 cell's Wilson 95% upper bound < 0.5. Simulated Fisher power (n = 24 vs 16, true 0.85 vs
  0.0/0.1/0.2/0.3) = 1.000 / 0.999 / 0.989 / 0.928.
- **S3 (multi-way, H2).** McNemar exact, data-chosen 3-way partition vs size-matched random
  partition best-of-10, at each H ∈ {24, 48}.
- **S4 (distractor-cue negative control).** Round-4's "noise-sensitivity" control, ported: build
  the mask from sensitivity to two **label-independent distractor** fields/columns (TF: fields F2
  vs F3; MLP: noise columns 4 vs 5) instead of the cues, and score it identically. Expectation:
  near zero. If it succeeds at a rate comparable to the cue-derived mask, the procedure is vacuous
  and FINDINGS.md will say the headline is void.
- **S5 (designated-only rate).** All arms rescored under the designated labelling only (the
  data-chosen mask names its own halves, so the either-mapping allowance is a generosity the
  method does not need). Reported, not decisive.

## 8. Anchors (not outcome tests; cannot rescue or discard anything)

- **A0 (MLP).** MLP-W's H = 16, f = 0 cell must reproduce `r4-break-tautology/results.json`'s
  `floor_control` rows for seeds 700–729 — fixed-mask and data-chosen accuracies, max abs
  diff < 1e-6, identical per-seed success booleans.
- **A1 (transformer).** TF-A's **fixed**-partition (heads {0,1}+units[0:64)) accuracies must
  reproduce `r3-lm-organism/results.json`'s `control` rows for seeds 700–711 (max abs diff
  reported; MPS float non-determinism means an exact match is not guaranteed and a tolerance
  of 1e-3 is treated as anchored, with the observed max diff always printed).

If an anchor fails, every number in that substrate is labeled a re-implementation rather than a
replication.

## 9. What would make me wrong / declared limitations

- The mask consumes the **cue oracle**; nothing here is oracle-free localization. Round 4's
  rotation result (`r4-asymmetry-localizer`) is the relevant open problem and is untouched.
- The transformer's masks are shared across layers and coarse (4 heads); a failure could be a
  granularity artifact rather than an absence of latent structure. **Pre-stated diagnostic:** for
  every TF failure, record which of the four criterion clauses failed and whether either half
  missed the competence gate. If failures are dominated by competence misses, the honest reading
  is "the halves are too small", not "no latent switch", and FINDINGS will say so.
- Toy scale throughout: ≤68.5k params, synthetic tasks, searched-for equal-cost cues.
- Wider MLPs have more units and therefore more possible splits; the random control's per-draw
  rate is reported per width so that any width trend can be checked against a shifting floor.

## 10. Analysis is fully specified

`experiment.py` writes `results.json` with every per-seed row and every test above computed
exactly as written. Anything in FINDINGS.md that is not in this file is labeled **POST-HOC**.
