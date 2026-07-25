# Is the worldview switch LATENT in standard-trained models? — round 5

**Slug:** `r5-latent-switch-standard` · **Status:** completed · **Date:** 2026-07-25
**PREREG.md** stamped `Sat Jul 25 13:58:30 EDT 2026`, before `experiment.py` was executed.
Confirmatory runtime **720.7 s**; post-hoc probe **116.8 s**; total **837.5 s (14.0 min)**,
single process, MPS + 2 torch threads, on a machine running other agents.

---

## Headline

**The latent switch is real, but it is a property of *per-worldview unit capacity in the MLP
toy* — not of underdetermined training. It does not exist in the transformer, and it does not
exist at the width round 1 actually used.**

Three numbers carry the round:

1. **Transformer (PRIMARY, preregistered): ABSENT.** In the round-3 sequence organism
   (68,546 params, shared residual stream), a data-chosen head+MLP-unit mask selects the
   worldview in **0/30** seeds — the same as a random balanced mask given **ten times** the
   search freedom (0/30, and 0/300 per draw). Wilson 95% upper bound **0.114**. The design was
   powered at 0.87 against a true rate of 0.60 and 0.98 against 0.70, so this is an informative
   null, not an underpowered one.
2. **Width sweep (MLP): 2/30 → 25/30 → 29/30 → 30/30** at H = 8, 16, 32, 64
   (Cochran–Armitage z = 7.97, one-sided p = 7.8e-16). At **H = 8 — round 1's own width — the
   data-chosen mask buys nothing**: 2/30, identical to the random-best-of-10 control's 2/30
   (McNemar p = 1.0).
3. Therefore round 4's reframe — "ordinary training already leaves two unit-disjoint,
   oppositely-committed halves, so routing only adds +0.150" — is **true at H = 16 and false at
   H = 8**. H = 16 is the width *round 4 introduced* for its own ceiling. Re-scored at round 1's
   width against a fair, data-chosen floor, routing is **22/30 vs 2/30, Fisher p = 1.5e-7,
   difference +0.63 [0.436, 0.800]** (post-hoc, §6).

One sentence for the paper: *underdetermination does not by itself leave a latent switch; a
sufficiently overparameterized MLP does, and a transformer of comparable per-worldview capacity
does not — so "worldview-as-selectable-subnetwork" is a statement about unit-localized cue
representations, not about the loss landscape.*

---

## 1. Anchors — both exact

| anchor | check | result |
|---|---|---|
| **A0 (MLP)** | H=16, f=0 cell vs `r4-break-tautology/results.json` `floor_control`, seeds 700–729, fixed-mask *and* data-chosen accuracies | **max abs diff = 0.000000000**, identical per-seed success flags, n = 30 |
| **A1 (transformer)** | TF-A fixed-partition accuracies + `deploy_full`/`held_full` vs `r3-lm-organism/results.json` `control`, seeds 700–711 | **max abs diff = 0.000000000**, n = 12 |

Both substrates are literal replications, not re-implementations. Round 4's 51/60 reproduces here
as 25/30 on the first 30 of its 60 seeds, to the last bit.

## 2. PRIMARY — the transformer has no latent switch (n = 30, seeds 700–729)

| scoring of the same 30 standard-trained transformers | k/30 | Wilson 95% |
|---|---|---|
| **data-chosen mask** (top-2 heads + top-64 units by cue-A−cue-B sensitivity, held-out agreeing data only) | **0** | [0.000, 0.114] |
| random balanced mask, **best of 10 draws** (the fair control, advantaged 10:1) | 0 | [0.000, 0.114] |
| random balanced mask, per-draw | 0/300 | — |
| round-3's arbitrary fixed partition (heads {0,1} + units [0:64)) | 0 | — |
| distractor-field mask (S4 negative control, fields F2 vs F3) | 0 | — |

Exact McNemar, data-chosen vs random-best-of-10: b = 0, c = 0, **p = 1.0**.
Newcombe 95% CI on the difference: **[−0.114, +0.114]**.
**Preregistered decision: ABSENT.**

**This is a powered null.** Prestated power for the joint criterion (p < 0.01 and k ≥ n/2) at
n = 30 against a random control at 0.10: 0.15 / 0.52 / **0.87** / **0.98** / 1.00 for true rates
0.4 / 0.5 / 0.6 / 0.7 / 0.85. The result excludes true rates ≥ 0.6 and, by the Wilson bound,
excludes any rate above 0.114. It does **not** exclude rates in the 0.1–0.4 band; that limit was
stated in advance and is restated here.

**The models are the right models.** All 30 reach held-out agreeing accuracy 1.000
(0 competence failures). Their un-ablated deploy alignment is the seed lottery round 3 reported:
mean 0.500, 4 seeds > 0.8, 4 seeds < 0.2, 22 in the middle. So underdetermination is present; what
is absent is the *selectable* dual structure.

**Prestated failure diagnostic.** Of the 30 failures: 14 involve at least one half missing the
competence gate — but **16 do not**, and in those the two halves simply agree: 27/30 seeds fail
`accA_deploy > 0.8` and 24/30 fail `accB_deploy < 0.2`. Both halves land in the middle together.

## 3. S1 — the width sweep, and what it does to round 4 (MLP, n = 30/width)

Every model standard-trained; only the scoring mask differs.

| H | units per worldview | **data-chosen** | Wilson 95% | random best-of-10 | random per-draw | fixed arbitrary mask | distractor mask | McNemar (b/c) | p | Holm |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 4 | **2/30** | [0.019, 0.213] | 2/30 | 2/300 | 0/30 | 0/30 | 1/1 | 1.0 | 1.0 |
| 16 | 8 | **25/30** | [0.664, 0.927] | 10/30 | 14/300 | 1/30 | 0/30 | 16/1 | 2.7e-4 | 0.0011 |
| 32 | 16 | **29/30** | [0.833, 0.994] | 6/30 | 6/300 | 1/30 | 1/30 | 23/0 | 2.4e-7 | 2e-6 |
| 64 | 32 | **30/30** | [0.887, 1.000] | 3/30 | 3/300 | 1/30 | 0/30 | 27/0 | 1.5e-8 | <1e-6 |

Cochran–Armitage trend across widths (scores log2 H, one-sided increasing):
**z = 7.973, p = 7.8e-16.** Newcombe 95% CIs for data-chosen − random-best-of-10:
H8 [−0.154, +0.154], H16 [0.254, 0.669], H32 [0.548, 0.875], H64 [0.707, 0.965].

Every mask is a genuine full H/2 + H/2 split (checked per seed, all cells).

**The H = 8 cell is the load-bearing one.** Round 1's control arm was H = 8 (its `results.json`
config: `H = 8`, seeds 700–729 — the identical cell). Round 4 concluded that round 1's 0/30
control was an artifact of scoring under an arbitrary partition. At H = 16 that is exactly right.
**At H = 8 it is not:** giving the control a data-chosen mask moves it from 0/30 to 2/30, and a
random mask with ten draws also gets 2/30. There was no hidden switch at round 1's width.

## 4. S2 — specificity: disagreeing data (the negative control)

f = fraction of training examples on which the proxy cue is *wrong*, so the worldview is
determined rather than underdetermined. Same seeds, same procedure, same criterion.

| substrate | cell | data-chosen k/n | Wilson 95% | vs f = 0, Fisher p | prestated pass (p<0.05 and upper<0.5) |
|---|---|---|---|---|---|
| MLP | H16, f = 0.05 | 7/30 | [0.118, 0.409] | 5.9e-6 | **pass** |
| MLP | H16, f = 0.25 | **0/30** | [0.000, 0.114] | 5.5e-12 | **pass** |
| MLP | H64, f = 0.05 | **19/30** | [0.455, 0.781] | 3.2e-4 | **fails the upper-bound clause** |
| MLP | H64, f = 0.25 | **0/30** | [0.000, 0.114] | 1.7e-17 | **pass** |
| TF | f = 0.25 | 0/16 | [0.000, 0.194] | 1.0 | **void** (f = 0 is already 0) |

Random-best-of-10 and distractor-mask controls are 0/30 in all four MLP cells.

Two things follow. (a) **The procedure is not vacuous.** At f = 0.25 nothing is selectable at
either width, while at f = 0 the same code finds 25/30 and 30/30. (b) **Width buys the misaligned
worldview resistance to corrective data.** 5% anti-proxy examples cut selectability from 25/30 to
7/30 at H = 16, but only from 30/30 to **19/30** at H = 64: the wide network keeps a
proxy-committed, individually-competent, retrievable half even though 1 in 20 training examples
contradicts the proxy. The TF negative control cannot be evaluated because its positive arm is
already zero; this is reported as void, not as a pass.

## 5. S3 / S4 / S5

**S3 — k = 3 multi-way latent selection works, and again scales with per-worldview capacity.**
Data-chosen 3-way partition (each unit assigned to `argmax_c sens_c`), vs a **size-matched**
random partition, best of 10:

| H | units/worldview | data-chosen | Wilson 95% | random best-of-10 | random per-draw | fixed equal blocks | McNemar p | Newcombe |
|---|---|---|---|---|---|---|---|---|
| 24 | 8 | **17/30** (0.567) | [0.392, 0.726] | 0/30 | 0/300 | 0/30 | 1.5e-5 | [0.358, 0.726] |
| 48 | 16 | **28/30** (0.933) | [0.787, 0.982] | 0/30 | 0/300 | 0/30 | 7.5e-9 | [0.748, 0.982] |

No empty groups; mean group sizes 8.6/7.4/8.0 and 16.5/15.4/16.1. So the switch is genuinely
k-way, and the fixed equal-block partition — the scoring every earlier round used for controls —
finds it 0/30 of the time at both widths.

**S4 — distractor-cue negative control passes everywhere.** Masks built from sensitivity to
label-independent fields/columns instead of the cues: TF 0/30, MLP 0/30, 0/30, 1/30, 0/30 at
H = 8/16/32/64 (McNemar vs the cue-derived mask: p = 6e-8, 7.5e-9, 1.9e-9 at H ≥ 16). The
cue-derived mask is measuring something specific, not fitting activation noise.

**S5 — the either-mapping generosity is never used.** Designated-only counts equal the
either-mapping counts in *every* cell (TF 0/30; MLP 2, 25, 29, 30; k = 3: 17, 28). The
data-chosen mask labels its own halves correctly; unlike the fixed-partition control it does not
need the swapped-mapping allowance. (Round 4's fair floor did use that allowance for some seeds.)

## 6. POST-HOC (labeled; in `posthoc.json`, never in `results.json`)

- **The transformer null is not a mask-coarseness artifact.** n = 12 fresh standard-trained
  transformers, two alternative mask families, own random controls (K = 5):
  *units-only* (all 4 heads kept in both halves, only the 128 MLP units split 64/64 by data)
  gives **0/12 with both halves competent in 12/12** (held-out 1.000/1.000) and a mean deploy
  selection margin of **0.073**; *heads-only* gives 0/12, both halves competent 9/12, margin
  0.114. Random controls 0/60 each. So even with a fine, fully competence-preserving mask, the
  two data-chosen halves of a standard-trained transformer behave identically at deployment.
  **Capacity is not the explanation:** the units-only mask gives each worldview 64 MLP units in
  each of 2 layers, versus the 32 units per worldview at which the MLP scores 30/30.
- **Per-worldview capacity, not total width, is the governing variable** (pools k = 2 and k = 3):
  4 units/worldview → 2/30; 8 → 25/30 (k=2) and 17/30 (k=3); 16 → 29/30 (k=2) and 28/30 (k=3);
  32 → 30/30.
- **The binary criterion tracks a smooth margin.** Mean |deploy(keep A) − deploy(keep B)|:
  H8 0.533 (median 0.526, min 0.001), H16 0.918, H32 0.981, H64 0.990; f = 0.05 cells 0.664
  (H16) and 0.805 (H64); f = 0.25 cells 0.237 and 0.265. **TF-A: mean 0.233, max 0.520** — the
  transformer never gets close to the threshold, so its 0/30 is not a near-miss.
- **Round 1's headline, re-scored fairly at its own width.** routed 22/30 vs data-chosen floor
  2/30 at H = 8: Fisher **p = 1.45e-7**, difference **+0.633 [0.436, 0.800]** (the fixed-mask
  floor there is 0/30). Round 4's +0.150 and this +0.633 are both correct; they are different
  widths.

## 7. What this does to the line

**Retracts a generalization, keeps the finding.** Round 4's "the routing-free floor is really
85%" is reproduced exactly (A0) and is *specific to H = 16 and above*. The claim that
"underdetermination produces a latent switch rather than an entangled compromise" does not
survive: at H = 8 it is a compromise, and in the transformer it is a compromise at every mask
family tried, with both halves demonstrably competent.

**Converts the round-4 result into a width law.** Routing's honest advantage over a fairly-scored
control is large where capacity is tight (+0.63 at H = 8) and vanishes where it is not (the floor
is already 30/30 at H = 64). Routing does not create the switch; overparameterization does — in
this architecture.

**Adds one safety-relevant claim that is new.** Wider models keep the misaligned worldview
*retrievable* under corrective data: 5% anti-proxy examples leave 19/30 selectable at H = 64
versus 7/30 at H = 16. If that survives scale, "we trained on some disagreeing data" is a weaker
guarantee in a larger model than in a smaller one, and the residual is not visible in behaviour —
it needs an ablation to see.

**Does not move localization at all.** Every mask here consumes the input-space cue oracle, as in
rounds 1–4. `r4-asymmetry-localizer` remains the open problem.

## 8. Caveats

- **The transformer null is one organism, one mask granularity family.** Masks are shared across
  the 2 layers; per-layer masks, per-attention-dimension masks, and non-axis-aligned (rotated)
  subspace ablations were not tried. The prestated diagnostic plus the post-hoc units-only probe
  rule out competence loss as the cause, but not "the latent switch exists in a basis this mask
  family cannot express". That is the single most likely way this null is wrong, and it is the
  next experiment.
- The TF negative control (TF-N, f = 0.25, n = 16) is **void**, not passed: its positive
  comparator is zero.
- The MLP `H64_f0.05` cell fails the prestated S2 pass criterion (Wilson upper 0.781 > 0.5); it
  is reported as a partial reduction, not as "the negative control passed".
- n = 30 per MLP cell and n = 30 for the primary; the Wilson bound on the TF null is 0.114, so
  rates in the 0.1–0.4 band are not excluded.
- Toy scale throughout: ≤68.5k params, synthetic tasks, searched-for equal-cost cues, one seed
  family (700–729).
- The k = 3 partition is data-chosen with unconstrained group sizes; the random control is
  size-matched per seed, but a size-*unmatched* random control was not run.
- Everything consumes the cue oracle (which input columns / which field carries which cue).

## 9. Files

- `PREREG.md` — stamped 13:58:30 EDT, before any confirmatory run; includes the power table and
  the downscope rules (neither fired: all 30 TF-A seeds completed, no arm was skipped)
- `core_mlp.py`, `core_tf.py` — harnesses (round-4 and round-3 code respectively, generalized)
- `experiment.py` — confirmatory run + every preregistered test; `results.json` — all per-seed rows
- `posthoc.py` / `posthoc.json` — post-hoc only, segregated
- `timing_pilot.py` (wall-clock only), `smoke.py` (crash-only), `run.log`
