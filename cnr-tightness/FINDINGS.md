# Certificate tightness audit — certified_nla_reachability

Analysis-only audit of the saved artifacts in `/Users/steven/certified_nla_reachability`
(read-only). Question: **how far are the sound certified bounds (CROWN/IBP) from the
best empirical attack, and where is the certificate binding vs vacuous?**

All eps values are in the repo's own units: per-dimension L-inf radius `eps * sigma_i`,
sigma = per-dim std of the layer-6 pooled GPT-2 activations (from probe meta). tau = 0.

## Pre-registered tests (written BEFORE running the analysis)

**T1 — shallow radius gap (primary).** For each of the n=8 anchors saved in
`artifacts/meta.json` (`anchor_ids`, h0 = bank.npz X[id]; the anchor set is fixed by the
artifact, no selection by us): compute
- `eps_cert`: certified radius, bisection (tol 1e-3) on U(eps) = min(native CROWN, IBP)
  using the repo's own `cnr.verify` bound functions (largest eps with U < tau);
- `eps_att`: smallest eps (bisection, tol 1e-3) at which multi-restart PGD
  (20 restarts, seeds 0..19, 400 sign-gradient steps, projected to the box) finds
  h in the ball with probe(h) >= tau. PGD is a *lower* bound on the true max, so the
  true reachability threshold lies in [eps_cert, eps_att].
- Primary metric: ratio `rho = eps_att / eps_cert` per anchor; report per-anchor values
  + median and min–max over the 8 anchors.
- Pre-stated interpretation: **binding** if rho <= 1.5; **loose** if 1.5 < rho <= 10;
  **vacuous as a guarantee about its threat model** if rho > 10.

**T2 — bound-value gap at fixed eps.** At the 13 eps values saved in
`artifacts/cert/cert_results.json` (tightness rows, anchor 0): report
`U_crown_saved − PGD_max` and `U_crown_saved − sampling_saved` (the repo's own unsound
sampling reference). Consistency gate: our recomputed U must match the saved U to 1e-3.

**T3 — token-space GCG attack vs certificate.** Recompute the layer-6 pooled activation
of the saved GCG attack prompt and its benign prefix (`artifacts/last_run.json`,
cached gpt2, repo harness). Gate: recomputed probe scores must match saved
(4.7246 / −8.949) to within 0.05 (tokenizer round-trip risk; if the gate fails we report
recomputed values and flag it). Then compute, in the certificate's own norm
(per-dim L-inf, std units), the displacement `delta_inf` of the attack activation from
each certified anchor, and the natural prompt-to-prompt variation baseline:
`delta_inf` over 2000 random deploy–deploy bank pairs (seed 0; median + 5–95 pct;
bootstrap 95% CI on the median, 1000 resamples, seed 1).
Pre-stated interpretation: if the real (token-space) attack and even benign
prompt-to-prompt variation sit at `delta_inf >= 10 × eps_cert`, the activation-space
certificate is **vacuous for the token-level threat model** (it certifies a ball no
prompt edit stays inside).

**T4 — deep (frozen-attention) certificate gap.** Anchor 0, token_index 12 (the setting
saved in cert_results.json `deep`). Gate: frozen-forward score matches saved −10.5330.
Compute `eps_cert_deep` (bisection, tol 2e-4, repo's deep certifier, min(CROWN,IBP)) and
`eps_att_deep` (PGD on the repo's exact `_frozen_forward_batch`, single-token box,
20 restarts, 400 steps), and `rho_deep`. Same interpretation thresholds as T1.

Notes on rigor: PGD is heuristic, so eps_att can only overstate the true reachability
threshold; rho is therefore an upper bound on the true tightness ratio. Monotonicity
(success at eps implies success at larger eps) holds for the true max (nested balls);
for PGD we verify by ascending coarse grid before bisection.

## Results

All consistency gates passed: 6/6 saved anchor probe scores reproduced to <1e-3; the
GCG attack prompt's probe score reproduced exactly (4.72458 vs saved 4.72458); the deep
frozen-forward score reproduced exactly (−10.53303); the recomputed sound bound equals
min(saved CROWN, saved IBP) to <1e-3 at all 13 saved eps. Wall time 449 s, CPU only.

**T1 (primary): the shallow activation-space certificate is BINDING.**
Across all n=8 anchors, rho = eps_att/eps_cert = **median 1.276, min 1.245, max 1.311**
(eps_cert median 0.334, PGD attack radius median 0.437, bisect tol 1e-3, 20 PGD
restarts seeds 0..19). By the pre-registered threshold (rho <= 1.5) the certified
radius is within ~28–31% of the best attack radius at every anchor. The true
reachability threshold lies in [eps_cert, eps_att] per anchor.

**T2: the bound-value gap is small at certification-relevant eps and the repo's own
sampling reference overstates looseness ~4x.** At anchor 0: U_sound − PGD_max rises
from ~0.00 at eps=0.02 to 0.93 at eps=0.26 and 2.22 at eps=0.31 (≈ the certified
radius), reaching 5.8 at eps=0.60. The repo's saved uniform-sampling reference sits
9–14 points below PGD (it never exceeds −9.2 anywhere), so the saved tightness.png
figure makes CROWN/IBP look far looser than they actually are: at eps=0.31 the real
relaxation gap is 2.2 points, not the ~9.0 the sampling curve suggests.

**T3: for the token-level threat model the certificate is VACUOUS (pre-registered
criterion met).** The saved GCG attack activation sits at delta_inf = 6.04–7.63 sigma
from the 8 certified anchors (min 6.04), i.e. ≥ 16x the largest per-anchor certified
radius (0.374) and ~18x the median. Even ordinary benign prompt-to-prompt variation
dwarfs the certified ball: deploy–deploy pair delta_inf median 4.51 sigma
(n=1996 pairs, 5–95 pct [2.04, 5.44], bootstrap 95% CI on the median [4.48, 4.54])
— ~13.5x the median certified radius. No prompt edit, indeed no prompt *resampling*,
stays inside any certified ball; the shallow certificate constrains a region that
token-space attacks never pass through.

**T4: the deep frozen-attention certificate is VACUOUS even on its own subnetwork.**
Anchor 0, single-token (index 12) ball: eps_cert_deep = 0.0082 (consistent with the
saved grid value 0.005, bracket [0.005, 0.01)), while PGD on the repo's *exact*
frozen-attention forward does not cross tau until eps ≈ 3.51. rho_deep ≈ **428**
(>10 = vacuous). At the config's operating eps = 0.25 the saved sound bound is
U = +932.9 while the best exact-subnetwork attack found stays below tau = 0 — a
bound ~940 points above anything attainable.

**Where binding vs vacuous (one line):** the certificate is tight exactly where the
threat model is weakest (tiny L-inf balls on the probe's immediate input, gap ~28%),
and vacuous everywhere the threat model gets real (one block deeper: ~430x gap;
actual token-space attacks: they operate 16x+ outside any certified radius, as does
benign prompt variation).

Machine-readable numbers: `results.json`. Figure: `tightness_summary.png`
(left: bound vs attacks vs eps, anchor 0; right: log-scale radius comparison).

## Caveats

- **PGD is heuristic**: eps_att and rho are upper bounds on the true threshold/ratio.
  For T1 this makes the "binding" conclusion conservative (true rho ≤ 1.31). For T4
  the true rho_deep could be < 428 if PGD badly underperforms on the deep subnetwork,
  but it is ≥ 1 by soundness, and the +932.9-vs-<0 bound-value gap at eps=0.25 is
  attack-independent evidence of vacuousness. PGD monotonicity across the bisection
  was assumed after an ascending coarse-grid bracket; non-monotone PGD failures could
  shift eps_att by less than the bracket width.
- **T2 gate, definitional (post-hoc note)**: the pre-registered gate compared against
  the saved pure-CROWN column; the repo's `crown_ibp_upper` returns min(CROWN, IBP),
  so 4/13 large-eps rows "mismatch" pure CROWN while matching min(CROWN, IBP) exactly
  (all 13 <1e-3). No number changed; only the label of which saved column we reproduce.
- **T3 prefix score**: the saved `score_start` (−8.95) is the score at GCG
  initialization (prefix + initial adversarial tokens), not the bare prefix (−11.29
  recomputed). The gate keyed on the attack prompt, which reproduced exactly. The
  attack-vs-own-prefix displacement (7.22 sigma) uses the bare prefix.
- **Small fixed n**: 8 anchors (all that the artifact defines — no selection by us,
  but no sampling either), 1 anchor for T2/T4 (the same single anchor the repo's own
  ablation used). Min–max ranges are reported instead of CIs where n=8; nothing here
  is a population estimate beyond this artifact set.
- **Norm caveat for T3**: delta_inf compares a token-attack displacement to an L-inf
  certified radius in the same per-dim std coordinates, which is the certificate's own
  norm; a certificate in a different norm (e.g. L2 with larger radius) could in
  principle cover more of the natural variation. This audits the certificate as built.
- The deep certificate covers a frozen-attention subnetwork only (the repo says so
  explicitly); T4 measures tightness *on that subnetwork*, which is the generous
  reading — the full-model gap can only be larger or equal in threat-model terms.
