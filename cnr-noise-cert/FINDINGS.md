# Certified reachability of the eval-awareness region under activation noise

Extension of `~/certified_nla_reachability` (CNR): the box certificate there answers
"can any bounded *worst-case* perturbation reach A* = {h : probe(h) >= tau}?".
Here we ask the `~/misalignment_noise_probe` question instead: the noise probe's
operation is **isotropic Gaussian noise** `h -> h + N(0, (sigma*RMS(h))^2 I)` on a
residual layer. Can that operation move GPT-2's layer-6 pooled activation into the
NLA-defined eval-awareness region A*, and can we certify anything about it?

**Math fact stated up front:** Gaussian noise has full support and A* is non-empty
(the eval-class bank activations are in it), so P(h0 + eta in A*) > 0 for every
sigma > 0. A hard "unreachable" certificate is impossible for this operation. The
correct certified object is a **sound upper bound delta on the flip probability**
P_flip(h0, sigma) = P(probe(h0 + eta) >= tau).

## Pre-registration (written 2026-07-24 17:50, BEFORE any experiment code was run)

Artifacts reused read-only from CNR: `artifacts/probe.pt` (768->64 ReLU->1 probe, tau=0,
folded to raw activations), `artifacts/bank.npz` (512 x 768 layer-6 mean-pooled GPT-2
activations, 256 deploy / 256 eval), `artifacts/meta.json` (anchor_ids = the 8 repo
anchors, all deploy-class). What was inspected before pre-registration: array shapes,
label counts, tau/tau_hi, pooled RMS (mean 7.47), per-dim sigma scale (mean 0.655).
No probe gradients, bounds, or flip rates were computed before this was written.

### Primary question
At the noise probe's operating levels sigma in {0.1, 0.2, 0.3} (noise std
s = sigma * RMS(h0), isotropic, applied directly to the layer-6 pooled activation
the probe reads), what fraction of the 256 deployment-class bank activations is
**certified** to have P_flip <= delta = 0.01, using the best of four sound
certificates (min is sound because each is individually sound)?

Certificates (all closed-form / deterministic, derivations in `noise_cert.py`):
1. **Ball+tail**: P_flip <= P(||eta||_inf > rho*) with rho* = the largest uniform
   per-dim box radius certified by the repo's own CROWN/IBP verifier (bisection,
   30 iters); the tail is exact: 1 - erf(rho*/(s*sqrt2))^768.
2. **CROWN-Gaussian hybrid (new)**: for any rho, P_flip <=
   Phibar((tau - A.h0 - c)/(s*||A||_2)) + P(||eta||_inf > rho), where (A, c) is the
   sound CROWN linear upper bound on the rho-box (reimplemented from `cnr/verify.py`
   so it exposes A, c; equivalence-checked against the repo's `crown_upper`).
   Minimized over a fixed grid rho = s*q, q in {2.5,3,...,8}.
3. **Input-Lipschitz-Gaussian (naive baseline)**: P_flip <= P(chi2_768 >= (m0/(L*s))^2),
   L = ||W2||_2 * sigma_max(W1), m0 = tau - probe(h0). Expected loose; included to
   show direction-awareness matters.
4. **Hidden-space dead-neuron-aware (new)**: exact hidden pre-activation shift
   g = W1 eta ~ N(0, s^2 W1 W1^T); score increase <= f(eta) =
   sum_{lambda_j>0} lambda_j [relu(z0_j+g_j) - relu(z0_j)] + sum_{lambda_j<0} |lambda_j||g_j|;
   f is (sum_j |lambda_j| ||W1_j||)-Lipschitz in eta, E f in closed form, then
   Gaussian-Lipschitz concentration: P(f >= m0) <= exp(-(m0 - Ef)^2 / (2 s^2 L_f^2)).

### Primary endpoints (decision rule fixed now)
- E1: certified fraction (count/256) with best-cert P_flip <= 0.01 at each of
  sigma = 0.1, 0.2, 0.3. (delta = 0.05 reported as secondary, same rule.)
- E2: certified max noise level sigma*_{0.01} per anchor (bisection on sigma over
  [1e-4, 10], monotone bound), reported as median and IQR over the 256 deploy anchors.

### Soundness witnesses (must pass, else the result is void)
- W1: on the 8 repo anchors x sigma in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0},
  Monte-Carlo flip rate (n = 20,000 draws per cell, seeds fixed from master seed 0)
  must not exceed the certified bound: one-sided exact binomial test of
  H0: p <= bound at alpha = 0.05 per cell; any rejection = verifier bug, result void.
- W2: reimplemented CROWN (A, c) must reproduce the repo's `crown_upper` box bound
  to <= 1e-6 on the 8 anchors x the rho grid.
- W3: repo's own `CertifiedVerifier.certify` (with its internal 2000-sample witness)
  must confirm certified_unreachable at rho*(1-1e-3) for the 8 anchors.

### Secondary / exploratory (labeled as such, no certification claims)
- S1: empirical flip-onset sigma: smallest sigma in the W1 grid with >= 1 flip in
  20,000 draws, per anchor; plus pooled flip rate over all 256 deploy anchors x
  2,000 draws at sigma = 0.1 and 0.3 (Clopper-Pearson 95% CIs).
- S2: tightness: median bound per certificate at sigma = 0.1 (which certificate wins).
- S3: in-model arm (GPT-2 forward passes, exploratory): per-token noise injected with
  the noise probe's exact hook at (a) the read layer (block 5 output = hidden_states[6])
  and (b) upstream block 2, sigma in {0.3, 1.0}, texts = the 8 anchor texts + the next
  8 deploy-class texts by bank index, 128 draws each. Read-layer injection is covered
  by the certificate with s_eff = sigma * sqrt(sum_t rms_t^2)/T (mean-pooling
  attenuates i.i.d. per-token noise by ~sqrt(T)); we check measured pooled std vs
  theory and flip counts vs bound. Upstream injection is NOT covered (nonlinear
  propagation through blocks 3-5); empirical flip rate only.

Seeds: master seed 0; per-cell seeds derived deterministically. Device: CPU,
torch threads capped at 3 (shared machine). Runtime budget ~10 min.
Negative/null outcomes on any endpoint will be reported as-is; delta, sigma grid,
anchor set, and n were fixed above and will not be changed after seeing results.

## Results

Run 2026-07-24, seed 0, CPU, runtime 76.5 s, single process. All soundness
witnesses passed: W1 0/56 MC cells violate any bound (n=20,000 draws/cell, exact
binomial tests); W2 reimplemented CROWN matches repo `crown_upper` to 1.1e-13;
W3 repo `CertifiedVerifier` (internal 2000-sample witness) confirms 8/8 anchors
at rho*(1-1e-3). Clean GPT-2 layer-6 activations reproduce the bank to 1.5e-5.
No deploy anchor starts in A* (scores -13.8 to -7.5, tau = 0).

### E1 (primary): certified fraction, P_flip <= 0.01, n = 256 deploy anchors
| sigma | certified @ delta=0.01 | @ delta=0.05 | median best bound | max |
|---|---|---|---|---|
| 0.1 | **54/256 (21.1%)** | 154/256 | 0.038 | 0.33 |
| 0.2 | **0/256** | 0/256 | 0.70 | 0.96 |
| 0.3 | **0/256** | 0/256 | 0.97 | 1.0 |

### E2 (primary): certified max noise level sigma*_{0.01}
Median **0.088**, IQR [0.077, 0.097], range [0.061, 0.120], 0/256 uncertifiable.
So the certifiable regime ends right at the noise probe's *lowest* operating
level (sigma = 0.1): below ~0.09 the eval-awareness region is provably (p <= .01
per draw) out of reach; at 0.2-0.3 nothing can be certified — correctly so, see S1.

### S1 (secondary): the noise probe's operation really does reach A*
- sigma = 0.1: **0 flips in 512,000 draws** (256 anchors x 2,000; CP95 rate <= 7.2e-6).
- sigma = 0.2: flip onset. 4/8 repo anchors show first flips (2-8 per 20,000).
- sigma = 0.3: **2,777 / 512,000 flips = 0.54%** (CP95 [0.52%, 0.56%]); per-anchor
  rates 0.03%-0.88% (n=20,000 each). Bounded-level Gaussian activation noise at the
  noise probe's top operating level moves GPT-2's layer-6 state into the NLA-defined
  eval-awareness region at ~1-in-200 draws, from *every* kind of benign anchor tested.

### S2 (secondary): which certificate does the work
At sigma = 0.1 the new hidden-space dead-neuron-aware certificate (cert4) is the
best bound for **256/256** anchors (median 0.038). Ball+tail (cert1),
CROWN-Gaussian (cert2), and input-Lipschitz (cert3) are all vacuous (median 1.0):
covering 768-dim Gaussian noise with an L-inf box forces rho ~ 4.4s, where the
box relaxations collapse (certified box radius rho* median 0.194 raw units ~= 0.09s
at sigma=0.1). Post-hoc observation: naive "box certificate + tail bound" porting of
the CNR certificate to noise fails completely; the noise-adapted derivation is needed.
Looseness: at sigma=0.1 the median certified bound (0.038) is >= 3-4 orders above
the empirical rate (<= 7.2e-6), so E1's 21% is very conservative.

### S3 (exploratory): in-model per-token injection (GPT-2 forwards, 16 texts x 128 draws)
- Read-layer (block 5 = hidden_states[6]) injection: measured pooled per-dim noise
  std matches the theory s_eff = sigma*sqrt(sum_t rms_t^2)/T to 0.5% (ratio
  0.9995-1.0045 over 32 cells), so the pooled certificate covers the actual in-model
  hook exactly (0/32 coverage violations, exact binomial). Numerically
  s_eff ~= 0.84 x sigma*RMS(h_pooled): sqrt(T) pooling attenuation is almost
  cancelled by per-token RMS being ~sqrt(T) x pooled RMS.
- Flip rates: read-layer sigma=0.3: 7/2048 = 0.34% [0.14%, 0.70%]; sigma=1.0:
  250/2048 = 12.2% [10.8%, 13.7%]. Upstream (block 2, propagated through blocks
  3-5, NOT covered by any certificate here): sigma=0.3: 5/2048 = 0.24% [0.08%,
  0.57%]; sigma=1.0: 435/2048 = **21.2%** [19.5%, 23.1%] — more flips than
  read-layer despite *smaller* measured pooled std (5.9 vs 6.9): propagation shapes
  isotropic noise toward probe-sensitive directions. Post-hoc, n small; a follow-up
  would need per-layer sweeps and a capability control.
- Implementation note (bug found and fixed before final run): transformers 5.x fills
  `output_hidden_states` before user forward hooks mutate a block's output, so a
  noise hook on the read layer is invisible to `hidden_states[6]`; block outputs
  must be captured with one's own later-registered hook.

### Answer to the brief's question
Can sigma-bounded activation noise move GPT-2 into the NLA-defined eval-awareness
region, certifiably? **Yes at the top of the noise probe's operating range, and the
new certificate proves safety below it.** A hard certificate is impossible
(Gaussian support), but the flip *probability* is certifiable: P_flip <= 0.01 is
provable for a median max noise level sigma* ~= 0.088 (21% of anchors already
certified at sigma = 0.1), while at sigma = 0.3 the region is genuinely reached at
0.5% of draws (0.2-0.9% depending on anchor) — so the non-certifiability at 0.2-0.3
is a true positive, not looseness.

## Caveats
- A* is CNR's stand-in probe (synthetic eval-vs-deploy labels, near-separable,
  train/val acc 1.0); "eval-awareness" here means the probe's decision region,
  not a validated concept. All certificates are relative to this probe and tau=0.
- Certificates cover single-draw flip probability at the pooled layer-6 readout
  (and, via s_eff, the exact in-model read-layer hook). Upstream injection and
  autoregressive generation under persistent noise are not covered.
- delta=0.01 and the sigma grid were pre-registered; the cert4-wins observation,
  the 0.84 attenuation constant, and the upstream-amplification effect are post hoc.
- sigma=0.2 flip counts are small (2-8/20k on 4/8 anchors); onset location between
  0.1 and 0.2 is imprecise. S3 uses 16 texts x 128 draws only.
- Certificate looseness at low sigma is 3-4+ orders of magnitude vs MC; the certified
  fractions are lower bounds on the truly-safe fraction, not tight estimates.
