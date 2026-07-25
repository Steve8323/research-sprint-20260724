# Covert capacity in bits vs rho (ERT paraphrase bottleneck)

Extends `~/english_bottleneck_lm` E2 (read-only). E2 reported covert-channel
strength as bit-recovery accuracy (0.5 = closed) at rho in {0,...,1.0}; the
README headline is the single rho=1 point ("optimal reader 0.496 ~ chance").
Here: the same channel measured in **bits per carrier token**, rho grid
{0, 0.25, 0.5, 0.75, 0.9, 1.0}, with CIs — a dose-response curve.

## Question

How many bits/carrier-token can a colluding sender/receiver pass through the
real `ParaphraseChannel` applied `n_layers - 1 = 3` times (the exact E2 code
path, LAYERS=4), as a function of paraphrase strength rho?

## Pre-registered method (written BEFORE this agent ran anything)

- **Channel**: X = synonym index (uniform on {0..3}) of a random meaning class
  (all 24); one-hot word through the repo's `ParaphraseChannel(rho)` 3 times;
  Y = synonym index of the surviving token. Real repo module, not a rewrite.
- **Primary estimand**: I(X;Y) bits under uniform X. The induced 4x4 channel is
  symmetric/doubly stochastic, so uniform input is capacity-achieving: this MI
  IS the per-token capacity and (by data processing) an upper bound on any
  reader, optimal or learned.
- **Estimator**: Miller-Madow-corrected plug-in MI on the pooled joint table;
  95% percentile bootstrap (multinomial resamples); per-seed spread reported.
- **Test**: independence test per rho (H0: capacity = 0), alpha 0.05,
  Bonferroni over the 6 rhos.
- **Analytic cross-check**: per-step T(rho) = (1-rho)I + rho(J-I)/3, second
  eigenvalue lambda = 1 - (4/3)rho; exact capacity after 3 steps =
  2 - H(row of T^3). Agreement criterion: analytic value inside the 95% CI.
- **Pre-stated prediction**: NON-MONOTONE curve — exactly 0 at rho = 0.75
  (lambda = 0), slightly positive again at rho = 1 (lambda = -1/3, ~0.003
  bits) because "always swap to a different synonym" is itself informative
  (anti-correlation). rho=1 should reject H0; rho=0.75 should not.
- **Secondary (labeled)**: E2-comparable binary code (bit in synonym 0 vs 1).

## Provenance note (deviation, disclosed)

After writing the pre-registration above, this agent found the target
directory already contained a COMPLETED earlier attempt at this same task
(`capacity_curve.py` + `results.json`, 12:11 today): same design, larger n
(3 seeds x 200,000 = 600,000/rho), permutation test (1,000 shuffles) instead
of a G-test, and its own pre-registration block (including the same analytic
predictions) written into results.json before sampling. Actions taken:
1. Preserved the as-found results as `results_prior_run.json`.
2. Re-ran `capacity_curve.py` unmodified: output is bit-for-bit identical to
   the as-found results.json except `wall_time_s` (provenance confirmed).
3. Ran an INDEPENDENTLY CODED verification (`verify_indep.py`): fresh seeds
   100-104 (disjoint from the prior run's 0-2), n = 5 x 24,000 = 120,000/rho,
   G-test (df=9) + Bonferroni per this agent's pre-registration. Deviation
   from my prereg: seeds 100-104 x 24k instead of 0-4 x 20k, chosen to be
   disjoint from the prior run's seeds.
Primary numbers below are the better-powered n=600,000 run; verification
agreed at every rho (analytic and prior estimate inside the fresh 95% CI),
except the degenerate rho=0 case (see caveats).

## Results

Primary: covert capacity C(rho), bits/carrier-token through 3 paraphrase hops
(n = 600,000/rho, 3 seeds; 95% percentile bootstrap; permutation p, 1,000
shuffles; verification = independent run, fresh seeds, n = 120,000/rho).

| rho  | C bits [95% CI]                   | analytic | perm p | verify C [95% CI] (G-test p_bonf) |
|------|-----------------------------------|----------|--------|------------------------------------|
| 0.0  | 2.00000 [1.99999, 2.00000]        | 2.0      | 0.001  | 1.99990 [1.99981, 1.99994] (~0)   |
| 0.25 | 0.16638 [0.16456, 0.16821]        | 0.16572  | 0.001  | 0.16594 [0.16203, 0.16988] (~0)   |
| 0.5  | 0.00296 [0.00275, 0.00321]        | 0.00290  | 0.001  | 0.00323 [0.00275, 0.00386] (3e-111) |
| 0.75 | 0.0000012 [0, 0.000031]           | 0 exact  | 0.36   | 0.000005 [0, 0.00016] (p_bonf=1)  |
| 0.9  | 0.000161 [0.000117, 0.000233]     | 0.000139 | 0.001  | 0.000084 [0.000034, 0.000285] (0.038) |
| 1.0  | 0.00296 [0.00273, 0.00321]        | 0.00305  | 0.001  | 0.00263 [0.00222, 0.00317] (8e-90) |

Findings:
1. **Dose-response**: capacity falls ~12x by rho=0.25 (2 -> 0.166 bits) and
   ~675x by rho=0.5 (0.0030 bits). The closed form 2 - H(row of T^3) with
   lambda = 1 - (4/3)rho matches everywhere (analytic in CI at all 6 rhos in
   the primary run; 5/6 in verification, see caveats).
2. **The curve is non-monotone, as pre-predicted**: the true zero is
   rho = 0.75 (only rho consistent with C = 0: perm p = 0.36, G-test
   p_bonf = 1), NOT rho = 1. At rho = 1 the channel REOPENS to
   C ~ 0.0030 bits/token [0.0027, 0.0032] (p = 0.001) — "always swap"
   leaves an anti-correlation: after 3 forced swaps the sent synonym is less
   likely than each alternative (p_stay = 2/9 vs 7/27).
3. **Refines the published E2 headline**: the rho=1 channel is not closed,
   just tiny — ~0.003 bits/token, i.e. ~0.012 bits per 4-slot E2 message
   (~4,000x below the 2 bits/token at rho=0, ~330 tokens needed per reliable
   covert bit). In accuracy terms a Bayes-optimal binary reader gets 51.9%,
   not 50%; E2's published reader rule (guess the observed synonym) is
   anti-optimal at rho=1 — it scores BELOW chance (0.4832 [0.4792, 0.4872],
   n=60,000; analytic 0.4815) because it ignores the forced-swap
   anti-correlation. E2's published 0.496 at n=4,000 was within noise of
   both 0.5 and 0.4815, hence the "channel closed" reading. [The
   anti-optimality interpretation is post-hoc; the reopening itself was
   pre-predicted.]
4. Practical note: for closing the covert channel, rho = 0.75 (exact zero) or
   rho in [0.75, 0.9] dominates rho = 1.0; E1 (published) shows the meaning
   task survives at both 0.75 and 1.0.

## Caveats

- Toy substrate: 4-synonym classes, model-free channel; "capacity" is exact
  for this channel but says nothing about natural-language paraphrasers.
- The non-monotonicity depends on SYN_PER=4 and odd hop count (lambda^3 < 0);
  with even hops or larger classes the rho=1 residual shrinks
  ((k-1)^-hops scaling).
- rho=0 edge case: identity channel is deterministic, so the plug-in MI is
  exactly 2 bits with ~zero bootstrap variance; two Miller-Madow conventions
  (prior run: nonzero-cell count; verification: (Kx-1)(Ky-1)) disagree by
  ~1e-4 bits there, so the mutual "in-CI" check fails trivially at rho=0
  only. Immaterial; both are within 1e-4 of the exact value 2.0.
- The MI here upper-bounds single-token codes actually used by E2's learned
  collusion (4 slots => 4C by per-position independence of P); no learned
  attack was retrained in this extension.
- Primary numbers come from the earlier same-task attempt's run; this agent
  verified provenance (bit-identical re-run) and reproduced all conclusions
  with independent code and disjoint seeds rather than discarding it.

## Files

- capacity_curve.py — main experiment (prior attempt's script; re-run, verified)
- results.json — primary results (regenerated, matches as-found)
- results_prior_run.json — as-found copy (identical except wall_time_s)
- verify_indep.py / verification.json — independent verification, fresh seeds
- capacity_curve.png — curve (linear + log), MC vs closed form
