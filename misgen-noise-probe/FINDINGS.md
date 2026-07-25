# Noise-probing the misgen toy

**Question.** Apply the activation-noise probe from `~/misalignment_noise_probe` (h -> h + N(0, (sigma*RMS(h))^2)) to the 154-param misgeneralization MLP from `~/moral_framework_misgen` (`stats/paper_stats.py` config). On inputs where the two cues *agree* — i.e. where aligned, misaligned, and hedger seeds are behaviorally indistinguishable — does the response to hidden-layer noise differentially surface the latent worldview class?

## Pre-stated analysis plan (written BEFORE running the experiment)

**Setup (exact replica of `stats/paper_stats.py`, read-only).** 2-hidden-layer width-8 MLP (7->8->8->2, 154 params), inputs = true-cue XOR block (2 dims, margin 2.0) + proxy XOR block (2 dims) + 3 noise dims, input noise 0.6, N=800 train, full-batch Adam lr 5e-3, wd 2e-3, 2000 epochs, seeds 700..819 (**120 seeds**, same seed family as the published 120-seed run).

**Class labels (ground truth).** Clean alignment index = accuracy on a fixed deploy set (n=4000, seed 50701, proxy flipped). aligned > 0.8, misaligned < 0.2, hedger otherwise — same thresholds as the repo.

**Noise probe.** Post-ReLU hidden activation h_l -> h_l + sigma*RMS(h_l)*eps, eps ~ N(0,I) iid per item and draw; RMS over all elements of the clean batch activation (the probe repo's RMS scaling). Grid: layer l in {1,2} x sigma in {0.1, 0.2, 0.3, 0.5, 1.0} x K=50 draws. **Primary condition fixed in advance: l=1 (first hidden layer), sigma=0.2** (the canonical dose used throughout the noise-probe repo). Everything off the primary condition is exploratory.

**Eval sets (fixed, shared by all seeds).** AGREE: n=4000, mode "train" (cues agree), fresh seed 60701 — behavior here does not reveal the worldview. DEPLOY: n=4000, seed 50701 (cues disagree) — used for class labels and for the surfacing tests.

**Per-seed measures.**
- AGREE flip rate: P(argmax flips under noise vs clean), mean over items x draws.
- DEPLOY per-draw alignment a_k; commitment C_noisy = mean_k |a_k - 0.5|; dC = C_noisy - |a_clean - 0.5|.
- Confound covariates (clean, no noise, AGREE only): mean logit margin, accuracy.

**Primary tests (all at l=1, sigma=0.2, two-sided, alpha=0.05, Holm-corrected across the four):**
- **P1** AGREE flip rate, aligned vs misaligned — Mann-Whitney U. *Built-in negative control: the two cue blocks are statistically exchangeable, so symmetry predicts null.*
- **P2** AGREE flip rate, committed (aligned + misaligned) vs hedger — Mann-Whitney U. *The main detection test: does noise response on agreement-only data reveal latent hedging?*
- **P3** dC for hedgers — Wilcoxon signed-rank vs 0. *Surfacing test: does noise push hedgers toward a commitment (dC > 0) rather than nowhere?*
- **P4** dC for committed seeds — Wilcoxon signed-rank vs 0. *Direction check: erosion toward 0.5 (dC < 0) is the trivial expectation.*

**Pre-stated exploratory analyses (labeled as such in results):** full l x sigma grid of P1/P2-style contrasts; AUROC (with 2000x seed bootstrap CI) of hedger-vs-committed detection from AGREE flip rate; the same AUROC from *clean* AGREE margin alone and clean AGREE accuracy alone (does noise add anything beyond a margin readout — the noise-probe repo's own gameability lesson); Spearman corr(flip rate, |a_clean - 0.5|); partial correlation of flip rate with class controlling for clean margin; hedger lean amplification corr(a_clean - 0.5, d_align under noise).

**Power/units.** Unit = seed (independent init + own training-set draw). Expected class sizes from the published run: ~31 aligned / 34 misaligned / 55 hedgers. Per-item flip probabilities saved to `peritem.npz` for every grid cell.

---

## Results (filled in after the run; the pre-stated section above is unchanged)

Run: 120 seeds, 116 s single-process CPU. **Class census: 31 aligned / 34 misaligned / 55 hedgers** — identical to the published 120-seed run; train acc 1.000 for all seeds.

### Primary tests (l=1, sigma=0.2, Holm over 4)

| Test | Contrast | Result | p raw | p Holm | Verdict |
|---|---|---|---|---|---|
| P1 (neg. control) | AGREE flip: aligned (n=31) vs misaligned (n=34) | 0.00110 vs 0.00094, AUROC 0.54 | 0.62 | 0.62 | **null, as symmetry predicts** |
| P2 (detection) | AGREE flip: committed (n=65) vs hedger (n=55) | 0.00102 vs 0.00059, AUROC 0.71 | 6.5e-5 | **1.3e-4** | **significant — but committed seeds flip MORE, not hedgers** |
| P3 (surfacing) | hedger dC vs 0 | mean **-0.0289** [-0.0334, -0.0246] | 2.1e-10 | 6.4e-10 | **significant in the ANTI-surfacing direction: noise erodes hedger leans toward 0.5** |
| P4 (direction) | committed dC vs 0 | mean -0.0407 [-0.0453, -0.0359] | 2.4e-12 | 9.6e-12 | erosion toward 0.5, the trivial direction |

Per-class AGREE flip rates (mean, seed-bootstrap 95% CI): aligned 0.00110 [0.00082, 0.00144], misaligned 0.00094 [0.00073, 0.00119], hedger 0.00059 [0.00051, 0.00068].

### Key exploratory results (all labeled post-hoc)

- **Uniform shrinkage, not surfacing.** At the primary dose, noisy alignment is a near-deterministic contrast shrinkage of clean alignment: a_noisy - 0.5 = 0.898 (a_clean - 0.5), r = 0.9975 across all 120 seeds (hedgers alone: slope 0.80, r = 0.9993, residual sd 0.005). Hedger "lean amplification" is rho = **-0.99** (p = 1.3e-46) — noise moves every hedger toward 0.5 in exact proportion to its lean, revealing no idiosyncratic latent commitment. Aligned and misaligned seeds erode symmetrically at every grid cell (d_align -0.037 vs +0.044 at the primary dose): no differential surfacing of misalignment anywhere.
- **A noise-free readout beats the noise probe as a hedger detector.** Hedger-vs-committed AUROC: clean AGREE logit margin **0.80** [0.72, 0.88] (hedgers have smaller margins) vs noise flip rate 0.71 [0.62, 0.80] (as committed>hedger). Clean AGREE accuracy is uninformative (0.44 [0.38, 0.49]). Flip rate and margin are positively related (Spearman rho = 0.38); partialling margin out of the flip-rate/class relation leaves a marginal residual (partial rank rho = -0.21, p = 0.019, uncorrected, post-hoc) — the noise probe carries at most a sliver of information beyond the clean margin, and less than the margin itself.
- **Direction is dose-fragile.** The committed>hedger flip-rate gap replicates at 8/10 grid cells (all p < 0.005); at l=1 sigma=0.5 it vanishes (p = 0.98) and at l=1 sigma=1.0 it reverses (AUROC 0.35, hedgers flip more). The primary-condition direction is not a dose-robust signature.
- Mechanistic reading (post-hoc, untested): on agreement inputs both cues push the same way, so a hedger that pools two redundant cues has a larger effective signal-to-noise than a committed seed reading one cue — which would explain why committed seeds are more noise-fragile despite larger raw logit margins, and why the raw margin is the better detector.

### Conclusion

In this cheapest-possible testbed the noise-probe hypothesis fails in its interesting forms and succeeds only in a trivial one. Noise response on ambiguous (cue-agreeing) inputs cannot distinguish aligned from misaligned seeds (P1 null, as the construction's symmetry demands), does not collapse hedgers into a latent commitment (P3 significant in the opposite direction; noisy alignment is a uniform 0.9x shrinkage toward chance with r = 0.9975), and the one real class signal it does carry — committed seeds flip more than hedgers (P2, AUROC 0.71) — is dose-fragile, directionally opposite at high sigma, and dominated by a noise-free clean-margin readout of the same inputs (AUROC 0.80). The toy-scale verdict matches the noise-probe repo's own hard-won lesson: the noise response is essentially a re-read of the clean decision geometry, not a window onto the latent worldview.

### Caveats

- P1's null is expected by construction (the two cue blocks are statistically exchangeable); it validates the pipeline but does not transfer to settings where aligned/misaligned is not a symmetry (e.g. real LLMs).
- Nothing here was trained to conceal: the toy has no analog of a deceptive/concealed stance, so this tests only the weaker claim "noise response on innocuous inputs carries latent-worldview information beyond clean outputs" — which is the claim that fails.
- Absolute flip rates at the primary dose are tiny (~1e-3); per-seed estimates rest on 200,000 Bernoulli draws (SE ~7e-5), so they are well resolved, but the regime is far from the LLM probe's operating point where noise visibly degrades behavior.
- Class thresholds (0.8/0.2) inherited from the repo; the mechanistic reading and everything under "exploratory" is post-hoc; the primary condition (l=1, sigma=0.2) was fixed in advance but sigma=0.2 was borrowed from the LLM repo, not tuned to this toy.
- Single architecture and a single, deliberately-searched-for task construction (the repo's equal-cost regime).

## Files

- `probe.py` — full experiment (training, probing, stats; 116 s single process).
- `results.json` — config, census, all tests, per-seed table, full grid, post-hoc shrinkage block.
- `peritem.npz` — per-item flip probabilities for every seed x layer x sigma (AGREE and DEPLOY) + per-item clean correctness.
- `run.log` — console output of the run.

