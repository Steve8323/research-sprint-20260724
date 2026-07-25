# Honest bimodality survey (misgen-bimodality-honest)

## Question

The published bimodal aligned/misaligned split in ~/moral_framework_misgen came from a
regime explicitly searched for bimodality (harder_core.py scores candidate widths by
min(nA, nB); harder_c/sweep.py scores a Sarle bimodality coefficient). If we sweep a
grid of cue-cost x width regimes WITHOUT selecting, what fraction of regimes is genuinely
non-unimodal (bimodal/multimodal) vs graded?

## Pre-registered design (written BEFORE running the grid)

Setup: exact replication of the headline stats/paper_stats.py recipe (2-bit XOR true cue
+ 2-bit XOR proxy cue + 3 noise dims, true-cue margin 2.0, noise 0.6, N=800, 2-hidden-layer
MLP, Adam lr 5e-3, wd 2e-3, 2000 full-batch epochs; score = deployment accuracy on 4000
held-out points where the proxy flips). Two regime knobs:

- Proxy cost ratio c in {0.5, 0.75, 1.0, 1.5, 2.0}: proxy block margin = 2.0 * c
  (c<1 proxy is lower-SNR/harder, c=1 equal cost = the published regime, c>1 proxy easier).
- Width W in {2, 4, 8, 16, 32}. (W=8, c=1.0) is a fresh-seed replication of the
  published cell.

Grid = 25 regimes, all reported, none dropped. Fresh seeds per regime
(3_000_000 + r*10_000 + s), disjoint from all published seed families (700+, 1000+, 2000+).

Seeds per regime: n = 100 if the timing benchmark (speed only, no score distributions
inspected) fits the 10-minute budget, else n = 60. [Resolved by the speed-only benchmark before the main run: n = 100 on MPS
(batched training is dispatch-overhead-bound; full grid ~5-6 min).]

PRIMARY TEST (pre-stated): Hartigan dip test on the n deployment-accuracy scores of each
regime (table-interpolated p, same test as stats/paper_stats.py), with Holm-Bonferroni
correction across the 25 regimes at family alpha = 0.05. A regime "rejects unimodality"
only if it survives Holm. Headline number = count (and fraction) of the 25 regimes that
reject unimodality after correction.

Sensitivity check (pre-stated): bootstrap dip p-value (10,000 draws, fixed seed) per
regime; report if it changes any Holm verdict.

Validity flag (pre-stated): the loss-equivalence framing requires fitting the training
data, so regimes with mean train acc < 0.95 are flagged "underfit" (still counted in the
Holm family; flagged in the table).

Secondary, DESCRIPTIVE ONLY (labeled post-hoc): per-regime GMM BIC (k=1,2,3), counts
nA (score>0.8), nB (score<0.2), mid, and a 5x5 histogram figure.

Power caveat (pre-stated): the published headline dip was p=0.016 at n=120 - marginal.
At n=100 per regime plus a 25-way correction, this survey has limited power; failure to
reject unimodality is NOT proof of unimodality. The replication cell (W=8, c=1.0) is the
internal yardstick: if even it does not survive, that says the published effect size is
too small to survive honest multiplicity, not that the mid-mode structure is absent.

Implementation note: seeds are trained in a batched tensor implementation (one process,
sum of per-seed full-batch losses; Adam is elementwise, so gradients and updates are
identical to per-seed training). Smoke test = train-acc and score-spread replication check
of the published cell before the grid.

## Result

Grid: 25 regimes x n=100 fresh seeds = 2500 models, runtime 160.8s (single process, MPS).

PRIMARY RESULT: 9/25 regimes reject unimodality after Holm-Bonferroni (family alpha=0.05); 11/25 at raw p<0.05. Bootstrap-p sensitivity check: 9/25 after Holm.
Holm-significant regimes: (c=0.5, W=2, p=0.0000), (c=0.75, W=2, p=0.0000), (c=1.0, W=2, p=0.0008), (c=1.5, W=2, p=0.0000), (c=2.0, W=2, p=0.0000), (c=0.5, W=8, p=0.0007), (c=1.0, W=8, p=0.0015), (c=0.5, W=16, p=0.0000), (c=0.5, W=32, p=0.0004)

Replication cell (c=1.0, W=8, the published regime, fresh seeds): dip p=0.0015 (boot 0.0008), nA/nB/mid = 27/27/46, best GMM k=3, train acc 1.000. Published: p=0.0156 at n=120 (verified by recomputation on the stored scores).

Per-regime table (cost, width, mean train acc, nA/nB/mid, dip p, boot p, Holm):

| c | W | train acc | nA/nB/mid | dip p | boot p | Holm | note |
|---|---|-----------|-----------|-------|--------|------|------|
| 0.5 | 2 | 0.853 | 50/0/50 | 0.0000 | 0.0000 | YES | UNDERFIT |
| 0.75 | 2 | 0.864 | 41/3/56 | 0.0000 | 0.0000 | YES | UNDERFIT |
| 1.0 | 2 | 0.848 | 16/7/77 | 0.0008 | 0.0005 | YES | UNDERFIT |
| 1.5 | 2 | 0.856 | 1/40/59 | 0.0000 | 0.0000 | YES | UNDERFIT |
| 2.0 | 2 | 0.853 | 7/33/60 | 0.0000 | 0.0000 | YES | UNDERFIT |
| 0.5 | 4 | 0.985 | 94/0/6 | 0.0084 | 0.0098 | no |  |
| 0.75 | 4 | 0.996 | 84/1/15 | 0.9922 | 0.9967 | no |  |
| 1.0 | 4 | 0.988 | 18/35/47 | 0.0238 | 0.0230 | no |  |
| 1.5 | 4 | 1.000 | 4/92/4 | 0.9950 | 0.9993 | no |  |
| 2.0 | 4 | 0.988 | 0/93/7 | 0.9890 | 0.9872 | no |  |
| 0.5 | 8 | 1.000 | 100/0/0 | 0.0007 | 0.0006 | YES |  |
| 0.75 | 8 | 1.000 | 96/0/4 | 0.7904 | 0.7907 | no |  |
| 1.0 | 8 | 1.000 | 27/27/46 | 0.0015 | 0.0008 | YES | replication cell |
| 1.5 | 8 | 1.000 | 0/100/0 | 1.0000 | 1.0000 | no |  |
| 2.0 | 8 | 1.000 | 0/100/0 | 1.0000 | 1.0000 | no |  |
| 0.5 | 16 | 1.000 | 100/0/0 | 0.0000 | 0.0000 | YES |  |
| 0.75 | 16 | 1.000 | 100/0/0 | 0.6665 | 0.6638 | no |  |
| 1.0 | 16 | 1.000 | 3/5/92 | 0.9907 | 0.9934 | no |  |
| 1.5 | 16 | 1.000 | 0/100/0 | 1.0000 | 1.0000 | no |  |
| 2.0 | 16 | 1.000 | 0/100/0 | 1.0000 | 1.0000 | no |  |
| 0.5 | 32 | 1.000 | 100/0/0 | 0.0004 | 0.0004 | YES |  |
| 0.75 | 32 | 1.000 | 100/0/0 | 0.5945 | 0.5844 | no |  |
| 1.0 | 32 | 1.000 | 1/0/99 | 0.9441 | 0.9417 | no |  |
| 1.5 | 32 | 1.000 | 0/100/0 | 1.0000 | 1.0000 | no |  |
| 2.0 | 32 | 1.000 | 0/100/0 | 1.0000 | 1.0000 | no |  |

Figure: grid_histograms.png (5x5 score histograms, Holm-significant in red).

## Post-hoc decomposition of the 9 Holm rejections (LABELED POST-HOC)

The 9 Holm-significant cells are not one phenomenon. A tie-breaking jitter sensitivity
check (uniform +-1/8000, i.e. below the 1/4000 deploy-accuracy resolution; 50 draws,
seed 777; posthoc_jitter.json) splits them three ways:

1. 3/9 are discreteness artifacts, not worldview splits: (c=0.5, W=8/16/32) have ALL
   100 seeds aligned (nA=100, nB=0) with total score range 0.0033-0.0071 - every seed
   sits at ~0.998. The dip fires on tie structure of near-ceiling quantized accuracies;
   under jitter the median p becomes 0.44-0.72 (vs <=0.0007 raw). No aligned/misaligned
   structure whatsoever.
2. 5/9 are the underfit W=2 row (train acc 0.848-0.864, pre-flagged): genuinely spread
   multimodality (jitter-robust, range 0.70-0.99), but these regimes violate the
   loss-equivalence premise (models do not fit the training data), so seeds differ in
   loss, not only in worldview.
3. 1/9 is the published cell (c=1.0, W=8): jitter-robust p=0.0015 with a substantive
   27 aligned / 27 misaligned / 46 hedger split - a genuine fresh-seed replication.

So among the 20 well-fit regimes, exactly 1 (5%) - the published equal-cost width-8
cell itself - shows jitter-robust aligned-vs-misaligned bimodality after Holm. The rest
of the grid: cost-imbalanced regimes (c=0.75, 1.5, 2.0) collapse to deterministic
capture by the cheaper cue (nA or nB ~ 100, dip p ~ 1); equal-cost at W=4 shows the
same 3-way split (18/35/47) at raw p=0.024 but does not survive Holm; equal-cost at
W=16/32 is hedger-dominated and unimodal (92-99 mid seeds, p>0.94).

Honest summary: the published bimodality replicates out-of-sample and survives 25-way
multiplicity correction - the earlier "post-selection, marginal p=0.016" worry is
resolved in the effect's favor for that cell - but it is a knife-edge regime: it needs
equal cue cost AND the width-8 capacity bottleneck, and it is the only well-fit cell in
the grid where the aligned/misaligned split passes an honest corrected test.

## Caveats

- The 25 regimes are a fixed grid census, not a random sample of "regimes"; 9/25 and
  1/20 are exact counts over this grid, and no CI across regimes is meaningful. The
  grid brackets the published cell by 4x in cost ratio and 16x in width but is still
  a choice.
- Power: n=100 seeds/regime plus a 25-way Holm correction. Non-rejection is not
  evidence of unimodality; (c=1.0, W=4) at raw p=0.024 with an 18/35/47 split is
  plausibly a true positive that the correction eats. The fraction of genuinely
  non-unimodal regimes is therefore a lower bound.
- The jitter decomposition and the GMM/nA/nB counts are post-hoc and descriptive;
  only the Holm-corrected dip count was pre-registered. The dip table p-values assume
  continuous data; with 1/4000-quantized scores they are anti-conservative under heavy
  ties (that is exactly what the jitter check exposes in the 3 artifact cells).
- Cost ratio is operationalized as proxy-cue margin scaling (SNR); other cost notions
  (sparsity, depth of computation) were not surveyed.
- Same toy family as the published repo (7-dim XOR cues, 2-layer MLP, Adam full batch,
  2000 epochs); this survey says nothing about realistic tasks, where the repo's own
  realdata results already show graded continua.
- Batched multi-seed training shares one Adam trajectory per stacked tensor; because
  the loss is a sum of per-seed means and Adam is elementwise, updates are exactly
  per-seed - verified conceptually, and empirically the replication cell reproduces
  the published spread (train acc 1.000, scores spanning 0.01-0.99).
- Runtime 161 s on MPS, single process; seeds 3_000_000+ disjoint from all published
  seed families.
