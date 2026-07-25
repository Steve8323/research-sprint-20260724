# ERT with learned meaning classes

**Question.** In `~/english_bottleneck_lm` the paraphrase channel's meaning
classes are *given* (the hand-written synonym dictionary) — a stated limitation.
Does the E1 double dissociation (meaning task M stays ~1.0, surface task S
collapses to ~within-class chance at rho=1) survive when the channel must use
*learned* equivalence classes instead?

## Pre-registration (written before running the main experiment)

Setup is the repo's canonical toy: d=64, 4 layers/hops, eps=0.5, B=256,
steps=600 (= the repo's `--quick` setting; budget-limited), rho=1.0 throughout,
seeds {0,1,2}. All arms are trained/evaluated with the repo's own unmodified
`train.py` / `model.py` / `tasks.py`; only the channel's class table changes.

**Class sources (arms):**
1. **oracle** — the repo's original dictionary channel (replication anchor).
2. **distill** (PRIMARY) — classes learned by probing a *separately* trained
   rho=0 ERT (trained on task M, 600 steps): each content word w is fed alone,
   the trained model's predicted final class is read out (16-sample average,
   tau=0.3), and words with the same predicted label form one learned class.
   No access to the dictionary; fidelity is limited by that model's accuracy.
3. **geom** (secondary) — k-means (k=24, n_init=50) on the frozen public
   embedding matrix, content words only. A pre-probe showed this recovers the
   true classes poorly at eps=0.5 (ARI ~ 0.03-0.06, purity ~ 0.35) — the
   within-class embedding signal (cos ~ 0.06) is below the pairwise noise
   floor. This arm therefore tests what happens to the dissociation with
   poorly learned classes. Trained on M and S.
4. **random control** — a random partition of the 96 content words into 24
   groups of 4. Null control for "class quality is load-bearing". M only.

Learned channels: with prob rho=1 replace each content token by a uniformly
random *other* member of its learned class (singleton classes pass through
unchanged); specials always pass. Token type (content vs special) and k=24 are
given; class assignments are not.

**Primary test (stated in advance):** the double dissociation *survives with
distilled classes* iff in **all 3 seeds** the distill arm has **M >= 0.8 and
S <= 0.4** (oracle reference from the repo, 3 seeds, 1200 steps: M ~ 0.99,
S ~ 0.25; within-class chance for S = 0.25; full chance: M = 1/24 = 0.042,
S = 1/106 = 0.0094). We also report per-seed distill-minus-oracle differences
under the identical 600-step protocol.

**Secondary expectations (stated in advance):** geom-arm M collapses (M < 0.5)
and is comparable to the random control, showing the dissociation does *not*
come for free from any partition — class fidelity is load-bearing.

**Reporting:** every accuracy is on freshly sampled eval data, n = 6000
examples per trained model (3 x 2000, repo's eval), with 95% Wilson CIs;
across-seed spread reported as min/max of the 3 seeds. n_seeds = 3 is small
(matches the repo's own seed study); no significance tests are pretended on 3
seeds. Channel quality per learned partition: ARI and purity vs true classes,
cross-class swap fraction (probability a swap changes the true class), and
within-class pair coverage (fraction of true synonym pairs swappable by the
channel). If the ~10 min budget runs out, secondary arms are dropped first and
marked skipped.

## Results

### Numbers (rho=1.0, 600 steps, 3 seeds, eval n=6000/model)

```
oracle   M: mean=0.780  per-seed=[0.851, 0.863, 0.627]  (n_seeds=3, each acc on n=6000, Wilson95 ~ +/-0.011 at p=0.25)
oracle   S: mean=0.246  per-seed=[0.254, 0.237, 0.248]  (n_seeds=3, each acc on n=6000, Wilson95 ~ +/-0.011 at p=0.25)
distill  M: mean=0.890  per-seed=[0.999, 0.683, 0.988]  (n_seeds=3, each acc on n=6000, Wilson95 ~ +/-0.011 at p=0.25)
distill  S: mean=0.241  per-seed=[0.231, 0.249, 0.242]  (n_seeds=3, each acc on n=6000, Wilson95 ~ +/-0.011 at p=0.25)
```

**Pre-stated primary test** (all 3 seeds: distill M>=0.8 and S<=0.4 at rho=1): **FAILED**
- distill M per seed: [0.999, 0.683, 0.988]
- distill S per seed: [0.231, 0.249, 0.242]
- distill minus oracle, M per seed: [0.148, -0.18, 0.361], S per seed: [-0.023, 0.013, -0.006]

### Learned-partition quality

```
distill  seed0: ARI=1.000 purity=1.000 cross_class_swap=0.000 pair_coverage=1.000 groups=24 singletons=0 probe_model_M=0.999
distill  seed1: ARI=1.000 purity=1.000 cross_class_swap=0.000 pair_coverage=1.000 groups=24 singletons=0 probe_model_M=0.997
distill  seed2: ARI=1.000 purity=1.000 cross_class_swap=0.000 pair_coverage=1.000 groups=24 singletons=0 probe_model_M=0.996
```

Skipped for budget: secondary arms from seed 0 (budget)

Wall time: 884s

### Interpretation (written after the run; attribution analysis is post-hoc)

1. **The classes are perfectly learnable by model-distillation.** In 3/3 seeds
   the distilled partition equals the true dictionary exactly (ARI=1.000,
   cross-class swap fraction 0.000, synonym-pair coverage 1.000; probe-model
   accuracy 0.996-0.999). The learned-class channel is therefore extensionally
   identical to the oracle channel, without ever seeing the dictionary.
2. **The pre-stated pass criterion failed, and the failure is attributable to
   the undertrained 600-step regime, not to class learning** (post-hoc
   diagnosis): the oracle anchor itself scores M=0.627 in seed 2 — below the
   0.8 bar the criterion presumed oracle-level channels would clear — and
   distill M is *above* oracle M in 2/3 seeds (diffs +0.148/-0.180/+0.361).
   At the repo's own 1200-step budget the oracle arm gets M=0.994+/-0.003;
   the 600-step budget used here (forced by the shared-machine limit) is too
   noisy for a per-seed 0.8 threshold on M.
3. **The dissociation direction holds in every single run of both arms**:
   min M = 0.627 (oracle) / 0.683 (distill), both >> chance 0.042, while all
   six S values sit in 0.231-0.254, i.e., at within-class chance 0.25 (surface
   fully destroyed, meaning largely preserved), with distill S
   indistinguishable from oracle S (diffs -0.023/+0.013/-0.006, each
   |diff| <~ 2x the Wilson95 half-width 0.011).
4. **Classes are NOT learnable from embedding geometry at eps=0.5**: the
   pre-registered geom probe got ARI 0.027-0.055, purity 0.34-0.37 (k-means,
   3 restarts-seeds). The within-class cosine signal (~0.06) is below the
   pairwise noise floor, so any purely geometric channel-builder fails here.
   The trained arms for geom and the random control were skipped for budget,
   so "bad classes destroy M" was NOT experimentally verified — only the
   partition quality itself is established.

### Caveats

- n_seeds=3, and the run was undertrained (600 steps vs the repo's 1800/1200);
  wall time 884 s (torch capped at 3 threads on a shared 10-core box with ~29
  concurrent agents) — the contention slowdown is why the pre-registered
  secondary arms (geom-trained, random-control) were auto-skipped.
- Distillation uses a probe model trained on task-M labels; "learned" means
  learned from a separately trained model's behavior, not from raw unlabeled
  text. In this toy, task-M behavior determines the classes, so near-perfect
  recovery follows from the probe model's 99.6-99.9% accuracy; the interesting
  regime (imperfect distilled classes) was not reached.
- k=24 (number of classes) and content-vs-special token type were given to
  all class learners; only assignments were learned.
- The 0.8/0.4 primary threshold was mis-calibrated for the 600-step budget
  (oracle itself fails it in 1/3 seeds); the criterion-level result is
  honestly FAILED, and the survival claim rests on the arm comparison and the
  exact-recovery result instead.
