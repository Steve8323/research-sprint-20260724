# Routing to make worldview selectable

**Question.** In the moral_framework_misgen ALIGN setup (two equal-cost XOR cues that agree on
every training example: cue A = true objective, cue B = misaligned proxy; deployment flips the
proxy), standard training leaves the adopted worldview to the seed (published baseline, seeds
700-759: 12/60 aligned, 17/60 misaligned, 31/60 hedgers). Can gradient routing during training on
the SAME agreeing data make the worldview *selectable* post-hoc by ablation?

**Method.** Same architecture and hyperparameters as `alignment.py` (7-dim input, 2-hidden-layer
width-8 MLP, Adam lr 5e-3, wd 2e-3, 2000 full-batch steps, N=800, MARGIN=2.0, NOISE=0.6).
Hidden units are partitioned into two width-4 columns (units 0-3 = subnet A, 4-7 = subnet B;
cross-column hidden-to-hidden weights zeroed at init, and they receive zero gradient by
construction). Each training step computes two *worldview-consistent gradient components* from the
same agreeing batch: view A = batch with proxy-cue input columns replaced by fresh noise, forward
with subnet B's units ablated (so the gradient only reaches subnet-A parameters); view B =
symmetric. Loss = mean of the two views. No example ever shows the cues disagreeing with a label
-- the training signal remains worldview-symmetric; only the *localization* of the two components
is controlled. Post-training selection = zero the other column's hidden units.

Arms (30 seeds each, seeds 700-729, matching the published baseline seeds):
1. `routed` -- routing as above; designated mapping (ablate B -> aligned, ablate A -> misaligned).
2. `control` -- standard training exactly as `alignment.py` (replication check against
   `results_align.json` scores), then the same ablation applied post-hoc.
3. `occlusion_only` -- the two occluded views but NO routing (no unit ablation in forward, all
   parameters updated by both views): tests whether the data augmentation alone explains anything.
4. `routing_no_occlusion` -- unit-ablated forwards and block-diagonal structure but full
   (un-occluded) inputs: each column faces the underdetermined tie itself; tests whether the
   occlusion (component definition) is load-bearing.

**Pre-stated success criteria and primary test (written before running).**
- Deployment alignment = accuracy on a fixed deploy set (n=4000, gen seed 50001) where the proxy
  is flipped; train-dist accuracy on a fixed held-out agreeing set (n=4000, gen seed 60001).
- Per-seed selection success = [ablate-B model: alignment > 0.8 AND train-dist acc > 0.9] AND
  [ablate-A model: alignment < 0.2 AND train-dist acc > 0.9]. Both directions must work: the
  claim is *selectability*, not an alignment bias.
- For the three non-routed arms the mapping of columns to worldviews is arbitrary, so they are
  scored generously: success under the designated OR the swapped mapping counts.
- PRIMARY TEST: two-sided Fisher exact test on selection-success counts, `routed` vs `control`,
  n=30 vs n=30. Wilson 95% CIs on all success rates. Secondary (same test, labeled secondary):
  `routed` vs `occlusion_only` and `routed` vs `routing_no_occlusion`.
- Also reported (descriptive, not primary): full-network (un-ablated) deployment-alignment
  distribution per arm -- routing is expected to convert seed-committed models into hedgers,
  which is the cost of making the choice selectable.

**Result.**

| arm | selection success | Wilson 95% CI | full-model align hi/lo/mid |
|---|---|---|---|
| routed | **22/30 (73%)** | [0.556, 0.858] | 0 / 1 / 29 (hedged, mean 0.491 +/- 0.143) |
| control (standard) | **0/30** | [0.000, 0.114] | 7 / 4 / 19 (seed lottery) |
| occlusion_only | 0/30 | [0.000, 0.114] | 0 / 0 / 30 |
| routing_no_occlusion | 6/30 (20%) | [0.095, 0.373] | 5 / 8 / 17 |

- PRIMARY: routed 22/30 vs control 0/30, Fisher exact two-sided **p = 8.3e-10**.
- Secondary: vs occlusion_only p = 8.3e-10; vs routing_no_occlusion p = 7.3e-5.
- Selection is sharp when it works: median ablate-B alignment 0.997 (aligned), median ablate-A
  alignment 0.003 (misaligned), with train-dist accuracy of the surviving column ~0.99.
- Replication anchor: the control arm reproduces the published ALIGN baseline exactly
  (max |score diff| = 0.0000 vs `results_align.json`, seeds 700-729, same deploy set).
- Failure mode of the 8 routed misses: in every one, a single width-4 column got stuck at the
  classic partial-XOR optimum (train-dist acc ~0.74, or 0.51 in one seed) while the other column
  learned perfectly; 52/60 columns reached > 0.97. This is a column-trainability failure, not a
  routing leak (no failed seed showed mixed/leaked worldviews).
- POST-HOC (labeled, run after seeing the failure mode): same routed procedure with H=16
  (width-8 columns) -> **30/30** selection success, Wilson 95% [0.886, 1.0].
- Both losses of routing are visible and reported: the un-ablated routed model *hedges*
  (mean deploy alignment 0.491), so a selection step is mandatory before deployment; and
  `routing_no_occlusion` shows the occlusion is load-bearing (without it each column just
  re-runs the seed lottery privately; 6/30 ~ chance-level double-commitment).

**Interpretation (one line).** In this toy, the loss cannot pick the worldview from agreeing data,
but routing the two worldview-consistent gradient components into disjoint columns moves the
choice from the seed to a post-hoc, reversible ablation switch -- deferred selection instead of
chance selection.

**Caveats.**
- Toy scale (7-dim synthetic inputs, <=770-param MLPs), inheriting the parent repo's searched-for
  equal-cost construction; nothing here is evidence about frontier models.
- The router uses oracle concept localization: it must know which input columns carry which cue
  in order to build the occluded views (the analog of gradient routing needing example tags).
  In a real model you cannot occlude a "worldview" by zeroing input dims; this is the main
  obstacle to any scaled version.
- The occluded views change the input distribution (cue columns replaced by noise), though they
  never break the cue-label tie; the `occlusion_only` arm (0/30, all hedgers) shows the
  augmentation alone does not produce selectability.
- The routed net is architecturally slightly different from control (32 cross-block weights
  frozen at 0); `routing_no_occlusion` (6/30) controls for that structure alone.
- The H=16 (30/30) number is post-hoc and should be treated as an unconfirmed follow-up.
- n=30 seeds/arm, single fixed deploy and held-out sets (n=4000 each); thresholds
  (0.8 / 0.2 / 0.9) were pre-stated but are inherited conventions from the parent repo.
- Total runtime ~292 s on 2 CPU threads (232 s main + 59 s post-hoc).

**Files.** `experiment.py` (all four arms + tests), `results.json` (per-seed numbers, tests,
post-hoc block), `posthoc_h16.py` / `posthoc_h16.json` (regenerates the post-hoc H=16 block —
added in the audit pass; the original run had no saved code for it), `verify_rerun.py` /
`verify_rerun.json` (audit spot-check). Parent repos read-only:
`~/moral_framework_misgen/alignment.py` (data + regime),
`~/gradient_routing_experiment/harness.py` / `fix_experiment.py` (routing + ablation-aware idea).

**Audit (second pass, 2026-07-24 late afternoon).**
- Reproducibility spot-check (`verify_rerun.py`): seeds 700-702 x {routed, control} re-trained
  from scratch with the committed code; all 6 runs match `results.json` per-seed metrics with
  max |diff| = 0.000000 (training is deterministic given the seed).
- The post-hoc H=16 claim originally had no saved code. `posthoc_h16.py` was written in the audit
  pass and re-run: 30/30 selection success, and every per-seed metric matches the stored
  `posthoc_routed_H16` block exactly (max |diff| = 0.000000). It remains labeled post-hoc.
- Aggregates recomputed independently from `per_seed` rows: routed 22/30, control 0/30,
  Fisher two-sided p = 8.27e-10 — matches the stored `primary_test`.
- Failure-mode claim recomputed: all 8 routed misses have exactly one column with train-dist acc
  < 0.9 (0.74 in seven seeds, 0.51 in seed 722) while the other column is > 0.97; 52/60 columns
  reach > 0.97. In every miss the *working* column still selects its worldview cleanly (e.g.
  seed 709 ablate-A alignment 0.003); success required both directions, so these count as misses.
