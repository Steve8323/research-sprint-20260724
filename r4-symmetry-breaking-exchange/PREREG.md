# PREREG — Exchange rate: how much supervision buys localization? (round 4)

Stamped: **Sat Jul 25 11:11:43 EDT 2026** (`date` output, before any confirmatory run)

## Background (read-only prior rounds)
In the `moral_framework_misgen` ALIGN toy, two equal-cost XOR cues agree on every training example:
cue A (true objective) and cue B (proxy that flips at deployment); 3 noise dims. Round-1/2 showed
that ROUTING the two worldview-consistent gradient components into disjoint hidden columns makes the
worldview post-hoc selectable by ablation (22-23/30 vs 0/30 standard), given a correct localization
of the two cue subspaces. Round-3 RETRACTED the "oracle-free localization" claim: round-2's
localizer only works in a coordinate system where the two concepts are already axis-separated.
With a rotation-equivariant localizer, selectability collapses to a coin flip in EVERY basis
(rot_ica 16/30, rot_dict 8/30 vs rot_oracle 26/30, rot_control 0/30), because at exactly equal cue
strength the labelled training distribution has an orthogonal symmetry group (|G|=64) whose orbit on
the true split has exactly two members (`true` and `entangled`) — an exact identifiability
obstruction. Round-3 also showed routing itself is basis-free given the true subspaces.

**This round.** Localization at the symmetric point is obstructed. Quantify the CHEAPEST information
that breaks the symmetry, and convert it into an exchange rate: m examples -> localization accuracy
-> end-to-end selectability. Everything runs in the `rot` (Haar-orthogonal, per-seed) basis, the
regime where unsupervised localization fails; the axis basis is oracle-privileged and is not used.

## Symmetry-breaking sources priced (all in the same currency: m examples)
- **(a) DEP-LAB — m labelled deployment examples** (cue A and cue B disagree there). PRIMARY source.
- **(a') DEP-UNLAB — m UNLABELLED deployment inputs** (secondary; cheaper still).
- **(b) CONCEPT-LAB — m training inputs annotated with an external concept label**: the latent first
  bit of the cue-B block (an internal-structure annotation an interpreter could supply). Secondary.
- **(c) ASYM — cue-strength asymmetry alone, m=0**: proxy margin ratio rho in {1.0,1.25,1.5,2.05,3.0}.
  Round-3 proved |G| drops to 16 and the orbit collapses to ONE member already at rho=2.05, so
  identifiability is restored in principle; round-3's ICA still coin-flipped. Secondary.

## The localizer L-SUP(m) — one family, supervision is the only thing that changes
Round-3's rotation-equivariant `dict` localizer, verbatim in structure: learn 4 orthonormal atoms
U (QR-parametrised, Adam lr 0.05, 400 steps) so that BOTH atom-pairs independently predict the
training label through a 2-D quadratic readout [p,q,p^2,q^2,pq]. **The only change** is an added
supervised term with fixed weight lambda = 1.0 (chosen a priori, NOT tuned; same at every m):
- (a) DEP-LAB:   + lambda * [ BCE(head1(x_dep), y_dep) + BCE(head2(x_dep), 1 - y_dep) ]
- (a') DEP-UNLAB:+ lambda * mean( -log P(head1 and head2 disagree) )  (no labels used)
- (b) CONCEPT-LAB:+ lambda * BCE(linear readout on atom-pair 2 of the m annotated inputs, latent bit)
- (c) ASYM: no supervised term at all (m=0); only the data-generating margin changes.
At m=0 with no supervised term L-SUP is EXACTLY round-3's `dict` localizer — the preregistered
unsupervised floor. Deployment/annotation examples are drawn fresh per seed (gen seed 80000+seed),
are disjoint from the evaluation deployment set (gen seed 50001), and are transformed by the same T.
Head-1 is designated as the subspace that must MATCH deployment labels, so labelled supervision also
picks the switch POSITION, not just the split; this is scored separately (see below).

## Arms
Basis = `rot` throughout. Seeds 700-729 (n=30) for every arm, matching rounds 1-3.
m grid **M = {0, 1, 2, 4, 8, 16, 32}** (stated before running; 4 examples = 0.5% of N=800, the
round-1 prevention dose).

ROUTED (localize -> routed training -> ablation scoring; identical machinery to round-3 `core.py`):
1. `sup_dep[m]`, m in M — routing with L-SUP(m) from labelled deployment examples. **ARM OF INTEREST.**
2. `rot_oracle` — true cue subspaces, exact conjugated occlusion. CEILING (must reproduce r3 26/30).
3. `rot_control` — standard training + post-hoc ablation. FLOOR (must reproduce r3 0/30).
4. `unsup_ica` — round-3's best unsupervised localizer (must reproduce r3 16/30). Second floor, so
   the m=0 baseline cannot be dismissed as a deliberately weak localizer.
5. `augment[m]`, m in M\{0} — **the alternative use of the same m examples**: standard (unrouted)
   training on the 800 training examples PLUS the m deployment examples. Outcome = "aligned"
   (deploy align > 0.8 AND held-out agreeing acc > 0.9). This prices the switch against just
   fixing the model with the same data.

LOCALIZATION-ONLY (no routed training; outcome = lands-true rate):
6. `unlab_dep[m]`, m in M — source (a').
7. `concept_lab[m]`, m in M\{0} — source (b).
8. `asym[rho]`, rho in {1.0,1.25,1.5,2.05,3.0} at m=0 — source (c).
9. `agree_ctrl[m]`, m in M\{0} — **NEGATIVE CONTROL**: the same supervised objective and the same
   lambda, but the m extra examples come from the AGREEING training distribution instead of
   deployment. Tests that the information comes from disagreement, not from m more examples.

## Outcomes
- **Localization accuracy** = fraction of seeds landing the TRUE split: basis-free subspace fidelity
  `split_fidelity(est, true) > 0.95` (round-3's exact definition and code, `core.orbit_subspaces`).
  Also reported: lands-entangled, lands-neither, mean fid_true.
- **Selectability** = round-1/2/3 success criterion verbatim: [keep-A model: deploy align > 0.8 and
  held-out agreeing acc > 0.9] AND [keep-B model: deploy align < 0.2 and held-out acc > 0.9], under
  the designated OR swapped mapping (`success_either`) — the metric all prior rounds used.
- **Position identification** (secondary): `success_designated` only, i.e. the arm also got the
  aligned subspace into subnet A. Meaningful only for labelled supervision (m>=1).

## PRIMARY tests and decision criterion
- **PRIMARY 1 (localization vs m):** Cochran-Armitage trend test, scores t(m)=log2(1+m), on
  `sup_dep` lands-true across the 7 m-levels; one-sided (increasing), permutation p (20,000
  seed-label permutations), alpha = 0.05.
- **PRIMARY 2 (selectability vs m):** identical trend test on `sup_dep` success_either.
- **DELIVERABLE k*** (the sentence the paper needs):
  - k*_sel = smallest m in M whose selectability Wilson-95 LOWER bound >= 0.50 (needs >= 21/30).
  - k*_loc = smallest m in M whose lands-true Wilson-95 LOWER bound >= 0.75 (needs >= 27/30).
  Report "none in M" if no level qualifies.
- **DECISION: SUPPORTED** ("cheap supervision buys a working switch") iff PRIMARY 1 p<0.05 AND
  PRIMARY 2 p<0.05 AND k*_sel exists AND k*_sel <= 8. **NARROWED** if the trends hold but
  k*_sel in {16,32}. **HONEST NEGATIVE** if either trend fails or no m in M reaches the bar.
- Power (simulated before running, n=30/level, 7 levels, alpha=0.05 one-sided): trend test type-I
  0.050; power 0.62 against a weak rise 0.27->0.45, 0.97 against 0.27->0.60, 1.00 against
  0.27->0.87 and against a step at m=16. Per-level Fisher vs m=0 at Bonferroni alpha=0.0083 has
  power 0.82 only against a jump to 0.75 and 0.24 against 0.55 — so per-level Fishers are
  SECONDARY/descriptive and the trend test is primary.

## Secondary tests (labelled secondary)
- S1. Fisher `sup_dep[m]` vs `augment[m]` at each m (Bonferroni over 6, alpha=0.0083): does the
  same m examples buy more as a switch or as a fix?
- S2. Trend tests for `unlab_dep`, `concept_lab`, `agree_ctrl` (localization only), same procedure.
- S3. `asym` lands-true vs rho: exact binomial test vs 0.5 at each rho and a trend test over rho.
- S4. success_designated for sup_dep across m (does supervision also pick the switch POSITION?).
- S5. Fisher `sup_dep[m*]` vs `rot_oracle` at the smallest passing m (is it at ceiling?).
- **Preregistered contingency:** if `unlab_dep` reaches lands-true >= 0.75 at some m <= 8, the routed
  arm at that single m (n=30) will additionally be run and reported as a preregistered contingency,
  not as post-hoc.

## Nulls and equivalence (round-3's methodological failure, explicitly guarded)
Every reported null carries either (i) its exact power against the preregistered effect size
delta = 0.20 in proportion difference, or (ii) a TOST equivalence bound at margin delta = 0.20.
No "no difference" claim will be made from a bare non-significant Fisher.

## Controls and anchors
- Positive control: `rot_oracle` must reproduce round-3's 26/30 (same seeds, same code path).
- Negative controls: `rot_control` must reproduce 0/30; `agree_ctrl` must NOT localize.
- Replication anchors: `unsup_ica` must reproduce round-3's rot_ica 16/30, and `sup_dep[0]` must
  reproduce round-3's rot_dict 8/30 (identical localizer at m=0).
- The latent-bit data generator used for source (b) is asserted bit-identical to `core.gen_align`.
- Reproducibility spot-check on 3 seeds re-run from scratch (max |diff| reported).

## Compute budget
Single process, torch 2 threads, CPU. Routed run ~1.0 s, localizer ~0.15 s (measured). Estimate
~620 s total. **Downscope rule if over budget:** drop `augment` at m in {16,32}, then drop
`asym` — stated here, and any downscope will be reported in FINDINGS.md.

## Anything not listed above is POST-HOC and will be labelled as such in FINDINGS.md.

---

# AMENDMENT 1 — stamped Sat Jul 25 11:15:09 EDT 2026 (still BEFORE any confirmatory run)

## What changed and why
A pre-run smoke test (`explore_sub.py`, 4 seeds, **fidelity only, no selectability measured** — the
same exploratory protocol round-3 used to pick its localizers: "selected on exploratory 4-seed
*fidelity only*") confirmed a degeneracy round-3 had **already documented independently of this
round**: the `dict` localizer does not land on either symmetry-orbit member — round-3 reported it
lands NEITHER on 21/30 seeds, and the smoke test reproduces this (fid_true 0.44, fid_entangled 0.68
at m=0 on seed 700; supervision moves it only to fid_true ~0.71). On a localizer that never lands on
an orbit member, the preregistered outcome "lands the TRUE split (fidelity > 0.95)" is close to
undefined, and an exchange rate measured through it would be uninterpretable.

**Amendment:** the PRIMARY localizer family becomes **L-SUP-S**, identical in every respect except
that the 4 atoms are searched *inside the unsupervised signal subspace* (range of the
class-conditional second-moment difference, whitened) instead of the full 7-D input space. That
signal-subspace step is not new and is not supervision: it is the first stage of round-3's own
PRIMARY localizer (`ica`), it uses only training data and training labels, and it is
rotation-equivariant. In the smoke test L-SUP-S lands crisply on orbit members (fidelities 0.998 or
0.51, never intermediate), so "lands true" is a clean binary, and it shows the dose-response the
primary tests are designed to measure.

**The originally preregistered family is NOT dropped.** L-SUP-D (dict-based, exactly as written
above) is retained and reported in full as a preregistered SECONDARY, localization-only, at
m in {0,1,2,4,8,32}.

**Nothing outcome-relevant was seen before this amendment:** no selectability, no routed run, and no
localization measurement on more than 4 seeds, all with the ORIGINAL (dict) family except the 4-seed
fidelity probe of the new family. Both families' full curves are reported.

## Downscope (stated before running, per the budget rule)
Measured costs: routed/standard training 1.0 s per run, localizer 0.45 s per call. To stay inside
~15 min single-process the secondary grids are thinned (the PRIMARY grid M = {0,1,2,4,8,16,32} is
untouched):
- `augment` at m in {1,2,4,8,32} (m=16 dropped; it is bracketed by 8 and 32).
- `unlab_dep`, `L-SUP-D` at m in {0,1,2,4,8,32}; `concept_lab` at m in {1,2,4,8,32}.
- `agree_ctrl` (negative control) at m in {1,4,32}; `asym` at rho in {1.0,1.5,2.05,3.0}.
Primary run (`experiment.py`) and secondary run (`sources.py`) are separate scripts, primary first,
so a budget overrun can only cost the secondaries.
