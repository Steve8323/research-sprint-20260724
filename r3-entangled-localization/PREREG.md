# PREREG — Does the worldview switch survive an ENTANGLED input basis? (round 3)

Stamped: **Sat Jul 25 10:19:06 EDT 2026** (`date` output, before any confirmatory run)

## Background (read-only prior rounds)
In the `moral_framework_misgen` ALIGN toy, two equal-cost XOR cues agree on every training example:
cue A (input columns [0,1], true objective) and cue B (columns [2,3], proxy that flips at deploy);
columns [4,5,6] are noise. Standard training = seed lottery (0/30 selectable). Round-1
(`routing-worldview-select/`) showed that routing the two worldview-consistent gradient components
into disjoint hidden columns — built by *occluding* the other cue's input columns — makes the
worldview post-hoc selectable by ablation (22/30, oracle cue locations). Round-2
(`r2-routing-switch-learned/`) removed the oracle with a **data-only XOR-pair probe over input
columns** (23/30, localization 30/30 exact).

**The weak point this round attacks:** round-2's localizer searches over *coordinate pairs*. It only
works because each cue occupies a disjoint, axis-aligned pair of input columns. Real concepts are
not axis-aligned. Here we destroy axis-alignment and ask whether localization + routing still gives
a switch.

## Manipulation
Every input (train, held-out, deploy) is mapped x → T x, with T drawn per seed:
- `axis`: T = I (control basis, reproduces rounds 1–2).
- `rot`: T = Haar-random orthogonal 7×7 (cues live in two random, mutually orthogonal 2-D planes).
- `mix`: T = random dense non-orthogonal 7×7, SVD-rescaled to condition number exactly 3.0 and
  Frobenius-normalised to √7 (so the test is about **non-orthogonality**, not conditioning).
Nothing else changes: same data generator, same 7→8→8→2 MLP, same Adam(5e-3, wd 2e-3), 2000
full-batch steps, N=800, same deploy set (n=4000, gen seed 50001), same held-out agreeing set
(n=4000, gen seed 60001), same success thresholds (0.8 / 0.2 / 0.9), same seeds 700–729 (n=30).

## Routing in a LEARNED BASIS (not coordinates)
A localizer returns an orthonormal 4-D signal basis `Vs` plus an invertible 4×4 source map `G`
(sources s = (x−μ)Vs G) split into two index pairs. Occlusion of one cue = replace its two source
coordinates with fresh Gaussian noise and reconstruct, leaving the orthogonal residual untouched.
Replacement scale is data-estimated as std(|s|) (for a well-separated bimodal source ±m+ε this
equals std(ε); it reproduces NOISE=0.6 exactly in the axis basis — no oracle constant).
The `oracle` arms instead use the exactly faithful conjugated occlusion T∘occlude_cols∘T⁻¹.

## Localizers (both rotation-equivariant; selected on exploratory 4-seed *fidelity only*)
- **PRIMARY `ica`** — signal subspace = range of the class-conditional second-moment difference
  E[xxᵀ|y=1]−E[xxᵀ|y=0] (top-4 |eigenvalues|); whiten inside it; FastICA with the cubic
  (kurtosis) contrast to recover 4 independent non-Gaussian source directions; pair the 4 sources
  into two 2-D subspaces by the max–min 2-D **quadratic** label probe [p,q,p²,q²,pq]
  (rotation-covariant within a pair).
- **SECONDARY `dict`** — a small learned dictionary: 4 orthonormal atoms U (QR-parametrised) trained
  by Adam so that *both* atom-pairs independently predict the label through a 2-D quadratic readout.
- **REFERENCE `pair`** — round-2's localizer verbatim in spirit (top-2 disjoint *column* pairs by
  XOR probe). Coordinate-dependent by construction; included to show what naive transfer does.

## Arms (n = 30 seeds each, 700–729)
| arm | basis | localizer | training |
|---|---|---|---|
| `axis_pair` | I | pair | routed — **round-2 anchor through this code path (expect ≈23/30)** |
| `axis_ica` | I | ica | routed — equivariant localizer *without* rotation |
| `rot_pair` | rot | pair | routed — naive transfer of round-2's localizer |
| **`rot_ica`** | rot | ica | routed — **ARM OF INTEREST** |
| `rot_dict` | rot | dict | routed — second localizer |
| `rot_oracle` | rot | true subspaces | routed — routing ceiling under rotation |
| `rot_control` | rot | — | standard training, post-hoc ablation — **floor** |
| `mix_ica` | mix | ica | routed |
| `mix_oracle` | mix | true subspaces | routed |
| `mix_control` | mix | — | standard training — floor |

Per-seed success (identical to rounds 1–2): [keep-A model: deploy-align > 0.8 AND held-out acc > 0.9]
AND [keep-B model: deploy-align < 0.2 AND held-out acc > 0.9], scored under the designated **OR**
swapped mapping for every arm (which learned cue is "aligned" is unidentifiable from agreeing data).

## PRIMARY test + decision criterion
**PRIMARY:** two-sided Fisher exact on selection-success counts, `rot_ica` vs `rot_control` (30 vs 30).
- **SWITCH SURVIVES UNDER ROTATION** iff p < 0.01 **AND** rot_ica ≥ 10/30.
- Otherwise **SWITCH DESTROYED BY ENTANGLEMENT** (a clean negative; it will be reported as such and
  NOT rescued post-hoc).

**SECONDARY 1 (degraded vs intact):** two-sided Fisher, `rot_ica` vs `axis_pair`. p < 0.05 with
rot_ica < axis_pair ⇒ "degraded"; p ≥ 0.05 ⇒ "not detectably degraded".
**SECONDARY 2:** Fisher `rot_ica` vs `rot_oracle` — separates *localization* failure from *routing*
failure under rotation.
**SECONDARY 3:** Fisher `rot_dict` vs `rot_control`; `rot_pair` vs `rot_control`; `axis_ica` vs
`axis_pair`; `mix_ica` vs `mix_control`; `mix_ica` vs `mix_oracle`.
**SECONDARY 4 (mechanism):** per-seed subspace fidelity (mean cosine of principal angles, best of
the two subspace↔cue assignments) to each member of the symmetry orbit below, and the fraction of
seeds landing on each member.
**SECONDARY 5 (symmetry-breaking diagnostic, localization only, NO training):** ICA fidelity to the
true split for cue-B margin ∈ {2.0, 2.2, 2.5, 3.0} × 30 seeds × {rot}. Tests whether making the two
cues unequal restores identifiability. This is a *diagnostic of the identifiability claim*, not a
rescue of the primary.

## Preregistered mechanism prediction (stated before the confirmatory run)
The labelled training distribution is **exactly invariant** under an orthogonal map Φ acting on the
4 cue coordinates (p=(e0+e1)/√2, q=(e2+e3)/√2, p'=(e0−e1)/√2, q'=(e2−e3)/√2; Φ: p→q, q→−p, p'→p',
q'→q'). Φ permutes the 8 class-conditional means and preserves their labels (verified algebraically
in `explore.py`), and Φ maps the true cue split {span(e0,e1), span(e2,e3)} to the **entangled** split
{span(q,p'), span(p,q')}. The orbit of the true split under the full symmetry group D4×D4 contains
**exactly 2 distinct splits**. Therefore any *equivariant* localizer — one that uses only the data
distribution and not the privileged coordinate basis — is at a **coin flip** between the true split
and the entangled split, no matter how good it is. Round-2's pair-probe escaped this only by being
coordinate-dependent.
Consequences predicted: (i) `rot_ica` ≈ 0.5·(oracle rate) + 0.5·(entangled-split rate). The
entangled split gives each column both a cue-A-only and a cue-B-only readout, so it should behave
like round-1's `routing_no_occlusion` arm (6/30) ⇒ point prediction **rot_ica ≈ 14/30**, i.e. above
the floor but well below 23/30. (ii) `axis_ica` ≈ `rot_ica` (the failure is caused by localizer
equivariance, not by the rotation itself). (iii) `rot_pair` collapses. (iv) `rot_oracle` ≈ 20–24/30.
(v) `mix_*`: uncertain — exploratory 4-seed fidelity (seeds 700–703, *fidelity only, no
selectability seen*) suggested ICA may recover the true split more often under a non-orthogonal
mixing, because orthogonal projection onto the estimated signal subspace is not equivariant to a
non-orthogonal Ψ = TΦT⁻¹. Recorded here so it cannot be claimed as a prediction after the fact.

## Controls / anchors
- Positive control: `axis_pair` must reproduce round-2's `learned_L1` (≈23/30) through the new
  learned-basis code path.
- Negative controls: `rot_control`, `mix_control` (expect 0/30).
- Deterministic control: the algebraic Φ check (orthogonality + labelled-centre permutation).
- Equivariance control: `axis_ica` and `rot_ica` per-seed fidelities should coincide.

## Compute budget
Single process, torch 2 threads, CPU. Measured: routed run 2.25 s, control run 1.1 s, localizers
< 0.15 s. Estimate 8×30×2.25 + 2×30×1.1 ≈ 10 min. If the run threatens the 15-min cap, arms are
dropped in this order (and the drop is reported): `mix_oracle`, `rot_pair`, `axis_ica`.

## Anything not listed above is POST-HOC and will be labelled as such in FINDINGS.md.
