# Connectivity graph vs worldview (theta-sweep of the 60x60 barrier census)

**Status: COMPLETE. The pre-registration below was written and saved before the analysis ran; results are appended at the bottom unchanged from the plan.**

## Question

The parent repo (`~/moral_framework_misgen`, READ-ONLY) claims "the landscape partially sees
the worldview" from a single ad hoc barrier threshold (0.05) on the 60x60 permutation-aligned
barrier matrix (`stats/results.json: basins/barrier_matrix`). Turn that single-threshold claim
into a curve: treat the census as a graph with edge iff barrier <= theta, sweep theta over the
full barrier range, and quantify (a) how separable worldview groups are in a threshold-free way,
(b) over what theta range direct vs chained (through-hedger) connectivity differs, (c) community
structure vs worldview labels.

## Data (fixed, no recomputation)

- `stats/results.json` -> `basins/barrier_matrix` (60x60, seeds 700..759), `multimodality/scores[:60]`
  (deploy alignment for the same seeds, same ordering: `deliverable_3_basins(models, scores, nseed=60)`
  slices the first 60 of the 120 deliverable-1 models).
- Worldview labels (repo's own convention): aligned A = score > 0.8, misaligned M = score < 0.2,
  hedger H = otherwise. Expected counts from saved `cluster_align`: 12 A / 17 M / 31 H.
- Sanity gate before any new analysis: reproduce the README numbers from the saved matrix
  (cross mean ~0.064, within mean ~0.029, ~49% of A-M pairs not directly connected at theta=0.05).
  If these do not reproduce, stop and report that instead.

## Pre-stated PRIMARY test

**Statistic:** threshold-free rank AUC = P(within-group barrier < cross-group barrier) + 0.5*P(tie),
where within = all A-A and M-M pairs (expected n = 66+136 = 202) and cross = all A-M pairs
(expected n = 204). Hedger pairs are excluded from the primary (they are the chains, tested secondarily).
AUC = 0.5 means the edge structure is blind to worldview at every theta simultaneously;
AUC > 0.5 means within-group pairs connect before cross-group pairs as theta rises.

**Test:** one-sided node-label permutation test. Permute the 60 worldview labels over nodes
(preserving counts 12/17/31), recompute AUC, 10,000 permutations, rng seed=0.
p = (1 + #{AUC_perm >= AUC_obs}) / (1 + 10000).
This respects the dependence between pairs sharing a node, which the repo's original
Mann-Whitney on pairs ignores (pairs are not independent -> naive MW is anticonservative).

**Decision rule (pre-stated):** p < 0.01 -> "the landscape sees worldview in a threshold-free
sense"; otherwise report as not established by this census.

## Pre-stated SECONDARY / descriptive analyses (curves; no decision rules, reported as-is)

1. Curves vs theta (theta grid = all unique off-diagonal barrier values, n<=1770):
   - frac of A-M pairs with a DIRECT edge (barrier <= theta)
   - frac of within (A-A + M-M) pairs with a direct edge
   - frac of A-M pairs PATH-connected in the full 60-node graph (chains through hedgers allowed)
   - median / max shortest-path hops over path-connected A-M pairs
   - modularity Q(theta) of the fixed 3-way worldview partition, with a 1,000-label-permutation
     null band (2.5/97.5 pct, same permutations across theta) at each theta
   - ARI between worldview labels (3-way) and connected components; and, where the graph has
     >=1 edge, NMI between labels and greedy-modularity communities (networkx).
2. Landmark thetas: first A-M direct edge; 50% and 100% A-M path-connected; 50% and 100% A-M
   direct-connected; theta at which A alone (and M alone) become internally path-connected.
3. Chain interpretation at the repo's legacy theta=0.05: hop-length distribution of A-M pairs,
   and hedger-mediation test: at each theta, frac of A-M pairs path-connected in the
   hedger-deleted 29-node subgraph vs in the full 60-node graph.

Anything not listed above and reported later is labelled POST-HOC.

## Compute budget

Analysis-only on a saved 60x60 matrix; < 1 minute CPU, single process. No training, no model loads.

---

# RESULTS (appended after the run)

## Sanity gate: PASSED

Reproduced from the saved matrix + `scores[:60]` with the repo's 0.8/0.2 label rule:
12 A / 17 M / 31 H; cross mean 0.0637, within mean 0.0285, naive MW p = 2.3e-15,
48.5% of A-M pairs not directly connected at theta = 0.05. All match the README.

## PRIMARY result

**AUC = P(within-group barrier < cross-group barrier) = 0.725** (within n = 202 pairs,
cross n = 204 pairs, 60 nodes / 60 seeds).
Node-label permutation test (10,000 perms, seed 0): **p = 1.0e-4** (the floor: 0 of 10,000
permuted labelings reached the observed AUC; null mean 0.501, 97.5th pct 0.564, max 0.641).
Pre-stated alpha 0.01 -> **the landscape sees worldview in a threshold-free sense**, and the
node-level permutation shows the repo's pair-level Mann-Whitney conclusion survives the
pair-dependence correction (though 2e-15 was anticonservative; the honest p is "<= 1e-4").

## SECONDARY curves (descriptive, as pre-stated)

Landmark thetas (barrier units; matrix min 0.00045, median 0.0244, max 0.356):

| event | theta |
|---|---|
| first direct A-M edge | 0.0019 |
| 50% of A-M pairs path-connected (full graph) | 0.0057 |
| M group internally path-connected (M-only subgraph) | 0.0088 |
| **100% of A-M pairs path-connected (full graph)** | **0.0109** |
| 100% of A-M pairs path-connected with ALL 31 hedgers deleted | 0.0152 |
| 50% of within pairs direct | 0.0214 |
| A group internally path-connected (A-only subgraph) | 0.0265 |
| 50% of A-M pairs direct | 0.0485 |
| Q(worldview partition) > 97.5pct perm null, window | [0.0015, 0.090] (30% of grid) |
| 100% of within pairs direct | 0.161 |
| 100% of A-M pairs direct | 0.341 |

The "partially sees" claim as a curve: the direct-edge curves for within vs cross are shifted
by ~2.3x at their medians (0.0214 vs 0.0485) and the cross curve completes at 0.341 vs 0.161
for within — but path connectivity across worldviews is complete by theta = 0.0109, i.e. at
0.45x the median barrier and 0.22x the legacy 0.05 threshold. Everything merges into one
component long before most direct edges exist; the worldview signal lives entirely in *which*
pairs get direct edges first, not in whether groups are reachable.

Chain lengths: at theta = 0.05 (legacy), A-M shortest paths are 1 hop for 105/204 pairs and
2 hops for 99/204 (median 1, max 2). At theta = 0.0109 (first full path connectivity) median
2, max 5 hops; the longest chain anywhere on the grid is 9 hops (theta = 0.0022).

Modularity / communities: Q of the fixed 3-way worldview partition exceeds its 95% permutation
band only for theta in [0.0015, 0.090] and peaks at just Q = 0.22 (very sparse graph,
theta = 0.00086). Detected structure does not recover worldview: max ARI(connected components,
labels) = 0.108 (theta = 0.0031); greedy-modularity NMI averages 0.104 in the significant-Q
window (its global max 0.401 at theta = 0.0005 is a degenerate 59-community near-empty graph).
At theta = 0.05: NMI 0.068, ARI 0.000, one component.

## A claim in the parent README this analysis contradicts

README (Landscape geometry row): all 60 models merge at 0.05 "but only via chains through
hedger seeds." **Not supported by the saved matrix**: with all 31 hedgers deleted, every A-M
pair is still path-connected at theta >= 0.0152 (and at 0.05 the max A-M chain is 2 hops,
running through A/M nodes). Hedgers are not needed as bridges; this was a pre-stated secondary
test (hedger-mediation), not post-hoc.

## Verdict in one sentence

Across the whole theta sweep the barrier graph orders within-worldview pairs ahead of
cross-worldview pairs (AUC 0.725, node-permutation p <= 1e-4), but the effect is purely an
edge-ordering signal: full cross-worldview path connectivity arrives by theta = 0.011 (with or
without hedgers), label-partition modularity is weak (peak Q 0.22) and significant only for
theta < 0.09, and no community detection recovers the worldview partition (ARI <= 0.11).

## Caveats

- n = 60 seeds from one searched-for construction (154-param MLP, equal-cost XOR cues); the
  repo itself flags the regime as post-selected for bimodality. Nothing here tests other
  architectures (the repo notes Transformers show no linear mode connectivity at all).
- Analysis-only reuse of the saved census: barriers are linear-path barriers after 2-layer
  permutation alignment on one reference batch; "path-connected" means chained *linear* segments,
  not true low-loss curves. Curved connectors could only lower barriers (claims are conservative
  in that direction), but the census itself may overstate barriers.
- The 0.8/0.2 worldview thresholds are inherited from the repo, not tuned here; 31/60 nodes are
  hedgers and are excluded from the primary statistic by design.
- POST-HOC observations: none beyond the pre-stated list; the README-contradiction paragraph
  uses only the pre-stated hedger-deletion test and hop histograms.
- Permutation p floor: with 10,000 permutations, p is reported as 1.0e-4 = (1+0)/(1+10000).
- Runtime ~86 s single process (grid of 1,770 thetas x 1,000 modularity perms dominates).

## Files

- `analyze.py` - full analysis (deterministic, seeds 0/1 fixed)
- `results.json` - all curves, landmarks, primary test, legacy-threshold details
- `curves.png` - connectivity fractions, modularity + null band, chain lengths vs theta
