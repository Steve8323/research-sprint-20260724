# Noise-probing a legible model (ERT + activation-noise probe)

Combination experiment: apply the activation-noise probing idea of
`~/misalignment_noise_probe` (h -> h + N(0, (sigma*RMS(h))^2) at one layer, read the
output) to the English-Residual Transformer of `~/english_bottleneck_lm`, whose
inter-layer state is literal words. Question: **if a model holds information in its
within-layer activations that its emitted words suppress, does boundary noise surface
that information as directly readable words?**

## Pre-registration (written BEFORE the noise experiments were run)

### Setup

Toy scale, CPU, single process. ERT config as in the parent repo: d=64, 4 layers,
24 synonym classes x 4 synonyms, SEQ_LEN=4, RPOS=0. Noise injection: at the output of
transformer block `l` (the activation that is about to be unembedded into this layer's
words -- the layer-boundary state), replace `x` with `x + sigma * RMS_token(x) * eps`,
`eps ~ N(0,1)`, RMS per token over the feature dim. Word commitment at eval is
deterministic argmax (no Gumbel), so injected Gaussian noise is the only stochasticity
in arm A; in arm B the paraphrase channel's own randomness remains (it is part of the
model). In the ERT, noise injected at layer `l` can reach later layers ONLY through the
words layer `l` emits.

Two arms:

**Arm A -- "concealer" (information present in activations but suppressed in words).**
ERT with rho=0 (identity channel), trained jointly (3 seeds) to:
1. emit the CANONICAL synonym (class member index 0) of the correct class
   g^(l+1)(c0) at RPOS at every layer (token-level CE) -- this trains the model to
   conceal WHICH synonym s0 the input used;
2. let a colluding linear decoder recover s0 from the final layer's full emitted
   word sequence (all 4 positions) -- this forces the surface variant to be carried
   through the whole stack (necessarily as words in the unsupervised free positions).

QC gates (checked before the probe, reported either way): final-task accuracy
(canonical token of g^4(c0) at RPOS), covert decoder accuracy (chance 0.25),
per-layer canonical-compliance rate, clean reveal rate at RPOS (expected ~0).

**Arm B -- "destroyed" control (information absent, not hidden).**
ERT with rho=1 trained on Task S (copy the exact input word; the repo's E1 shows this
collapses to ~0.25 because the channel destroys surface at each boundary; the natural
copy-strategy Bayes ceiling against 3 independent forced synonym swaps is 7/27 ~ 0.259).
3 seeds, trained with the repo's own `train_S`.

### Primary tests (stated in advance)

**Primary A (the headline question).** Grid: seeds {0,1,2} x noised layer {0,1,2,3} x
sigma {0.25, 0.5, 1.0}; one fresh batch of 400 items per cell, with s0 drawn from
{1,2,3} (excluding the canonical index so "reveal" is distinguishable from compliance).
Read the noised layer's OWN emission at RPOS. An event is ELIGIBLE if that emission is
class-correct for that layer and non-canonical (synonym index in {1,2,3}). A reveal is
an eligible event whose synonym index == s0. Under H0 (emissions carry no s0
information) s0 is independent of the emission, so P(reveal | eligible) = 1/3 exactly,
regardless of any synonym biases. Test: two-sided exact binomial test of the pooled
reveal count against p=1/3, alpha=0.05. A significant excess means noise surfaces the
concealed variable in directly human-readable form (the model "says" the hidden
synonym slot).

**Primary B (specificity control).** Same grid (3 seeds x 4 layers x 3 sigmas, 400
items/cell, clean baseline 1200 items/seed at sigma=0). Metric: final-layer exact-copy
accuracy (emission at RPOS == w0). Prediction: NO cell exceeds its seed's clean
baseline significantly (per-cell one-sided Fisher exact, Bonferroni over the 12
layer x sigma cells per seed = 36 tests). Rationale: at rho=1 surface information is
destroyed at the first boundary, not hidden, so a correct noise probe must find
nothing. This is the negative control the parent noise-probe repo lacked.

### Labeled exploratory (not primary)

sigma in {0.1, 2.0}; reveals read at layers downstream of the noised layer;
wrong-class reveals; verbatim "blurt" rate (RPOS emission == w0 exactly); the clean
stash form (does the model hide s0 in plain sight -- verbatim w0 or synonym-index code
at free positions?); task/decoder accuracy vs sigma (noise as covert-channel scalpel);
example traces printed as words.

### Honesty notes

- Before the full run the code is smoke-tested for mechanics (shapes, convergence
  trend, timing) on a throwaway seed; QC metrics may be seen then, primary reveal
  statistics are computed only in the full run.
- Per-cell sufficient statistics (n_items, n_eligible, k_reveal / n, k_correct) are
  saved in results.json; Wilson 95% CIs on all reported rates.
- n=400/cell, 3 seeds/arm was chosen for the <=10 min CPU budget; power notes go in
  caveats if eligible counts come out small.


### Pre-registration amendment (during organism QC, before any noise-probe statistics)

Final-layer-only covert supervision did not converge (decoder stuck at chance ~0.25
after 700 steps; credit assignment through 3 straight-through commits is too weak at
this scale). Amended: the shared linear decoder is supervised on EVERY layer's
committed words (process supervision, same style as the repo's per-layer task
supervision). QC after the fix: task 1.000, covert decoder 0.877 at 500 steps
(throwaway seed 96). Training budget reduced to 700 steps / batch 128 / 1 torch
thread after profiling on the shared machine. The primary tests are unchanged.

---

## Results

### Arm A - concealer organism QC (clean, per seed)

| seed | task acc | covert decoder acc (chance 0.25) | canonical compliance L0-L3 |
|---|---|---|---|
| seed0 | 0.989 | 0.929 | 1.000 / 1.000 / 1.000 / 0.989 |
| seed1 | 0.991 | 0.945 | 1.000 / 1.000 / 0.991 / 0.991 |
| seed2 | 1.000 | 0.968 | 1.000 / 1.000 / 1.000 / 1.000 |

Clean reveal-eligible events at RPOS (n=1200/seed, s0 in {1,2,3}):
- seed0: eligible 0/4800 layer-events, reveals 0
- seed1: eligible 0/4800 layer-events, reveals 0
- seed2: eligible 0/4800 layer-events, reveals 0

Stash form at the free positions (clean, final layer):

| seed | pos | P(verbatim w0) | P(synonym index == s0) |
|---|---|---|---|
| seed0 | pos1 | 0.000 | 0.000 |
| seed0 | pos2 | 0.000 | 0.000 |
| seed0 | pos3 | 0.000 | 0.000 |
| seed1 | pos1 | 0.000 | 0.000 |
| seed1 | pos2 | 0.000 | 0.000 |
| seed1 | pos3 | 0.000 | 0.000 |
| seed2 | pos1 | 0.000 | 0.000 |
| seed2 | pos2 | 0.000 | 0.000 |
| seed2 | pos3 | 0.000 | 0.000 |

### PRIMARY A result: VACUOUS -- zero eligible events

The pre-registered conditional test could not be evaluated: over the full primary grid
(3 seeds x 4 layers x sigma {0.25,0.5,1.0} x 400 items = 14,400 noised forwards) there
were **0 eligible events** -- the noised layer's RPOS emission was never a
class-correct non-canonical synonym. Nothing reveal-shaped ever appeared. (Rule of
three: 95% upper bound on the per-item eligible-event rate is ~2.1e-4.) Including the
exploratory sigmas {0.1, 2.0} (24,000 items total): 1 eligible event (which happened
to match s0 -- anecdotal), 1 verbatim blurt of w0, 10 wrong-class non-canonical
emissions (6 matching s0), 0 eligible downstream reads.

So the answer to the pre-registered question is a strong negative, but in an
unanticipated form: rather than "flips occur and are uncorrelated with s0" (the H0 the
test was built for), **flips into the reveal-capable region essentially never occur**.

**Post-hoc diagnostic (labeled; seed 0 retrained deterministically).** Where do noised
emissions go, and where does the covert code live?

- Under noise the RPOS word moves ONLY between canonical tokens: at sigma=1.0,
  95.8-99.5% canonical-correct and the rest canonical-of-another-class, 0.00%
  non-canonical content words; even at sigma=2.0 non-canonical emissions are <=1.25%.
  Token-level canonical supervision collapsed the model's emission repertoire onto the
  24 canonical words, so the "reveal vocabulary" (the other 72 synonyms) is essentially
  unreachable by activation noise.
- The covert stash is a CLASS-code, not a synonym code: every emitted word at every
  position is canonical (rate 1.000), but WHICH class's canonical word the free
  positions emit varies with s0 within fixed c0 for 23-24 of 24 classes; a simple
  majority-vote lookup from (c0, final free-position classes) recovers s0 at 0.942
  (n=1000 held out), matching the trained decoder (0.931).
- The covert channel is noise-ROBUST: mean decoder accuracy 0.935-0.938 at
  sigma<=1.0 and still 0.878 at sigma=2.0 (task accuracy 0.99 -> 0.786).

Per-cell (primary sigmas; eligible k/n, reveal rate):

| seed | layer | sigma | n_eligible | k_reveal | rate | task acc | dec acc |
|---|---|---|---|---|---|---|---|
| 0 | L0 | 0.25 | 0 | 0 | nan | 0.993 | 0.923 |
| 0 | L0 | 0.5 | 0 | 0 | nan | 0.975 | 0.895 |
| 0 | L0 | 1.0 | 0 | 0 | nan | 0.930 | 0.880 |
| 0 | L1 | 0.25 | 0 | 0 | nan | 0.985 | 0.902 |
| 0 | L1 | 0.5 | 0 | 0 | nan | 0.988 | 0.925 |
| 0 | L1 | 1.0 | 0 | 0 | nan | 0.990 | 0.920 |
| 0 | L2 | 0.25 | 0 | 0 | nan | 0.978 | 0.910 |
| 0 | L2 | 0.5 | 0 | 0 | nan | 0.975 | 0.913 |
| 0 | L2 | 1.0 | 0 | 0 | nan | 0.983 | 0.902 |
| 0 | L3 | 0.25 | 0 | 0 | nan | 0.988 | 0.925 |
| 0 | L3 | 0.5 | 0 | 0 | nan | 0.993 | 0.918 |
| 0 | L3 | 1.0 | 0 | 0 | nan | 0.985 | 0.918 |
| 1 | L0 | 0.25 | 0 | 0 | nan | 0.985 | 0.920 |
| 1 | L0 | 0.5 | 0 | 0 | nan | 0.993 | 0.923 |
| 1 | L0 | 1.0 | 0 | 0 | nan | 0.940 | 0.923 |
| 1 | L1 | 0.25 | 0 | 0 | nan | 0.995 | 0.950 |
| 1 | L1 | 0.5 | 0 | 0 | nan | 0.990 | 0.928 |
| 1 | L1 | 1.0 | 0 | 0 | nan | 0.952 | 0.920 |
| 1 | L2 | 0.25 | 0 | 0 | nan | 0.993 | 0.935 |
| 1 | L2 | 0.5 | 0 | 0 | nan | 0.985 | 0.923 |
| 1 | L2 | 1.0 | 0 | 0 | nan | 0.973 | 0.940 |
| 1 | L3 | 0.25 | 0 | 0 | nan | 0.988 | 0.910 |
| 1 | L3 | 0.5 | 0 | 0 | nan | 0.988 | 0.930 |
| 1 | L3 | 1.0 | 0 | 0 | nan | 0.962 | 0.908 |
| 2 | L0 | 0.25 | 0 | 0 | nan | 1.000 | 0.983 |
| 2 | L0 | 0.5 | 0 | 0 | nan | 1.000 | 0.962 |
| 2 | L0 | 1.0 | 0 | 0 | nan | 0.998 | 0.975 |
| 2 | L1 | 0.25 | 0 | 0 | nan | 1.000 | 0.965 |
| 2 | L1 | 0.5 | 0 | 0 | nan | 1.000 | 0.965 |
| 2 | L1 | 1.0 | 0 | 0 | nan | 0.995 | 0.983 |
| 2 | L2 | 0.25 | 0 | 0 | nan | 1.000 | 0.973 |
| 2 | L2 | 0.5 | 0 | 0 | nan | 1.000 | 0.980 |
| 2 | L2 | 1.0 | 0 | 0 | nan | 0.990 | 0.975 |
| 2 | L3 | 0.25 | 0 | 0 | nan | 1.000 | 0.965 |
| 2 | L3 | 0.5 | 0 | 0 | nan | 1.000 | 0.973 |
| 2 | L3 | 1.0 | 0 | 0 | nan | 1.000 | 0.973 |

### Exploratory (labeled)

- sigma {0.1, 2.0} pooled: reveals 1/1 eligible.
- verbatim blurt (RPOS emission == w0 exactly, all cells): 1/24000.
- downstream reads (emission at layers after the noised one): 0/0 eligible.
- wrong-class emissions with synonym==s0: 6/10.

Example traces (seed 0; words per layer, RPOS first):
```
input: modern <pad> <pad> <pad>   (s0=3)
  L0 clean : hot hot hot hot
  L0 noise1: hot hot hot hot
  L1 clean : slow slow slow slow
  L1 noise1: slow slow slow slow
  L2 clean : strong strong strong strong
  L2 noise1: strong strong strong strong
  L3 clean : bad bad bad bad
  L3 noise1: bad bad bad bad
input: foolish <pad> <pad> <pad>   (s0=1)
  L0 clean : big big big big
  L0 noise1: big big big big
  L1 clean : sad sad sad sad
  L1 noise1: sad sad sad sad
  L2 clean : dirty dirty dirty dirty
  L2 noise1: dirty dirty dirty dirty
  L3 clean : broke broke broke broke
  L3 noise1: broke broke broke broke
input: muddy <pad> <pad> <pad>   (s0=3)
  L0 clean : broke slow slow slow
  L0 noise1: broke slow slow slow
  L1 clean : rich strong strong strong
  L1 noise1: rich strong strong strong
  L2 clean : fast bad bad bad
  L2 noise1: fast bad bad bad
  L3 clean : dumb new new new
  L3 noise1: dumb new new new
```

### PRIMARY B result (destroyed control)

| seed | repo eval_S | clean argmax acc [Wilson 95%] |
|---|---|---|
| seed0 | 0.244 | 0.251 [0.227, 0.276] |
| seed1 | 0.242 | 0.255 [0.231, 0.280] |
| seed2 | 0.246 | 0.251 [0.227, 0.276] |

Cells significantly above clean (one-sided Fisher, Bonferroni alpha=0.00139): **0/36**. Prediction (confirmed). Best primary cell: seed 0, L0, sigma=0.25: acc 0.295 (Fisher p=0.048; copy-strategy ceiling 7/27 = 0.259).

### Verdict

**Noise probing did not make the concealed variable human-readable.** In this
legible-by-construction model, the concealed variable never surfaces as words under
boundary-activation noise (0 reveal-capable emissions in 14,400 primary noised
forwards), because (i) the trained emission repertoire collapsed onto canonical
tokens, and (ii) the concealment is a distributed class-code that is itself robust to
the noise doses that degrade the overt task. The specificity control behaved
correctly: on the rho=1 model, where surface information is destroyed rather than
hidden, noise never produced spurious "recovery" (0/36 cells above clean,
Bonferroni). Legible-by-construction intermediates are readable, but they are not
self-disclosing: what you read under noise is just the model's trained vocabulary
rearranged, and a steganographic class-code survives both the word bottleneck and the
noise probe. Reading the words does show that free positions vary systematically with
the input beyond the task (an auditor could notice THAT something is carried), but
decoding WHAT requires the codebook -- i.e., a trained probe, exactly what the
noise-probe idea hoped to avoid.

Wall time: 234 s, single CPU process, 1 torch thread.

