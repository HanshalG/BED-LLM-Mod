# ChemBench Continuous Policy-Ladder Development Result

Date: 2026-08-15 (Europe/London)

## Status

**Failed closed at the numerical dependency.** No LLM/API call was made, the
cost was $0, and no proposal-efficacy or sealed evaluation stage is authorized.

This screen tested the MDA-inspired repair proposed in the third-pass decision:
raw continuous rates, a shared log-normal likelihood for simulated and realized
updates, independent inference and planning parameter banks, predictive
observation branches, and conservative d1/d2/d3 policy improvement.

The final run follows a pre-commit consistency audit: known and outside models
now use the same transformed-rate Gaussian density convention. The first draft
had subtracted a Jacobian only from the outside component. Regression tests were
added and all three slices were rerun; the correction changed the aggregate
numbers only in the fourth decimal place and did not change the conclusion.

## Frozen Setup

- Official LLM-AutoSciLab source at commit `acf160eb6c96897748dd92b152703b59b74efc05`.
- Opened v4 development truths on easy, medium, and hard; 48 truths per slice.
- Candidate structures and parameter ranges use v0-v3 only.
- Sixteen deterministic broadened LHS particles per structure, expansion 1.5.
- Independent inference seed `2026081900` and planning seed `2026081901`.
- Eighteen fixed assays, four sequential experiments, and 1,000 held-out queries.
- Three common-random-number policy-improvement replicates and a 5% adoption floor.
- Registry-oracle structure proposer; zero model calls.

## Result

| Slice | d1 MSE | d2 MSE | d3 MSE | d2 vs d1 | d3 vs d2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 0.15301842 | 0.13310907 | 0.11550939 | -13.01% | -13.22% |
| Medium | 0.15151565 | 0.17233974 | 0.16782573 | +13.74% | -2.62% |
| Hard | 0.24634384 | 0.24836259 | 0.27639266 | +0.82% | +11.29% |
| **Aggregate** | **0.18362597** | **0.18460380** | **0.18657593** | **+0.53%** | **+1.07%** |

At practical tolerance `1e-6`, aggregate d2 versus d1 was 30/86/28
wins/ties/losses and d3 versus d2 was 29/85/30. Easy was strongly monotonic,
but medium failed at d2 and hard failed at d3. Aggregate d3 was 1.61% worse
than d1.

All levels chose `T=368` at the easy and medium roots. Hard d1/d2 also chose
`T=368`, while hard d3 changed to `C_I=50,C_A=100`. Thus most depth effects
come from continuation policies rather than superficial root diversity.

The failures are mechanism-specific rather than a uniform loss of signal. For
example, medium d2 improves `c79_anticoop_arrhenius` by 0.175 MSE but worsens
`c41_hill_noncompetitive_arrhenius` by 0.443 and
`c12_mm_uncompetitive_arrhenius` by 0.328; d3 then reverses some of those
choices. This is consistent with planning-distribution and parameter-posterior
misspecification, not merely independent observation noise.

The bounded predictive policy table also reaches its 10,000-state cap at d3,
with 53,734/45,953/42,323 evictions on easy/medium/hard. The earlier nearest
state fallback covered most realized states but regressed held-out risk, so it
remains disabled. Coarse global state matching is not an acceptable repair.

## Murphy-Paper Fourth-Pass Decision

The paper's strongest transferable lesson is that the LLM should propose
mechanistic structures while numerical machinery owns likelihoods, evidence,
posterior updates, pruning, design, and forecasts. We keep that split. The new
result shows that our numerical approximation is still too weak to support an
LLM efficacy experiment.

### Priority 0: posterior and planner transfer

1. Replace fixed LHS reweighting with per-structure adaptive-tempered SMC:
   roughly 100 parameter particles, ESS target 0.6, and three bounded
   rejuvenation moves per temperature rung. Real updates refresh SMC; rollout
   branches reuse immutable posterior snapshots plus exact incremental
   likelihoods.
2. Validate every policy improvement across multiple independent planning
   banks, not only multiple rollout-noise replicates from one bank. Adopt a
   challenger only when a paired lower confidence bound is positive and each
   source-only bank is nonworse. This directly targets the medium/hard transfer
   failure without inspecting truth outcomes.
3. Replace the global quantized policy table with a local particle scenario
   tree using double progressive widening. Root-sampled world particles and
   common random numbers avoid repeatedly matching unrelated posterior states;
   nested d(k-1) rollouts preserve the conservative policy ladder.
4. Separate between-structure disagreement from within-structure parameter
   uncertainty. Expose explicit `discover/discriminate`, `refine-parameters`,
   and `replicate` planner modes, then let the non-myopic meta-policy choose a
   sequence of modes.
5. After the fixed-assay numerical gate passes, generate continuous designs
   over the official 7D box with a fixed 48-evaluation CMA-ES/Sobol budget.
   Shortlist by model-discrimination or parameter-refinement mode, then score
   the common top set with the exact rollout objective.

### Priority 1: an LLM-native proposal transition

1. Use a typed mechanism-edit language rather than asking for free-form plans.
   Each proposal contains an executable AST, parameter bounds, the residual
   motif it explains, and one exposing and one falsifying design region. Code
   compiles, fits, deduplicates, and rejects it.
2. Build a source-only proposal atlas on synthetic residual histories before
   evaluation. Distill or retrieve a calibrated proposal kernel
   `q(new structure | residual summary)` for rollout use; call the real LLM only
   after realized observations. Gate atlas predictions against held-out actual
   LLM proposals before any efficacy endpoint.
3. Trigger M-open expansion with a prequential, target-containing predictive
   check. When fit is adequate and model mass is concentrated, shrink the pool;
   when the check fails, expand it. Preserve a novelty reserve and prune by
   marginal evidence plus executable equivalence.
4. Hold out mechanism compositions or channel families, not only parameter
   versions. The present v4 split stresses parameter transfer but does not by
   itself establish open-ended structure generation.

## Environment Changes

### ChemBench: retain as the numerical sandbox

Keep the official raw-rate task, but use six to eight experiments, continuous
designs, repeatable assays, and structural holdouts. The intended horizon event
is: acquire a residual that causes a useful mechanism edit, discriminate the
new structure, then refine its parameters. The current four-step v4 screen is a
useful stress test but is not yet the LLM-native headline.

### NeuronBench: promote to the next zero-call opportunity audit

Deterministic NeuronBench is the strongest paper-derived headline candidate.
The LLM receives compact protocol-to-spike-count summaries and proposes an
open parameterized ion-current mechanism; numerical ODE simulation and a
feature likelihood own inference. Construct source-only worlds where an early
phenotype probe changes which channel family the proposer generates and a later
conditioning protocol identifies it. First verify that d2/d3 oracle planning
changes roots and improves held-out spike/feature forecasts against myopic.

If that passes, add the stochastic extension as a second stage: make protocol,
repeat count, and observation model (feature synthetic likelihood versus
particle filter) joint costed design choices. This supplies a natural
non-myopic tradeoff between learning what signal matters and spending repeats
to resolve it.

### ForceBench and location finding: supporting hybrids

ForceBench can test a two-stage `expose residual -> estimate mechanism scale`
sequence, especially screened force laws. Location finding becomes LLM-native
only if the source count or field/sensor law is also unknown: let the LLM
propose an executable kernel while trans-dimensional SMC localizes sources.
Pure known-kernel localization remains a classical supporting task.

## Next Gates

1. Adaptive-SMC posterior calibration and evidence tests on source-only
   synthetic histories.
2. Multi-bank policy-improvement screen on the already-open v4 rows; require
   at least 5% aggregate improvement on both links and nonworse performance on
   every tier before another LLM gate.
3. Zero-call NeuronBench horizon-opportunity audit with fixed-support,
   dynamic-oracle, myopic, and random controls.
4. Only after one numerical route passes: a small proposal semantic/transition
   gate, followed by a sealed LLM efficacy comparison.

## Artifacts

- Easy JSON SHA256: `a2a596e345b0540a6c58d0a045f7d7b8fa466433b818fad3ffb008c081888080`
- Medium JSON SHA256: `bda4a987217176e6b64353dced8db638f836d36d45324165d4342ed1e6cdeb41`
- Hard JSON SHA256: `3d2eb6c0af16761d4def0cf7e8ed01cd3b30686e2f6e84648e4011b6d7e720d0`
- Implementation SHA256: `9a8d16399dcab2d4b750e1d42bcb1b6bfa980668fc2bb9b372f6d26cdbd3a23a`
- Runner SHA256: `7065607d45dea6fa36dc92493bc532f8afae5e08ec307d3cb05984ee518986d5`
- Test SHA256: `4145cb8edaa0dcc836138ac42b929bcfdcbf3a3714fc167ff6d1a3e5aab5107e`
