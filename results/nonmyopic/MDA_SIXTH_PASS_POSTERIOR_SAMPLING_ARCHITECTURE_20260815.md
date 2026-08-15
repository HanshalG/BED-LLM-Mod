# MDA Sixth Pass: Posterior-Sampling Architecture For Non-Myopic LLM BED

Date: 2026-08-15 (Europe/London)

## Executive Decision

Kevin Murphy's Model Discovery Agent should be treated as an interface
decomposition, not copied as a planner:

1. the LLM proposes executable mechanistic model edits;
2. Bayesian computation owns parameter inference, evidence, and prediction;
3. numerical value of information owns experiment choice.

MDA is explicitly myopic. Our workshop contribution should be a Bayes-adaptive
planner whose state includes the mechanism pool and whose future observations
can trigger branch-conditioned typed edits. The two frozen ChemBench branch
nulls show that a small fixed quadrature tree is not reliable enough across all
candidate actions. Replace it with common-random-number posterior-sampling
tree search and progressive widening.

## Proposed Architecture

### 1. Belief state

Use

```text
B_t = (mechanism graph, structure weights, per-structure SMC particles,
       experiment history, residual summary, remaining budget)
```

The canonical numerical posterior is the pooled 1,024-particle representation
from the two calibrated V3 banks for each structure. Do not ask an LLM to emit
posterior probabilities or likelihoods.

### 2. Typed mechanism-edit graph

Represent an LLM proposal as a validated edit to an executable mechanism:

```text
add factor | remove factor | replace kinetic law | add interaction
add temperature term | add pH term | change inhibition mode
```

Each edit must compile, satisfy units and positivity constraints, declare new
parameter priors, and deduplicate to a canonical graph hash. Preserve the
parent model, so proposal failure cannot erase support.

The structural transition is explicit:

```text
q_phi(edit | current pool, residual summary, action, observation, budget)
```

This is the LLM-native part of the method. MDA proposes after real experiments;
we additionally need a frozen approximation to this transition inside future
branches.

### 3. Proposal atlas, not live rollout calls

Before planning, generate and validate edits for a frozen set of residual
archetypes: saturation mismatch, inhibitor mismatch, temperature mismatch,
pH mismatch, two-substrate mismatch, cooperative curvature, and unexplained
activation. Cache the accepted typed edits and transition frequencies as a
proposal atlas.

During a simulated trajectory, map the branch residual summary to an archetype
and sample an atlas edit. This allows branch-conditioned support expansion
without making an API call at every rollout node. At the next real observation,
the LLM may refresh the atlas from the actual history.

### 4. Posterior-sampling tree search

Replace nine-way quadrature with sparse Monte Carlo tree search:

- sample a structure and parameter particle from the current pooled belief;
- sample an observation under the executable likelihood;
- update all structure weights and per-structure particles numerically;
- apply the frozen proposal-atlas transition when its residual trigger fires;
- recurse until the remaining budget is exhausted;
- back up terminal task Bayes risk plus experiment cost.

Use common random numbers for d1/d2/d3 and competing first actions. Use
progressive widening for continuous observations and merge nearby nodes using
the tested posterior-state mean/variance features. Node merging is a compute
optimization; trajectory values still come from sampled numerical updates.

This architecture does not require a fixed branch set to rank all actions.
Increasing depth can reuse the same root samples, making depth comparisons less
noisy and exposing where a deeper continuation changes the first action.

### 5. Action generation

Do not evaluate every assay uniformly at every node. Build a diverse shortlist:

- discovery actions maximize residual-trigger probability;
- discrimination actions separate current structures;
- refinement actions reduce parameter uncertainty within the leading model;
- one task-greedy action protects immediate predictive performance.

Keep the shortlist fixed before outcomes and include the myopic best action at
every node. A depth-two planner can then never lose the myopic option because
of action pruning.

### 6. Objective and controls

Optimize expected terminal predictive Bayes risk on a sealed query distribution,
not structure entropy alone. Report predictive log loss, log-rate MSE, and
model-structure calibration as secondary metrics.

Required controls:

- numerical myopic VoI with the same action shortlist and posterior samples;
- compute-matched myopic ensemble;
- random valid action;
- oracle fixed support;
- fixed misspecified support with no LLM transition;
- random typed edit transition with the same edit count;
- MDA-style real-history-only proposer without branch-conditioned transitions.

The final control isolates the non-myopic contribution from simply proposing
more models.

## Environment Changes

### ChemBench primary: compositional source worlds

Use only prospectively sampled in-prior source worlds for the primary result.
The failed v4 stress cohort is an out-of-prior robustness appendix, not a depth
endpoint.

Construct truths compositionally from the same typed grammar exposed to the
LLM. Start with a deliberately incomplete but plausible mechanism pool. The
missing composition must be recoverable through one or two typed edits rather
than an unconstrained program-generation leap.

Create a real horizon gap through experiment costs and edit observability:

- a cheap diagnostic assay produces a residual pattern that identifies which
  edit family should be opened;
- a second targeted assay then discriminates the newly added mechanisms;
- a costly direct assay gives higher immediate risk reduction but leaves too
  little budget for structural discovery and refinement.

Freeze these worlds only after an oracle-support opportunity audit confirms
that d2 changes the first action and lowers expected terminal risk relative to
myopic. Do not tune the gap after LLM results.

### NeuronBench secondary: costed intervention sequences

Use the paper's neural-dynamics setting as a secondary transfer environment.
Expose a typed library of channel, adaptation, threshold, and interaction
edits; use stimulation protocols as actions; and charge different costs for
short diagnostic pulses and long response curves. The intended horizon gap is
diagnostic stimulation followed by a model-specific intervention.

Run this only after the ChemBench numerical and proposal gates pass. It tests
whether typed branch-conditioned discovery transfers beyond enzyme kinetics.

### Location finding: mechanics control only

Retain location finding as a planner and likelihood sanity check, not the
LLM-native headline. Its known likelihood and fixed source family leave little
scientific model-discovery role for an LLM, and the prior Gemma experiments
showed that language-model spatial reasoning can obscure rather than validate
the planner. A thinking naive policy remains a useful behavioral baseline.

## Prompt And Interface Changes From MDA

Give the proposer the full legal mechanism grammar, current executable pool,
parameter-prior forms, observation likelihood, action costs, remaining budget,
experiment history, and a compact numerical residual table. Require JSON typed
patches with a short mechanistic rationale and predicted residual signature.

Do not ask for a free-form strategy, posterior distribution, EIG score, or
future numeric observation. The model's hard task is compositional scientific
hypothesis generation, where language priors can help; simulation and scoring
stay deterministic and testable.

## Dependency-Ordered Gates

1. **One-step sampling fidelity:** freeze 32/64/128/256 CRN posterior samples
   against the saved 2,048-outcome pooled reference. Require the existing
   Spearman and regret thresholds before tree depth.
2. **Oracle horizon opportunity:** with truth support available and no LLM,
   require d2 to change the first action in a meaningful fraction of prospective
   worlds and improve paired terminal risk.
3. **Proposal validity and recall:** on sealed residual archetypes, require high
   compile/constraint validity and recovery of the true edit family. Compare
   against random typed edits.
4. **Transition fidelity:** compare atlas-predicted branch edits with held-out
   real-history LLM edits before endpoint outcomes.
5. **Numerical d1/d2/d3 development:** common worlds and random numbers;
   require nonworse value estimates and positive paired terminal-risk gains.
6. **LLM-native efficacy:** dynamic typed support must beat fixed support,
   random edits, and real-history-only MDA at matched calls and compute.
7. **Sealed confirmation:** new worlds, seeds, and outcome files; no threshold
   changes after development.

## Immediate Next Experiment

Do not run another fixed branch-count sweep. Freeze and implement the one-step
CRN posterior-sampling fidelity gate at 32/64/128/256 samples per action against
the already banked pooled 2,048-outcome reference. If 256 samples do not pass,
the MCTS route is too noisy at a practical budget and the environment/action
panel must be reduced before any LLM spend.
