# MDA Seventh Pass: A Factored M-open Planner

Date: 2026-08-15 (Europe/London)

## Decision

The next serious route should not be a more accurate version of the failed
continuous tree. It should use the verified categorical ChemBench policy ladder
as the structural planning layer and replace only its registry-oracle proposal
transition with an MDA-style LLM proposer.

The resulting method has three deliberately separate components:

1. an LLM proposes typed, executable mechanism edits from pool-wide residuals;
2. numerical Bayesian code fits, weights, validates, and prunes those edits;
3. an exact small tree plans over future residual categories and the support
   expansions they would trigger.

Continuous parameter inference and continuous assay optimization should be an
inner, myopic layer added after this structural result works. Long-horizon value
comes from path-dependent model discovery, not from trying to resolve every
continuous posterior fluctuation several steps ahead.

This is a stronger reading of Murphy's Model Discovery Agent than either asking
an LLM for a strategy or copying MDA's one-step VoI. MDA supplies the division
of labor. Our contribution is to plan over its M-open support transition.

## What The Paper Says That Matters Most

### The LLM is a proposal kernel over structures

MDA's only LLM call in its discovery loop proposes executable structures. It
does not ask the LLM for posterior probabilities, likelihoods, experiment
scores, or forecasts. The proposer sees the complete current pool, negative
evidence in the form of per-model residuals, the input most correlated with the
best model's residual, the remaining budget, and an explore/refine phase.

This is more specific than our earlier residual label or strategy interfaces.
The semantic object is

```text
q(edit | current pool, pool-wide residual report, experiment history,
         remaining budget, phase).
```

### Support expansion and contraction are both essential

The ChemBench ablation shows that M-open expansion is the only route to any
compound-mechanism recovery, but expansion alone creates over-elaborated and
near-duplicate laws. Evidence-based pool contraction is what recovers much of
the hard-tier performance. A proposal-aware planner that only unions support
will therefore overestimate discovery value and progressively dilute its
posterior.

### The trigger must be externally anchored

MDA describes a prequential query check precisely because a learned summary and
the candidate models can otherwise agree while both miss the task. For our
method, support expansion should be triggered by pre-update predictive surprise
on the newly observed assay, with a second check on the held-out task lens. It
must not be triggered merely by posterior entropy or by an LLM saying that the
current support looks inadequate.

### Structural and parameter information are different

MDA's default VoI scores between-model disagreement using per-structure
posterior means. It intentionally excludes within-structure parameter spread
from the structural acquisition. Our continuous experiments became unstable
when one tree tried to rank all actions while carrying both uncertainties.
This suggests a factored planner rather than a larger Monte Carlo budget.

### Interventional prediction is the scientific endpoint

The paper finds natural-language explanation scores unreliable and instead
uses held-out interventional forecasts. We should likewise make terminal
held-out log-rate risk primary. Symbolic recovery and edit-family recall are
mechanism diagnostics, not substitutes for predictive efficacy.

## Proposed Architecture

### 1. Two-level belief state

Use a structural state for long-horizon planning:

```text
S_t = (live mechanism graph,
       evidence weights,
       categorical predictive table,
       pool-wide residual report,
       proposal-atlas state,
       remaining experiment budget).
```

Each live mechanism may additionally own a continuous parameter posterior, but
the deep tree does not branch on its full particle state. The inner numerical
layer supplies per-mechanism predictive means, outcome-bin probabilities, and
evidence. It can be refreshed after real observations and approximately updated
inside branches without changing the finite structural state.

This makes the abstraction explicit and testable. Before using it with raw-rate
endpoints, measure the regret of the categorical structural policy under the raw
held-out loss. If that value-equivalence gate fails, the abstraction does not
open the continuous result.

### 2. Typed mechanism-edit graph

Start from the nine primitive ChemBench mechanisms. An LLM proposal is a patch,
not a free-form strategy:

```json
{
  "parent_id": "...",
  "edit": {
    "op": "add_factor",
    "mechanism": "arrhenius"
  },
  "residual_motif": "underprediction increases with temperature",
  "exposing_assay_group": "temperature",
  "falsifying_assay_group": "inhibitor"
}
```

Code canonicalizes the graph, compiles the law, checks units and positivity,
assigns declared parameter priors, evaluates it over the design box, rejects
duplicates, and preserves the parent if the edit fails. The common grammar can
include add/remove/replace operations for saturation, inhibition, Arrhenius,
Hill, pH, and second-substrate terms.

For the first controlled result, compiling to the existing 57-model executable
registry is acceptable if reported honestly as typed compositional retrieval.
Free-form RateLaw generation is a stronger follow-up, not a prerequisite for
testing non-myopic support expansion.

### 3. Pool-wide residual reports

Follow MDA's prompt more faithfully. Every proposer input should contain:

- every live structure and its canonical form;
- its posterior/evidence weight;
- a robust prequential residual over the history;
- signed residual association with each assay family;
- forms already tried and rejected;
- the latest action and observation category;
- remaining budget and `explore` or `refine` phase.

In the categorical layer, use a numerical report derived from each model's
predictive probabilities: observed-category surprise, signed category
innovation, and family-aggregated calibration error. Do not invent a continuous
rate for the LLM or ask it to estimate one.

### 4. A frozen proposal atlas for simulated futures

Live LLM calls at every rollout node are unaffordable and irreproducible. Build
a prospective atlas from source-only synthetic residual states before opening
the efficacy cohort:

1. generate representative pool-wide residual states from the public model
   grammar and source parameter versions;
2. query the LLM with the exact MDA-style proposer interface;
3. parse, compile, deduplicate, and evidence-screen each edit;
4. embed each residual state with fixed numerical features;
5. retrieve or interpolate the banked proposal distribution for a simulated
   branch.

The atlas must contain multiple residual histories, not merely the 18 labels
`(assay group, low/mid/high)`. History, current support, negative evidence, and
remaining budget can change the correct edit after the same latest outcome.

Use a simple, auditable transition model such as nearest medoids plus Dirichlet
smoothed edit frequencies. Do not train a flexible neural transition on a tiny
bank. A held-out transition-fidelity test should compare atlas proposals with
fresh, sealed LLM responses on unseen synthetic residual histories.

### 5. A calibrated M-open trigger

After each branch outcome:

1. record the pre-update posterior-predictive probability of that outcome;
2. update current model weights numerically;
3. compute the pool-wide residual report;
4. expand only if predictive surprise exceeds a prospectively calibrated
   threshold or the task-anchored predictive check fails;
5. add at most four compiled edits, as in MDA's ChemBench setup;
6. refit/reweight and prune back to a fixed maximum pool size.

The threshold should be calibrated on source-only histories for a declared
false-expansion rate. It should not be tuned to maximize depth gains.

### 6. Evidence and diversity-aware contraction

Use MDA's lesson but avoid deleting every alternative from the same family:

- keep the highest-evidence models;
- reserve a small number of slots for distinct edit families;
- canonicalize algebraically equivalent forms;
- shrink aggressively when posterior concentration is high and the predictive
  check passes;
- re-expand when a later observation violates the predictive check.

An ablation with union-only support is required. If union-only appears better,
inspect whether the planner is exploiting duplicate mass rather than learning a
better mechanism.

### 7. Exact structural policy improvement

Reuse the existing three-outcome, 18-assay, four-experiment exact mechanics.
Define `d1`, `d2`, and `d3` by conservative policy improvement over the previous
level, with the shallower action and continuation always available as fallback.
The terminal objective remains expected held-out log-rate MSE.

This avoids the failed continuous first-link estimators. The IID, RQMC, and
moment-shortlist gates all found low action regret but unstable full rankings;
more samples would make the method expensive without making the scientific
claim cleaner. Exact discrete planning is a feature for the first paper result,
not an embarrassment.

### 8. Hierarchical actions for later continuous transfer

After the structural result passes, let the deep planner select an option:

```text
discover family | discriminate structures | refine parameters | task-greedy
```

A one-step numerical optimizer then selects the continuous assay inside that
option using CMA-ES or a fixed Sobol shortlist. This mirrors MDA's successful
continuous design optimization while keeping the long-horizon branching small.
The inner optimizer must always include the current myopic best assay.

## Environment Changes

### Primary now: controlled ChemBench structural discovery

Use the already verified structural environment as the first LLM-native result:

- nine primitive mechanisms in the initial live support;
- compound mechanisms available only through typed proposals;
- 18 fixed assays across six mechanistic families;
- three categorical rate outcomes;
- four sequential experiments;
- terminal held-out log-rate MSE;
- opened v4 for development and untouched v5 for one-shot confirmation.

Call this a controlled structural BED benchmark. Do not claim it reproduces the
full continuous ChemBench benchmark. Its purpose is to isolate whether an LLM
proposal transition and non-myopic planning reinforce each other.

Before confirmation, stratify v5 by missing edit family and freeze all gates per
stratum. Parameter-version transfer alone is weaker than composition transfer,
so report both aggregate and edit-family results.

### Stronger follow-up: a costed ChemBench discovery corridor

Add heterogeneous experiment costs only in a prospectively new environment:

- cheap diagnostic assays expose a residual family but do not identify its
  parameterization;
- targeted assays discriminate or calibrate the newly proposed mechanism;
- an expensive broad assay gives larger immediate improvement but consumes the
  budget needed for the second-stage targeted test.

This produces an interpretable horizon gap: diagnose, propose, then target. Run
a source-only oracle opportunity audit before any LLM response. Do not tune the
costs after observing LLM behavior.

### Secondary transfer: standardized, costed NeuronBench

The paper's neuron environment naturally separates phenotype discovery from
channel identification. Repair our previous compositional formulation by:

- using standardized feature risk rather than raw MSE across currents;
- using `(protocol, repeat_count)` as the costed action;
- constructing worlds where a cheap phenotype screen identifies a channel
  family and a branch-specific pre-pulse discriminates the subtype;
- holding out complete channel compositions;
- proposing typed channel/current edits while the ODE and likelihood remain
  numerical.

Require the depth opportunity to be distributed across worlds and medians, not
driven by one slow-channel outlier.

### Small architecture validator: M-open location finding

The current known-kernel location task should remain a mechanics baseline. A
new variant can validate the architecture cheaply by hiding:

- source count;
- field kernel family;
- sensor saturation or bias.

The LLM proposes a typed kernel/source-topology edit from a signed residual map;
exact numerical inference localizes sources. Score held-out field prediction,
not source-coordinate RMSE, because source count varies. A coarse wide-area
measurement followed by a targeted local measurement can create the horizon
gap. This is useful as an ablation, but less LLM-native than mechanism discovery
in ChemBench.

### Do not use the failed Bongard interface as the next environment

The visual serving gate failed answer-conditioned calibration before any policy
endpoint. More planning cannot repair a belief model that does not obey the
simulated answer. The compiled numerical transition proposed here addresses that
failure by asking the LLM for executable hypotheses and letting the simulator
determine their consequences.

## Required Controls

Every final comparison should reuse the same assays, histories, random numbers,
proposal count, and numerical inference where applicable:

1. dynamic LLM transition with d1/d2/d3;
2. exact compute-matched myopic planning;
3. MDA-style proposer used only after realized observations, with no proposal
   transition inside simulated futures;
4. history-blind LLM proposals with matched calls;
5. random valid typed edits with matched proposal count;
6. fixed primitive support with no expansion;
7. registry-oracle transition as a structural upper bound;
8. naive thinking as a behavioral LLM baseline.

The comparison against real-history-only MDA is central. It isolates the value
of non-myopically anticipating future model discovery from the value of simply
having a good LLM proposer.

## Gate Sequence

### Gate 1: revised oracle mechanics, zero calls

Add the predictive trigger, pool cap, evidence/diversity pruning, and residual
features to the exact categorical planner. Require d1/d2/d3 monotonicity and at
least 5% successive aggregate improvement on opened v4 under the registry
oracle. Also require no truth identifiers in policy state or proposer features.

### Gate 2: proposal semantics, small paid block

On source-only synthetic residual histories, require:

- strict schema and executable validity;
- low duplicate/reproposal rate;
- meaningful outcome-conditioned movement;
- held-out missing-edit-family recall;
- positive proposal-action ranking relative to random edits;
- no endpoint labels or v5 outcomes in prompts.

Failure closes the interface without opening a policy endpoint.

### Gate 3: atlas transition fidelity

On unseen synthetic residual histories, compare atlas proposals with fresh
LLM responses. Require edit-family agreement and low induced one-step action
regret. This is the first-link gate for the proposal world model.

### Gate 4: opened-v4 policy development

Require dynamic LLM d2 and d3 to improve successively, beat the history-blind
and random-edit controls, and beat real-history-only MDA at matched calls. Read
per-family paired losses, first-action changes, proposal triggers, accepted
edits, and pruning events.

### Gate 5: sealed-v5 confirmation

Freeze the complete v5 runner before opening outcomes. Require monotonic
d1/d2/d3 terminal risk, positive paired gains, nonworse log loss, and gains over
the compute-matched myopic and real-history-only MDA controls. Report the oracle
gap, not just whether the LLM arm wins.

## Immediate Build Order

1. Extend the existing categorical `DynamicState` with prequential surprise,
   pool-wide residual features, tried edits, and phase.
2. Add a typed registry-edit compiler and evidence/diversity pool contraction.
3. Re-run the zero-call registry-oracle ladder with trigger and pruning.
4. Freeze a small DeepSeek nonreasoning proposer-atlas protocol using exact JSON
   edits and matched history-blind controls.
5. Only if semantics and transition fidelity pass, evaluate the LLM transition
   on opened v4 and then open v5 once.

Do not reopen the continuous posterior-sampling tree, increase rollout counts,
or spend on another free-form strategy interface before these gates.

## Paper-Level Claim If It Works

The clean claim is not that LLMs can numerically plan better than Bayesian
methods. It is:

> LLMs can serve as executable, residual-conditioned hypothesis proposers, and
> a Bayes-adaptive planner can improve experiment choice by anticipating how
> future observations will change the available scientific model class.

That is a genuine non-myopic extension of MDA, and every term in the claim has
a matched control and an inspectable intermediate artifact.

## Source

Kevin Murphy, *Model Discovery Agent: LLM-assisted Bayesian experiment design
for data-efficient discovery of mechanistic world models*, arXiv:2608.09696,
2026: https://arxiv.org/abs/2608.09696
