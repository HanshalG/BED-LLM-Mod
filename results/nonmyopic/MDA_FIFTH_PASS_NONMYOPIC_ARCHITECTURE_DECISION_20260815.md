# MDA Fifth-Pass Non-Myopic Architecture Decision

Date: 2026-08-15 (Europe/London)

## Bottom Line

Murphy's Model Discovery Agent (MDA) strengthens the case for keeping the LLM
out of numerical inference and experiment scoring. Its successful division of
labor is:

1. the LLM proposes executable mechanistic structures;
2. adaptive SMC fits their parameters and computes marginal evidence;
3. a predictive check decides when the current support is inadequate;
4. numerical VoI chooses an experiment;
5. an evidence- and fit-aware controller expands or shrinks the model pool.

MDA's acquisition itself is explicitly one-step and myopic. Our strongest
extension is therefore not a deeper version of its VoI formula. It is a
Bayes-adaptive planner whose transition includes the path-dependent creation of
new executable hypotheses. An early experiment can be valuable because its
residual causes the LLM to propose a useful mechanism that did not exist in the
root support, making a later targeted experiment possible.

ChemBench remains the primary environment. The verified categorical oracle
ladder already establishes the required opportunity on an untouched cohort:
d1/d2/d3 terminal MSE 0.03919425/0.03203636/0.02924680. The continuous screen
failed because fixed 16-particle parameter banks and a global quantized policy
table do not transfer reliably to medium and hard truths. The next action is a
zero-call posterior-calibration gate, not another LLM proposal experiment.

## What The Paper Changes

### 1. Treat the LLM as a proposal kernel, not a policy

The LLM should not output a natural-language multi-step strategy, posterior
weights, entropy, or query scores. It should emit a typed structural mutation
that numerical code can compile, fit, compare, and reject. This is both more
LLM-native and more testable: the irreducible semantic work is naming a useful
mechanism, while Bayesian machinery owns every quantitative claim.

### 2. Model open-world support as a transition, not a scalar

The current scalar `outside_mass` says that the support may be wrong, but it
does not predict what support will appear after a particular residual. That is
insufficient for non-myopic planning. Replace it with a calibrated proposal
transition:

```text
q(new executable edit | current pool, per-model residual map, experiment,
                        observation, remaining budget and phase)
```

This transition is the LLM-native world model inside the planner. It must be
measured directly: the same public residual code should induce similar edit
distributions in the banked atlas and in held-out live LLM calls.

This follows the paper's actual ChemBench proposer more closely than a generic
strategy prompt: MDA shows the LLM every form already tried, each form's
residual, the input most correlated with the best form's residual, and the
remaining budget/phase. It also supplies the complete mechanism vocabulary.
The honest LLM contribution is therefore residual-conditioned compositional
selection and editing, not discovery from no scientific prior.

### 3. Use evidence to control structural complexity

Every structure must have a parameter posterior and marginal evidence. Model
probabilities must not be equal counts of versions or point-fit scores. Add a
small explicit complexity prior to handle equivalent over-elaborated laws,
because Murphy's ablation shows that M-open exploration creates near-duplicate
structures and that adaptive pool contraction is needed to recover hard-tier
performance.

### 4. Separate discovery, discrimination, and parameter refinement

Murphy's Eq. 10 intentionally scores between-structure disagreement using
per-class posterior means. Our terminal forecast loss also depends on
within-structure parameter uncertainty. Candidate experiments should therefore
be generated under three numerical modes:

- `discover`: maximize expected predictive-check violation or proposal change;
- `discriminate`: maximize between-structure disagreement;
- `refine`: maximize within-structure parameter reduction.

The exact non-myopic objective then scores the union of these candidates by
expected terminal held-out log-rate risk. The modes are shortlist generators,
not separate hand-written policies.

The predictive check should contain the evaluation target. MDA describes an
external `QUERY` check but reports its experiments with the `SUMMARY` check;
that is safe in its benchmarks because the summary contains the target. In our
NeuronBench variants this condition must be tested explicitly so that a lossy
summary and proposal model cannot agree with each other while missing the
forecast quantity.

### 5. Make approximation fidelity a first-class gate

The paper uses adaptive-tempered SMC with 100 parameter particles per ChemBench
structure, target ESS 0.6, three random-walk Metropolis moves per tempering
rung, and up to 80 rungs. Our 16 fixed LHS particles are not a comparable
posterior approximation. Before planner work, require posterior-predictive and
evidence stability across independent particle banks.

## Proposed Architecture

### A. Executable mechanism-edit graph

Represent candidate laws as a graph of typed edits rather than unrelated free
form expressions. Initial nodes are the nine primitive mechanisms. Edges apply
semantic edits such as:

- add Arrhenius temperature dependence;
- add competitive, uncompetitive, noncompetitive, product, or substrate
  inhibition;
- replace Michaelis-Menten saturation with Hill cooperativity;
- add a second-substrate ping-pong term;
- remove a factor contradicted by residuals.

The LLM may still propose a genuinely new typed subtree, but common compound
mechanisms become local, deduplicable edits. Each proposal returns:

```json
{
  "parent_id": "...",
  "edit": {"op": "multiply", "mechanism": "arrhenius", "parameters": []},
  "residual_motif": "underprediction grows with inverse temperature",
  "exposing_region": {},
  "falsifying_region": {}
}
```

Code compiles the edit to the existing safe `RateLaw` IR, canonicalizes it,
checks finite behavior over the design box, fits it with SMC, and rejects
duplicates or unsupported parameter bounds. This graph gives the planner a
compact, structured proposal state and lets us measure whether residuals route
to the right edit family.

### B. Per-structure adaptive SMC

Add a transformed-coordinate SMC backend with:

- 100 particles per live structure for the first serious calibration;
- adaptive likelihood tempering to ESS 0.6;
- systematic resampling at every rung;
- three bounded random-walk rejuvenation moves per rung;
- proposal scales based on current transformed particle spread;
- log-evidence accumulation per structure;
- independent inference and planning banks;
- diagnostics for temperature rungs, ESS, acceptance, boundary hits, and
  evidence variability.

Real observations should refit or rejuvenate the live structures. Planning
branches should begin from an immutable posterior snapshot, apply exact
incremental likelihoods, and perform only a short branch-local rejuvenation if
the branch ESS collapses. Never share mutable particles between branches.

### C. Bayes-adaptive M-open belief state

Use the following planning state:

```text
B = (history,
     live structures and SMC posteriors,
     reserve edit nodes,
     structure evidence weights,
     prequential residual map,
     proposal-transition state,
     remaining experimental budget)
```

An action-observation transition is:

```text
experiment -> raw observation -> SMC update -> predictive check
           -> conditional structural edits -> fit/evidence/prune
           -> next belief state
```

This transition, rather than a static likelihood-only update, is the scientific
object that makes depth meaningful.

### D. Local particle scenario tree

Remove the global 10,000-state quantized policy table. It evicted tens of
thousands of states and transferred policies between posterior states that were
only superficially similar.

Use a local root-sampled scenario tree instead:

- sample a structure, parameter particle, and observation-noise path at root;
- reuse common random numbers across actions and policy depths;
- branch on posterior-predictive quantiles or sampled outcomes;
- run the proposal transition after each branch observation;
- use double progressive widening for experiments and observations;
- cache only exact local belief descendants within one root decision;
- define d(k) as policy improvement over the frozen d(k-1) continuation.

This preserves the verified conservative policy ladder while eliminating
cross-state policy aliasing.

### E. Continuous design search

Keep the frozen 18-assay menu until the SMC and local-tree gates pass. Then use
the official seven-dimensional design box with a fixed evaluation budget:

1. common Sobol designs for global coverage;
2. exposing and falsifying regions supplied by accepted edits;
3. local CMA-ES refinement of the three shortlist modes;
4. exact common-random-number rollout scoring on the shared finalist set.

This imports the part of Murphy's ChemBench ablation that specifically helped
hard mechanisms without conflating design optimization with the posterior
repair.

### F. Proposal atlas for simulated futures

Calls inside a rollout tree are neither affordable nor reproducible. Before
outcomes are opened, construct a seed-bound atlas of LLM edits for synthetic
public residual histories generated from source mechanisms. Index it by a
continuous residual embedding plus the current edit graph, not by a coarse
hand-written label alone.

At evaluation time, realized observations may call the LLM once when the
predictive gate opens. Simulated branches sample from the frozen atlas. A
separate transition-fidelity gate compares atlas and live proposal distributions
on held-out synthetic residual histories. The dynamic planner, compute-matched
myopic control, fixed-atlas control, history-blind control, and random-proposal
control must all reuse the same banked responses.

## Environment Changes

### Primary: ChemBench compositional M-open

Use a prospectively frozen split by mechanism composition, not merely by
parameter version.

- Initial support: the nine primitive laws.
- Hidden truths: held-out two- and three-factor compositions.
- Development compositions: disjoint from confirmation compositions.
- Parameters: sampled from declared broad priors, with truth draws held out
  from inference particles but inside the prospectively declared support.
- Observation: raw reaction rate under the released multiplicative-noise model.
- Budget: six to eight sequential experiments.
- Evaluation: disjoint 1,000-query log-rate MSE, standardized per task for the
  primary aggregate; raw RMSLE and symbolic equivalence as secondary metrics.
- Structural event: an early assay exposes a residual motif, the proposal
  transition adds an edit, and a later assay discriminates or calibrates it.

The source-only oracle gate must first show that the exact dynamic-support d1,
d2, and d3 policies are monotonic under the raw likelihood on every tier. Then
the proposal-transition gate asks whether the LLM recovers enough of that
opportunity.

### Secondary: costed NeuronBench protocol design

The prior compositional extension had a large mean depth effect but was
dominated by one slow-Na+T outlier; median behavior barely moved. Repair the
environment rather than claiming that result.

- Use the paper's standardized feature MSE, not unnormalized raw MSE across
  currents with very different scales.
- Make `(protocol, repeat_count)` the action and charge repeats against a fixed
  budget, as in the paper's stochastic extension.
- Include channel pairs for which a cheap phenotype screen identifies the
  relevant family, then a branch-specific conditioning pre-pulse identifies the
  current.
- Hold out complete channel compositions across development and confirmation.
- Let the LLM propose a typed current/channel edit; ODE simulation, feature
  likelihood, and evidence remain numerical.
- Compare dynamic depth against a compute-matched myopic policy using the same
  protocol and proposal candidates.

This environment has the cleanest natural story for non-myopia: spend little
to learn which signal matters, then spend repeats and a specialized protocol on
that signal.

### Supporting: ForceBench hidden-source discovery

Combine uncertainty over the force-law family with uncertainty over hidden
source count and location. A wide exploratory launch can reveal a residual
region; after a screened-law or hidden-source proposal, a targeted launch
estimates its scale or location. This is a good low-dimensional validator for
the proposal-aware tree, but the semantic LLM contribution is narrower than in
ChemBench.

### Supporting: location finding with an unknown field model

Known-kernel source localization is not LLM-native. A useful extension would
hide source count and the field/sensor law, with the LLM proposing a safe kernel
AST and trans-dimensional SMC localizing sources. Keep this as an ablation of
the architecture, not the headline: it overlaps substantially with the
ForceBench hidden-source story.

## Frozen Gate Sequence

### Gate 1: posterior calibration, zero calls

On source-only synthetic histories and already-open v4 development data,
compare fixed 16-particle LHS, fixed 100-particle importance sampling, and
adaptive SMC with 100 particles. Initially fit the correct structure only to
isolate parameter inference.

Require:

- finite normalized posteriors and evidence;
- no truth leakage from parameter versions or query rows;
- posterior-predictive log-rate MSE better than 16-particle LHS and nonworse
  than fixed 100-particle importance sampling on every tier;
- stable evidence ordering across two independent SMC seeds;
- non-collapsed ESS and nondegenerate Metropolis acceptance;
- simulation-based calibration or rank coverage consistent with the declared
  prior.

### Gate 2: branch fidelity, zero calls

Against high-sample Monte Carlo one-step values, require predictive branches to
reach Spearman rho at least 0.8, low top-one regret, and correct root action on
most source worlds. This must pass before any deeper policy result is read.

### Gate 3: raw continuous oracle ladder, zero calls

Use the local scenario tree and dynamic registry-oracle edit transition on
opened development data. Require at least 5% aggregate improvement on d2 over
d1 and d3 over d2, practical paired wins greater than losses, and nonworse
performance on every tier across independent planning banks.

### Gate 4: LLM proposal-transition semantics, small paid block

On synthetic residual histories with sealed structure labels, require strict
schema validity, executable validity, meaningful residual-conditioned movement,
held-out edit-family recall, calibrated proposal probabilities, and positive
proposal-action ranking fidelity. Failure closes the interface without opening
an efficacy endpoint.

### Gate 5: sealed efficacy

Open a disjoint composition cohort once. Compare dynamic d1/d2/d3, exact
compute-matched myopic, fixed/history-blind proposal, random proposal, random
design, and naive thinking. Use paired common-random-number observations and
report terminal standardized log-rate risk, raw RMSLE, symbolic recovery,
proposal recall, cost, and uncertainty.

## What Not To Try Next

- Do not increase StrategyEIG depth, rollout count, or thinking tokens on the
  current fixed-particle continuous planner.
- Do not use entropy reduction as a proxy for terminal forecast risk.
- Do not retain categorical rate bins for the efficacy environment.
- Do not make unconstrained live LLM calls inside rollout branches.
- Do not evaluate a new LLM proposal interface before SMC and branch-fidelity
  gates pass.
- Do not rely on a scalar unknown model to stand in for future structural
  proposals.
- Do not claim the NeuronBench compositional mean as a positive depth result;
  its effect is tail-dominated.

## Immediate Action

Implement the adaptive-SMC posterior calibration as a standalone numerical
module and runner. Keep it correct-structure-only, fixed-assay, and planner-free
for the first gate. If it passes, integrate immutable SMC snapshots into the
local scenario tree. If it fails, repair prior transforms or likelihoods before
spending on any LLM call.

No API/model call or new sealed scientific endpoint was opened for this
decision.

## Source

Kevin Murphy, *Model Discovery Agent: LLM-assisted Bayesian experiment design
for data-efficient discovery of mechanistic world models*, arXiv:2608.09696,
2026: https://arxiv.org/abs/2608.09696
