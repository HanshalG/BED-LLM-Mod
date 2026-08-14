# ChemBench Non-Myopic M-open Architecture Protocol

Date frozen: 2026-08-14 (Europe/London)

## Research Claim

Test whether non-myopic Bayesian experiment design improves held-out scientific
forecasting when the belief transition itself includes residual-conditioned LLM
model discovery.

The LLM must do one irreducible job: propose new executable mechanism
structures that are absent from the current support. Numerical code owns
parameter inference, evidence, posterior updates, experiment scoring, and final
forecasting. The headline comparison is not an LLM that directly chooses an
experiment against another LLM. It is dynamic-support planning versus
call-matched myopic planning over the same bank of LLM-proposed branch
transitions.

This protocol authorizes zero-call mechanics implementation only. A separate
prospective serving protocol must bind prompts, model, seeds, cohorts, caps, and
semantic gates before any LLM response.

## Source Bindings

- MDA paper: arXiv `2608.09696`, v3 dated 2026-08-13.
- Official ActiveSciBench release: `scientific-discovery/LLM-AutoSciLab`.
- Release commit: `acf160eb6c96897748dd92b152703b59b74efc05`.
- Release tree: `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`.
- Structural opportunity result:
  `CHEMBENCH_MOPEN_NONMYOPIC_OPPORTUNITY_V2_RESULT_20260814.md`.

The official active source set has 57 worlds. In the pinned release, 21 of
these are marked as novel mechanisms outside the standard compositional
library. This is stronger for an LLM-native claim than a closed menu containing
all 57 labels, but it must be disclosed because the paper describes its fair
ChemBench set more abstractly as canonical and compound rules.

## Changes From MDA

MDA uses residual-triggered support expansion but chooses each experiment with
one-step model information gain. We retain its division of labor and change the
controller:

1. The planner simulates future observations.
2. Each simulated branch computes the residual that would be visible then.
3. A branch-conditioned proposal kernel may add new executable structures.
4. Those structures are fitted and weighted on the simulated branch history.
5. The action is scored by expected terminal held-out predictive risk after
   one, two, or three such transitions.

Thus the belief transition is

```text
B' = UPDATE(B, action, observation, PROPOSE(history, residual_report))
```

rather than a Bayes update on fixed support. Depth has scientific meaning only
if this transition is path dependent and calibrated.

## Environment Changes

### Primary Environment: M-open ChemBench

- Use the released seven-dimensional assay interface and 1% multiplicative
  observation noise.
- Use a four-experiment policy budget so d1, d2, and d3 can be compared without
  making the endpoint trivial.
- Score predictions on 1,000 sealed, seed-bound assay points using mean squared
  log-rate error; report RMSLE and exact-accuracy thresholds only as secondary
  views.
- Give the agent the physical vocabulary and safe expression language, never
  the active world list, world identifier, source function, parameter values,
  or endpoint outcomes.
- Initialize with only simple single-mechanism structures. Compound and novel
  truth worlds are deliberately outside the live support.
- Include both standard compound worlds and the pinned release's active novel
  worlds. Report them as separate strata.

The initial fixed-support audit used 18 assays and categorical three-bin
observations to establish horizon opportunity. Efficacy should restore noisy
real-valued observations. Planning may discretize each posterior predictive
into three common-random-number quadrature outcomes, but fitting and realized
updates use the raw rate.

### Design Pool

Do not expose all source-optimal probes or optimize against the private truth.
At each real or simulated belief state, numerical code constructs a small
action set from the public bounds:

- one assay maximizing current between-model predictive disagreement;
- one assay maximizing posterior predictive variance within the live support;
- one residual-directed assay for each of the two strongest unexplained input
  dependencies;
- one broad space-filling assay;
- one deterministic carry-over assay selected at the parent state.

Deduplicate to at most six actions. A future continuous CMA-ES variant is an
ablation after the discrete mechanics passes; it must not be introduced only
after seeing an unfavorable endpoint.

### Secondary Environments

- Do not use deterministic NeuronBench as the main depth result: its small
  protocol menu often saturates within three experiments.
- Stochastic NeuronBench is a useful portability test after ChemBench because
  repeated measurements create a nontrivial design tradeoff, but it requires a
  simulation-based likelihood and should not delay the primary result.
- ActiveSciBench-GRN is the next candidate if ChemBench proposal semantics
  pass but its finite grammar makes the LLM look replaceable. Graph edits give
  a larger, naturally compositional support while keeping executable forward
  models.

## Executable Model Representation

Use a typed rate-law intermediate representation, not unrestricted Python and
not a closed 57-way class label.

```text
RateLaw
  name: string
  expression: arithmetic AST
  parameters: [Parameter(name, lower, upper, transform)]
  rationale: short residual-to-mechanism account
```

The AST permits only the seven public inputs, declared parameters, numeric
constants, `+ - * / **`, and allowlisted `exp`, `log`, and `sqrt`. Compilation
must reject undeclared names, mutation, indexing, attributes, calls outside the
allowlist, non-finite evaluation, invalid bounds, and expressions that fail a
public-bound stress grid. Canonicalize commutative operations and parameter
renamings before deduplication.

This representation can express known primitives, new compositions, and the
release's exotic active mechanisms. A factor-enum-only representation is a
useful closed-grammar ablation but cannot be the headline because enumeration
would remove the need for an LLM.

## Belief State

```text
MOpenBelief
  live_models: evidence-weighted fitted executable structures
  reserve_models: valid proposals retained at low prior weight
  parameter_particles: per-structure parameter posterior
  residual_report: prequential and cross-validated residual diagnostics
  outside_support_mass: calibrated missing-model alarm
  history: raw assay/observation pairs
```

The live pool is capped at 12 structures, matching the scale used by MDA. Keep
up to 12 additional reserve structures so evidence pruning cannot permanently
erase a useful but not-yet-identified mechanism. When the MAP weight is above
0.9 and endpoint-aligned residual is below the frozen threshold, shrink the
live pool to four while retaining the reserve. Re-expand before proposing if
the predictive check fails.

`outside_support_mass` is a disclosed meta-controller score, not fake Bayesian
mass. It is calibrated on source-only synthetic histories from prequential
surprise, lack-of-fit, and proposal coverage. It prevents reporting false
certainty and triggers expansion, but it is not included as a mechanism in
symbolic-accuracy scoring.

## Residual Report

Use endpoint-aligned prequential residuals rather than only in-sample fit:

- standardized error on the newly observed rate predicted before acquisition;
- leave-one-experiment-out log-rate residual for each live model;
- signed rank correlation of MAP residuals with each public input;
- regions where high-evidence models disagree;
- forms already tried, their evidence, and their residual signatures;
- remaining experiment budget and current explore/refine phase.

Do not expose the true mechanism, source equation, hidden parameter values, or
held-out endpoint labels. The proposer receives negative evidence about failed
forms, following MDA, and must return structurally different candidates.

## Proposal Transition

The interface is a pure function of public branch state and a seed:

```text
propose(history, residual_report, tried_canonical_forms, seed)
    -> up to 4 RateLaw objects
```

For every planning state, cache accepted raw responses and compiled models by
the exact prompt hash, branch-history hash, model route, schema hash, and seed.
Dynamic and matched controls must consume the same immutable cache. No policy
may request an extra proposal because its first result was inconvenient.

Expansion runs when either prequential surprise or outside-support mass exceeds
the frozen threshold. For the initial mechanics, always calling the scripted
proposer is allowed as a stress test, but the eventual LLM policy must use one
prospectively calibrated trigger shared by all methods.

## Numerical Inference

- Fit each structure with transformed positive parameters and broad physical
  priors.
- Use adaptive-tempered SMC per structure for the final method; a Laplace or
  importance approximation is allowed only in zero-call mechanics and must be
  compared against SMC on a fixed synthetic suite.
- Weight structures by marginal evidence with a complexity prior. Never use an
  LLM self-reported confidence as posterior mass.
- Use common random numbers for parameter particles, simulated outcomes, and
  endpoint assays across depths and controls.
- Refit every newly proposed structure on the entire branch history. Do not
  assign new hypotheses zero historical mass by aligning old vectors.

## Planner

The objective is expected terminal held-out predictive MSE, not an entropy sum.
At a branch leaf, average the squared log-rate error of the posterior predictive
over a public seed-bound proxy query set disjoint from final evaluation. The
private evaluation set is opened only after the policy terminates.

Use exact enumeration over at most six candidate actions and three predictive
outcomes per action. Use progressive widening at deeper nodes: six root actions,
at most three child actions, and at most two grandchild actions. Branches below
frozen predictive-mass tolerance are omitted and their probability is recorded.

Depth means the number of future dynamic-support transitions scored. d1 must
still simulate the proposal/update caused by its candidate observation; it is
not a fixed-support greedy shortcut.

## Controls

All comparisons are paired by truth world, parameter version, initial support,
proposal seeds, observation noise, parameter particles, and endpoint assays.

1. `dynamic_d1`, `dynamic_d2`, `dynamic_d3`: the primary depth comparison.
2. `call_matched_myopic`: receives every cached proposal generated for d3 but
   chooses using only expected post-first-step terminal risk.
3. `mda_myopic_voi`: residual-triggered proposal plus the paper's one-step
   between-model information objective.
4. `fixed_support_d3`: same planner with support refresh disabled.
5. `history_blind_refresh_d3`: proposal prompts omit branch observations and
   residuals but keep grammar, call count, and seeds.
6. `dictionary_refresh_d3`: a no-LLM residual-to-factor heuristic.
7. `random`: random assays with the same realized support update.
8. `naive_thinking`: a separately labelled LLM baseline that directly chooses
   assays and forecasts from history.
9. `complete_support_oracle`: non-deployable upper yardstick only.

The call-matched control is primary. A depth win that disappears against it is
a compute-width result, not evidence for non-myopic planning.

## Zero-Call Mechanics Gate

Before model serving, implement two proposal substitutes:

- `oracle_transition`, which can add the hidden source structure and defines an
  upper ceiling;
- `scripted_residual_transition`, which sees only the same public residual
  report as the future LLM and maps signatures to safe grammar edits.

On untouched `v3` source states, require:

1. finite, normalized branch beliefs and forecasts;
2. no private truth data in prompts, transitions, or design construction;
3. exact cache reuse across dynamic and matched controls;
4. d2 aggregate terminal risk at least 5% below d1 under oracle transition;
5. d3 aggregate risk at least 5% below d2 under oracle transition;
6. each successive comparison wins more paired truth cells than it loses;
7. at least 25% of d2 root actions differ from d1;
8. fixed-support and history-blind ablations do not spuriously receive the
   branch-conditioned candidates;
9. adversarial invalid proposals fail closed without changing prior support;
10. scorer replay from banked transition artifacts is producer independent.

Failure means redesign the environment or mechanics before any model call. It
does not authorize lowering a gate.

## Future LLM Semantic Gate

Freeze a small development cohort from untouched `v4` states before responses.
The exact thresholds belong in its serving protocol, but the gate must be
conjunctive and include:

- strict schema and compile validity;
- unique structural proposal rate;
- simulated-answer obedience across low/mid/high branches;
- residual-conditioned proposal advantage over history-blind prompts;
- proposal evidence or held-out fit improvement over the parent support;
- coverage of genuinely outside-support truth components;
- positive candidate-action rank correlation with realized terminal risk;
- no worse calibration than the dictionary proposer;
- clean serving, complete banks, fixed retries, and exact cost accounting.

Only a semantic and ranking pass can authorize paired efficacy. A codec pass or
interesting qualitative strategy is insufficient.

## Efficacy Gate

Use untouched `v5` parameter states and predeclare the world strata, sample
size, seeds, and power calculation. The primary endpoint is paired terminal
log-rate MSE for `dynamic_d2` versus `call_matched_myopic`; d3 monotonicity is a
co-primary or secondary endpoint depending on the mechanics power audit.

Require a prospective improvement threshold, paired bootstrap interval,
nonworse calibration/log loss, positive ranking fidelity, and complete
per-world reporting. Report standard compounds and novel mechanisms separately.
No symbolic-accuracy or depth headline may be rendered if the call-matched
comparison is null.

## Implementation Layout

```text
environments/chembench_mopen/
  source.py       pinned source adapter and privacy boundary
  ir.py           typed RateLaw parser, canonicalizer, compiler
  residuals.py    endpoint-aligned public residual report
  inference.py    parameter SMC, evidence, posterior predictive
  belief.py       live/reserve pools and expansion controller
  proposer.py     pure proposal protocol plus cache adapters
  planner.py      dynamic d1/d2/d3 and matched objectives
  evaluation.py   paired endpoints, controls, and diagnostics
scripts/
  chembench_mopen_mechanics.py
  chembench_mopen_semantic_gate.py
tests/
  test_chembench_mopen_ir.py
  test_chembench_mopen_belief.py
  test_chembench_mopen_planner.py
  test_chembench_mopen_privacy.py
```

Keep this route separate from `location_finding`. The location implementation
is useful evidence about rollout diagnostics, but its free-form strategy text,
fixed analytical posterior, and entropy-based score are the three architectural
choices this protocol deliberately replaces.
