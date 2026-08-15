# NeuronBench Compositional M-Open Opportunity Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Test, without an LLM call, whether a compositional extension of the released
NeuronBench simulator contains a genuine dynamic-support planning-horizon
opportunity. Stock deterministic NeuronBench is already closed as the primary
depth environment because its single-current worlds and nine-protocol menu
often saturate after one revealing experiment. This protocol changes the
scientific task prospectively rather than repeating or rescuing that screen.

The intended LLM-native transition is a typed mechanism edit: after seeing a
protocol-to-spike-count residual, the LLM may add one executable ion-current
component. Numerical code owns simulation, likelihoods, posterior weights,
experiment design, and held-out forecasts. This audit replaces the future LLM
with a registry oracle and asks whether planning through two sequential edits
can beat one-step dynamic support.

## Source Binding

Official repository:

```text
https://github.com/murphyk/neuronbench
commit: c354622458c460b419cab821d482c879f0578377
tree:   4b0fba903168abcdd85db9336a37944670a990c9
```

Required source-file SHA-256 values:

```text
neuronbench/worlds.py
  d462834c969cb5e20b103a14971e3ecc9db49cc90696af478e90d4d2f5b95d64
neuronbench/stochastic.py
  20f897c97521fe30e8157ec31a227428fc4af826d94b5bb0a595aa98d0879b96
neuronbench/protocols.py
  38a38610a7269236486abea159da706148cd62b52b52b175136a2a8aa29e9b04
neuronbench/features.py
  7d4f161e3dd4cec1e0dd56a44c11cc6f7687a892932d3fdb43f0fb6ced801aea
neuronbench/evaluator.py
  7e559789b33c1e6412e443affb3b5de65fdedc124facd302a3372d4de7849ad7
```

The audit imports this checkout by explicit path. It must not vendor or modify
the upstream source.

## Mechanism Population

The six public primitive edits are, in this fixed order:

```text
0 z_rebound: add Z
1 h_sag: add Ih
2 na_fatigue: enable slow Na inactivation
3 ca_rebound: add T-type Ca
4 d_type: add D-type K
5 textbook_M: add M-current
```

A mechanism is a six-bit mask. The executable candidate universe contains the
plain mask, all six singleton masks, and all fifteen unordered two-edit masks.
The truth population is exactly the fifteen two-edit masks with a uniform
prior. No singleton or plain truth is included. Multiple extra `Chan` objects
are passed together to the released deterministic simulator; the slow-Na edit
sets its released `slow_na` flag.

Support begins with only the plain mechanism. After every observation, the
registry-oracle proposer enumerates every undiscovered mechanism that adds
exactly one primitive to any discovered support member, scores each candidate
on the complete history, and adds only the highest-evidence candidate. Ties use
ascending mask order. Support is never handed a two-edit mechanism directly
from the plain state and is not pruned in this audit.

This one-edit restriction is the prospective contract for the later LLM codec;
it is not selected from response outcomes.

## Designs And Budget

The action set is the exact ordered nine-protocol `neuronbench.protocols.POOL`:
four textbook steps followed by five advanced conditioning or paired-pulse
protocols. Each action may be used once. Every policy executes four experiments
and replans after each observation.

The observation is the deterministic released test-window spike count at
`dt=0.01`. Belief updates use a Gaussian count kernel with fixed standard
deviation `1.0` spike. The same kernel updates the full speculative truth
distribution and computes evidence within the dynamically discovered support.
All log weights are normalized stably; malformed or nonfinite responses close
the audit.

## Held-Out Forecast Objective

The common query battery is constructed before simulation by taking the union
of all six stock worlds' released `test` protocol segment lists, preserving
world and within-world order, removing exact segment duplicates, and removing
any segment list equal to one of the nine design actions. Human-readable source
labels are discarded. The resulting battery must be identical for every truth
and nonempty.

For an inference support posterior `p(m|D)`, the forecast at a query protocol is
the posterior mean deterministic spike count. Terminal loss for speculative
truth `w` is mean squared error over this common battery. Planning value is the
posterior expectation of that same loss under the full fifteen-world
speculative distribution. Entropy, structure accuracy, and truth mass are
diagnostics only.

## Policies

Define a conservative finite-budget policy ladder:

- `d1` chooses the action with the smallest expected leaf risk after one
  observation and one support refresh, then replans greedily at the next real
  state.
- `d(k)` evaluates each action followed by the complete remaining-budget
  continuation under `d(k-1)`, and chooses the lowest expected terminal risk.
  Executing `d(k)` applies the same policy improvement at every reached state.
- Branches merge speculative worlds that produce the same integer spike count.
  All depths use the same exact branches, priors, likelihoods, proposer, action
  order, and tie tolerance `1e-12`.

The primary comparison is dynamic-support d1/d2/d3. Required controls are:

- full-support d1, initialized with all 22 mechanisms and no proposal;
- plain fixed-support d1;
- dynamic random action selection with the same support refresh;
- a future history-blind one-edit proposer using the same cached candidates.

No LLM baseline is called by this audit.

## Frozen Gate

All conditions are conjunctive:

1. Source commit, tree, hashes, primitive order, 22 candidate masks, 15 truth
   masks, nine actions, and query construction match this protocol.
2. Aggregate expected held-out MSE improves by at least 5% for d2 versus d1.
3. Aggregate expected held-out MSE improves by at least 5% for d3 versus d2.
4. Each successive comparison wins on more of the fifteen paired truths than
   it loses after ties within `1e-9` are removed.
5. d3 is strictly better than d1 on at least twelve of fifteen truths and is
   not worse in aggregate than full-support d1 by more than 10%.
6. Dynamic d2 and d3 are behaviorally distinct from their predecessors: the
   root changes or at least 20% of prior-reachable nonterminal histories choose
   a different action.
7. Planned prior risk and explicit uniform truth replay agree within `1e-10`
   for every dynamic depth.

Failure closes this exact compositional deterministic formulation. Truth
pairs, query protocols, likelihood scale, budget, proposer width, action menu,
and thresholds may not be changed after response matrices are inspected to
rescue it.

A pass authorizes only a small LLM mechanism-edit semantic and transition gate.
It does not authorize an efficacy claim, a stochastic endpoint, or a paper
headline.

## Required LLM Descendant

The proposer must receive only public mechanism grammar, discovered executable
models, protocol-to-count history, numerical residual summaries, and remaining
budget. It returns one typed edit with channel reversal class, activation
direction, optional inactivation, kinetic parameter bounds, and rationale.
Code compiles and fits the edit. Dynamic d1/d2/d3, call-matched myopic,
history-blind, fixed-support, random-edit, random-action, and separately labeled
naive-thinking controls must reuse the same accepted proposal bank.

Model/API calls and cost before this protocol was frozen: `0` / `$0`.
