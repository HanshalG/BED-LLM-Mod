# NewtonBench Snell-Law Non-Myopic Opportunity Audit

## Status

Frozen before computing any EIG, posterior entropy, selected action, invalid
outcome partition, or depth-two planning outcome. This is a zero-call
structural gate following the separately closed sound-speed null. Failure
closes this Snell-law construction before model use; passing authorizes only a
fresh, separately preregistered development smoke.

## External Environment

- Official repository: `HKUST-KnowComp/NewtonBench`.
- Pinned commit:
  `912a4ba5f4356ddd06acc16e44460ca30be4abc2`.
- Module: `m4_snell_law`.
- System: `vanilla_equation`.
- The task prompt does not disclose equation difficulty or law version, so the
  prior is uniform over the nine released laws: `easy`, `medium`, and `hard`,
  crossed with `v0`, `v1`, and `v2`.
- Finite observation means come from the pinned official law functions.
- Finite observations use the benchmark's relative Gaussian model:
  `Normal(mean, max(abs(mean * noise_level), 1e-9))`.
- An official law-domain failure (`NaN`) is modeled as the exact categorical
  observation `invalid`, matching the benchmark's user-facing non-vanilla
  experiment endpoints and its formatted vanilla `nan` response.
- Declared noise strata are `0.0001`, `0.01`, and `0.1`. All are reported and
  none may be dropped after seeing results.

The qualitative invalid/finite branch is part of the released simulator, not
an added constraint or hidden reward.

## Frozen Variant Split

A content-blind Python `random.Random(24367)` shuffle produced:

1. `medium:v0`
2. `medium:v2`
3. `hard:v2`
4. `easy:v0`
5. `hard:v0`
6. `hard:v1`
7. `medium:v1`
8. `easy:v2`
9. `easy:v1`

The first six are development worlds and the final three are holdout worlds
for any later LLM experiment. The structural audit integrates over all nine
hypotheses because design value requires the complete prior.

## Frozen Action Bank

There are exactly 32 single-experiment actions. A fresh
`random.Random(24367)` instance constructs a Latin-hypercube design:

1. independently shuffle bins `0..31` for each dimension;
2. draw one uniform jitter within every selected bin;
3. map both refractive indices linearly to `[1.0, 1.5]`; and
4. map incidence angle linearly to `[0, 90]` degrees.

Actions are serialized as compact, key-sorted JSON in bank order. The expected
SHA-256 is:
`4b38f1249b3a4753304eb569c4f43f1e2109b2d796832b64550861d20f97d440`.
No endpoint, hand-selected critical angle, local perturbation, new action, or
bank seed may be added after outcomes.

## Exact Planning Audit

Entropy is measured in nats. Finite Gaussian likelihoods are evaluated in log
space. For an `invalid` observation, invalid hypotheses have likelihood one
and finite hypotheses likelihood zero; for a finite observation the converse
holds.

Expected finite-observation entropies use deterministic Gauss-Hermite
quadrature. The primary calculation uses 15 nodes and the convergence check
uses 9. Invalid outcomes are integrated exactly.

For every root action:

- immediate EIG is prior entropy minus expected one-step posterior entropy;
- depth-two EIG is prior entropy minus expected terminal entropy when the
  second action is selected adaptively after the exact first observation from
  the same 32-action bank; and
- repeating an action is allowed, matching the official interface.

The greedy root maximizes immediate EIG, then depth-two EIG, then frozen bank
order. The non-myopic root maximizes depth-two EIG, then immediate EIG, then
bank order. Values within `1e-12` are ties.

## Strict Opportunity And Gate

A noise stratum is strict only if:

1. greedy and non-myopic roots differ;
2. non-myopic immediate EIG is at least `0.01` nats lower;
3. non-myopic depth-two EIG is at least `0.01` nats higher than the greedy
   root followed by its own optimal observation-conditioned continuation;
4. both margins retain their signs under 9-node quadrature;
5. both quadrature orders select the same roots; and
6. maximum immediate- and depth-two-value differences between orders are at
   most `0.005` nats.

The gate passes if at least one declared stratum is strict. A failure permits
only a deterministic result write-up, with no OpenRouter call, threshold
repair, support restriction, action-bank edit, or same-domain retry.

## Conditional LLM-Native Smoke

Only a pass may authorize a later development smoke. The LLM must generate
semantic optical-law hypotheses, predict finite versus invalid branches and
their likelihoods, and regenerate path-dependent beliefs after simulated
observations. Exact code may score the shared generated tree, but the released
nine-law registry must remain an oracle diagnostic rather than the policy's
fixed support.

Reasoning is reserved for the naive thinking baseline. Any smoke must preserve
the `$25` OpenRouter reserve, fit within the separate through-Monday operating
cap, and use no OatML resources.
