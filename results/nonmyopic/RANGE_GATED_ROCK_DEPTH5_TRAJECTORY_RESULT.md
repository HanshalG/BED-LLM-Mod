# Focused Range-Gated Rock Hierarchical H5 Trajectory Result

The preregistered 50-pair Gemma 4 26B hierarchical h5 trajectory confirmation
and independent exact audit passed every frozen scientific and mechanics gate.

## Frozen Setup

- Environment: standard RockSample[7,8], start `(6,6)`, `.55` remote and `.95`
  on-site sensor accuracy.
- Prior: rock 6 has `p_good=.5`; every other rock has `p_good=.005`.
- Model: thinking `google/gemma-4-26b-a4b-it`.
- Seed: `24245`; 50 paired truths and eight real rounds.
- Audit bootstrap seed: `24246`.
- Architecture: cached semantic target assignment, deterministic shortest-path
  compiler, exact h5 verifier.
- Projection: none.
- Preregistration:
  `RANGE_GATED_ROCK_DEPTH5_TRAJECTORY_PREREGISTRATION.md`.

## Primary Results

| Paired comparison | Mean gain | Producer 95% CI | Audit 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Entropy AUC vs identical plans scored at h4 | +.283217 | [+.271013, +.293970] | [+.271131, +.293759] | 50/0/0 |
| Truth-log AUC vs identical plans scored at h4 | +.285520 | [+.225623, +.343566] | [+.227322, +.343035] | 48/0/2 |
| Entropy AUC vs matched-random targets at h5 | +.204126 | [+.184325, +.222931] | [+.184791, +.223006] | 49/1/0 |
| Truth-log AUC vs matched-random targets at h5 | +.206389 | [+.173750, +.238715] | [+.173197, +.239135] | 48/1/1 |
| Entropy AUC vs exhaustive d4 | +.283217 | [+.270930, +.294118] | [+.271092, +.293913] | 50/0/0 |
| Truth-log AUC vs exhaustive d4 | +.285520 | [+.227278, +.342626] | [+.227365, +.342387] | 48/0/2 |

All six producer and independently bootstrapped lower bounds are positive.

## Policy Outcome

Hierarchical h5 exactly matched the exhaustive d5 policy on this formal seed:

| Arm | Mean entropy AUC | Final entropy | Final MAP accuracy | Mean truth-log AUC |
| --- | ---: | ---: | ---: | ---: |
| Hierarchical LLM h5 | .608839 | .224239 | 96% | -.633701 |
| Exhaustive d5 | .608839 | .224239 | 96% | -.633701 |
| Matched-random targets h5 | .812964 | .449920 | 92% | -.840090 |
| Shared compiled h4 | .892056 | .875089 | 62% | -.919221 |
| Exhaustive d4 | .892056 | .875089 | 62% | -.919221 |

- Exact d5-over-d4 entropy-AUC gain: `.283217`.
- Hierarchical h5 recovery of that gain: `1.0`.
- Registered `North, West, West, West, check-6` route rate: 50/50.
- On-site rock-6 check by round five: 50/50.

Thus the same semantic plans fail when their roots are judged at h4 but succeed
when exact h5 can value all four enabling moves before the on-site check.

## Cache And Serving

The run retained 400 logical LLM decisions across the h5 and shared-h4 arms but
needed only 18 unique physical prompt cells. Exact scoring, controls, posterior
updates, and rollout branching made no LLM calls.

One first response assigned rock 6 to all four roots and failed the distinct
target constraint. The single preregistered correction returned a valid target
set. All other target sets were valid first attempt. Every reasoning pass used a
bounded non-reasoning final:

- Accepted unique target cells: 18.
- Invalid first responses: 1.
- Reasoning attempts / forced exits: 19/19.
- Forced-final requests/successes: 19/19.
- Physical OpenRouter requests: 38.
- Prompt tokens: 55,553.
- Completion tokens: 80,719.
- Reasoning tokens: 63,770.
- Cost: `$0.03407972`.
- Project spend after S1: `$40.398566282459846 / $110`.

## Independent Audit

The audit independently:

- recomputed the focused source prior and all paired truth indices;
- recompiled every unique target assignment;
- verified all cache identities and physical/logical request accounting;
- replayed all 2,000 arm trajectories and 16,000 real decisions;
- recomputed every plan value, stable selection, exact d4/d5 action,
  observation, posterior, entropy, and truth log probability;
- reproduced every aggregate mean and independently bootstrapped all six
  positive lower bounds.

Every audit mechanics and endpoint gate passed.

## Interpretation

This is a positive, paired, realized non-myopic sequential BED result at horizon
five. It isolates both load-bearing components:

- semantic LLM target selection beats matched-random targets under the same
  compiler, roots, and exact h5 verifier;
- exact h5 scoring beats the identical LLM-generated plans scored at h4.

The result is LLM-Modulo rather than unaided LLM planning: routing, Bayesian
updates, likelihoods, and policy verification are exact. It is also an engineered
focused-prior finite task, not evidence that five-step LLM planning transfers to
learned simulators or open-response environments. Within that boundary, it
closes the previous gap between proposal-quality evidence and a complete
receding-horizon trajectory claim.
