# Initial-history ambiguity without a measured horizon advantage

Protocol: ROBUSTFILL_INITIAL_OPPORTUNITY_PROTOCOL_20260909.md, written before
execution. The one-shot process completed both panels; no retries or cap changes.
Input source used only e1 labels plus e2..10 inputs from the previously sealed
forecast artifact. No later actual outcomes were loaded or scored.

| Development task | Distinct sampled behaviors / 256 | Initial risk | h1 | h2 | h3 | Fixed open-loop | Random |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | .224375407 | .010257719 | .010257719 | .010257719 | .010257719 | .060344514 |
| 10 | 124 | .487345378 | .008289931 | .008289931 | .008289931 | .008289931 | .014000700 |

Risk is terminal half-Brier Bayes risk under each fixed empirical belief, not actual
held-out error or full-prior risk. Both h2-vs-h1 and h3-vs-h2 gains are exactly zero.
Thus uncertainty and query-selection value exist, but no advantage from adaptive
lookahead is measured in this configuration. This closes it as the immediate paid
depth-sweep candidate, without claiming impossibility across all priors or tasks.

## Prior and interpretation

Sampling is exact conditional sampling from the declared bounded syntax prior:
uniform length1..3, IID uniform among162043 source atom expressions. Dynamic
programming retains syntax multiplicity, including empty-producing atoms. Length
posterior probabilities are [.25395929,.36105521,.38498550] for task1 and
[0,.33559672,.66440328] for task10. These are not learned semantic priors, and many
syntaxes can share behaviors. The256 draws are not an exhaustive posterior.
Exact optimization of this empirical law must not be described as exact full-prior
planning or true-data calibration. No smoothing or posterior repair was performed.

Saved panels SHA256 b1446a16400547b842803afa6e300f55066a2b88056de02c6f618cbe221cf840;
implementation SHA256 038481763145fa2bc04128e9c04e0d518138b5d07f51d3593e81a068cf5e1674.
Three focused tests pass (.20s), covering exact conditional length weights,
concatenation, empty pieces, syntax multiplicity, missing support, reproducibility.
Lint passes. Forecast panels and terminal results are banked unchanged.

## Next decision

Do not rotate seeds, tune this prior or enlarge this opened pair until a positive
curve appears. A representative set of untouched source tasks with a fixed
selection rule and a stronger symbolic baseline is needed to estimate opportunity
prevalence. Before opening it, audit the source's task-family/derivative overlap
metadata and predeclare separate development and sealed evaluation partitions.
The task selection must not use the realized h1/h2/h3 results. In parallel with
that source design, specify how semantic LLM proposals could change the joint
answer/target law beyond short syntax and how a compute-matched myopic control will
isolate planning over those updates. Do not conflate program synthesis length with
experimental planning depth. No paid block is authorized by these two diagnostics.

Previous goal turn produced verified representation evidence; this turn produced
new measured opportunity evidence. The complete LLM-native non-myopic objective is
still unachieved. Account245/221.179955339/23.820044661 unchanged, cost0, London-day
remaining4.23832314 including retained .04 uncertainty. Automation stays paused.
