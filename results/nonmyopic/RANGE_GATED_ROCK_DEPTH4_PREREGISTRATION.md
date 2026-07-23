# Range-Gated RockSample[7,8] Depth-Four Preregistration

## Motivation and Exploratory Screen

The existing range-gated task makes two-step travel load-bearing for h3. To test a
strictly deeper horizon without changing the standard Smith--Simmons rock layout,
move only the rover start from `(0,3)` to the opposite corner `(6,6)`. Rock 4 at
`(6,3)` is then three moves away. The sensor remains 0.55 accurate remotely and
0.95 accurate exactly on site.

A zero-LLM exploratory screen at seed `24200` used 50 paired truths and eight
rounds. Exact d3 checked remotely and never reached an on-site inspection. Exact d4
moved north three times and reached an on-site check in all 50 trials. D4-minus-d3
entropy-AUC gain was `+0.348436` with 50/0/0 wins/ties/losses; truth-log gain was
positive on average. These exploratory cases are excluded from the qualification.

## Frozen Qualification

- Fresh seed `24201`.
- 500 paired truth states sampled independently from the uniform 256-state prior.
- Eight executed rounds.
- Exact exhaustive receding-horizon d3 and d4 over all legal actions and
  positive-probability observation branches.
- Terminal-history EIG planning utility, matching the existing exact depth gates.
- Common deterministic observations keyed by seed, trial, position, rock, and
  repeat count.
- 10,000 paired bootstrap replicates.

Primary endpoint: d4-minus-d3 entropy-AUC gain.

Corroborating endpoint: d4-minus-d3 truth-log-posterior-AUC gain.

The producer gate passes only if both paired 95% lower bounds are strictly positive
and every mechanic passes:

1. truths are paired and all 1,000 traces contain eight legal actions;
2. every d3 trace starts with a check;
3. every d4 trace starts with `move-NORTH`;
4. every d4 trace reaches an on-site inspection; and
5. no model call occurs.

A separate audit must replay every history, exact d3/d4 action, common-random
observation, posterior, metric, aggregate, and fresh confidence interval. A pass
authorizes only a separately preregistered h4 proposal interface; it is not an LLM
policy result.
