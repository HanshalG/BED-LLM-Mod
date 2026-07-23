# Focused-Prior Range-Gated Rock H5 Exact Result

The preregistered exact d5-over-d4 qualification and independent audit passed
every endpoint and mechanics gate.

## Frozen Setup

- Environment: standard RockSample[7,8], start position `(6,6)`.
- Sensor: `.55` remote accuracy and `.95` on-site accuracy.
- Prior: rock 6 has `p_good=.5`; each other rock has `p_good=.005`.
- Initial entropy: `.9135006422` nats.
- Protocol: 500 paired truths, eight rounds, exact receding-horizon d4 and d5
  policies, and 10,000 paired bootstrap replicates.
- Producer seed: `24235`; independent audit bootstrap seed: `24236`.
- LLM calls: zero.

## Result

| Comparison | Mean paired gain | Producer 95% CI | Audit 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Entropy AUC, d5 over d4 | +.288994 | [+.285778, +.292045] | [+.285770, +.292033] | 500/0/0 |
| Truth-log AUC, d5 over d4 | +.284228 | [+.267473, +.300362] | [+.267312, +.300517] | - |

The d5 policy reduced final entropy by `.653184` nats relative to d4
(producer 95% CI `[+.647664, +.658061]`). Mean final entropy was `.226708`
for d5 and `.879892` for d4. Final MAP accuracy was `97.2%` for d5 and
`61.4%` for d4.

At the initial belief, exact d4 selected the remote `check-6` action with value
`.0197373562`. Exact d5 selected `move-NORTH` with value `.4946319372` and
then followed:

```text
move-NORTH, move-WEST, move-WEST, move-WEST, check-6
```

Every d5 trajectory used this registered prefix and checked rock 6 on site in
round five. Every d4 trajectory began with a remote check of rock 6.

## Audit

The independent audit:

- replayed all 1,000 trajectories and paired truth indices;
- recomputed the source prior, initial action values, posteriors, metrics,
  mechanics, and producer comparisons;
- independently recomputed positive entropy-AUC and truth-log-AUC lower bounds;
- verified all actions were legal and that no LLM call occurred.

The initial audit run exposed only a JSON representation mismatch between
tuples in memory and lists after serialization. Canonicalizing both source-prior
representations fixed that audit comparison; no action, observation, posterior,
endpoint, interval, or gate changed.

## Interpretation

This is a large, exact, independently audited h5-over-h4 structural opportunity:
the shorter planner cannot value four enabling moves before the informative
on-site check, while d5 can. The focused prior is engineered but not
single-hypothesis or deterministic; the seven secondary rocks retain about
`.22035` nats of initial uncertainty.

This result does not establish an LLM proposal or trajectory result. It
authorizes a separately preregistered hierarchical h5 LLM gate under fresh
seeds. The next claim is whether an LLM can identify the semantic sensing target
while deterministic routing and exact EIG handle low-level composition and
verification.
