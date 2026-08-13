# AutumnBench LLM-Native Source Admission Result

Date: 2026-08-13

Status: **source failed closed; exact public-release route closed.**

## Result

The audit bound the official Autumn interpreter and MARA protocol repositories,
then downloaded only the public metadata manifest. The frozen population gate
required the paper's 129 tasks over 43 base worlds before any task payload or local
fixture execution.

The manifest instead declares and contains 60 tasks over 20 base worlds. It is
internally balanced: each observed world has one masked-frame-prediction, one
change-detection, and one planning task. But it is not the frozen complete benchmark
population.

| Quantity | Frozen requirement | Observed |
|---|---:|---:|
| public tasks | 129 | 60 |
| unique base worlds | 43 | 20 |
| MFP / change-detection / planning | 43 / 43 / 43 | 20 / 20 / 20 |
| non-fixture programs/prompts/answers downloaded | 0 | 0 |
| fixture executions / model calls / endpoint outcomes | 0 / 0 / 0 | 0 / 0 / 0 |

Per the preregistered ordering, the audit stopped at that population failure. It did
not run the deterministic `ice` handshake, inspect repository task programs, or open
any downloaded prompt, answer, goal, observation, or correct option.

## Interpretation

AutumnBench remains conceptually well matched to non-myopic BED: reward-free
interaction, strategic reset, and sealed derived tests are exactly the right shape.
This result does not evaluate those mechanics. It establishes only that the public
release available at the frozen URL is a 20-world subset rather than the claimed
43-world benchmark.

The exact source protocol closes. We do not post-hoc redefine admission around the
observed 60-task subset. No mechanics, opportunity, model-serving, development,
confirmation, or paper-efficacy claim is authorized. The route may be reconsidered
only if the authors publish a complete versioned 129-task manifest under a fresh
prospective protocol.

## Integrity

- protocol SHA-256:
  `0980ca9f9f189577c154367475a7a94bbb120dbd6cc9bb53b8e2f1dff4292856`;
- public manifest SHA-256:
  `c3f17e4d51318994dcd169a8ab3c4db1212dc30b83ad6b0dec88e933edb2071a`;
- aggregate source result SHA-256:
  `2ca706d4ee88619bcc8f70b687d6f10e7cc485f504209da37aa9d3809d21bc03`;
- audit implementation SHA-256:
  `42d2ff477ecbb6d16a1ca1f2e0a6453735e883466995fb166d46beb69edd41d6`;
- focused test SHA-256:
  `3b261e99c281226a6ba77af39abfbbe22c5144e2676777ee6beea411b933d7e7`.
