# Number Game Qwen First-Link Serving Smoke V2 Preregistration

Date frozen: 2026-07-29, after V1 failed and before any response from seed
`60001`.

## Why V2 Is Distinct

V1 required every newly generated conditioned child support to contain at
least eight unique rules. The planned 64-tree policy does not use a child
support alone: it filters the parent by the observation, merges the surviving
parent hypotheses with the generated child, and deduplicates the result.

V1 therefore tested a stricter object than the deployed retained-rejuvenation
state. It remains failed and cannot authorize the confirmation.

## Sole Instrument Change

Use the same exact ten histories, Qwen model, non-reasoning setting,
temperature, response schema, parser, and `$0.10` cap. Link them into:

- two initial supports;
- four first-step merged retained-rejuvenation supports;
- four second-step merged retained-rejuvenation supports.

V2 passes only if:

- exactly ten responses parse;
- exactly ten accepted requests and HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- both initial supports contain at least 16 valid unique hypotheses;
- every generated conditioned child contains at least four;
- every merged first-step support contains at least eight;
- every merged second-step support contains at least four;
- cost is at most `$0.10`.

Seed is `60001`. There is no response repair or reissue.

## Conditional Confirmation

A full V2 pass authorizes the already frozen 64-tree confirmation without
changing:

- tree, target, validation, or bootstrap seeds;
- model, prompts, parser, retained-rejuvenation policy, or exact endpoint;
- 3,712-call accounting and `$5.75` cap;
- any first-link, policy-efficacy, mechanics, or diagnostic threshold.

Failure closes this confirmation route.
