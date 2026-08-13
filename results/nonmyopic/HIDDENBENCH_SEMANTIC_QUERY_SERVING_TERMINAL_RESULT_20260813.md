# HiddenBench Semantic-Query Serving Terminal Result

Date: 2026-08-13

Status: **pre-serving implementation null; exact interface and mechanics cohort
closed.**

## Result

The frozen ten-call HiddenBench semantic-query gate did not reach the model or
selected source rows. Its authenticated read-only preflight passed all source,
binding, account-wide budget, catalog, price, and pristine-path checks. The exact
dated DeepSeek endpoint was available at `$0.08/M` prompt and `$0.18/M`
completion tokens, and the full `$0.015` stage reservation fit under the Aug-13
account-wide `$5` cap.

Execution then failed while constructing `Config`: the frozen producer set
`openrouter_backoff_seconds=0.0`, while the repository requires this value to be
strictly positive even when `openrouter_max_retries=0`. The adapter was never
constructed.

| Quantity | Result |
|---|---:|
| accepted model requests / HTTP attempts | 0 / 0 |
| OpenRouter cost | `$0` |
| selected mechanics rows projected | 0 |
| raw model responses | 0 |
| registered answers opened | 0 |
| opportunity endpoints opened | 0 |
| terminal-audit gates | 5 / 5 |
| focused tests after banking | 16 / 16 |

Authenticated cumulative usage was unchanged at `$220.178352166` before and
after the attempt. Aug-13 conservative account-wide spend therefore remains
`$0.044223286` from the frozen `$220.134128880` boundary.

## Interpretation

This is not a language-model serving failure, a planner failure, or evidence
against HiddenBench as a non-myopic BED substrate. No model response or task
value existed to evaluate. It is an implementation-gate null.

The exact protocol, seeds, semantic-query interface, and four-task mechanics
cohort are nevertheless terminal under the frozen no-repair/no-retry rule. They
must not be relaunched with a positive backoff or another seed. A scientifically
distinct revisit would require a new prospective cohort and protocol, and it
must include a real adapter-construction rehearsal before its execution binding
is pushed.

No development, confirmation, endpoint, or paper-efficacy claim is authorized.

## Integrity

- serving protocol SHA-256:
  `c363595920ac6b37ce0846fe283ef38524ac8fce934983927613bc7fbc270cc0`;
- execution binding SHA-256:
  `272e0c9daf3d0556cc0888f0c37a854abdf5f8ca8c9aace3fea9915483e5dc75`;
- terminal failure SHA-256:
  `567b2afb35d5f6cf7645119f661d8563a7b78168c73771d34fba4b5fc9f4adc3`;
- account ledger SHA-256:
  `e6071c353d1d43db931f4a79d1bc838e10ec1f4d51fac98526bed90bead520a3`;
- terminal audit SHA-256:
  `d36b616d1cabda2845ac5301038cf1c61482bad8e2743d5e79f3b419a257f23c`;
- terminal-audit implementation SHA-256:
  `aa67b652995d6a27484ea51fa03ee97d66c3accb5b7c38938e9a2885feaa0b4b`.
