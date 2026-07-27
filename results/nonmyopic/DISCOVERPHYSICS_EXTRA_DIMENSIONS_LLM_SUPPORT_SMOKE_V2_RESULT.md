# DiscoverPhysics Extra-Dimensions LLM Support Smoke V2 Result

## Outcome

**Failed closed during branch-support parsing. No score or scientific endpoint
was computed.**

The ordinary-chat fixed-record transport worked: all ten frozen requests
completed with `stop`, without retries, reasoning tokens, or forced exits.
Preflight and initial support parsed and compiled. Only two of the eight branch
refreshes were valid, however. The other six emitted one or more
`log_amplitude` values above the frozen upper bound `-0.5`; several branches
placed most or all laws outside the interval.

The parser performed no clipping, coercion, repair, resampling, partial-subset
selection, or score calculation. Under the preregistration, the
extra-dimensions LLM-support route is closed without a V3 transport.

## Exact Accounting

| Quantity | Value |
| --- | ---: |
| Physical requests | 10 |
| HTTP attempts | 10 |
| Retries | 0 |
| Prompt tokens | 3,887 |
| Completion tokens | 5,396 |
| Reasoning tokens | 0 |
| Local adapter cost | `$0.0906575` |
| Valid supports | 4 / 10 |
| Valid branch refreshes | 2 / 8 |

The authenticated credits endpoint had accounted for only the first `$0.00706`
when checked immediately after the batch. Budgeting therefore uses the more
conservative pre-run balance minus the complete local adapter ledger:
`$33.574042594` remaining, or `$8.574042594` above the protected `$25`
reserve.

## Interpretation

This is not evidence against the frozen structural opportunity. The exact
18-law oracle still has a verified non-myopic scout-then-target advantage.
The failure is evidence that GPT-5.4's branch-conditioned executable support
generation is not sufficiently constraint-faithful under this interface. Since
the dynamic score requires all eight branch supports on a common frozen
grammar, using only the valid branches would change the estimand and is
forbidden.

The result also separates the two serving problems cleanly:

- V1 failed before a response because strict JSON Schema was unroutable.
- V2 served reliably, but the generated scientific parameters violated the
  frozen domain on six branches.

No paired hidden-law endpoint is authorized.

## Artifacts

- Frozen commit: `fd1824a`
- Public failure:
  `discoverphysics_extra_dimensions_llm_support_smoke_v2/discoverphysics-extra-dimensions-llm-support-smoke-v2-20260727T105700Z/FAILURE.json`
- Private raw SHA-256:
  `11df0b6edea7af4d2d53caeeda45c6d27dcb3afe899af3a1537e564d11d67932`
- Run log SHA-256:
  `9ee3ab2523af1a774ceeafbae0ad6a5089480f8e52fdaa47a4bc459f30c2cc6a`
