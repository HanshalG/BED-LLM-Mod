# Tau2 Account Joint-Likelihood V5 Result

Date: 2026-07-24

## Decision

The two-ticket serving smoke failed the frozen semantic mechanism gate. The
twelve-ticket formal ranking gate and twenty-four-ticket paired confirmation
remain untouched. V5 is closed without a prompt repair, resample, or threshold
change.

## Integrity and Cost

Run `tau2-account-joint-v5-smoke-20260724T214705Z`:

- exactly two full GPT-5.4 requests;
- one complete 4-hypothesis by 9-action table per request;
- zero reasoning tokens, retries, forced exits, parser errors, or missing rows;
- temperature zero; and
- cost `$0.02483750`.

Official Tau2 outputs were read only after both model tables were fixed. Raw
responses remain private and the parsed tables are public.

## Frozen Gates

| Criterion | Required | Observed | Pass |
|---|---:|---:|:---:|
| Complete tables | 2/2 | 2/2 | yes |
| Exact physical requests | 2 | 2 | yes |
| Reasoning tokens | 0 | 0 | yes |
| Every initial action observationally equivalent | 2/2 | 1/2 | **no** |
| Four distinct line-detail outcomes | 2/2 | 1/2 | **no** |
| Depth-one lookup selections | 0/2 | 0/2 | yes |
| Depth-two lookup selections | 2/2 | 1/2 | **no** |
| Lookup successor is line details | 2/2 | 1/2 | **no** |

## Failure Localization

The second ticket produced the exact intended model:

- every initial action had one outcome;
- `line_details` represented all four allowance-by-roaming combinations;
- lookup had predicted depth-one value zero and depth-two value `ln(4)`; and
- depth two selected lookup followed by line details.

The first ticket was internally coherent but wrong:

- `payment_request` was predicted absent for allowance-available worlds and
  present for allowance-exhausted worlds, creating a spurious immediate
  `ln(2)` partition;
- `line_details` exposed only account roaming, not allowance, giving two
  outcomes;
- `data_usage` separately exposed only allowance;
- lookup therefore had only `ln(2)` predicted two-step value; and
- the exact planner selected a direct root whose continuation used the
  hallucinated payment-request partition.

The planner did what its learned semantic model requested. Joint elicitation
removed V4's branch-action inconsistency but did not make the forward model
invariant to ticket wording.

## Consequence

No formal or confirmation run follows. A further Tau2 likelihood-prompt repair
would tune directly on the observed miss.

The complementary LLM-Modulo route remains scientifically distinct: let the LLM
propose a semantic fault prior from unstructured evidence, then use the official
Tau2 simulator for likelihoods and exact planning. That follows BED-LLM's
prior-likelihood construction while assigning deterministic state transitions
and tool behavior to code rather than the language model.
