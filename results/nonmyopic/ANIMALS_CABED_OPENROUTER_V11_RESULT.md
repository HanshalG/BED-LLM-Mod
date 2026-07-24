# Animals CA-BED OpenRouter V11 Result

Date: 2026-07-24

## Outcome

V11 failed closed during the two-target serving smoke. No formal target was
used.

- Run ID:
  `animals-cabed-openrouter-v11-smoke-20260724T210105Z`
- Requests: 82
- Reasoning tokens: 0
- Cost: $0.01282883
- Completed semantic batches: 30

Every completed semantic batch contained all 64 animals and a strict binary
label. The run failed on the second smoke target when duplicated independent
answer calls disagreed:

`Aardvark`, question `Is it a carnivore?`: `Yes`, then `No`.

The target-blind semantic table for that same question labeled Aardvark
`Yes`. No answer was selected, repaired, or resampled.

## Interpretation

The OpenRouter batching mechanism works and makes the semantic likelihood
table inexpensive and auditable. The exact V11 observation interface does not:
independent temperature-zero answer calls are not a stable realization of the
same semantic response function.

This reproduces the likelihood/answer mismatch diagnosed in Detective Cases
in a simpler form. A coherent LLM-induced environment must use one frozen
target-blind response table for both simulated likelihoods and realized
observations. That is a distinct interface, not a recovery of V11.
