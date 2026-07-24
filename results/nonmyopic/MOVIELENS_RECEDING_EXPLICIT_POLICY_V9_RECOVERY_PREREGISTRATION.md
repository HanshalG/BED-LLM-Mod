# MovieLens Receding Explicit Policy v9 Recovery Preregistration

Date: 2026-07-24

Source run: `movielens-receding-policy-v9-formal-20260724T124635Z`

Source private raw SHA-256:
`bec011754b9f7c90e01f484b1754a8ba86e53dac0b1e358536474febd6909ee0`

## Failure

The formal process stopped after exactly 256 physical requests and `$0.73673414`,
before first-round candidate ratings, second-round policy states, held-out
ratings, or any efficacy metric were read.

All 48 initial profile responses and 48 initial likelihood responses parsed.
Exactly one of the 160 first-round hypothetical profile responses, frozen index
13, was truncated inside the sixth profile's final string. The other 159
responses parsed under the preregistered schema. Reasoning-token count was zero.

## Frozen Recovery

The recovery is an operational continuation, not a new sample:

- Require the exact source hash above.
- Reuse all 48 initial profiles, 48 initial likelihoods, and 159 valid
  first-round hypothetical profiles without changing their order.
- Verify that index 13 is the only invalid first-round profile response.
- Reissue exactly that one prompt with the unchanged model, messages,
  temperature, and output-token limit.
- Require the replacement to pass the original six-profile parser. No second
  replacement is allowed.
- Continue the frozen policy protocol from the reconstructed first-round tree.
- Preserve the failed raw file and write recovery responses to a separate
  private checkpoint.

The normal scientific request count is `416 + 40U`, where `U` is the number of
unique round-two policy states. The recovered run must therefore have exactly
`417 + 40U` total requests across the failed and recovery invocations. The
single extra request is logged as an operational replacement. All original
efficacy gates, cohort, policies, thresholds, and endpoint protections remain
unchanged.

Any source-hash mismatch, different invalid-response set, failed replacement,
additional malformed response, nonzero reasoning token, request-count
mismatch, or budget failure stops the run without another retry.
