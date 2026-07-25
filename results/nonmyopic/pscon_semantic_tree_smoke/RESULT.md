# PSCon Semantic-Tree Development Smoke Result

## Outcome

The preregistered smoke failed closed at the first generator batch, before any
scientific score or hidden-target endpoint existed.

- Run: `pscon-semantic-tree-smoke-20260725T140030Z`
- Physical requests / HTTP attempts: `5 / 5`
- Model: `openai/gpt-5.4-mini`
- Prompt / completion tokens: `6,820 / 433`
- Reasoning tokens / retries / forced exits: `0 / 0 / 0`
- Cost: `$0.0070635`
- Responder calls: `0`
- Followup calls: `0`
- Hidden target loaded: `false`

At least one of the five root JSON objects supplied an assignment value whose type
or value was not a canonical integer option index in `{1,2,3}`. The frozen parser
rejected the batch immediately. There was no coercion, repair, reissue, alternate
case, target lookup, partition score, policy choice, or endpoint.

## Audit Limitation

The V1 harness assigned the returned batch to its raw checkpoint only after parsing
the whole batch. Consequently, the failure checkpoint contains the support and
`target_loaded: false`, but not the five raw root strings. The adapter usage record
and exception establish the exact failure stage, but the specific malformed value
cannot be reconstructed from the artifact.

Checkpoint order was fixed prospectively after closure so future batches are
persisted before parsing. This does not recover the missing V1 outputs and does not
authorize a rerun.

## Interpretation

This is a serving/schema null, not evidence for or against non-myopic semantic
planning. The exact conversation-64937 prompt, JSON schema, model pair, and efficacy
smoke are closed under the preregistration. PSCon remains a promising substrate
because its external product pool preserved the hidden truth, but a distinct future
test must first validate a simpler output interface without accessing a new target.

The English-to-Chinese confirmation split remains untouched. OatML was not used.
