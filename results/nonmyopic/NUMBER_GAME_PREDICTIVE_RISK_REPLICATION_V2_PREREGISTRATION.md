# Number Game Predictive-Risk Replication V2 Preregistration

Date frozen: 2026-07-28, before any V2 response.

V1 ended after one completed tree because an HTTP-success Gemini response had
`finish_reason="error"`, zero tokens, zero cost, and malformed content. No V1
aggregate scientific metric was computed.

## Sole Transport Change

V2 retries a response only when all of the following hold:

- the provider explicitly returns `finish_reason="error"`;
- reported response cost is zero; and
- the per-request transport retry limit has not been exhausted.

The retry uses the identical payload, including model, prompt, temperature,
schema, and seed. It is counted in HTTP attempts and retry accounting. Malformed
model output with a normal finish reason is not retried. A provider-error
response with nonzero cost fails closed.

## Fresh Seeds

The partially exposed V1 tree is not retained or selected. V2 uses:

- Gemini planning seeds `26080..26087`;
- GPT-5.4 target seeds `26180..26187`; and
- the unchanged bootstrap seed 26270.

## Unchanged Science

Every other item in
`NUMBER_GAME_PREDICTIVE_RISK_REPLICATION_PREREGISTRATION.md` remains binding:

- eight independent trees;
- temperature `0.7`, nonreasoning strict-schema proposals;
- 17 planning plus one target call per tree;
- unchanged grammar, filtering, support, candidates, policy, baselines,
  endpoints, equal-tree aggregation, 50,000-tree bootstrap, structural gates,
  scientific thresholds, and `$2.00` run cap.

The request gate counts 144 successfully returned model responses. Additional
zero-cost provider-error attempts are permitted only through the transport rule
above, and exact attempt accounting must still hold.
