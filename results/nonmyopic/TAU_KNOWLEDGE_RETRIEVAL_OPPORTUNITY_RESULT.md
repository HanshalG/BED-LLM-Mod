# tau-Knowledge Retrieval Opportunity V1 Result

## Decision

The two-task serving smoke passed. The six-task opportunity stage then failed
closed on one malformed followup response before any retrieval endpoint was
computed. V1 is closed; its 20 confirmation tasks remain sealed.

## Serving smoke

The smoke completed the exact 12 calls with zero reasoning tokens, retries,
forced exits, or malformed responses. It cost `$0.09392250`. First-query top-1
document diversity was 3/5 and 5/5, passing the frozen serving gate.

Both tasks retrieved required documents. One had pair gain of one required
document over its best first search. The non-myopic gap was zero on both
mechanics-only cases because the oracle-strength myopic root had an equally good
continuation. These descriptive values did not alter the opportunity protocol.

Artifact SHA-256:
`edfa25310b32d65f04135cf79ef4f86d716aed36528e7cfc22350f562908f95b`.

## Opportunity serving failure

The opportunity stage issued its exact 36 calls with zero reasoning tokens,
retries, or forced exits and cost `$0.28362500`. All six initial responses and
29 of 30 followup responses parsed. The final response, for `task_065` root 4,
opened a second `information_need_hypotheses` key inside the first hypotheses
array, producing invalid JSON.

The frozen protocol forbids response repair or replacement. Consequently:

- no BM25 opportunity endpoint was computed or read;
- no task, threshold, or response was replaced;
- V1 was not rerun; and
- all 20 confirmation tasks remain untouched.

Public failure artifact SHA-256:
`9dfcb235af63303bda11bd9ab2058c4661cc52cd6c7177f0893e31b65f4e76ef`.

Private raw checkpoint SHA-256:
`7110d7bbb4c0824787baa51137c4182986ff5d2d4f349610ab17b5c15248c902`.

## Next interface

The externally grounded route remains scientifically promising: local
development showed structural gains, the smoke passed, and V1 failed only at
serialization. A distinct V2 may use:

- a flat numbered response schema rather than nested string arrays;
- two new serving tasks and six new opportunity tasks that lack scripted
  openings and were not used by V1;
- a one-time target-blind opening utterance generated from each official user
  script; and
- the same official corpus, BM25 transition, exact required-document endpoint,
  frozen V1 gates, and sealed 20-task confirmation set.

V2 must be preregistered and committed before calls. It is not a repair or
reissue of the malformed V1 response.

## Budget

Combined V1 smoke and opportunity serving cost was `$0.37754750`. The project
ledger is `$70.72250216` spent with `$34.66230053` headroom. The live account
has `$59.66230054` remaining, or `$34.66230054` above the protected `$25`
Monday reserve. OatML was not used.
