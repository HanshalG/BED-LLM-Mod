# PSCon Role-Separated Binary Serving Result

## Outcome

The final target-free PSCon serving gate failed closed.

- Run: `pscon-binary-query-serving-20260725T141100Z`
- Requests / HTTP attempts: `10 / 10`
- Prompt / completion tokens: `11,605 / 161`
- Reasoning tokens / retries / forced exits: `0 / 0 / 0`
- Cost: `$0.00942825`
- Hidden target loaded: `false`
- Responder, policy score, and endpoint calls: `0`

All five one-line questions parsed. Three of five likelihood strings contained the
required 20 labels. Two contained only 19 labels:

- question 1: `YNNUNNUNNUYUUNNUUUU`
- question 2: `NYUNYYNYYNUNNUUNNNU`

The strict source-order likelihood map is therefore undefined for those responses.
No missing label was inserted or inferred and no response was repaired or reissued.

The five questions were distinct strings, but all concerned screen size. This also
indicates weak semantic action diversity even apart from the length failure.

## Decision

Per preregistration, this closes the current PSCon semantic-tree route. There is no
binary serving V2, stronger-model substitution, parser tolerance, per-product reissue,
or efficacy run. Across the three PSCon attempts, the external product support
successfully guaranteed the target, but Mini did not provide a sufficiently reliable
joint or role-separated semantic partition interface under exact fail-closed
measurement.

This remains serving evidence only. No PSCon result bears on non-myopic efficacy.
Chinese PSCon data and every hidden confirmation endpoint remain untouched. OatML was
not used.
