# tau-Knowledge Target-Blind First-Link Scorer V2

## Status

V1 failed closed after two myopic smoke calls because complete flat responses
encoded every integer score as a canonical digit string while the parser
required JSON numbers. V2 is a distinct serving interface. It does not coerce,
repair, or reissue either V1 response.

## Frozen V2 change

- Every root score and best-followup index must be a canonical decimal digit
  string (`"0"` through `"100"` for scores and `"1"` through `"4"` for
  followups).
- The parser accepts only those representations: JSON numbers, signs, decimals,
  whitespace, and leading zeroes are invalid.
- The fresh mechanics smoke uses the first two records in the already-public V2
  opportunity artifact: `task_018` and `task_008`.
- Smoke remains exactly four calls: two isolated myopic scores followed by two
  isolated non-myopic scores.

Everything scientific remains unchanged from V1:

- GPT-5.4, temperature zero, provider reasoning disabled;
- myopic scorer sees first results only;
- non-myopic scorer sees complete depth-2 trees;
- neutral document references, titles, and 700-character excerpts;
- exact official BM25 transitions and required-document endpoints;
- the original untouched 20-task confirmation split;
- oracle continuation beneath each selected root for the primary endpoint; and
- all confirmation efficacy thresholds.

## Gates

Smoke requires exact four calls, zero reasoning, complete canonical schemas, and
score variation in both scorer views on both tasks.

Only a passing smoke releases exact160-call confirmation. Confirmation passes
only with complete trees and scores, exact calls, zero reasoning, structural gap
on at least 4/20 tasks and mean >=0.20, at least 50 comparable root pairs,
non-myopic pairwise accuracy >=0.60 and >=0.05 above myopic, at least four
changed roots, at least three root-policy wins, at most two losses, total
endpoint advantage >=2, and >=50% oracle-root hits on opportunity tasks.

No response, task, query, score, or threshold may be repaired, replaced, or
changed.

## Budget

Before V2 calls, project ledger headroom is `$34.26354053`. The protected `$25`
Monday reserve remains inviolate and OatML remains paused. Smoke is projected
at `$0.20` with a `$1` cap; confirmation is projected at `$3` with an `$8` cap.
