# Bamboogle Flat-Protocol Mechanics Result

## Decision

The final Bamboogle interface fails closed during the five fresh initial
model calls. Per the preregistration, there is no repair, reissue, parser
relaxation, or fourth serving attempt. Bamboogle is closed for this project.

No retrieval transition or scientific endpoint was evaluated, so this is not
evidence for or against non-myopic semantic BED.

## Exact Failure

- Run: `bamboogle-flat-mechanics-20260725T191928Z`.
- Interface: `bamboogle-cached-search-mechanics-flat-3`.
- Physical requests / HTTP attempts: `5 / 5`.
- Model retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Four initial responses satisfy the full parser.
- Task `test_87` repeats the normalized query
  `eliezer ben-yehuda father` across its eight root/fixed queries.
- This violates the unchanged global query-diversity requirement.
- Wikipedia logical actions / physical requests: `0 / 0`.
- Gold endpoints: not loaded or evaluated.
- Cost: `$0.0146675`.
- Public failure SHA-256:
  `5d8e5743e6708be5503406272d1482dc954c01467dda06e1688f3a4029e039cb`.
- Private raw SHA-256:
  `288fc5ea2383c08d2316632387a780c9677b623a9a09cdc71459bd8d76e02b34`.
- OatML use: none.

## Interpretation

The flat protocol solved JSON transport but not exact semantic constraint
adherence on fresh samples. Deduplicating, substituting, or regenerating the
query would alter the frozen tree and introduce a repair policy after seeing
the failure. The clean decision is therefore to close this route without
spending the remaining mechanics budget.
