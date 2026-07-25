# tau-Knowledge GPT-5.4 Mini Scorer Replication

## Status

Frozen before any GPT-5.4 Mini tau response. This is a post hoc cross-size
replication on open V3.1 trees, not a new held-out environment or cross-family
result. Required-document endpoints remain absent from all prompts.

## Fixed Protocol

- Smoke artifact SHA-256:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- Confirmation artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Nonsemantic analysis SHA-256:
  `d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4`.
- Model: `openai/gpt-5.4-mini` through OpenRouter.
- Temperature: `0`; explicit `reasoning_effort="none"`.
- Maximum output: 4,096 tokens.
- Any reported reasoning token fails the stage.
- The query trees, beliefs, retrieval results, endpoints, prompts, and
  nonsemantic controls are not regenerated.
- Parsing accepts JSON integers or digit strings and the V3.1 focused
  zero-padding convention; semantic keys and ranges remain strict.

## Calls

The two-tree smoke uses exactly 14 calls: two myopic root scorers, two full-tree
root scorers, and ten focused continuation scorers. Passing every smoke gate
unlocks the exact 140-call confirmation on the 20 open V3.1 trees.

No semantic response is retried, repaired, replaced, or manually interpreted.
Raw responses remain private and untracked.

## Smoke Gates

- exactly two cases, ten focused roots, and 14 physical requests;
- zero reasoning tokens;
- nonconstant focused scores on at least 8/10 roots;
- focused pairwise accuracy at least `.55`; and
- at least 7/10 oracle-optimal focused selections.

## Confirmation Gates

All original V3.1 gates remain:

- root comparable pairs at least 50;
- non-myopic root accuracy at least `.60` and at least `.05` above myopic;
- continuation comparable pairs at least 200, accuracy at least `.60`, at
  least 70/100 optimal, and mean regret at most `.30`;
- selected-root continuation loss at most 5;
- at least 4 wins, at most 2 losses, and total gain at least 4 over myopic;
- at least 6 wins, at most 4 losses, and total gain at least 5 over random; and
- focused improvement over the original joint selector at least 4 documents.

The Mini scorer must also beat all frozen nonsemantic controls:

- root accuracy greater than `.5455`;
- continuation accuracy greater than `.6316`; and
- endpoint coverage at least 25 documents.

Passing supports cross-size semantic-scorer robustness on open trees. It does
not establish cross-family transfer, repair the refreshed-belief alignment null,
or constitute a second held-out policy test.

## Budget

Smoke is capped at `$0.50`; confirmation at `$2.00`. OpenRouter reports
`$50.375775036` remaining before this protocol, or `$25.375775036` above the
protected reserve. Estimated total cost is below `$1`. OatML remains paused.
