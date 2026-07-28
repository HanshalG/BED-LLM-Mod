# Number Game Multi-Draw Prior Development Preregistration

Date frozen: 2026-07-28, before any extra-prior response.

The eight-tree V2 target endpoints are exposed. This is a development test, not
a confirmation.

## Frozen Change

- Keep all eight published V2 branch supports, candidate roots, GPT-5.4 target
  supports, likelihoods, second-query policy, and controls fixed.
- For each tree, request exactly two additional no-observation Gemini 2.5 Flash
  supports using seeds `26300..26315`, temperature `0.7`, nonreasoning strict
  schema.
- Pool the original planning support and two fresh supports, deduplicating by
  complete `0..100` extension.
- Recompute terminal predictive Bayes risk uniformly over that three-draw
  empirical prior, restricted to the tree's already-generated eight roots.
- The extra supports never receive the exposed GPT target rules or metrics.
- Compare the newly selected roots on the unchanged exposed endpoints against
  the original one-draw predictive-risk policy, myopic EIG, and PTS.
- Weight trees equally and use the unchanged 50,000 whole-tree bootstrap.

Exactly 16 accepted calls are permitted. Every draw must retain at least 16
valid unique rules and every pooled prior at least 32 unique extensions. Cost
cap is `$0.50`.

## Development Gates

All must pass to authorize a fully fresh multi-draw replication:

1. Versus PTS: at least 5% aggregate Brier gain, wholly negative whole-tree
   interval, and at least six of eight strict tree wins.
2. Versus the original one-draw predictive-risk policy: at least 2% aggregate
   Brier gain, interval not above zero, and at least three strict wins.
3. Versus myopic EIG: at least 10% aggregate Brier gain, wholly negative
   interval, and no mean exact-extension coverage loss.
4. All transport, validity, and budget gates pass.

No number of draws, support weight, tree, root, endpoint, or criterion will be
changed after the extra-prior responses.
