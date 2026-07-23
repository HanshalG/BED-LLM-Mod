# Range-Gated Rock 26B Named-Plan S0 Failure

The preregistered S0 serving/mechanics gate failed closed on its first belief cell,
so the S1 proposal-quality gate was not launched.

On the first attempt, Gemma 4 26B returned four syntactically valid three-action
plans, but all four roots were checks. The frozen contract required exactly two
distinct movement roots and two distinct check roots. The single correction attempt
then duplicated two check-root plans twice, violating both distinctness and the root
mix. No plan was accepted and no exact proposal-quality score was computed or viewed.

Usage:

- 2 direct-vLLM requests on A100 job `106376`.
- 1,103 prompt tokens and 124 completion tokens.
- 0 reasoning tokens, 0 forced exits, 0 rollout/scoring LLM calls.
- `$0` API cost.

The failure is narrower than the earlier indexed-menu failures: named action strings
were legal and correctly shaped, but the model did not follow the set-level diversity
constraint even after exact validation feedback. The frozen protocol explicitly
allowed no prompt repair after S0, so this interface line stops without S1 or a paired
trajectory endpoint. The banked exact d3-over-d2 structural result remains valid; this
run supplies no LLM-policy evidence for it.

Artifact:

- `results/nonmyopic/range_gated_rock_26b_smoke_20260723/SMOKE_FAILURE.json`
