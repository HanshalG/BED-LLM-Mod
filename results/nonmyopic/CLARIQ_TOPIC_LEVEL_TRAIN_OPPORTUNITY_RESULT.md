# ClariQ Topic-Level Train Opportunity Result

## Decision

The corrected fresh-train opportunity gate passes every preregistered
condition. A small multisample semantic-likelihood development gate is
authorized. No holdout policy run is authorized yet.

## Results

The 93 frozen opportunity topics yield:

| Metric | Result | Gate |
|---|---:|---:|
| Usable topic-level beliefs | 70 | >=50 |
| Greedy/depth-two root changes | 36 | >=30 |
| Positive terminal gains | 30 | >=25 |
| Gains at least `.005` | 21 | >=15 |
| Mean terminal NDCG@20 gain | `.005982` | >=`.005` |
| Maximum terminal gain | `.037024` | >=`.02` |

Seventeen topics were excluded for having fewer than two facets and six for
having fewer than two valid common roots. No replacement topics were drawn.

## Interpretation

ClariQ contains a native two-turn information-acquisition problem even after
correcting the decision unit. All facets sharing one observable request are
treated as alternative worlds under one uniform prior, and one first question
must be selected for that shared belief. The exact human answer then determines
the second-stage branch.

The effect is small in absolute NDCG but broad: 30 of 70 usable topics reward a
different first question when its answer-conditioned continuation is valued.
Actions, observations, transitions, and retrieval utility are externally
authored rather than generated to manufacture the planning gap.

This is an oracle structural result, not LLM policy efficacy. The next gate
tests whether multisample LLM semantic likelihoods rank these externally
grounded roots better than a compute-matched myopic policy.

## Budget

- API calls: `0`.
- OpenRouter spend: `$0`.
- OatML use: none.

## Artifacts

- Preregistration:
  `results/nonmyopic/CLARIQ_TOPIC_LEVEL_TRAIN_OPPORTUNITY_PREREGISTRATION.md`
- Opportunity audit:
  `results/nonmyopic/clariq_topic_level_train_opportunity/OPPORTUNITY.json`
- Audit SHA-256:
  `3342229d0d31477c4fb7cff1abe8758aceea47208071986ae1cececf504633a3`
