# ClariQ Multisample Likelihood Development Preregistration

## Status

Frozen after the fresh train opportunity gate passed and before any new
OpenRouter response or development synthetic-transition/NDCG access.

This is a three-topic first-link development gate. It does not use the 63-topic
holdout and cannot establish a headline efficacy claim by itself.

## Development Topics

From the frozen 31-topic development list, select the first three topics in
split order having three through six facets and at least six human-authored
questions. Selection uses `train.tsv` metadata and text only:

| Topic | Facets | Questions | Initial request |
|---|---:|---:|---|
| `46` | 3 | 14 | Give me information about Alexian Brothers hospitals. |
| `177` | 4 | 14 | what is the best long term care insurance |
| `117` | 3 | 15 | What are specific dangers of asbestos? |

No synthetic answer, successor context, retrieval score, greedy root, or
depth-two root was loaded to select these topics.

## LLM Semantic Likelihood

- Model: `openai/gpt-5.4`, non-reasoning.
- Every human question is paired with the topic's human facet descriptions.
- The model returns exactly one `Y`, `N`, or `U` per facet.
- `Y` means the facet implies an affirmative response, `N` a negative
  response, and `U` that the facet does not determine a yes/no response.
- Five independent samples are drawn per question at temperature `.7`.
- The 215 requests are shuffled once with seed `24400`.
- Per facet/question likelihoods use Jeffreys smoothing:
  `(count + .5) / 6.5` over `Y/N/U`.

All 215 responses must parse. There is no response repair, normalization,
reissue, or first-object extraction.

## Policies

The prior is uniform over a topic's facets. For each question:

- myopic value is one-step mutual information;
- depth-two value is first-step mutual information plus expected maximum
  second-step mutual information under each `Y/N/U` posterior;
- the root question cannot repeat at step two;
- both policies use the identical five-sample likelihood table and action
  bank, making myopic compute- and width-matched;
- exact score ties choose the lexicographically smallest question ID; and
- seeded random uses seed `24401`.

The primary endpoint is the selected root's exact external oracle-tail
`NDCG20.with_answer`, computed from human answers and best legal second
questions only after every LLM response, likelihood, score, and selected root
has frozen. This isolates the first link from noisy policy execution.

Secondary diagnostics include:

- pooled Spearman correlation between LLM depth-two scores and external
  root oracle-tail values;
- myopic-score correlation with the same endpoint;
- selected depth-two versus random oracle-tail value; and
- five single-sample depth-two selections per topic, measuring whether at
  least three agree on a modal root.

Development and holdout endpoint entries are not available to the LLM.

## Frozen Gates

All must pass:

- exactly 215 physical requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all 215 maps parse with exact lengths;
- every topic has at least three distinct modal question partitions;
- every topic has one-step EIG range at least `.05` nats;
- full-table depth two differs from myopic on at least one topic;
- single-sample depth-two selections have modal count at least three on every
  topic;
- all three selected roots have valid external oracle-tail endpoints;
- depth two loses to myopic on at most one topic;
- depth two strictly beats myopic on at least one topic;
- mean depth-two minus myopic oracle-tail gain is at least `.003`;
- mean depth-two minus random oracle-tail gain is nonnegative;
- pooled depth-two-score versus oracle-tail Spearman is at least `.20`; and
- adapter cost is at most `$0.50`.

Failure closes this exact multisample fixed-support likelihood method without
sample-count, smoothing, prompt, topic, threshold, label, or endpoint repair.

## Conditional Next Stage

Passage authorizes a separately frozen confirmation on untouched holdout
topics. That confirmation must add the project's sharp mechanism:
answer-conditioned LLM facet regeneration, then compare regenerated-support
depth two against this fixed-support depth two, myopic, matched-width myopic,
and random under paired external transitions.

## Budget

Projected cost is below `$0.20`; hard cap is `$0.50`. The authenticated balance
before this gate is `$41.051936`, with the `$25` Monday reserve protected.
OatML is prohibited.
