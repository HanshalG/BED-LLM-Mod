# Detective CA-BED Shared-Tree Depth Ranking Preregistration

Date frozen: 2026-07-24

## Question

Can depth-two CA-BED rank and select better first interrogation questions than
one-step EIG when both methods use the same LLM-generated questions, textual
semantic likelihoods, simulated suspect answers, and exact Bayesian updates?

This is an LLM-native gate. The released case text is unstructured, candidate
interrogation questions are open-ended, and both
`p(answer | murderer, question)` and the hidden suspect's answer require semantic
inference by an LLM. The exact tree arithmetic is deliberately external and
auditable.

## Sources

- CA-BED paper: <https://arxiv.org/abs/2606.01182>
- Released implementation:
  <https://github.com/DanielArnould/ca-bed> at commit
  `bd4bbd7a6c518c5ebfa3f8791ca22e9799520a75`
- Released `DetectiveCases.json` SHA-256:
  `049ea3003753b15e3319483d15993591b5d50ac7eedb3187dca6ef3951cd2a57`
- AR-Bench source:
  <https://github.com/tmlr-group/AR-Bench> at commit
  `9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1`

The paper describes five suspects, while the released CA-BED artifact contains
four hypothesis-bearing suspects per case. Inspection shows that it retains the
first four AR-Bench suspects and excludes the fifth confuser role. This experiment
uses the exact released CA-BED artifact and its four-hypothesis support.

## Frozen Splits

Python `random.Random(24312).shuffle(range(100))` defines disjoint splits.

- Serving smoke: `27, 94`
- Ranking gate: `31, 96, 91, 65, 99, 56, 2, 4, 50, 6, 87, 34`
- Untouched sequential-confirmation reserve:
  `75, 70, 89, 98, 86, 88, 8, 74, 82, 19, 78, 72, 18, 97, 76, 46, 24,
  67, 42, 3, 80, 1, 79, 14`
- The remaining 62 cases are unused reserve.

No failed or incomplete case may be replaced.

## Frozen Method

- Questioner, textual likelihood estimator, and hidden-role answerer:
  `deepseek/deepseek-v3.2` through OpenRouter, non-thinking.
- Generation temperature: `1.0`, matching CA-BED.
- Hypothesis prior: uniform over the four released suspects.
- Root width: `3`.
- Follow-up width: `3` for each root and each `Yes`/`No` branch.
- Estimator confidence: `0.7`, so each textual likelihood is smoothed as
  `0.7 * p_LLM + 0.3 * 0.5`.
- Depth-one score: immediate EIG.
- Depth-two score: expected total entropy reduction after the root and the best
  branch-specific one-step follow-up.
- All entropy and truth-log-posterior metrics are in nats.
- Root and follow-up questions are generated once and shared by depth one,
  depth two, and the seeded-random control.
- Textual likelihoods follow the released CA-BED prompt: one response supplies
  `p(Yes | murderer, question)` for every suspect. Missing, duplicate,
  non-finite, or out-of-range rows fail closed.
- Every generated question must target exactly one released suspect, end as a
  question, and be unique within its history. Incomplete menus fail closed.
- Hidden-role answers use the released public case context, the targeted
  suspect's private story, and their murderer/innocent role. Answers must parse
  as exactly `Yes` or `No`.
- Each root is evaluated with four independent two-turn hidden-role answer
  rollouts. Each rollout follows the immediate-EIG-optimal follow-up for the
  realized root answer. The mean two-turn truth-log-posterior gain is the
  realized root utility used for ranking.
- No content retries, parser repairs, case replacements, threshold changes, or
  post-result support/model changes are allowed. Transport retries retain the
  identical request.
- Raw responses, prompts or prompt hashes, questions, likelihoods, selected
  roots, answer rollouts, token usage, and cost are persisted.

## Serving Smoke

The two smoke cases use the complete formal request shape: `52` requests per
case and `104` total. It passes only if:

1. Both complete with all `3 x 2 x 3` question-tree cells and four answer
   rollouts per root.
2. Every question, likelihood row, and answer parses without a content retry.
3. All smoothed likelihoods are finite and strictly between zero and one.
4. All score and endpoint values are finite.
5. OpenRouter reports zero reasoning tokens, zero forced exits, and exactly
   `104` requests.

The smoke is mechanics-only and cannot contribute to the ranking endpoint.
Projected cost is `$0.20`; hard run cap is `$0.75`.

## Ranking Gates

All gates are conjunctive:

1. All 12 frozen cases complete with exactly `624` requests and no content
   retry, replacement, forced exit, or reasoning token.
2. At least 9/12 cases have nonconstant depth-two root scores and nonconstant
   realized root utilities, making their within-case rank correlation defined.
3. Depth two selects a different root from depth one on at least 4/12 cases.
4. Mean within-case Spearman correlation between depth-two score and realized
   truth-log-posterior gain is at least `0.20`.
5. The mean depth-two correlation exceeds the depth-one correlation by at
   least `0.10`.
6. Mean paired truth-NLL improvement of the depth-two-selected root over the
   depth-one-selected root is at least `0.02` nats, its paired 90% bootstrap
   lower bound is above zero, and depth two wins at least 7/12 cases.
7. Mean paired truth-NLL improvement over the seeded-random root is at least
   `0.02` nats with a paired 90% bootstrap lower bound above zero.
8. Mean final posterior entropy for the depth-two-selected root is no more than
   `0.02` nats worse than depth one.

Projected ranking cost is `$1.20`; hard run cap is `$3.00`. The live OpenRouter
balance must be checked before launch. Passing authorizes only the frozen
24-case sequential confirmation. Failure closes this exact model, prompt,
support, width, smoothing, and split; it does not authorize tuning on these
cases.

## Pre-Endpoint Serving Amendment

The first serving attempt
`detective-cabed-depth-smoke-20260724T191300Z` stopped after exactly eight
requests and `$0.0028929176`, before follow-up generation, answers, or any
endpoint. All six root-likelihood responses reached the 512-token output limit
while still giving the requested explanation, so none reached the required
likelihood rows. This is a serving-budget failure with no scientific read.

Before any V2 response, freeze one format-only repair: increase
`openrouter_max_output_tokens` from `512` to `2048`. The model, prompts,
temperature, cases, split, tree shape, smoothing, parser, gates, and cost cap are
unchanged; failed responses are not reused or repaired. If V2 still fails to
produce the frozen rows, close this DeepSeek/published-prompt interface.
