# RegretBench Typed-Action Source Protocol

Frozen: 2026-08-11 Europe/London, after the option-ID exact-eight terminal punctuation null and before any typed-action model response.

## Purpose

Freeze a genuinely new RegretBench cohort for a typed clarification-action interface. Exclude the disjoint union of every task in the original LLM-native, factorized-v2, and option-ID source manifests: 396 tasks total, including every prior calibration, mechanics, development, and confirmation task.

An eligible typed-action task must satisfy the existing official RegretBench source gates and expose exactly four executable clarification facets. An executable facet has:

- a public facet/action ID;
- at least one nonempty official reference question; and
- at least two distinct nonempty source values across official intents.

For each facet, code freezes one canonical rendered question by sorting official reference-question text by normalized text and then literal text. Source values are used only inside the zero-call source audit to establish executability; they are absent from every public artifact and policy prompt.

Hash-rank the remaining typed-eligible tasks with salt `regretbench-typed-action-v1|` and freeze, in order:

- `typed_calibration`: 2 tasks;
- `mechanics`: 2 tasks;
- `development`: 64 tasks; and
- `confirmation`: 64 tasks.

All 132 selected tasks must be disjoint from all 396 prior tasks, mutually disjoint, prompt-unique, answer-alias nonleaking, checksum-valid, and have exact zero fixed-support depth-two gain under the established source audit. Calibration responses may never select, tune, replace, or reveal mechanics tasks.

Passing this source audit authorizes only a separately frozen typed-action calibration gate on the two calibration tasks. It authorizes no mechanics call, endpoint, development, confirmation, or paper claim.

## Visibility

Proposal and likelihood-evaluator calls may see only:

- public task ID and ambiguous prompt;
- four public records containing `action_id` and code-rendered `question`; and
- model-generated hypotheses, answers, action options, or likelihoods from the same request chain.

They may not see source facet values, intent descriptions or probabilities, final source answers or aliases, hidden truth, endpoint outcomes, candidate policy scores, or any mechanics/development/confirmation record.

Only after code selects an action from model-generated categorical mutual information may it load the selected facet's unique source values. Two independently seeded environment-codec calls may then see only the selected public action/question, its fixed model-generated option labels, and indexed unique values. No source value or mapping is written to the public result.

## Scientific Boundary

Typed actions repair only query executability. A later mechanics protocol must still establish answer-conditioned support regeneration, calibrated categorical likelihoods, a real non-myopic action gap, and paired endpoint value versus compute-matched myopic and random controls. This source pass alone is not evidence for any of those claims.
