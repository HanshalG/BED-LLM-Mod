# tau-Knowledge Refreshed-Belief Alignment Ablation

## Status

Frozen before any ablation-model response. This is a post hoc mechanism test on
the already open V3.1 trees, not a second held-out policy result. Researchers
know the exact required-document endpoints, but no endpoint or label enters a
model prompt.

## Question

Does the successful semantic policy depend on aligning each retrieved branch
with the LLM's own path-dependent refreshed information-need hypotheses, or can
the same scorer obtain the result from the opening, initial beliefs, and
retrieved evidence alone?

## Fixed Intervention

For each task independently, seed `24343` generates a derangement of its five
first-search branches. Branch `i` keeps its exact query, first results, four
followup queries, and all followup results, but receives
`refreshed_information_need_hypotheses` from a different branch in the same
task.

The intervention therefore preserves:

- customer opening and initial information-need hypotheses;
- every query and retrieved document;
- the multiset, style, length, and task topic of refreshed beliefs;
- scorer prompts, model, temperature, parser, tie-breaking, and endpoint; and
- the number of model calls.

Only the within-task alignment between a branch history and its refreshed
belief state is broken. Every permutation must be a derangement, and canonical
records with the refreshed-belief fields removed must hash identically before
and after transformation.

## Frozen Inputs

- V3 smoke artifact SHA-256:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- V3.1 confirmation artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Model: `openai/gpt-5.4` through OpenRouter.
- Temperature: `0`.
- Reasoning is not requested and any reported reasoning token fails the stage.
- Root scorer and count-dominant focused scorer are byte-identical to V3.1.
- Root scores use the frozen canonical-string parser. Focused scores use the
  frozen V3.1 one- or two-digit zero-padding parser.
- Existing myopic scores and full-alignment results are reused from the
  hash-locked source artifacts; they are not regenerated.

## Pre-Response Sensitivity Audit

The 20 confirmation tasks contain 100 refreshed branch states. All tasks have
five distinct states, no exact pair is duplicated, and all 200 within-task
token-set comparisons have Jaccard below `.8` (mean `.2879`). The intervention
therefore changes substantive branch-conditioned text rather than permuting
near duplicates.

## Calls and Stages

The two-tree serving smoke uses exactly 12 physical calls:

- two shuffled-belief non-myopic root scorers; and
- ten shuffled-belief focused continuation scorers.

It passes only if all responses parse, exactly 12 requests and zero reasoning
tokens are recorded, every permutation is a derangement, non-belief fields are
unchanged, and the refreshed-belief multiset is preserved per task. No efficacy
threshold is imposed because degradation is the estimand.

Only a complete serving pass unlocks the 20-task confirmation, which uses
exactly 120 calls: 20 root scorers and 100 focused continuation scorers. No
response is retried, repaired, replaced, or manually interpreted. Raw text is
stored privately outside git.

## Frozen Mechanism Criteria

The primary contrasts are full alignment minus shuffled alignment on the same
20 tasks:

- non-myopic root pairwise accuracy decreases by at least `.05`;
- focused continuation pairwise accuracy decreases by at least `.05`; and
- exact end-to-end required-document coverage decreases by at least `3`.

All three thresholds must pass to call refreshed-belief alignment load-bearing
for this policy. Exact task-level one-sided sign-flip tests are reported for
root ranking, continuation ranking, and endpoint differences, but are
descriptive because the trees and full-alignment endpoints were already open.
Failure of any effect threshold means the current result does not isolate
path-dependent belief alignment from semantic evidence scoring.

## Budget

Smoke is capped at `$0.50`; confirmation is capped at `$3.00`. The last live
balance before this protocol was `$52.069693036`, leaving `$27.069693036` above
the protected `$25` reserve. Estimated total cost is below `$2`. OatML remains
paused.
