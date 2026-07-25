# ClariQ Multisample Likelihood Holdout Preregistration

## Status

Frozen after the three-topic V2 development result and its disclosed post hoc
stability analysis, before holdout graph selection, holdout utility access, or
new OpenRouter calls.

V2 failed its conjunction because the unchanged-root topic had no `3/5`
single-sample modal root. Both topics where depth two changed the deployed root
passed `3/5` single-sample and leave-one-out stability and improved the external
first-link endpoint. This holdout protocol prospectively measures stability on
changed decisions, where instability can affect the non-myopic claim.

## Holdout Manifest

Use only the frozen 63-topic holdout split. In split order, a topic is eligible
when:

- it has three through six facets and at least six human questions;
- at least six roots are structurally evaluable under the V2 definition:
  official first-step evaluation keys, exact synthetic successors, and at
  least one common non-root followup with evaluation keys in every identical-
  answer branch.

The selector may inspect graph structure and evaluation key presence only. It
must not read or compare utility values.

Select the first 12 eligible topics. If only 8 through 11 are eligible, use all
of them. Fewer than eight fails before calls. Freeze the complete task/action
manifest, SHA-256, topic count, root count, and exact five-times-root request
count in an amendment before any model call.

## Frozen LLM Method

The V2 estimator is unchanged:

- `openai/gpt-5.4`, non-reasoning;
- the same exact `Y/N/U` semantic likelihood prompt;
- five temperature-`.7` samples per human root;
- request shuffle seed `24400`;
- Jeffreys smoothing `(count + .5) / 6.5`;
- uniform facet prior;
- exact one-step and depth-two mutual information;
- identical action width and likelihood calls for myopic and depth two;
- root cannot repeat at step two;
- smallest question-ID tie break; and
- random control seed `24401`.

The first-link endpoint remains selected-root exact external oracle-tail
`NDCG20.with_answer`, loaded after every map, likelihood, score, and selected
root freezes.

## Changed-Root Stability

For each topic where full five-sample depth two selects a different root from
five-sample myopic:

- recompute depth two five times, each omitting one likelihood sample from every
  question;
- the full selected root must appear in at least three of five leave-one-out
  fits.

Topics where depth two equals myopic do not enter this gate because no
non-myopic action change is deployed. All topics remain in endpoint and
correlation analyses.

## Frozen Gates

All must pass:

- manifest contains at least eight and at most 12 topics;
- exact manifest physical-request and HTTP-attempt counts;
- zero retries, reasoning tokens, and forced exits;
- every map parses exactly;
- every topic has at least three modal partitions and EIG range `.05`;
- depth two changes roots on at least `max(4, ceil(.25 * n))` topics;
- every changed root passes `3/5` leave-one-out agreement;
- every selected myopic, depth-two, and random root has an endpoint;
- depth two wins at least four topics and loses at most two versus myopic;
- mean depth-two oracle-tail gain over myopic is at least `.003`;
- exact one-sided paired sign-flip `p <= .10`;
- mean depth-two gain over random is nonnegative;
- pooled depth-two score/oracle-tail Spearman is at least `.20`; and
- adapter cost is at most `$1.25`.

The paired topic is the statistical unit. Zero-gain ties remain in means and
correlations and contribute no sign to the exact test.

Failure closes this exact fixed-support multisample method without task,
sample-count, prompt, smoothing, stability, threshold, or endpoint repair.

## Conditional Path-Dependent Stage

Only a full holdout pass authorizes a separate answer-conditioned support-
regeneration experiment. That stage must compare regenerated-support depth two
against the confirmed fixed-support depth two, compute-matched myopic,
fixed-support myopic, and random using paired external transitions.

## Budget

Projected cost is below `$1`; hard cap `$1.25`. The local pre-Monday allowance
after V2 is `$11.689054`, preserving the protected `$25` reserve. OatML is
prohibited.
