# ClariQ Multisample Likelihood Development V2 Preregistration

## Status

Frozen after V1 failed its deterministic preflight and before selecting V2
topics, reading any additional development utility value, or making an
OpenRouter call.

V1 topics `46`, `177`, and `117` are permanently excluded. V2 changes only
prospective structural eligibility and the action manifest. The five-sample
semantic likelihood, policies, controls, endpoint, and scientific thresholds
remain unchanged.

## Structural Manifest

Process the remaining topics in the frozen development split order. A topic is
eligible when:

- it has three through six facets in `train.tsv`;
- it has at least six human-authored questions;
- every retained root question exists for every facet at empty history;
- every facet has an official first-step evaluation key and exact synthetic
  successor after that root;
- for every identical-answer observation group, at least one common non-root
  second question has an official evaluation key for every member; and
- at least six roots meet all conditions above.

The manifest selector may inspect synthetic graph structure and evaluation
dictionary key presence. It must not index, compare, aggregate, rank, or output
any `with_answer`, `no_answer`, NDCG, or other utility value.

Select the first three eligible topics. Their action bank is exactly their
structurally valid roots, sorted by question ID. Freeze and hash the complete
manifest before any LLM request. If fewer than three topics are eligible, V2
fails before calls.

## LLM and Policies

Unchanged from V1:

- `openai/gpt-5.4`, non-reasoning;
- five independent temperature-`.7` exact `Y/N/U` maps per question;
- request shuffle seed `24400`;
- Jeffreys smoothing `(count + .5) / 6.5`;
- uniform facet prior;
- myopic one-step mutual information;
- depth-two mutual information with answer-conditioned best non-root
  continuation;
- identical action width and calls for myopic and depth two;
- smallest question-ID tie break; and
- seeded random control `24401`.

The exact expected request count is five times the manifest root count and is
frozen in a post-selection amendment before calls.

## Endpoint and Gates

The primary endpoint remains selected-root exact external oracle-tail
`NDCG20.with_answer`, loaded only after all maps, likelihoods, scores, and roots
freeze.

All V1 gates remain:

- exact manifest request and HTTP-attempt counts;
- zero retries, reasoning tokens, and forced exits;
- every map parses exactly;
- at least three distinct modal partitions and EIG range `.05` per topic;
- at least one depth-two root change;
- single-sample modal root count at least three per topic;
- all selected roots have endpoints;
- at most one depth-two loss and at least one strict win over myopic;
- mean depth-two gain over myopic at least `.003`;
- mean gain over random nonnegative;
- pooled depth-two score/oracle-tail Spearman at least `.20`; and
- cost at most `$0.50`.

Failure closes V2 without another topic, action, sample-count, smoothing,
prompt, label, threshold, or endpoint repair.

## Conditional Next Stage

Passage authorizes a separately frozen holdout confirmation with
answer-conditioned LLM facet regeneration and fixed-support/myopic/random
controls. No holdout access is authorized before V2 passes.

## Budget

Manifest selection uses zero API calls. The paid gate remains capped at `$0.50`
and uses no OatML.
