# ClariQ Dynamic-Support Mechanics Smoke Preregistration

## Claim Boundary

This smoke tests whether path-dependent LLM-generated intent supports improve
first-question ranking on already-disclosed topic `38`. Topic `38` was selected
after its exact ClariQ utility gap was known, so this is mechanics evidence only.
No development or holdout claim can follow directly from its endpoint.

The manifest SHA is
`8871d72aca825aa5b34cf52295eb93fe79caf631c0bba070d82abf6c3bec698d`.
Topic `148` and sealed holdouts `102/11/103/141` are not loaded.

## Model and Interface

- Model: `openai/gpt-5.4` through OpenRouter, non-reasoning.
- Temperature: `.7`.
- Exactly eight natural-language intent hypotheses per state.
- Flat grammar:
  `Hxx|0..100 unnormalized mass|response-code string|short intent`.
- Zero masses and sums other than 100 are valid prospectively; all-zero support,
  invalid codes, duplicate normalized intents, extra lines, or malformed fields
  fail.
- One initial call plus one independent support-regeneration call for every
  root-answer branch: exactly `40` requests.
- Raw responses checkpoint before parsing. No repair, normalization of malformed
  output, response reissue, scientific retry, or substitution.

The model sees the request, human question text, coded answer strings, and the
realized root answer. It never sees facet descriptions or the hidden
cross-question facet profiles.

## Policies

All policies share the same generated tree.

- **Myopic:** initial-support response-partition EIG.
- **Fixed depth two:** immediate EIG plus the best conditional follow-up EIG
  under the filtered initial support.
- **Dynamic depth two:** immediate EIG plus expected best follow-up EIG under
  the independently regenerated support for each root answer.
- **Shuffled dynamic:** same branch gains, but root-answer links are
  deterministically shuffled within each root.
- **Random:** seeded random legal root.

Every score and selected root freezes before loading the already-open NDCG@20
terminal utilities. The primary mechanics endpoint is the exact oracle-tail
NDCG@20 of the selected first question.

## Conjunctive Gates

Serving:

- exactly `40` physical requests and HTTP attempts;
- all `40` supports parse;
- zero retries, reasoning tokens, and forced exits;
- cost at most `$0.50`.

Belief dynamics:

- at least `35/39` branch supports differ from the initial support;
- at least `30/39` branches contain at least two positive-mass response
  profiles;
- at least `30/39` branches have positive best continuation EIG;
- myopic score range and dynamic future-value range are each at least `.05`
  nats; and
- at least one dynamic/fixed root score differs by `.02` nats.

First-link fidelity:

- dynamic depth two changes the selected root from myopic;
- dynamic score versus all-root exact terminal NDCG Spearman is at least `.20`;
- dynamic Spearman exceeds both myopic and fixed-support Spearman by at least
  `.02`;
- the dynamic selected root strictly exceeds both myopic and fixed-support
  selected roots on exact terminal NDCG; and
- dynamic is nonworse than the answer-link-shuffled control.

All conditions must pass to authorize the separately frozen topic-`148`
development run. Failure closes this exact method/prompt/topic smoke without
repair or rerun. Initial and branch exact hidden-profile coverage are diagnostic,
not gates.

## Budget

Projected cost is `$0.25`, with a hard `$0.50` run cap. The authenticated
OpenRouter balance is `$38.483769494`; `$25` remains protected through Monday
2026-07-27, and the stricter local project allowance is `$13.098966791`.
OpenRouter is the only compute path; OatML/cluster use is prohibited.
