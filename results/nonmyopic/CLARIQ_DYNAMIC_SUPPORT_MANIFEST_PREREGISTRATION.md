# ClariQ Dynamic-Support Manifest Preregistration

## Purpose

Freeze a scientifically distinct ClariQ route before any new model response or
new development utility value. The LLM will generate an open natural-language
intent support initially and regenerate that support after each observed human
answer. This differs from the closed fixed-facet `Y/N/U` interface that failed
its prior holdout serving gate.

## Source Boundary

- Official ClariQ repository commit and file hashes remain those verified by
  `scripts/clariq_topic_level_train_opportunity.py`.
- Topic `38` is an already-disclosed positive-gap opportunity topic and is used
  only for parser, serving, and mechanics development. It cannot support an
  efficacy or generalization claim.
- Topic `148` is the sole remaining structurally eligible, unused development
  topic. Its utility values remain unread when the manifest is generated.
- Holdout topics `102`, `11`, `103`, and `141` are the four structurally
  eligible topics not consumed by the failed fixed-support holdout. Their
  content and utility values remain unopened.

## Emitted Information

For mechanics and development tasks, the manifest may emit:

- the initial request;
- the human-authored clarification-question bank;
- the distinct human answer strings for each question, assigned deterministic
  `A`, `B`, ... response codes; and
- which follow-up question IDs are legal after each root answer.

It must not emit:

- facet descriptions;
- facet IDs;
- the cross-question answer profile belonging to any facet;
- any NDCG value; or
- any holdout content.

The hidden cross-question profiles are the semantic latent worlds. Coded answer
strings define the observation alphabet, not the latent support.

## Frozen Counts

| Stage | Topic | Roots | Root-answer branches | Model calls in full tree |
|---|---:|---:|---:|---:|
| mechanics | `38` | 13 | 39 | 40 |
| development | `148` | 13 | 52 | 53 |

One call generates the initial eight-hypothesis support. Each root-answer branch
gets one independent regenerated eight-hypothesis support over its legal
follow-up questions.

## Gates

Manifest generation passes only if:

- all source hashes and split memberships reproduce;
- topic `148` and the four sealed holdout IDs are disjoint from every prior
  fixed-support topic;
- root, branch, and request counts reproduce exactly;
- no latent facet description or cross-question profile is emitted;
- no development utility value is read; and
- no holdout content or utility value is read.

This stage uses zero API calls and authorizes only implementation and
deterministic testing. A separate preregistration and capped mechanics smoke are
required before any model call.

## Budget

OpenRouter is the only authorized compute route. The authenticated balance before
implementation is `$38.483769494`; `$25` is protected through Monday
2026-07-27. The stricter local project-ledger allowance is `$13.098966791`.
Manifest generation costs `$0`, and OatML/cluster use is prohibited.
