# DiscoverLLM Priority-World Tier Serving Preregistration

Frozen before loading the designated tier-serving artifact or obtaining any
tier-interface response.

## Motivation And Claim Boundary

The complete-permutation serving interface passed, but a zero-call
mathematical audit showed that fixed weights over every complete permutation
have invariant posterior entropy. It therefore cannot compare actions by
entropy EIG without an unregistered confidence scale.

This is a scientifically distinct one-task realistic serving gate for coarse
ordinal semantic likelihoods. It is not a mechanics, opportunity, policy, or
endpoint result. It does not reuse or rescore the permutation response.

The interface asks for `H`, `M`, or `L` likelihood tiers rather than numeric
scores. Ties are allowed, so different observations can express different
degrees of ambiguity and induce different posterior entropies. This follows
evidence that comparative/coarse judgments avoid some direct numeric-score
calibration failures:

- [Licht et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1635/)
- [Kola et al., ACL Industry 2026](https://aclanthology.org/2026.acl-industry.45/)

## Frozen Source And Split

- Pinned V2 manifest SHA-256:
  `9edfd3b20f762491db78087c95bccb1d345063af3423d4d7ccf6c481aa97ad3a`
- Tier-serving artifact: `creative_writing:artifact_385`.
- Reserved tier-mechanics artifacts:
  `technical_writing:artifact_333`,
  `creative_writing:artifact_367`,
  `technical_writing:artifact_249`.
- Prior permutation-serving artifact `svg_drawing:artifact_347` is excluded.
- All 60 opportunity and 162 holdout artifacts remain sealed.
- Released scores and winner labels are never read.

## Exact Interface

Use `openai/gpt-5.4`, temperature zero, without reasoning. Run the same five
semantic stages as the passed serving gate. Text stages use strict flat JSON.
Each likelihood stage returns exactly eight lines:

```text
ACTION_OBSERVATION|W1:X,W2:X,W3:X,W4:X
```

Each `X` is exactly `H`, `M`, or `L`; every world appears in fixed output order;
ties and all-equal assignments are valid. Candidate-world presentation is
deterministically shuffled by task and stage using seed `24415`. Observation
labels retain the independently shuffled truth mapping. The likelihood scorer
and continuation policy never receive that map.

The future mechanics primary weights are frozen now as `H=4`, `M=2`, `L=1`.
Sensitivity maps are `3/2/1` and `9/3/1`. No map may be selected after seeing
responses.

## Serving Gate

Pass requires:

- exactly five logical requests and all five stages parsing completely;
- no semantic retry, repair, coercion, or partial analysis;
- at most two logged adapter-level transport retries, with HTTP attempts equal
  to logical requests plus retries;
- zero reasoning tokens and forced exits; and
- cost at most `$0.15` (projected `$0.10`).

Passing authorizes only a separately frozen three-task tier mechanics smoke on
the reserved development artifacts. Failure closes this exact interface and
task without a format-only rerun.

OpenRouter only. OatML jobs: `0`.
