# DiscoverLLM Priority-World Ordinal Serving Preregistration

Frozen before loading the designated development artifact or obtaining any new
model response.

## Claim Boundary

This is a one-task realistic-input transport and grammar gate for a
scientifically distinct ordinal-likelihood construction. It is not a repair or
rescore of the closed cardinal-score mechanics run, a non-myopic opportunity
test, or a policy result.

The previous construction requested cardinal scores in `[0,100]` and is
permanently closed. This construction asks the LLM to order four
source-grounded semantic priority worlds. Comparative judgments are preferable
here because direct numeric LLM scores have known calibration and output-format
failure modes. Any future Bayes calculation will map ranks to fixed weights
specified before its mechanics run.

## Frozen Source And Split

- Pinned DiscoverLLM manifest SHA-256:
  `9edfd3b20f762491db78087c95bccb1d345063af3423d4d7ccf6c481aa97ad3a`
- Serving artifact: `svg_drawing:artifact_347`, the first V2 development item.
- Reserved mechanics artifacts, in order:
  `creative_writing:artifact_385`,
  `technical_writing:artifact_333`,
  `creative_writing:artifact_367`.
- All 60 opportunity and 162 holdout artifacts remain sealed.
- Released scores and winner labels must not be read.

## Exact Interface

Use `openai/gpt-5.4`, temperature zero, without reasoning. Run the same five
semantic stages as the closed mechanics protocol:

1. world-conditioned root user feedback;
2. root semantic-likelihood ordering;
3. target-blind branch-conditioned assistant continuation;
4. world-conditioned follow-up user feedback; and
5. follow-up semantic-likelihood ordering.

Text stages retain the already passing strict flat-JSON grammar. Each
likelihood stage returns exactly eight lines:

```text
ACTION_OBSERVATION|Wn>Wn>Wn>Wn
```

Every key must appear once and every line must contain a permutation of all
four worlds. Candidate-world presentation is deterministically shuffled by
task and stage with seed `24414`. Observation labels are independently shuffled
as in the cardinal protocol. The likelihood scorer and continuation policy
never receive the observation-to-truth map.

## Serving Gate

The gate passes only if:

- all five logical requests return and all five stages parse exactly;
- no semantic retry, repair, coercion, or partial analysis occurs;
- at most two adapter-level transport retries occur, with HTTP attempts exactly
  equal to logical requests plus retries;
- reasoning tokens and forced exits are zero; and
- authenticated run cost is at most `$0.15`.

Projected cost is `$0.10`. Passing authorizes only a separately frozen
three-task ordinal mechanics smoke on the reserved development artifacts.
Failure closes this exact interface and serving task without a format-only
rerun.

OpenRouter only. OatML jobs: `0`.
