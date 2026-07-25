# AmbigDocs Mini Binary Serving Result

## Outcome

The target-free gate completed cleanly but failed the frozen semantic usefulness
conjunction.

- Run: `ambigdocs-binary-serving-20260725T141838Z`
- Requests / HTTP attempts: `10 / 10`
- Reasoning tokens / retries / forced exits: `0 / 0 / 0`
- Cost: `$0.00826575`
- All five questions and all five six-character maps parsed.
- Unique questions: `3 / 5` (required at least 4)
- Unique partitions: `5 / 5`
- Informative partitions: `4 / 5`
- Informative entropy range: `0.0566` nats (required at least `.10`)
- Hidden target sampled: `false`

No target, responder, policy score, or endpoint was accessed.

## Likelihood Instability

Three calls generated the identical question:

> Are you asking about the city in Belarus?

The separate temperature-zero classifier returned three different likelihood maps:

```text
YNNNYN
NNYYYN
NNNYNY
```

Their pairwise Hamming distances are `3/6`, `4/6`, and `3/6`: mean disagreement is
`10/18 = 55.6%`. Because each classifier request received the same question and the
same six documents, this is direct evidence that Mini's semantic likelihood map is
not stable enough for a one-call BED score on this case.

The first generated question was also classified as `YYYYYY`, despite contrasting the
city with things named after it, showing a semantic rather than serialization failure.

## Decision

The exact Mini interface is closed and does not authorize an efficacy run. AmbigDocs
itself remains structurally promising because support preservation and all serving
syntax succeeded.

One distinct follow-up is justified before closing the environment: a target-free
GPT-5.4 stability gate with four generated questions and repeated classification of
the exact same question. It must establish question diversity, informative likelihoods,
and exact repeated-map agreement before any target is sampled. This is a model-capacity
test, not a parser or threshold repair.

The official test split remains sealed. OatML was not used.
