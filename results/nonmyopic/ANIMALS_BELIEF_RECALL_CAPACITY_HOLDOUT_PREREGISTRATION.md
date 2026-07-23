# Animals Capacity-Gated Belief-Recall Holdout Preregistration

Status: **frozen before any response involving the new 60-target set**.

## Motivation

The ungated seed-24271 holdout failed its positive-CI gate despite a positive
mean. Post-hoc analysis found that the ranker's largest losses occurred after
the current support had already expanded well beyond one generation batch.
This motivates a distinct confidence-gated selector:

```text
if current support size <= max_num_samples:
    select the belief-recall ranker's top candidate
else:
    select immediate EIG's top candidate
```

The threshold is not numerically tuned. It is exactly the configured
hypothesis-generation capacity, `max_num_samples=16`. The interpretation is
that recall assistance is used only when the merged current support has failed
to expand beyond what one generation call can produce.

The post-hoc seed-24271 result is quarantined and remains a failed ungated
holdout. It is not pooled with or relabeled by this test.

## Fresh Design

- Producer seed `24275`.
- Sixty new distinct targets, disjoint from both development traces and the
  ungated holdout.
- One attempt per target, three ordinary production candidates per state.
- Unchanged non-thinking Gemma 4 26B questioner, answerer, belief generator,
  validator, history filter, likelihood model, and frozen ranker prompt.
- The ranker is called for every state, so the EIG fallback does not receive
  less model compute.
- Target and truth measurements remain excluded from every model-visible
  payload by the existing explicit allowlist.
- No intermediate endpoint is written or inspected.

## Frozen Endpoint And Gates

The primary per-state quantity is expected truth coverage of the
capacity-gated top candidate minus immediate EIG's top candidate. Producer
bootstrap seed `24276` uses 10,000 paired replicates.

Pass requires:

1. all 60 distinct states complete;
2. at least 15 states have nonzero candidate coverage spread;
3. the paired 95% interval has a strictly positive lower bound;
4. capacity-gated wins exceed losses;
5. active-state regret is lower than immediate EIG;
6. the selector threshold exactly equals `max_num_samples=16`.

Ungated ranker results are diagnostic only. Independent audit seed `24277`
rebuilds all payloads, reparses all responses, reconstructs the capacity gate,
and repeats the bootstrap and scientific gates.

Any serving, mechanics, producer, or audit failure stops this capacity-gated
line without prompt, model, target, seed, threshold, endpoint, or replacement
repair.

## Spend

- Coverage run cap: `$1.50`; projected cost: `$0.75`.
- Ranker cap: `$0.10`.
- OpenRouter concurrency: `256`.
- Project spend before responses: `$41.08183780` of `$110`.
