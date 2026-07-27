# LongVidSearch Four-Hop Tradeoff Opportunity Result

## Decision

The exact V1 opportunity conjunction **fails**. The frozen confirmation block
must not be opened under V1, and no model call is authorized.

The audit completed all 40 opportunity trees with zero model calls and zero
spend. The public artifact SHA-256 is
`3cb882b1facbef2ababe2f8be68529423773c362a1eb4402b68d0bb240cc3ce0`.

## Frozen-Gate Results

| Gate | Result | Pass |
|---|---:|:---:|
| Complete tasks | 39/40 | **no** |
| Diverse-root tasks | 40/40 | yes |
| Depth-four gain tasks | 39/40 | yes |
| Mean oracle four-clip coverage | 0.6500 | yes |
| Mean coverage gain | 0.4125 | yes |
| Strict tradeoffs | 10/40 | yes |
| Strict total gap | 11 clips | yes |
| Mean strict answer sacrifice | 0.2650 | yes |

V1 fails solely because all 40 tasks were required to have at least two
non-stopword answer terms.

## Completeness Failure

Row `1148` is a State Mutation task with:

- 75 captions;
- 20 roots;
- zero non-stopword answer terms;
- greedy and oracle final count both four; and
- no strict tradeoff.

The retrieval tree is complete, but the frozen direct-answer diagnostic is
undefined for that answer. The row cannot be silently accepted, removed, or
retokenized after outcomes. The all-or-nothing V1 gate is therefore false even
though every scientific prevalence and effect-size gate passes.

## Structural Signal

The tradeoff is distributed across all three categories:

| Category | Tasks | Complete | Strict | Gap | Same root | Mean oracle coverage | Mean gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Causal Inference | 16 | 16 | 4 | 4 | 12 | 0.5938 | 0.3438 |
| Global Summary | 10 | 10 | 3 | 4 | 7 | 0.7000 | 0.4500 |
| State Mutation | 14 | 13 | 3 | 3 | 11 | 0.6786 | 0.4643 |

This is the strongest LongVid structural signal so far: one quarter of tasks
have a lower-immediate root with higher final necessary-clip coverage. It is
still development-only and cannot override the failed conjunction.

## Next Admissible Step

A materially versioned confirmation protocol may prospectively define
answer-scorable eligibility on the untouched 40-video confirmation block:
all tasks must remain in retrieval aggregates, only tasks with at least two
answer terms may enter the direct-answer tradeoff endpoint, and a frozen
minimum scorable count must be met. Such a protocol would be a V2 confirmation
motivated by a schema defect, not a V1 pass or rerun.

Until that protocol is committed and passes, model spend remains forbidden.
The 40 confirmation and 22 reserve videos remain caption-unopened.

Calls: `0`. Cost: `$0`. OatML/Slurm: `0`.

