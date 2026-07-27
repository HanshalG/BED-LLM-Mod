# LongVidSearch Four-Hop Ranking Mechanics Result

## Decision

The disclosed-development ranking mechanics gate **fails closed before
scientific scoring**. The exact entropy-ranking route is closed. No reserve
policy or confirmation is authorized.

Run ID:
`longvid-four-hop-ranking-mechanics-20260727T150132Z`.

Public failure artifact contains private raw SHA-256:
`1af086f7e3aae783b95277d7781f5e1504fdf573c85c420efafd71c5afd92ad0`.

## Failure

The run completed and checkpointed:

- 6/6 initial-support responses; and
- 12/12 first-caption refresh responses.

Parsing then stopped at refresh response 12, line `H2`. The model emitted
anchor:

`€4bn`

The token was copied from the visible caption and is semantically suitable,
but the frozen grammar permits only `[A-Za-z0-9_-]+`. No punctuation cleanup,
Unicode normalization, token extraction, reparse, partial subset, repair, or
reissue is allowed.

The remaining 36 refresh calls were not made.

## Accounting

| Item | Result |
|---|---:|
| Physical requests | 18 |
| HTTP attempts | 18 |
| Prompt tokens | 10,695 |
| Completion tokens | 5,948 |
| Reasoning tokens | 0 |
| Retries | 0 |
| Forced exits | 0 |
| Cost | `$0.1159575` |

The hidden necessary-clip endpoint remained unloaded. No four-step path,
entropy score, root choice, coverage, pairwise accuracy, correlation, or policy
comparison exists.

This is a transport/grammar failure, not evidence for or against LLM
non-myopic ranking.

## Scope

The replicated classical four-hop opportunity and successful synthetic
support-regeneration smoke remain valid. What remains unmeasured is the
scientific first link: whether entropy changes in the LLM's path-dependent
semantic support rank realized evidence coverage.

The exact route cannot be rescued by accepting currency-prefixed anchors or
changing the anchor field to a token index after this outcome. A future method
must be materially different rather than a rerun of this disclosed task set.

## Budget

Using provider balance immediately before the run minus exact local cost:

- conservative remaining balance: `$33.403250094`;
- protected through Monday: `$25`;
- remaining pre-Monday ceiling: `$8.3292075`;
- local project ledger: `$96.98155260922341`.

OpenRouter only. OatML/Slurm: `0`.

