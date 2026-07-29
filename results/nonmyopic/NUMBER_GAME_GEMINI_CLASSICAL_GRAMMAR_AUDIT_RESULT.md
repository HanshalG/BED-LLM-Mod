# Number Game Gemini Classical-Grammar Audit Result

Status: **endpoint-power inconclusive; efficacy unopened**.

Date: 2026-07-29

Model calls and cost: `0` / `$0`.

Public result:
`results/nonmyopic/number_game_gemini_classical_grammar_audit/number-game-gemini-classical-grammar-audit-20260729/RESULT.json`.
SHA-256:
`3eff0be96756adb15d61867b62d1bd0e3e7fad9112214dda788f9a6789634e08`.

## Bound Evidence

The audit uses two independent 32-tree studies and their 16 Gemini endpoint
draws per tree. Every source tree, result, and endpoint hash matches the
preregistration. The classical bank rebuilds to exactly 416,366 unique
nonconstant extensions with the expected canonical SHA-256
`6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0`.

The earlier generated-support result remains bound and unchanged:
grammar-novel second-refresh support occurs on all 64 trees, accounting for
20.17% of occurrences and 50.98% of unique extensions.

## Endpoint Power

The 1,024 Gemini endpoint draws contain 23,054 valid hypothesis occurrences.
Only 60 are outside the classical bank:

| Source study | Total occurrences | Grammar-novel |
|---|---:|---:|
| Fixed policies, fresh endpoints | 11,532 | 21 |
| Wholly fresh replication | 11,522 | 39 |
| **Total** | **23,054** | **60 (0.2603%)** |

There are only 23 unique grammar-novel endpoint extensions. Thirty-five of 64
trees contain any novel endpoint occurrence, but per-tree nonempty-draw counts
range from 0 to 4:

| Nonempty draws out of 16 | Trees |
|---:|---:|
| 0 | 29 |
| 1 | 19 |
| 2 | 13 |
| 3 | 2 |
| 4 | 1 |

No tree reaches the frozen eight-draw analyzability threshold. Therefore all
four endpoint-power gates fail:

- novelty fraction is `.2603%`, below `5%`;
- novel occurrences are `60`, below `512`;
- analyzable trees are `0`, below `48`; and
- each source contributes `0`, below `24`.

## Decision

Fixed-policy depth-three versus depth-two efficacy is not evaluated. We do not
pool the sparse targets across trees, lower the draw threshold, narrow the
grammar, or report a post-hoc underpowered subset score.

No fresh endpoint-only confirmation is authorized. Together with the Qwen
audit, this establishes a useful claim boundary: path-conditioned LLM planning
supports escape a very broad classical closure, but independent Gemini and
Qwen endpoint generators overwhelmingly sample concepts inside it. The
existing depth effect is LLM-native through generated belief dynamics, but an
out-of-bank depth advantage remains unproven.
