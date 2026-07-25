# tau-Knowledge Qwen3 14B Compact Thinking Result

Date: 2026-07-25

## Decision

**The serving-and-efficacy smoke passed, but confirmation failed closed and the
Qwen interface is closed.** The model retained useful local continuation
ranking on valid rows, but did not transfer the load-bearing full-tree root
ranking mechanism.

Smoke artifact:

`results/nonmyopic/tau_knowledge_qwen14b_compact_scorer_smoke/tau-knowledge-qwen14b-compact-smoke-20260725T050950Z/SERVING_SMOKE.json`

Confirmation failure:

`results/nonmyopic/tau_knowledge_qwen14b_compact_scorer_confirmation/tau-knowledge-qwen14b-compact-confirmation-20260725T051319Z/CONFIRMATION_FAILURE.json`

## Passed Smoke

The two-tree smoke completed exactly:

| Metric | Result |
|---|---:|
| Logical / physical / HTTP requests | 14 / 14 / 14 |
| Retries / forced exits | 0 / 0 |
| Reasoning tokens | 27,451 |
| Focused pairwise accuracy | `.6471` |
| Oracle-optimal focused choices | `7/10` |
| Focused mean regret | `.30` |
| Varying focused vectors | `10/10` |
| Cost | $0.01728746 |

Every frozen smoke gate passed, authorizing the unchanged 20-task
confirmation.

## Confirmation Serving Failure

| Metric | Result |
|---|---:|
| Logical prompts returned | 140 / 140 |
| Physical requests / HTTP attempts | 151 / 151 |
| Transport retries | 0 |
| Forced exits | 21 |
| Forced-final requests / nonempty successes | 11 / 1 |
| Reasoning tokens | 364,376 |
| Prompt / completion tokens | 513,127 / 367,448 |
| Cost | $0.17470294 |
| Valid myopic root objects | 20 / 20 |
| Valid non-myopic root objects | 20 / 20 |
| Valid focused objects | 90 / 100 |

Ten focused first passes exhausted the provider ceiling and their nominally
reasoning-disabled 512-token continuations still spent the allowance without
returning a visible JSON object. The frozen parser therefore stops before
policy scoring. No missing row was retried, imputed, normalized, or replaced.

## Diagnostic Ranking

The fully observed root block provides a valid diagnostic independent of the
missing focused rows:

| Root signal | Qwen result | Frozen requirement |
|---|---:|---:|
| Comparable pairs | 121 | at least 50 |
| Myopic accuracy | `.5537` | descriptive |
| Non-myopic accuracy | `.5000` | at least `.60` |
| Non-myopic minus myopic | `-.0537` | at least `+.05` |
| Strongest nonsemantic root control | `.5455` | Qwen must exceed |

Thus the load-bearing root-ranking gates fail even if serving were repaired.

On the 90 valid focused rows only:

| Focused signal | Diagnostic value |
|---|---:|
| Pairwise accuracy | `.6292` |
| Oracle-optimal choices | `68/90` (`.7556`) |
| Mean regret | `.2556` documents |
| Strongest nonsemantic control | `.6316` |

The valid-row continuation metrics satisfy the generic `.60`, `.70`, and
`.30` thresholds but narrowly miss the frozen nonsemantic accuracy control.
Because ten rows are missing non-randomly after length exits, these are
diagnostic rather than confirmation statistics.

## Interpretation

Qwen3 14B thinking can perform useful local document-evidence comparison, as
shown by the passed smoke and 90 valid confirmation rows. It does not reproduce
GPT-5.4's global ranking of first-search roots from complete semantic trees.
The failure is therefore not only serialization: the full-tree causal link is
negative on every available root score.

No reasoning-budget, forced-final, parser, prompt, model-size, or task repair
follows. Cross-family transfer remains unsupported. The headline remains the
GPT-5.4 held-out result, with model specificity stated as a limitation.
