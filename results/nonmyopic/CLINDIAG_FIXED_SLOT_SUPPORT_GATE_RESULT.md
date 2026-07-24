# ClinDiag Fixed-Slot Support-Refresh Gate Result

Date: 2026-07-24

Status: **v2 passed every frozen serving gate; a small structural opportunity screen
is authorized.**

## Result

The format-repaired v2 run made exactly 10 physical requests with zero reasoning,
zero retries, and no parser/runtime failures. All eight generated supports parsed to
12 distinct diagnoses. The two exact final-prompt replays were semantically stable:

| Case | Subset | Worse directional overlap | Truth-score gap |
|---|---|---:|---:|
| `25992750` | Challenging | 0.8333 | 0.0000 |
| `rare167` | Rare | 0.9167 | 0.0000 |

Both overlap values exceed the frozen `0.80` threshold and both truth-score gaps are
below `0.05`. The duplicate prompts were exactly equal, and neither stored path
contained the hidden target literally.

## Manual Audit

The supports were coherent with the visible evidence. For `25992750`, both final
supports concentrated on seizure/encephalitis, trauma, and inguinal-pain mechanisms.
For `rare167`, the lists were almost identical pulmonary infection, vasculitis, and
malignancy differentials.

Neither case recovered its hidden truth at any of these two deliberately limited
actions: all four semantic truth scores were zero for both cases. This does not violate
the serving gate, which preregistered truth-score changes as descriptive. It does mean
the pass establishes only stable path-dependent support regeneration, not an efficacy
or non-myopic result.

## Cost

- v1 format failure: `$0.03129325`;
- passing v2: `$0.02651550`;
- complete fixed-slot support-serving line: `$0.05780875`;
- v2 tokens: 4,296 prompt and 1,583 completion;
- conservative project-ledger remainder after v2: `$23.77172447`.

The provider credits endpoint had not yet posted the v2 charge when checked
immediately after completion, so the lower ledger remainder is authoritative.

## Decision

The exact support-refresh interface is qualified. Proceed only to a small fresh
truth-anchored screen over all eight generic stored-evidence actions. That screen must
show a replicated two-step structural advantage over the best one-step-first
continuation before any target-blind scorer or policy comparison is built.
