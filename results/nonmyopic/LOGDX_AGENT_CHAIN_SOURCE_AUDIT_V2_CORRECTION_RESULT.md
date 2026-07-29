# LogDx-CI Agent-Chain Audit V2 Correction Result

Date: 2026-07-29

**Status: all frozen corrected gates pass.**

## Correction Boundary

V2 preserves the exact LogDx-CI v1.2 source, `35` cases, `420` same-Sonnet
agent/single-shot pairs, deterministic replay, case-level averaging, score
definitions, and every V1 threshold.

It adds one exclusion: a later line number or search literal cannot be called
observation-dependent if it already appeared in any prior tool argument. This
removes repeated query patterns echoed by the `grep` observation header.

- superseded V1 audit SHA-256:
  `f7f7289f33434a7e9b3a69bc7dd54abbaedbd55a92fb28be8e29090500c1d9ab`;
- corrected V2 audit SHA-256:
  `dadc701229327e145d5fc01c71f76fe1320c4c4ecfe096d52c56abe0ce39194d`.

## Corrected Result

| Frozen metric | Required | V1 | Corrected V2 |
|---|---:|---:|---:|
| Matched distinct cases | at least 25 | 35 | 35 |
| Cases using at least one tool | at least 12 | 35 | 35 |
| Cases using at least two tools | at least 8 | 29 | 29 |
| Cases with an observation-dependent later call | at least 6 | 21 | 20 |
| Dependency cases with paired gain at least `.10` | at least 5 | 15 | 14 |
| Mean paired score gain across cases | positive | `+.1738` | `+.1738` |
| Mean paired gain on dependency cases | at least `+.10` | `+.3300` | `+.3434` |
| Distinct dependency types | at least 2 | 5 | 5 |

All source, leakage, opportunity, gain, and dependency-diversity gates still
pass. The correction removes one dependency case and one improved dependency
case. The corrected dependency subset has a slightly larger mean paired gain.

The five corrected dependency types remain line number, file/path,
test/symbol, error token, and other literal.

## Decision

The V1 aggregate is superseded. LogDx-CI remains admitted under the corrected
V2 evidence as an executable LLM-native non-myopic BED candidate.

This result authorizes only a separately frozen ten-call serving test. It is
not a StrategyEIG efficacy result.

Model calls: `0`. OpenRouter spend: `$0`.
