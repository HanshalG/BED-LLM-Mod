# GuessingGame Path-BED Source Audit Result

Date: 2026-07-29

Status: **all frozen source gates pass**.

## Source

- official repository commit:
  `df56f1f13fefc4a8ba2c1f89d5026c027f5f42b3`;
- object vocabulary SHA-256:
  `a55a5f9410c8fe3e7cb34e3f57f51b0ad96a6d52db4b261204a48b9685e9fa58`;
- released GPT-4o open-game log SHA-256:
  `a41a4dd3bccc444555c054f362c41e0fe747de79fda9183a893bf195cf0db6d4`.

The source repository has no license file. No vocabulary or raw trajectory is
redistributed by this project.

## Result

| Metric | Value |
|---|---:|
| released objects | 858 |
| released games | 858 |
| eligible two-action rows | 837 |
| excluded rows | 21 |
| unique material answers | 553 |
| material collision excess | 284 |
| unique function answers | 709 |
| function collision excess | 128 |

Exclusions comprise six rows without a primary-function second question, one
empty function answer, 14 material answers that literally name the target,
and one function answer that literally names the target. Error categories can
overlap.

Seed `39400` freezes opaque, disjoint splits:

| Split | Count |
|---|---:|
| serving smoke | 5 |
| mechanics | 10 |
| development | 32 |
| confirmation | 64 |
| unused | 726 |

The public manifest contains only opaque case IDs and source-row hashes. It
contains no target, question, or answer text.

## Interpretation

Every selected target has immutable material and function observations. Asking
them in opposite orders exposes the same evidence set, while the bounded LLM
hypothesis retrieval process can remain order dependent. This supports a
clean test of whether depth-two planning can choose the order that produces a
better terminal semantic belief than a myopic one-step selector.

This source pass authorizes only an exact ten-call structured retrieval
serving smoke. It does not authorize mechanics, development, or confirmation.

## Artifact

- Public manifest:
  `results/nonmyopic/guessinggame_path_bed_source_audit/guessinggame-path-bed-source-audit-20260729T020019Z/MANIFEST.json`
- SHA-256:
  `8d1269e197c9cf05865852353b16eeadb36cc8db6231af1d429b7650b263e7b8`
- Model calls and cost: `0`, `$0`.
