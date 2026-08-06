# Bongard-OpenWorld Source/Protocol Audit Result

Date: 2026-08-06

## Decision

`source_protocol_pass`; `scientific_opportunity_status=untested`.

Bongard-OpenWorld is the strongest fresh second-environment candidate found in
the zero-call audit. It supplies open-ended visual concepts, externally fixed
image labels, a clean support/query separation, and enough untouched tasks for
development and confirmation. The planner can receive real image content
while every label-bearing source filename and semantic answer remains hidden.

This result authorizes no paid calls. A source/protocol pass does not show that
path-dependent depth-two planning outperforms myopic image selection.

## Bound Result

- Official code commit: `0462ba15f3be2f7f9a7e6cdd0d24314a1fabb5fe`
- Official tree: `b527dfa22222cab3beef5c4023a8b3837d8d7ee5`
- Metadata: 1,010 tasks with exact 610/200/200 official splits
- Images: 14,140 unique references; every task has seven positive and seven
  negative images in the official layout
- Sequential interface: four initially labelled images, eight selectable
  images, two adaptive label queries, and two untouched endpoint images
- Planning space: eight myopic first actions versus 56 ordered depth-two
  action sequences
- Validation partitions: 4 mechanics, 32 development, 64 confirmation, 100
  reserve
- Sealed official test set: 199 tasks after excluding audit-exposed UID `0008`
- Official backup probe: exact 5,124,375,111-byte size and all required ZIP64
  boundary records present
- Source/protocol gates: 11/11 pass
- Model calls: 0
- OpenRouter cost: $0

Public manifest:
`results/nonmyopic/bongard_openworld_source_protocol_audit/bongard-openworld-source-protocol-audit-20260806/MANIFEST.json`

Manifest SHA-256:
`7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763`

## Scientific Boundary

The released `concept` and `caption` fields are hidden from every planner and
may not be used for task selection, planning, or primary scoring. The primary
endpoint is exact classification of the official positive/negative query
images. Concepts may be used only after a protocol is frozen for descriptive
semantic analysis.

Before any VLM experiment, download the official image archive, freeze its
full SHA-256, verify all selected images decode, and keep the filenames outside
the model-facing payload. Then use only the four validation mechanics tasks for
a cheap transport/parser smoke. A later opportunity gate must compare myopic
current-support scoring, fixed-support depth two, path-dependent refreshed-
support depth two, and random selection before a confirmation partition opens.
