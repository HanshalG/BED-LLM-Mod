# Bongard-OpenWorld Sequential BED Source/Protocol Audit

Date frozen: 2026-08-06

## Question

Can Bongard-OpenWorld supply a fresh, externally labelled environment for
non-myopic sequential BED in which a vision-language model performs
irreducible semantic hypothesis and belief work?

This is a zero-model-call source and interface audit. It cannot demonstrate a
non-myopic performance advantage. It may establish only that a later
mechanics/opportunity experiment is reproducible and leakage controlled.

## Bound Source

- Official repository: `rujiewu/Bongard-OpenWorld`
- Commit: `0462ba15f3be2f7f9a7e6cdd0d24314a1fabb5fe`
- Tree: `b527dfa22222cab3beef5c4023a8b3837d8d7ee5`
- Metadata: the official all/train/validation/test JSON files, each bound by
  SHA-256 in the audit implementation
- Image backup: official Google Drive file
  `1aXr3ihVq0mtzbl6ZNJMogYEyEY-WALNr`

The audit will verify the backup with byte-range requests for its exact size,
ZIP local header, ZIP64 end record and locator, and ZIP end record. This is an
availability check, not a full archive hash. No paid experiment is authorized
until the archive is downloaded and its full hash is frozen.

## Sequential Protocol

Each task must contain exactly seven positive followed by seven negative
images. The official loader's positive index 6 and negative index 6 remain the
two untouched endpoint images. The other twelve images form the support pool.

For each task, a hash-derived RNG under seed `20260806` will:

1. reveal two randomly selected positive and two negative support labels;
2. place the remaining eight support images in an unlabeled candidate pool;
3. allow two sequential label queries without replacement;
4. randomly order the untouched positive/negative endpoint pair for final
   classification.

The planner sees the image content and opaque IDs. It never sees source UIDs,
file paths, original positions, concepts, captions, candidate labels, or
endpoint labels. This is essential because official filenames encode
`pos`/`neg` directly.

## Frozen Partitions

The official training split remains training-only. The 200 official
validation tasks are deterministically partitioned into:

| Partition | Tasks | Purpose |
| --- | ---: | --- |
| mechanics | 4 | parser, image transport, and executable-action checks |
| development | 32 | prompt and scoring development |
| confirmation | 64 | fresh validation confirmation |
| reserve | 100 | untouched reserve |

The official 200-task test split remains sealed. UID `0008` is excluded from
that sealed set because its source filenames were inspected during this audit,
leaving 199 sealed test tasks.

## Gates

All are conjunctive:

- source commit, tree, and metadata hashes match;
- exact 1,010-row and 610/200/200 split sizes;
- official splits are disjoint and exhaustive;
- every task has the exact seven-positive/seven-negative path structure;
- all 14,140 image references are unique;
- validation partitions have exact sizes and no overlap;
- every planner payload hides semantic truth, source paths, and unrevealed
  labels;
- official endpoint positions never enter the selectable support pool;
- the official backup passes the frozen ZIP64 range-availability checks;
- eight first actions and 56 ordered depth-two action sequences provide a
  nontrivial structural planning space;
- the audit-exposed test UID is excluded from the sealed test set.

## Interpretation

A pass means `source_protocol_pass`, while
`scientific_opportunity_status` remains `untested` and
`authorizes_paid_calls` remains false. The next admissible step is a cheap
validation-only VLM mechanics/opportunity smoke after the already frozen
OpenRouter schedule. It must compare myopic current-support scoring, fixed
support depth two, path-dependent refreshed-support depth two, and random
selection against the held-out endpoint labels.
