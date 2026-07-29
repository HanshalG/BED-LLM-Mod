# LogDx-CI Agent-Chain Source Audit Result

Date: 2026-07-29

**Status: all frozen source and published-trace gates pass.**

## Pinned Release

- official repository: `https://github.com/eyuansu62/LogDx`;
- release: annotated tag `v1.2`;
- checked-out commit:
  `99591c1471118c95155976346df72f520a05f100`;
- protocol lock SHA-256:
  `c5e77e91267036ceb1e80df1d035caa78eb0c831ae690ab5eb0149215fb36087`;
- public audit SHA-256:
  `f7f7289f33434a7e9b3a69bc7dd54abbaedbd55a92fb28be8e29090500c1d9ab`.

The official protocol validator reproduces all `27` locked hashes and all
`35` cases across six splits. The audit independently verifies every committed
raw-log, case-metadata, and ground-truth hash from the six split manifests.

## Source Boundary

All `35` ground truths contain:

- a nonempty concrete root-cause summary and category;
- one or more required diagnostic signals;
- one or more line-indexed evidence spans.

The released agent receives safe case metadata and a reduced initial context.
Its four tools (`grep`, `read_file`, `tail`, and `view_log_lines`) read only
the case's `raw.log`. The shim rejects payloads containing ground truth,
failure category, or required signals. Diagnosis scoring is deterministic for
a fixed released diagnosis and ground truth.

All source and leakage gates pass.

## Published-Trace Opportunity

The audit matches the released Sonnet 4.6 `real-agent-v1` rows against the
same-model `real-debugger-v2` rows for all `12` context providers and all
`35` cases, yielding `420` paired rows.

Tool observations are not stored in the released diagnosis rows. The audit
therefore replays the released deterministic tools on each hash-verified raw
log using the recorded tool arguments. A later call is classified as dependent
only when a substantive line number or search literal:

1. is absent from the assistant-visible initial context; and
2. appears literally in an earlier reconstructed tool observation.

No root-cause or raw-log text is emitted in the public result.

| Frozen metric | Required | Observed |
|---|---:|---:|
| Matched distinct cases | at least 25 | 35 |
| Cases using at least one tool | at least 12 | 35 |
| Cases using at least two tools | at least 8 | 29 |
| Cases with an observation-dependent later call | at least 6 | 21 |
| Dependency cases with paired gain at least `.10` | at least 5 | 15 |
| Mean paired score gain across cases | positive | `+.1738` |
| Mean paired gain on dependency cases | at least `+.10` | `+.3300` |
| Distinct dependency types | at least 2 | 5 |

The five observed dependency types are:

- discovered line number;
- file or path;
- test or symbol;
- error token;
- other literal search term.

Every frozen published-trace gate passes.

## Interpretation

This is strong zero-call evidence for the needed **unlock mechanism**. A first
tool observation often exposes the coordinate or semantic token needed to form
a useful second query, and those chains are associated with materially better
external diagnosis scores.

It is not yet evidence that StrategyEIG selects those chains better than a
myopic policy. The source traces come from an existing reactive agent, and the
benchmark contains only `35` public cases. The next experiment must therefore
test the first link directly: whether an LLM-generated depth-two strategy score
ranks realized diagnosis improvement better than one-step EIG, under the same
two-query budget and deterministic observations.

## Decision

LogDx-CI is admitted as an LLM-native non-myopic BED candidate.

This pass authorizes only a separately frozen ten-call serving test for:

- semantic root-cause support generation;
- syntactically valid deterministic tool queries;
- observation-conditioned follow-up generation;
- hypothesis-conditioned likelihood scores.

No policy efficacy or holdout endpoint has been opened by this audit.
OpenRouter spend: `$0`.
