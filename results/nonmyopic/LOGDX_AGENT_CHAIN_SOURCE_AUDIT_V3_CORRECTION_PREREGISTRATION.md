# LogDx-CI Agent-Chain Audit V3 Correction Preregistration

Date: 2026-07-29

Status: **frozen before the V3 aggregate is computed**.

## Correction

The V2 audit correctly excludes literals and line numbers already supplied in
prior tool arguments. During deterministic serving-fixture work, a second
boundary issue was identified before any paid response: V2 searches the entire
tool observation for later regex literals. Released tool observations include
generated headers such as the query pattern, match count, range count, and
line-range description. A model could therefore use a tool-metadata word rather
than a literal returned from the raw CI log.

The public V2 audit SHA-256
`dadc701229327e145d5fc01c71f76fe1320c4c4ecfe096d52c56abe0ce39194d`
is preserved and superseded, not overwritten.

## Frozen V3 Rule

V3 keeps the exact source, `420` rows, scores, case aggregation, V2 prior-
argument exclusion, and every original threshold.

Before testing a dependency, each earlier tool observation is reduced to lines
matching:

```text
^\s*\d+\s*:
```

These are the released tools' numbered raw-log content lines. Tool-generated
headers, separators, no-match messages, and error messages are excluded.

A later action is dependent only if its line number or substantive search
literal:

1. occurs in those numbered raw-log lines;
2. is absent from the initial reduced context; and
3. is absent from all prior tool arguments.

All V1/V2 thresholds remain unchanged. Failure closes LogDx before serving.
Pass authorizes only the already drafted but not yet committed exact ten-call
serving qualification, updated to bind the V3 audit.

No model call, case removal, threshold change, or OpenRouter spend is allowed.
