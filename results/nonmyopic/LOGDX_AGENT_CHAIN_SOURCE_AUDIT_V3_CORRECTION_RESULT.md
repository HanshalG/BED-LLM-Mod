# LogDx-CI Agent-Chain Audit V3 Correction Result

Date: 2026-07-29

**Status: all frozen final corrected gates pass.**

## Final Dependency Boundary

V3 preserves the exact source, `35` cases, `420` same-Sonnet pairs, scores,
case aggregation, thresholds, and V2 exclusion of values already present in
prior tool arguments.

Before dependency matching, it reduces every earlier tool response to numbered
raw-log lines matching `^\s*\d+\s*:`. Tool-generated headers, separators,
no-match messages, and error messages cannot supply a dependency literal.

- superseded V2 audit SHA-256:
  `dadc701229327e145d5fc01c71f76fe1320c4c4ecfe096d52c56abe0ce39194d`;
- final V3 audit SHA-256:
  `7f6666a19b635087b78015a1ff98b8082c897c60a2eaca20ab8e73440931c4ff`.

## Result

The stricter raw-content rule produces the same aggregate as V2:

- `35` matched cases and `420` matched context rows;
- `35` cases use a tool somewhere;
- `29` cases use at least two tools;
- `20` cases have a raw-log-content-dependent later action;
- `14` dependency cases have mean paired diagnosis gain at least `.10`;
- all-case mean paired gain: `+.1738`;
- dependency-case mean paired gain: `+.3434`;
- five dependency types: line number, file/path, test/symbol, error token, and
  other raw-log literal.

Every original source, leakage, opportunity, gain, and diversity gate passes.
The equality with V2 confirms that generated tool headers did not drive any
case-level admitted result.

## Decision

V1 and V2 are superseded. V3 is the binding LogDx-CI source admission result.
It authorizes only the separately frozen exact ten-call serving smoke.

Model calls: `0`. OpenRouter spend: `$0`.
