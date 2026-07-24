# ClinDiag Fixed-Slot Support Gate Format Amendment

Date: 2026-07-24

Status: **frozen before the v2 serving calls.**

## Trigger

The preregistered v1 smoke made exactly 10 requests with zero reasoning and spent
`$0.03129325`. All eight support-generation calls completed, but one of the two
semantic-audit responses failed the strict parser because it did not contain one row
for every support. The run failed closed and no scientific endpoint was inspected.

The audit prompt requested four rows in prose but its JSON example showed only the
`initial` row. The failure is therefore compatible with a schema ambiguity in the
instrument, not evidence about support stability. V1 remains a banked interface
failure.

## Frozen V2 Change

V2 makes only two observability/format changes:

1. the JSON example explicitly enumerates all four frozen support IDs in order;
2. raw semantic-audit responses and parsed support generations are persisted even if
   strict parsing fails.

No scientific or serving variable changes:

- same seed and cases;
- same sequential stored-evidence path;
- same GPT-5.4 generator and GPT-5.4 Mini evaluator;
- same temperatures, support size, and zero-retry rule;
- same exact 10 physical requests;
- same semantic-overlap and truth-score-gap thresholds;
- same `$0.50` run ceiling and `$0.15` ledger reservation.

V2 is the sole format repair. If it fails parsing or any frozen gate, this exact
support-refresh interface stops.
