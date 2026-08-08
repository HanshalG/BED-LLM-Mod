# Bongard OpenWorld Confirmation-96 Power Audit

Date: 2026-08-08

## Decision

Freeze the endpoint-blind confirmation-only expansion from 64 to 96 tasks. Preserve
mechanics-4, development-32, all original confirmation-64 rows, every scientific
effect threshold, the four fixed confirmation dates, and the `$5` account-wide daily
cap.

The original 3% relative-Brier gate is a minimum claim size, not an 80%-powered target.
For paired-difference SD equal to 20--30% of control mean Brier, the estimated true
gain required for 80% marginal power is:

| Stage | Tasks | Required true relative gain |
|---|---:|---:|
| development | 32 | 5.98--8.93% |
| candidate development | 64 | 5.10--6.31% |
| original confirmation | 64 | 7.00--10.51% |
| expanded confirmation | 96 | 5.72--8.58% |

The calculation covers one paired effect gate. Ranking, log-loss, control, and
mechanics conjunctions can only lower full-tier power.

## Partition Integrity

The expansion appends the first 32 byte-valid rows from the already frozen reserve
order. It does not inspect concept, caption, label, model response, or endpoint outcome.
The resulting split is exactly `4/32/96/68` for mechanics/development/confirmation/
reserve.

- expanded confirmation UID SHA256:
  `3826a64b46668226c996afa92e81cf270bf59f99a373813e37196552300ecb26`;
- expansion manifest SHA256:
  `809621dd848741848d25c2ddc8603751d43b24486ef1e666260248f9195cea37`;
- exact selected-image byte overlaps: `0`;
- exact normalized concept/caption development-confirmation overlaps: `0`;
- strict cross-boundary aHash<=2 and dHash<=2 overlaps: `0`.

## Execution Boundary

Confirmation remains conditional on the unchanged literal full LLM-native development
tier. If authorized, it runs as four blocks of 24 tasks. Each block permits at most
1,032 accepted responses, 21 bounded same-payload transport retries, 1,053 HTTP
attempts, and `$4.212` precharged exposure under the unchanged `$4.75` run cap.

Changed-path count gates scale from `24/64` to `36/96`, preserving 37.5%. The 3%
relative-Brier gate, 20,000-bootstrap upper bound, ranking, log-loss, shuffled,
history-blind, fixed-support, matched fixed-score, label-obedience, endpoint-sealing,
and claim gates are unchanged.

## Verification

- protocol manifest V9 SHA256:
  `ad1ddefffb8340dd2ba2b86c86f4d48a1fed05595e5b395ed328a86ba41d05d4`;
- independent expansion-manifest replay: pass;
- independent V9 protocol verification: pass;
- execution binding verification: pass;
- complete Bongard test suite: `142 passed`;
- confirmation block-A preflight: `waiting_for_development`, zero calls/files;
- Aug10 serving/mechanics preflight: `ready_without_paid_calls`, zero calls/files.

No image endpoint, scientific endpoint, or paid model call was opened by this audit.
