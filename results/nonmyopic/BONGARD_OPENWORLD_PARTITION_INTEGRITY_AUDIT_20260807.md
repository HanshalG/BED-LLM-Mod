# Bongard-OpenWorld Partition Integrity Audit

Date: 2026-08-07

## Decision

`partition_integrity_pass`; model calls 0; cost $0.

The original row-disjoint split was not image-independent. It contained four
exact development-to-confirmation duplicate-image groups, two duplicate-image
development tasks, and one additional confirmation-to-confirmation duplicate
group. The deterministic pre-response repair in
`BONGARD_OPENWORLD_PARTITION_INTEGRITY_AMENDMENT.md` removes all such reuse
from mechanics, development, and confirmation while preserving seeded order
and exact 4/32/64 task counts.

## Result

- mechanics unchanged: 4/4 rows;
- development retained 30/32 original rows and deterministically filled 2;
- confirmation retained 57/64 original rows and deterministically filled 7;
- exact image-byte overlap among selected experimental tasks: 0;
- exact development/confirmation concept overlap: 0;
- exact development/confirmation caption overlap: 0;
- strict development/confirmation aHash+dHash near pairs: 0;
- labels, concepts, captions, responses, and endpoints used for selection: no.

Public manifest:
`results/nonmyopic/bongard_openworld_partition_integrity_audit/bongard-openworld-partition-integrity-audit-20260807/MANIFEST.json`

Manifest SHA-256:
`9d9dc695924e728a2653af0b05c575eb4c04a62be933e497afa1bc1ac2bbcbc9`

## Rebound Execution

- development interface v10 manifest:
  `7b96e8c0b86e6ccadbdddb521e150d687218e959cd382c3cdb4f82fa8c4f7973`;
- confirmation freeze V6 manifest:
  `7c14a11e1d3d469697c8545abb58c41d01fd5275f6bfa3f5eb4a60783e6d3ed0`;
- independent development, confirmation, and confirmation-execution replay:
  pass;
- full Bongard regression suite: 119 passed;
- authenticated August 10 read-only preflight: `ready_without_paid_calls`,
  Luna live at `$0.10/$0.60` per million input/output tokens, balance
  `$24.886393846`, model calls 0, files 0.

## Interpretation

This repairs exact-content reuse before any Bongard model response. It does
not show that non-myopic planning works, and it does not imply that all visual
or semantic families are statistically independent. The future reserve and
sealed test require the same earlier-partition exclusion rule before use.
