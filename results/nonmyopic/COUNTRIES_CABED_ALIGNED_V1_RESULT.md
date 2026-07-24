# Countries CA-BED Aligned Semantic V1 Result

Date: 2026-07-24

## Decision

The serving smoke failed during branch-menu validation. No follow-up semantic
table, depth score, target endpoint, or formal-tree response was produced. The
exact branch-specific V1 interface is closed without retry or content repair.

## Integrity and Cost

Run `countries-cabed-aligned-v1-smoke-20260724T221213Z`:

- 34 physical requests: 18 full GPT-5.4 question requests and 16 GPT-5.4 Mini
  semantic-table requests;
- zero reasoning tokens, retries, forced exits, or transport errors;
- 21,092 prompt and 10,326 completion tokens;
- cost `$0.08660050`; and
- all eight root semantic tables for both smoke trees completed before branch
  generation.

The parser then rejected a follow-up that duplicated a question already used in
the same shared tree. The frozen protocol forbids dropping, repairing, or
regenerating that row.

## Audit Defect

V1 checkpointed raw responses only after all four generation phases. Because the
exception occurred after branch generation, the private raw checkpoint was not
written. Usage logs and the public failed-closed artifact remain, but the exact
offending text cannot be recovered.

The harness is corrected prospectively to checkpoint after every completed
phase. V1 is not rerun and no endpoint is reconstructed.

## Consequence

The failure is serving-only, not evidence for or against a country-domain depth
gap. A distinct final architecture may avoid branch-menu global constraints by
generating one target-blind question bank, labeling it once, and giving every
root access to the same remaining follow-up bank. That changes the candidate
process rather than repairing the failed response.
