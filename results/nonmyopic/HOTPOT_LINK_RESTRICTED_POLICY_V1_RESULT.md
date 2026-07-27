# HotpotQA Link-Restricted Policy V1 Result

## Decision

Serving passed, but development **fails closed before policy scoring** on one
out-of-order myopic response. No aligned, fixed, shuffled, final-answer,
support endpoint, or confirmation stage was reached.

## Failure

Initial beliefs (`20/20`) and all root-conditioned refreshes (`80/80`)
completed. All 20 myopic responses contained four two-field root rows. One
response used prefix order:

```text
R1, R2, R4, R3
```

instead of frozen numeric order `R1, R2, R3, R4`. The rows were not reordered
or coerced. The strict parser raised `ValueError: myopic rank row is invalid`.

This is an interface-only failure. It provides no evidence about non-myopic
efficacy or path-conditioned belief value.

## Integrity

- Development run:
  `hotpot-link-restricted-development-20260727T223005Z`
- Accepted logical requests / HTTP attempts / retries: `120 / 120 / 0`
- Prompt / completion / reasoning tokens: `66,788 / 40,413 / 0`
- Forced exits: `0`
- Adapter-recorded cost: `$0.773165`
- Private raw SHA-256:
  `156480e007ff46d24ca3e21ba2f8e9e6bd341f9e9c007a8bcf103f7f49b29dd0`
- Confirmation endpoint rows opened: `0`

The 20 V1 development rows are consumed and cannot be rerun.

## Admissible Successor

A transport-distinct V2 may freeze a root-ID keyed parser that accepts the four
rows in any order while still requiring:

- exactly one row for each `R1` through `R4`;
- exactly one rank `1` through `4`;
- only locally available action IDs; and
- no extraction, repair, or semantic reissue.

V2 must use unqueried rows and freeze a new development/confirmation boundary
before opening endpoints.

Authenticated remaining balance: `$28.377802094`. No fixed reserve.
