# HiddenBench Dynamic-Belief V2 Ordering Terminal Result

Date: 2026-08-13

Status: **pre-serving privacy-ordering failure; exact V2 cohort closed.**

## Result

The V2 reserve source admission passed all seven metadata-only gates and fixed the
first four rows of the original nine-row reserve. No semantic value was used to
select the interface or cohort.

During implementation, a local custodian test called `endpoint_view` on those four
real rows in the same test process as the planner/router projection. This
materialized all four registered correct answers before the ten label-free serving
responses or serving gates existed, violating the frozen ordering requirement.
One answer string also appeared in the test failure diff because that string was
already naturally present in the router-view scenario/facts. The four answer
designations, not merely that one string, are considered opened.

| Quantity | Result |
|---|---:|
| V2 source gates | 7 / 7 |
| registered answer designations materialized | 4 / 4 |
| model requests / HTTP attempts | 0 / 0 |
| OpenRouter cost | `$0` |
| model responses / planner scores / policy endpoints | 0 / 0 / 0 |
| remaining original reserve rows not selected by V2 | 5 |

Authenticated cumulative credits/usage remained
`$245.000000000 / $220.178352166`; Aug-13 conservative account-wide spend remains
`$0.044223286` from the frozen `$220.134128880` boundary.

## Interpretation

This is a privacy-ordering implementation null. It provides no evidence about the
LLM, regenerated beliefs, depth-two planning, or HiddenBench policy efficacy.

The exact V2 protocol, first-four reserve cohort, prompts, model, and seeds are
closed. They must not be repaired or run. A distinct successor may use only an
untouched subset of the remaining five reserve rows and must satisfy two stronger
requirements before real-source projection:

1. endpoint code is rehearsed only on synthetic fixtures;
2. the real endpoint custodian is a separate executable that cannot be invoked by
   source, serving, or verifier tests and is called only after a banked label-free
   pass token has been independently verified.

No development, confirmation, endpoint, or paper-efficacy claim is authorized.

## Integrity

- V2 mechanics protocol SHA-256:
  `f7e414eb844cbcfb8974254b54691fff5c8350fe1efefbe61473dab573ad83a3`;
- V2 source manifest SHA-256:
  `fb0ced41fef4448e3c2fc93038f499977b00d1ec7ec362284a7d5c8b9e4686d5`;
- V2 source audit SHA-256:
  `27aecb38fe9ca7698edd034b181d040915924bad16fa66be34f5bff06d137610`;
- failed custodian implementation SHA-256:
  `cb0499cde29b88230e103248394dc40e0eda6b0ee73f7422c0f7ecac5d9e6988`;
- failing test implementation SHA-256:
  `3aa2ec99ec6d4afa0a7b2b4adfbc157807bbc231a6d95923ed210928160070ce`.
