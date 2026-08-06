# Fully Fresh Qwen Control Verification Protocol

Date frozen: 2026-08-06

## Purpose

Independently verify the later-day fully fresh history-blind control and its
composite result without model calls. The verifier is run only after the paid
control stage writes complete public artifacts. It does not change, rescue, or
relabel any scientific gate.

## Bound Inputs

The verifier consumes one completed daily-stage output directory containing:

- `SOURCE_STAGE.json` and `CONTROL_AUTHORIZATION.json`;
- `source/{RESULT,TREES,TARGETS}.json`;
- `control/{RESULT,CONTROLS}.json`;
- root `RESULT.json` and `CONTROL_STAGE.json`.

The source artifact set is already hash-bound by the authorization generated on
2026-08-06. The future control and composite hashes are learned only after the
registered control completes.

## Exact Checks

Verification passes only if all of the following hold:

- source hashes equal the source-stage and authorization bindings;
- the authorization hash equals the source-stage binding;
- control hashes equal the composite protocol bindings;
- the control's embedded source hashes equal the actual source hashes;
- the control `CONTROLS.json` hash equals its own protocol binding;
- the control-stage result hash equals the root composite result;
- source/control calendar dates preserve the later-day requirement;
- source, control, and total accepted calls equal `3680`, `3072`, and `6752`;
- source and control mechanics pass and total cost is within the registered
  `$9.25` daily-stage composite cap;
- the public control trees and all bootstrap analysis are recomputed from
  `TREES`, `TARGETS`, and `CONTROLS` and match exactly;
- second-draw novelty is recomputed and matches exactly;
- the source/control summaries embedded in the composite equal their component
  results;
- composite gates, usage, decision, and status equal an independent
  recomputation of the frozen conjunction.

The source scientific null remains part of the conjunction. A control science
pass therefore does not make the composite pass when the source dynamic-vs-fixed
gate failed.

## Outputs

The verifier writes `CONTROL_VERIFICATION.json` and
`CONTROL_VERIFICATION.md` in the completed run directory. Raw responses are
neither read nor copied. Verification makes zero provider calls and costs `$0`.
