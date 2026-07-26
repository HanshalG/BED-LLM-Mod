# VoI Medical Dynamic-Support Recovery Result

## Decision

The frozen 32-call recovery gate failed every delayed truth-coverage
opportunity criterion. The MedDG dynamic-support route is closed before
opening opportunity, development, or holdout values.

## Execution

- Preregistered implementation commit: `4039879`
- Run ID: `voi-medical-dynamic-support-recovery-20260726T062411Z`
- Requests and HTTP attempts: `32 / 32`
- Retries, reasoning tokens, forced exits, forced finalization: `0`
- Eight strict answer maps and 24 strict branch supports: complete
- Every root had at least two positive outcomes
- Cost: `$0.11908`
- Public artifact SHA-256:
  `43501eb3b2179a7a8e9a73d03714fea071ffb1d544f69d489369cdfa5c2b287d`
- Private raw SHA-256:
  `5313fcb47e0f6ea3e4ba34fa050ac43e0ad68133d281cbefaa12a465e509f727`

## Frozen Metrics

For both task `50` and task `284`:

- no positive-probability branch recovered the exact target phrase;
- root expected-coverage range: `0`;
- oracle-minus-myopic expected coverage: `0`.

Mean oracle-minus-myopic coverage was therefore `0`, below the frozen `0.10`
gate.

## Endpoint Audit

The exact failure reveals that the MedDG target labels are not a suitable
external endpoint for open-space clinical support.

Row `50` is labeled `Gastric ulcer`, while the initial generated support already
contained `Peptic ulcer disease`. Branch supports repeatedly contained
`Peptic ulcer disease`, `Duodenal ulcer disease`, or an H. pylori-associated
peptic ulcer. The frozen phrase matcher marks these as misses, but a semantic
clinical endpoint would make the case initially covered and largely saturated,
not a delayed recovery.

Row `284` is labeled `Cold` despite a self-report centered on acute stomach
discomfort and bloating. The free-form generator repeatedly produced clinically
plausible gastrointestinal hypotheses and never produced `Cold`. Rewarding the
source label would favor corpus-specific label mimicry over coherent clinical
hypothesis generation.

This audit does not alter or rescue the preregistered result. It explains why a
semantic judge is not a justified V2: it would remove row `50` from the missing
set and leave only a clinically misaligned row `284`.

## Consequence

Do not:

- rerun with another seed;
- add a post hoc synonym mapper or semantic judge;
- open the opportunity40, development20, or holdout434 values;
- build a patient-level policy on this source.

The target-blind serving pass remains evidence that GPT-5.4 can generate diverse
open clinical support and questions. It does not establish a usable
path-dependent truth-coverage endpoint.

## Budget

- run cost: `$0.11908`
- remaining frozen research allowance: `$0.98204155`
- last authenticated live balance: `$34.115627594`
- balance above protected `$25`: `$9.115627594`
- OatML jobs: `0`
