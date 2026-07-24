# ClinDiag De-Anchored Full-Refresh Serving Gate

Date: 2026-07-24

Status: **preregistered before any serving response.**

## Motivation

The prior-retaining ordered-pair line failed with strong support inertia. Its refresh
prompt explicitly asked GPT-5.4 to retain plausible diagnoses from the previous
support. This distinct line changes that single mechanism: the previous support is
context only, and every refresh must rebuild the differential from all visible
evidence, replacing unsupported items and introducing diagnoses unlocked by combined
findings.

No failed pair case is reused.

## Frozen Cases And Path

Seed `24297` selected:

- challenging `21991897`;
- rare `rare70`.

Each case follows the same serving path:

1. initial 12-diagnosis support;
2. de-anchored refresh after stored `present_illness`;
3. de-anchored refresh after stored `present_illness` plus `lab_1`;
4. exact replay of the step-3 prompt through an independent adapter;
5. one strict semantic stability audit.

## Frozen Gate

The models, temperatures, parser, and thresholds match the previously qualified
serving gate:

- GPT-5.4 non-reasoning support generation at temperature `0.5`;
- GPT-5.4 Mini non-reasoning measurement at `0.0`;
- exactly 10 physical requests;
- zero retries and zero reasoning;
- all eight supports parse to size 12;
- exact duplicate prompts;
- no full hidden target in source evidence;
- worse directional duplicate semantic overlap at least `0.80` for both cases;
- duplicate truth-score gap at most `0.05` for both cases;
- no parser or runtime failure.

The run ceiling is `$0.50` and projected ledger reservation is `$0.15`. Live provider
credits and the stricter project ledger are checked immediately before launch.

Passing establishes that de-anchoring does not destroy reproducibility and authorizes
only a fresh, small all-one-step headroom screen. Failure closes this prompt line.
