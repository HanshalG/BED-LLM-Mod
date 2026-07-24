# ClinDiag Branch-Opportunity Interface Amendment

Date: 2026-07-24

Status: **frozen after the serving smoke and before any opportunity-case response.**

## Observation

The exact 10-request serving smoke passed every interface contract:

- eight generated supports parsed to 12 unique diagnoses;
- exactly 10 physical requests;
- zero retries, reasoning tokens, forced exits, or runtime failures;
- both identical-prompt replicates had exactly the same semantic truth-match score as
  their originals.

The two identical replicates nevertheless had exact-name support Jaccards of `0.60`
and `0.4118`. Inspection showed clinically equivalent paraphrases and changed subtype
wording, for example:

- `Metastatic angiosarcoma involving the lung` versus
  `Metastatic angiosarcoma of the lung`;
- `Schizophrenia associated with Darier disease` with alternate parenthetical
  expansions;
- an otherwise identical infective-endocarditis diagnosis with an added
  `and brain abscess` qualifier.

Exact string identity is therefore not a valid noise measure for the semantic
truth-match endpoint.

## Amendment

Before any formal opportunity response:

- retain exact-name support Jaccard as a descriptive diagnostic;
- remove the preregistered mean-Jaccard `>=0.75` pass criterion;
- retain both endpoint-aligned identity controls unchanged:
  - mean duplicate semantic-score gap `<=0.05`;
  - maximum duplicate semantic-score gap `<=0.15`.

No action, case, support width, model, temperature, semantic threshold, opportunity
metric, non-myopic threshold, or formal split changes. The formal run remains a single
frozen 12-case measurement.

This amendment prevents harmless lexical paraphrase from vetoing the experiment while
preserving the control that directly tests whether serving noise can imitate the
reverse-order and non-myopic semantic-score effects.
