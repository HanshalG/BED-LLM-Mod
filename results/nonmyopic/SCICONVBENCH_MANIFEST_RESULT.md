# SciConvBench LLM-Native BED Manifest Result

## Verdict

The value-blind source and split manifest passes every frozen gate. Inspection
of exactly the five preregistered mechanics cases is authorized. No opportunity,
development, or holdout semantic value is authorized yet, and no OpenRouter
call is authorized.

## Reproducible Result

- Source commit:
  `0a87f8e755968a57d4e0ce063a6063fbc18ec87c`
- Protocol:
  `sciconvbench-value-blind-manifest-1`
- Seed: `24420`
- Eligible cases with at least three aligned hidden ontology components: `318`
- Split sizes: mechanics `5`, opportunity `40`, development `20`, holdout `253`
- Combined split hash:
  `4aa6af551369a43d9a29d71d1343627369e96def0df9cf4f2c137958a362791f`
- Public manifest SHA-256:
  `163f019824c2f6d9f3bbe0c33b4039946a06cec99f10bd60de97c238f16ee125`

Eligible counts by domain are:

| Domain | Eligible |
| --- | ---: |
| Fluids | 27 |
| OpenFOAM | 83 |
| Materials tool use | 19 |
| Solid mechanics | 143 |
| Solid-mechanics tool use | 46 |

The exact source file hashes, component-count histograms, domain-stratified
split IDs, and frozen split hashes are recorded in
`results/nonmyopic/sciconvbench_manifest/MANIFEST.json`.

## Access Accounting

The manifest emitted no incomplete requirement, complete requirement,
missing-entity text, ontology-component text, conversation outcome, or model
score. It made zero OpenRouter calls, cost `$0`, and submitted zero OatML jobs.

The only newly authorized semantic inspection is:

- `fluids:case_001`
- `foam:case_001`
- `matToolUse:case_005`
- `solMech:case_001`
- `solToolUse:case_017`

That mechanics audit must close the route unless at least three cases expose a
real prerequisite or answer-conditioned parameter dependency and an
independently scoreable response/final-specification endpoint.
