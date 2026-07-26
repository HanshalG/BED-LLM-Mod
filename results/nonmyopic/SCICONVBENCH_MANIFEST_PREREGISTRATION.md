# SciConvBench LLM-Native BED Manifest Preregistration

Frozen before inspecting any released incomplete requirement, complete
requirement, missing-entity text, ontology-component text, conversation, or
model outcome.

## Question

Does SciConvBench expose enough structured, multi-component scientific
underspecification to support a genuinely LLM-native non-myopic BED
construction?

This is a source and split gate only. It does not establish a planning gap and
does not authorize model calls. The intended scientific mechanism is that an
LLM must maintain and regenerate semantic hypotheses over a hidden scientific
specification, while an atomic clarification can expose a prerequisite that
changes which later scientific parameter is meaningful to ask about.

## Frozen Source

- Repository: `https://github.com/csml-rpi/SciConvBench`
- Commit: `0a87f8e755968a57d4e0ce063a6063fbc18ec87c`
- Included disambiguation domains:
  `fluids`, `foam`, `matToolUse`, `solMech`, `solToolUse`
- Source hashes are frozen in `scripts/sciconvbench_manifest.py`.

The release contains 570 disambiguation cases across seven domains. This
protocol excludes `matSci` and `pde` because only 10 and 9 cases respectively
hide at least three ontology components. The five included domains contain 318
such cases.

## Access Boundary

Before the manifest passes, public output may contain only:

- repository, commit, file names, byte hashes, and row counts;
- record IDs;
- missing-component count histograms;
- split IDs, sizes, and hashes;
- booleans confirming that semantic content and prior outcomes were not
  emitted;
- zero-call accounting.

It must not emit or summarize incomplete requirements, complete requirements,
missing entities, ontology components, conversations, generated
specifications, or model/judge outcomes.

## Frozen Eligibility And Split

Eligibility requires at least three aligned `missing_entities` and
`ontology_components`.

Use seed `24420` independently within each domain:

1. sort eligible IDs and assign the first to mechanics;
2. rank remaining IDs by
   `SHA256("24420:<domain>:<id>")`;
3. assign the first 8 to opportunity;
4. assign the next 4 to development;
5. assign the remainder to holdout.

| Split | Records | SHA-256 |
| --- | ---: | --- |
| Mechanics | 5 | `8c5117faf027e1a20d80b30e0aad832213fe73e40374dfe05694158d958a05fc` |
| Opportunity | 40 | `841250d0690a8ee43922065a98cf67aeb1e109ad177a4b94519e06d2c7eed926` |
| Development | 20 | `73528ea7d1b4832d3977337bd3b1a765fd9c74c97c288eeaa17dbfb783888dde` |
| Holdout | 253 | `eda8815ca90ee5243e406d19555f52b42d21079b12b6f8bcab690cd71024dd49` |

Combined ordered split hash:
`4aa6af551369a43d9a29d71d1343627369e96def0df9cf4f2c137958a362791f`.

The mechanics IDs are value-blind first-eligible records:
`fluids:case_001`, `foam:case_001`, `matToolUse:case_005`,
`solMech:case_001`, and `solToolUse:case_017`.

## Conjunctive Gates

1. source commit and all five file hashes match;
2. exact row counts are 51, 100, 48, 143, and 85;
3. IDs are nonempty and unique within domain;
4. required source fields have exact container types and nonempty values;
5. missing entities and ontology components are aligned one-to-one;
6. eligible counts total exactly 318;
7. split sizes are exactly 5/40/20/253, are disjoint within each domain, and
   exhaust every eligible record;
8. all frozen split hashes match;
9. no semantic content, conversation outcome, or model score is emitted;
10. OpenRouter calls/cost and OatML jobs are all zero.

A pass authorizes inspection of only the five mechanics records and source
runtime. That inspection must answer, before any opportunity value access:

- whether ontology entries distinguish parameter names from hidden values;
- whether at least three cases contain a real prerequisite or
  answer-conditioned parameter dependency rather than an additive checklist;
- whether a target-blind response can be grounded in released truth without
  exposing the full complete requirement;
- whether final specification fidelity can be scored without using the same
  model that generated the belief.

Only a positive mechanics result may authorize a separately preregistered
zero-call opportunity audit. No OpenRouter spend is authorized by this
manifest.
