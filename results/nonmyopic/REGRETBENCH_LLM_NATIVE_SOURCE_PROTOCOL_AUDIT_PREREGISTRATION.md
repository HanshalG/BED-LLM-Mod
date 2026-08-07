# RegretBench LLM-Native Source And Protocol Audit Preregistration

Date frozen: 2026-08-07

## Question

Can the newly public RegretBench release support a fresh sequential BED study
in which an LLM, rather than the benchmark's hidden finite CIG, generates the
policy's hypothesis support and clarification questions?

This is a zero-model-call source audit. It cannot establish support recovery,
policy efficacy, or a paper claim.

## Source Binding

- Official repository: <https://github.com/ngocminhta/RegretBench>
- Required commit:
  `b2978e1c2e31b7a7c4e1508ee3e1fa1cb98f4aa7`
- Dataset: released `OpenDomainQA` version `1.0.0`.
- Only repository-tracked `test/*.json` CIGs may enter this construction.

The audit must report, rather than silently repair, any disagreement between
the release manifest, checksum inventory, and Git tree. It independently
checks every available test CIG against its published checksum.

## Eligible Source Cohort

A CIG is eligible only when all of the following are true:

1. its ID begins with `ambigdocs_`;
2. it has three through six hidden intents;
3. it has two through four semantic facets;
4. every intent has a nonempty `answer_aliases` value;
5. every intent has a nonempty value for every facet; and
6. every facet takes at least two distinct values across intents.

Eligibility uses benchmark-side hidden structure only to define a coherent
environment. The model-facing payload will contain the ambiguous prompt and
subsequent dialogue only. Hidden intents, aliases, descriptions, slots,
facets, reference questions, metadata, and benchmark belief are forbidden.

## Frozen Split

Eligible CIGs are ordered by
`SHA256("regretbench-llm-native-v1|" + cig_id)`. The first 132 become:

- mechanics: first 4;
- development: next 64;
- confirmation: next 64.

No task is selected using a model response, support-coverage outcome, or policy
endpoint. The confirmation cohort remains unopened to model calls until a
separately frozen development mechanism passes.

## Structural Control

For every available CIG and every selected CIG, compute the exact noiseless
fixed-support two-question objective from the released facet partitions:

- greedy chooses the first facet with maximum one-step information gain;
- depth two chooses the first facet minimizing expected entropy after an
  adaptive second facet.

This is a source diagnostic, not the proposed policy. A clean result for this
project requires no strict depth-two gain on the selected cohort. That makes a
later dynamic-support advantage attributable to LLM regeneration rather than
to a hidden classical decision-tree opportunity.

## Passing Audit

The source audit passes only if:

- the required commit and version match;
- exactly 6,286 tracked test CIGs exist and all match published checksums;
- the manifest/Git-tree train-file discrepancy is explicitly detected;
- at least 132 CIGs satisfy eligibility;
- the 4/64/64 split is deterministic, disjoint, and prompt-unique;
- no selected prompt contains a hidden answer alias;
- model-facing payload fields are exactly `task_id`, `prompt`, and later
  dialogue records; and
- the selected fixed-support depth-two gain is exactly zero within numerical
  tolerance.

A pass authorizes only a separately preregistered exact serving and
answer-conditioned support-recovery gate. It does not authorize a policy
comparison or opening the confirmation cohort.
