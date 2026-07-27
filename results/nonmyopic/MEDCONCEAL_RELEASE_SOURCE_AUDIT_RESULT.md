# MedConceal Release Source Audit Result

Date: 2026-07-28

**Status: failed before mechanics-value access and before any model call.**

## Pinned Source

- official repository: `https://github.com/FAIRHealth/MedConceal`;
- commit: `f98d02c1eb9819325f091c0afd0dc4d63d90a21b`;
- commit date: `2026-07-25`;
- tracked files: `106`;
- tracked-content manifest SHA256:
  `8795c51bc64ce4e50fd09f7356dd5c365e991ce876de05dca9faa47aabe25fb3`;
- checkout state: clean.

The release contains `300` case records, human and model dialogue traces, derived
per-case metrics, offline evaluation/aggregation scripts, and plotting scripts.

## Frozen Case Boundary

Only the `case_id` field was extracted from `data/cases.jsonl`. All `300` IDs are
nonempty and unique. Before opening any hidden concern, recommended plan, source URL,
or trace, the preregistered hash ordering produced:

- mechanics: `20`;
- development: `80`;
- confirmation: `100`;
- retained: `100`.

The public manifest is
`results/nonmyopic/medconceal_release_source_manifest.json`, SHA256
`83643be0cd1d305457a59f9b1ea08958e84799ba39077f0e3fe2bd15d98e5458`.
Its source case-file SHA256 is
`9d92d1f6450eeac4f2af8c54a9552d5435961a4598d8154578be76db95862b6a`.

No mechanics, development, confirmation, or retained task value was opened during
this audit. The README's illustrative schema was visible as release documentation.

## Missing Load-Bearing Artifacts

The release README states:

> The patient simulator source code is not included in this release. It will be
> released after paper publication.

It separately states that latent policy weights and hidden-state transitions were
removed from the released traces. The tracked tree has no replacement implementation.
In particular, it does not include:

- patient-response generation code or patient prompt templates;
- concern-revelation state transitions;
- intervention-state transitions;
- the paper's history-conditioned turn scorer, smoothing, or hysteresis logic;
- a clinician-policy runner;
- simulator model/version/temperature/seed configuration;
- dependency or environment lock files;
- a repository or data license.

The released evaluation code operates after interaction. It reads recorded doctor and
patient turns plus already-produced `evaluator_analysis` fields, then computes metrics
or makes judge calls for semantic matching and communication style. It cannot generate
a new patient observation or advance the simulator's hidden state.

The case file also does not redistribute the visible Reddit post text; it provides
source URLs and asks users to verify the source-content license themselves. Therefore
the released case rows alone are not self-contained simulator initial states.

## Gate Decision

The exact release fails the preregistered admission conditions:

- licensing boundaries are not fully specified;
- no executable patient interaction can be instantiated;
- history-dependent response and transition semantics are absent;
- the LLM's load-bearing simulator role cannot be reproduced;
- common-random-number pairing over new trajectories cannot be implemented.

Recorded traces are useful for retrospective policy or evaluation analysis, but they
do not permit counterfactual action selection. A non-myopic policy cannot ask a
different first question and receive the corresponding patient response. Building a
new simulator from the paper, from general medical prompting, or from the observed
traces would define a new benchmark and leak the released endpoint data into its
construction.

Accordingly:

- no mechanics values were opened;
- no OpenRouter call was made;
- development and confirmation remain sealed;
- no local reconstruction or third-party substitution is allowed.

MedConceal remains one of the strongest conceptual fits for LLM-native non-myopic BED,
but this initial release is evaluator-only. Reconsider it when the authors publish the
promised simulator source, with a new version-pinned source audit.

OpenRouter spend: `$0`.
