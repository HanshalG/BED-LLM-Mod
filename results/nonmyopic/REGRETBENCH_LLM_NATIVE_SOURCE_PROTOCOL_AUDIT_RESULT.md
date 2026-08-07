# RegretBench LLM-Native Source And Protocol Audit Result

Date: 2026-08-07

**Status: passed. Only an exact support-recovery serving smoke is authorized.**

## Source Result

- Official repository: `https://github.com/ngocminhta/RegretBench`
- Commit: `b2978e1c2e31b7a7c4e1508ee3e1fa1cb98f4aa7`
- Dataset: `OpenDomainQA` version `1.0.0`
- Available test CIGs: `6,286`
- Eligible AmbigDocs CIGs: `2,419`
- Frozen split: `4` mechanics, `64` development, `64` confirmation
- Model calls: `0`
- Cost: `$0`

Every available test CIG matches its published SHA256. The deterministic split
is disjoint and prompt-unique, and no selected prompt contains a hidden answer
alias. The model-facing contract contains only `task_id`, `prompt`, and dialogue.

## Packaging Caveat

The release manifest and `SHA256SUMS` describe `21,252` train files that are
absent from the Git tree. This is recorded as a release-packaging discrepancy,
not silently repaired. The new protocol deliberately uses only repository-
tracked test CIGs and never calls that cohort pristine held-out data. Fresh
hash-based development and confirmation partitions were frozen before any model
call or endpoint inspection.

## Structural Control

The exact released facet partitions yield zero strict depth-two information
gain over greedy one-step information gain on every available CIG and every
selected CIG. Thus a later positive result cannot be attributed to a hidden
finite-support decision-tree opportunity. It would have to arise from the LLM's
history-conditioned support regeneration.

## Bindings

- Audit result SHA256:
  `d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de`
- Source protocol manifest SHA256:
  `8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97`
- Audit script SHA256:
  `6906a6db52f26fb781c59b76d29ff095adaef0d85154c6b28d8a396402028163`
- Preregistration SHA256:
  `7d68263cb75e120bccf69a892ed08ec459343b44cf51ded61b240398e0fd0b5c`

This pass does not authorize a policy comparison, confirmation access, or any
claim that clarification improves support. Those require separate prospective
gates.
