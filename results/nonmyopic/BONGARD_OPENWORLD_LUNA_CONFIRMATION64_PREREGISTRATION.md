# Bongard-OpenWorld Luna Confirmation64 Preregistration

Date frozen: 2026-08-06

## Purpose

This protocol freezes an independent 64-task confirmation before any Bongard
Luna mechanics or development response exists. It confirms the headline object:
non-myopic depth-two selection over answer-conditioned VLM-generated semantic
hypotheses, including the matched history-blind regeneration control.

The confirmation can execute only if the independent 32-task development result
reaches the already frozen
`full_path_dependent_llm_native_development_signal` claim tier: the conjunction
of the policy, matched-mechanism, and path-dependent-support families. A
development null or partial tier forbids confirmation execution.

## Data Boundary

- Official Bongard-OpenWorld validation confirmation partition: 64 tasks.
- Source partition UID hash:
  `27da2cc656add724bffbc43ea04ab28fa22bf8564ab9cba4d60e4e23df6facd0`.
- Task order: deterministic sort of the existing opaque task IDs.
- Four fixed 16-task blocks use offsets 0, 16, 32, and 48.
- Public protocol rows contain only opaque task ID, source-row hash, and block.
- Confirmation images, labels, responses, and endpoints remain unopened.
- The official test and 100-task reserve remain unopened.

## Execution

| Block | Earliest London date | Model seed | Max requests | Max precharged exposure |
|---|---|---:|---:|---:|
| A | 2026-08-15 | 2026081501 | 688 | $2.752 |
| B | 2026-08-16 | 2026081601 | 688 | $2.752 |
| C | 2026-08-17 | 2026081701 | 688 | $2.752 |
| D | 2026-08-18 | 2026081801 | 688 | $2.752 |

The model remains `openai/gpt-5.6-luna`, nonreasoning, with the unchanged
interface-v6 task-atomic CRN transport and matched history-blind control. Every
block uses the account-wide `$5` London-day ledger and a `$4.75` run cap.

Once confirmation starts, all mechanically valid blocks are mandatory.
Intermediate scientific endpoints remain sealed. Endpoint labels load only
after all four blocks independently replay.

## Confirmatory Gates

Shared gates require exact 64 disjoint tasks, independent replay of all four
endpoint-blind blocks, finite metrics, and root candidate Brier below the
constant-half predictor.

The policy family requires:

- at least 24 dynamic/myopic final-history changes, all clearing the numerical
  tie margin, with at least one change in every block;
- positive dynamic score-to-endpoint ranking fidelity, not worse than myopic;
- at least 3% dynamic Brier improvement over myopic;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
- dynamic log loss no worse than myopic; and
- dynamic Brier no worse than the shuffled-continuation control.

The matched-mechanism family independently requires:

- at least 24 dynamic/history-blind final-history changes and at least one in
  every block;
- at least 3% dynamic Brier improvement over history-blind regeneration;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
- dynamic log loss no worse than history blind; and
- dynamic ranking fidelity no worse than history blind.

The path-dependent-support family independently requires:

- at least 24 dynamic/fixed final-history and margin-clearing first-action
  changes, with at least one change in every block;
- at least 3% dynamic Brier improvement over fixed-support depth two;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
- dynamic log loss no worse than fixed; and
- dynamic ranking fidelity no worse than fixed.

Only the conjunction supports the full confirmation claim. No result authorizes
opening the official test, the reserve, an unregistered model swap, or a causal
claim stronger than the matched prompt-conditioning intervention.

## Budget and Status

At the frozen `$0.004` per-attempt precharge ceiling, a full 16-task block has
at most `688` requests and `$2.752` exposure, fitting the `$5` day without using
an optimistic average-cost assumption. This freeze makes zero model calls and
costs `$0`.

The original interface-v1 manifest is preserved as superseded because it used a
descriptive tier name that did not equal the claim classifier's literal output.
The authorization correction is frozen in
`BONGARD_OPENWORLD_LUNA_CONFIRMATION64_AUTHORIZATION_AMENDMENT.md`.

Authoritative frozen protocol manifest:

`results/nonmyopic/bongard_openworld_luna_confirmation64/PROTOCOL_MANIFEST_V4.json`

SHA-256:
`622ad102a2ed22a7e67722532902a4720012abf4852a60af1282f433d0f2317f`.

This manifest supersedes the earlier pre-response freeze after the
positive-present/negative-absent contrastive prompt clarification. The task
UIDs, seeds, policies, endpoints, sample sizes, and statistical gates are
unchanged; only the prompt and its transitive implementation bindings changed.
V3 further strengthens the claim boundary before responses: dynamic support
must beat fixed-support depth two on changed paths, relative Brier, paired
uncertainty, log loss, and ranking fidelity. Tasks, calls, seeds, endpoints,
budgets, and execution dates remain unchanged.
V4 adds the pre-response simulated-branch obedience mechanics gate. Positive
and negative conditioned branches must separately beat constant-half Brier on
the label supplied to the branch. This changes no confirmatory endpoint or
request.
