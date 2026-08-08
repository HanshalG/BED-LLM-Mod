# Bongard-OpenWorld Luna Confirmation96 Preregistration

Date frozen: 2026-08-06

## Purpose

This protocol freezes an independent 96-task confirmation before any Bongard
Luna mechanics or development response exists. It confirms the headline object:
non-myopic depth-two selection over answer-conditioned VLM-generated semantic
hypotheses, including the matched history-blind regeneration control.

The confirmation can execute only if the independent 64-task development result
reaches the already frozen
`full_path_dependent_llm_native_development_signal` claim tier: the conjunction
of the policy, matched-mechanism, and path-dependent-support families. A
development null or partial tier forbids confirmation execution.

## Data Boundary

- Official Bongard-OpenWorld validation confirmation partition: 96 tasks.
- Repaired partition UID hash:
  `3826a64b46668226c996afa92e81cf270bf59f99a373813e37196552300ecb26`.
- The byte-only repair is frozen in
  `BONGARD_OPENWORLD_PARTITION_INTEGRITY_AMENDMENT.md` and binds manifest
  `9d9dc695...2bbcbc9`.
- The confirmation-only power expansion is frozen in
  `BONGARD_OPENWORLD_CONFIRMATION96_POWER_AMENDMENT.md`; it retains the exact
  original 64 confirmation tasks and appends 32 byte-clean reserve tasks.
- The subsequent zero-endpoint development power amendment is frozen in
  `BONGARD_OPENWORLD_DEVELOPMENT64_POWER_AMENDMENT.md`; it retains all original
  32 development tasks and appends 32 byte-clean reserve tasks while preserving
  this exact confirmation partition.
- Task order: deterministic sort of the expanded opaque task IDs.
- Four fixed 24-task blocks use offsets 0, 24, 48, and 72.
- Public protocol rows contain only opaque task ID, source-row hash, and block.
- Confirmation images, labels, responses, and endpoints remain unopened.
- The official test and remaining 36-task reserve remain unopened.

## Execution

| Block | Earliest London date | Model seed | Max accepted | Max HTTP attempts | Max precharged exposure |
|---|---|---:|---:|---:|---:|
| A | 2026-08-15 | 2026081501 | 1,032 | 1,053 | $4.212 |
| B | 2026-08-16 | 2026081601 | 1,032 | 1,053 | $4.212 |
| C | 2026-08-17 | 2026081701 | 1,032 | 1,053 | $4.212 |
| D | 2026-08-18 | 2026081801 | 1,032 | 1,053 | $4.212 |

The model remains `openai/gpt-5.6-luna`, nonreasoning, with the unchanged
interface-v6 task-atomic CRN transport and matched history-blind control. Every
block uses the account-wide `$5` London-day ledger and a `$4.75` run cap.

Once confirmation starts, all mechanically valid blocks are mandatory.
Intermediate scientific endpoints remain sealed. Endpoint labels load only
after all four blocks independently replay.

## Confirmatory Gates

Shared gates require exact 96 disjoint tasks, independent replay of all four
endpoint-blind blocks, finite metrics, and root candidate Brier below the
constant-half predictor.

The policy family requires:

- at least 36 dynamic/myopic final-history changes, all clearing the numerical
  tie margin, with at least one change in every block;
- positive dynamic score-to-endpoint ranking fidelity, not worse than myopic;
- at least 3% dynamic Brier improvement over myopic;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
- dynamic log loss no worse than myopic; and
- dynamic Brier no worse than the shuffled-continuation control.

The matched-mechanism family independently requires:

- at least 36 dynamic/history-blind final-history changes and at least one in
  every block;
- at least 3% dynamic Brier improvement over history-blind regeneration;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
- dynamic log loss no worse than history blind; and
- dynamic ranking fidelity no worse than history blind.

The path-dependent-support family independently requires:

- at least 36 dynamic/fixed final-history and margin-clearing first-action
  changes, with at least one change in every block;
- at least 3% dynamic Brier improvement over fixed-support depth two;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
- dynamic log loss no worse than fixed; and
- dynamic ranking fidelity no worse than fixed.

It additionally requires the matched first-action comparison against
`fixed_score_dynamic_update`:

- at least 36 different final histories and margin-clearing first actions,
  with at least one change in every block;
- at least 3% dynamic Brier improvement;
- a paired complete-task 20,000-bootstrap 95% interval strictly below zero;
  and
- dynamic log loss no worse than the matched fixed-score control.

Only the conjunction supports the full confirmation claim. No result authorizes
opening the official test, the reserve, an unregistered model swap, or a causal
claim stronger than the matched prompt-conditioning intervention.

## Budget and Status

Under the prospective transport-retry amendment, a full 24-task block has at
most `1,032` accepted responses and `1,053` HTTP attempts. At the frozen `$0.004`
per-attempt precharge ceiling, its maximum exposure is `$4.212`, fitting the
`$5` day without using an optimistic average-cost assumption. This freeze makes
zero model calls and costs `$0`.

The original interface-v1 manifest is preserved as superseded because it used a
descriptive tier name that did not equal the claim classifier's literal output.
The authorization correction is frozen in
`BONGARD_OPENWORLD_LUNA_CONFIRMATION64_AUTHORIZATION_AMENDMENT.md`.

Authoritative frozen protocol manifest:

`results/nonmyopic/bongard_openworld_luna_confirmation64/PROTOCOL_MANIFEST_V10.json`

SHA-256:
`dce0a42e77447ebe289a9a13495058869ccee978380dd407a3460b72e2316c4c`.

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
V5 adds the pre-response matched fixed-score/dynamic-update control. It makes
the strongest tier isolate path-dependent first-query selection while adding
no request because every matched terminal history was already frozen in the
all-first-action cache.
V6 applies the pre-response image-byte integrity repair. It preserves all
models, prompts, policies, seeds, counts, dates, endpoints, thresholds, and
claim gates while replacing seven confirmation rows under the frozen seeded
ordering. The repaired development and confirmation partitions have no exact
image-byte reuse and pass the supplementary strict perceptual screen.
V7 adds the pre-response terminal label-obedience validity gate. Every
terminal belief must retain both newly queried label classes with
class-conditional Brier below constant half. It adds no request and changes no
scientific endpoint or threshold.
V8 prospectively permits the frozen bounded same-payload transport retries and
separates accepted-response from HTTP-attempt ceilings without changing the
scientific estimand.
V9 applies the zero-endpoint confirmation-only power amendment. It preserves
development exactly, retains all original confirmation tasks, expands to 96
tasks in four 24-task blocks, and scales changed-path counts from 24/64 to
36/96 while leaving every efficacy threshold and control unchanged.
V10 applies the subsequent zero-endpoint development power amendment. It
expands the unopened development partition from 32 to 64 tasks, preserves all
96 confirmation tasks, and leaves confirmation dates, seeds, requests,
thresholds, controls, and endpoint handling unchanged.
