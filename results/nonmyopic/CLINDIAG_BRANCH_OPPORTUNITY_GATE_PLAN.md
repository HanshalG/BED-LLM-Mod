# ClinDiag Branch-Opportunity Gate

Date frozen: 2026-07-24

Status: **preregistered before any response on these cases.**

## Purpose

The preceding ClinDiag gate showed that native full workups recover 12/14 diagnoses
omitted from the initial LLM-generated support. That is necessary but not sufficient
for non-myopic BED. A planner also needs:

1. evidence choices whose realized support value differs;
2. path dependence, so the order of the same evidence pair can change the regenerated
   support;
3. a positive two-step opportunity beyond committing to the action with the best
   realized one-step support value.

This gate measures those properties exhaustively before building a target-blind
planner. It is an oracle structural diagnostic, not a deployable policy.

## Frozen Data

- Same pinned ClinDiag source commit and archive hash as the generator gate.
- Seed `24290` selects 12 entirely fresh opportunity cases: six challenging and six
  rare.
- Every selected case has nonempty medical history, physical examination, laboratory,
  imaging, and other-test channels.
- Two additional fresh cases are reserved for the serving smoke.
- These cases are disjoint from the prior four smoke, 20 generator-development, and
  60 sealed generator-holdout cases.
- Exact IDs are frozen in `scripts/clindiag_branch_opportunity_gate.py`.

## Actions And Paths

The initial state contains only `initial_information` and a 12-diagnosis support.
Five actions reveal one native evidence block:

- `history`;
- `physical_exam`;
- `laboratory_tests`;
- `imaging`;
- `other_tests`.

Non-reasoning GPT-5.4 regenerates 12 diagnoses after each action. The gate exhaustively
evaluates all 20 ordered pairs of distinct actions. A second-step prompt receives the
ordered cumulative evidence and the first-step regenerated support, making belief
dynamics explicitly path-dependent.

Temperature is fixed at `0.0`, with the same provider seed. One exact second-step prompt
is repeated per case. This identity control distinguishes serving noise from reverse-
order effects. Non-reasoning GPT-5.4 Mini sees the hidden diagnosis only after every
support is generated and assigns strict semantic-equivalence scores.

## Structural Metrics

For each case:

- one-step spread: max minus min semantic truth-match score over five actions;
- reverse-order gap: maximum score difference between `A>B` and `B>A`;
- greedy action: action with the best realized one-step score;
- greedy continuation: best two-step score among paths starting with that action;
- oracle two-step score: best score among all 20 paths;
- non-myopic gap: oracle two-step minus greedy continuation;
- two-step gain: oracle two-step minus best one-step score;
- identity score gap and exact-name support Jaccard.

Truth is used only for post-generation measurement. The oracle metrics establish
opportunity; they are not available to a future policy.

## Serving Smoke

Run exactly 10 requests on two smoke cases:

- two initial supports;
- two one-step supports;
- two second-step supports;
- two exact identity duplicates;
- two semantic measurements.

Pass requires all eight generated supports to contain 12 unique diagnoses, exactly 10
requests, zero retries/reasoning/runtime failures, and valid semantic rows.

## Frozen Opportunity Gate

All criteria must pass on the 12 opportunity cases:

| Criterion | Requirement |
|---|---:|
| Complete cases | 12 |
| Initial coverage | at most 5/12 |
| One-step spread at least 0.20 | at least 8/12 |
| Reverse-order gap at least 0.15 | at least 5/12 |
| Non-myopic gap at least 0.10 | at least 4/12 |
| Mean non-myopic gap | at least +0.05 |
| Mean oracle two-step gain | at least +0.10 |
| Mean identity score gap | at most 0.05 |
| Maximum identity score gap | at most 0.15 |
| Mean identity support Jaccard | at least 0.75 |

Failure closes this exact five-action branch-opportunity line before target-blind
ranking, policy evaluation, or holdout use. Pass authorizes a separately frozen
target-blind planner/ranking-fidelity development set; it does not authorize the
60-case holdout directly.

## Cost

- Project-ledger spend before this gate: `$45.02778097`.
- Conservative remaining budget: `$25.35702172`.
- Per-run cap: `$3.00`; projected reservation: `$2.00`.
- Expected formal request count: 336 physical calls
  (12 initial, 60 one-step, 240 second-step, 12 identity, 12 semantic).
