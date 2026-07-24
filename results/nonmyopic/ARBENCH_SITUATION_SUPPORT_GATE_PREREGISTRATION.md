# AR-Bench Situation-Puzzle Support-Recovery Gate

Date: 2026-07-24

Status: preregistered mechanism gate. This is not a policy or depth comparison.

## Motivation

AR-Bench Situation Puzzles provide the structure missing from the closed Paprika and
ClinDiag lines: each case starts with deliberately incomplete public evidence, has an
open natural-language explanation, and answers atomic questions with
`Yes`/`No`/`Unknown` from a hidden full story. The benchmark paper reports that the
cases were generated as trees and human-verified for logical consistency. This gate
tests whether one observed answer can cause an LLM-maintained explanation support to
recover a previously omitted true mechanism.

Sources:

- AR-Bench paper: https://arxiv.org/abs/2506.08295
- Official repository: https://github.com/tmlr-group/AR-Bench
- Repository commit: `9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1`
- Situation Puzzle test SHA-256:
  `e3c34d58b8bad06d0152fee771f2ee19a01c17dd5ba65c3a6dfb69c02b52f485`

CA-BED later reported positive depth-two probabilistic dialog planning on AR-Bench's
Detective Cases, but did not isolate depth in its main control. This gate therefore
qualifies the open-hypothesis mechanism independently before any non-myopic claim.

## Frozen Sample And Roles

NumPy seed `24301` selected 12 test-row positions without replacement:
`10, 12, 19, 20, 28, 34, 47, 60, 62, 67, 91, 96`.
Their released dataset indices are:
`11, 13, 20, 21, 29, 35, 48, 61, 63, 68, 92, 97`.

- Non-thinking `google/gemma-4-26b-a4b-it` generates eight initial explanations,
  four target-blind binary questions, and eight refreshed explanations per branch.
- Non-thinking `openai/gpt-5.4-mini` answers each question from the hidden story and
  measures semantic coverage after all supports are generated.
- The released `key_question` fields are never used.
- The hidden story is absent from every explanation and question prompt. It enters
  only the answer oracle and final measurement.
- OpenRouter project ledger budget: `$70.38480269545715`.
- Per-run cap: `$0.50`; projected reservation: `$0.25`.
- Concurrency: `64`.

## Serving Gate

Before the formal gate, run the first two frozen cases through one branch each:

- two initial-support requests;
- two question-generation requests;
- two oracle-answer requests;
- two refresh requests;
- two semantic-coverage requests.

The smoke passes only with exactly 10 physical requests, zero reasoning tokens, valid
schemas, and two complete branches. Failure closes before the formal gate.

## Formal Endpoint

The formal run has exactly 132 physical requests:

- 12 initial-support requests;
- 12 question-generation requests;
- 48 oracle-answer requests;
- 48 support-refresh requests;
- 12 semantic-coverage requests.

Coverage score at least `0.80` requires the same central mechanism and enough correct
causal links to explain every public fact. Incidental hidden-story details may be
omitted.

All conditions must pass:

1. all 12 cases and 48 branches complete;
2. exactly 132 physical requests and zero reasoning tokens;
3. at least 6/12 hidden explanations are absent initially;
4. at least 3 initially omitted explanations are recovered by some branch;
5. mean oracle best-match gain is at least `0.10`;
6. at least 4/12 cases have branch score spread at least `0.15`.

Passage authorizes a separate semantic-likelihood and depth-1/depth-2 ranking-fidelity
gate. Failure closes this AR-Bench support-recovery apparatus before policy evaluation.
