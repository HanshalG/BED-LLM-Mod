# ICAE-Bench LLM-Native Opportunity Audit

Date: 2026-07-29

Status: **all zero-call source/opportunity gates pass; a separately frozen
10-call semantic serving smoke is authorized**.

## Source

The audit binds the official ICAE-Bench release:

- repository commit:
  `66bbabb20a2138d066ac7d6f7ba6768b57c2f79b`;
- official PRD/Oracle bundle SHA-256:
  `b054e8f03b3c434ffaec3c4ee6cf712d3f25e5bc7cd7ef9eb0a913026fd827d7`;
- authoritative test archive SHA-256:
  `f746c560e724d0dad12d512ead771d91de91d4b3e6051335728c305e2adff204`;
- frozen value-blind manifest SHA-256:
  `47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f`.
- public opportunity audit SHA-256:
  `f63a313adb8bc3bd1090fe4542d1b38b4c2e63abca30c559447a70a6e6348552`.

Only the 12 mechanics aliases, one per programming language, were opened.
Development (36), confirmation (48), and retained (384) remain sealed.

## Result

The released Oracle has the required scientific shape:

- it sees only the task's injected hidden requirement record;
- free-form questions are matched semantically by an LLM;
- complete conversation history conditions every later reply;
- unknown questions return a fixed fallback and reveal no automatic progress;
- one reply addresses at most three matched technical points; and
- the query budget is enforced by the official server.

The mechanics slice contains:

| Gate | Result | Required |
|---|---:|---:|
| tasks with at least 8 substantive hidden requirements | 11/12 | >=10/12 |
| tasks with at least 2 answer-introduced followup targets | 11/12 | >=10/12 |
| tasks with public, hidden, and enhanced executable tests | 12/12 | 12/12 |
| tasks with a literal Oracle-answer leak in the fuzzy PRD | 0/12 | 0/12 |

The answer-introduced metric is target-blind and deterministic: an edge exists
when a source answer and another hidden requirement's trigger phrases share at
least two task-specific content tokens absent from the fuzzy PRD, with
corpus-IDF score at least six. This establishes delayed semantic query
opportunity in 11 languages. The frozen Rust mechanics task is the adverse
case: six substantive requirements and no qualifying lexical unlock.

## Interpretation

ICAE-Bench is the strongest distinct LLM-native environment found in the
current source search. Unlike finite numeric POMDPs, the LLM owns the
free-form requirement hypotheses, semantic question matching, and
history-conditioned followup generation. Unlike checklist simulators that
advance after any turn, ICAE reveals nothing when the policy asks an
unmatched question. Final utility is independently grounded in released
black-box tests.

The hidden requirement table itself is static. Therefore non-myopic value must
come from the intended first link: an early answer changes the LLM's generated
belief/query support, making a useful later experiment available. This audit
does not show that a depth-two planner ranks those roots correctly.

## Consequence

Authorize only a separately preregistered exact 10-call smoke on frozen
mechanics tasks. It must test:

1. strict generation of task-specific requirement hypotheses and candidate
   clarification questions from the fuzzy PRD;
2. deterministic Oracle replay across fresh sessions;
3. no progress for a generic unmatched question;
4. branch-dependent generation of followup questions after real Oracle
   answers; and
5. independent semantic coverage scoring without opening executable endpoint
   results.

No policy or coding-agent efficacy run is authorized yet. A later efficacy
claim must compare equal-query policies and use the official executable test
endpoint; hidden-constraint recall alone is insufficient.

## Accounting

- model requests: `0`
- OpenRouter cost: `$0`
- OatML, Slurm, SSH, or cluster use: `0`
