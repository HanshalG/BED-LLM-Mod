# AskBench LLM-Native Source Audit Preregistration

Status: frozen before aggregate data inspection or row selection.

Date: 2026-07-29

Cost: zero model calls and zero OpenRouter spend.

## Question

Can the released AskBench AskMind environment support a new non-myopic BED
test in which the LLM generates and updates plausible missing-information
worlds, rather than merely searching an enumerable arithmetic rule set?

This stage is source and control-flow inspection only. It cannot establish an
opportunity or policy result.

## Frozen Source

- official repository: `https://github.com/jialeuuz/askbench`;
- commit:
  `f35da92feda34504f10413313554438e7abaeb08`;
- tree:
  `bfa4bf929f5ebd61a465be4b12f42a212d38685a`;
- AskMind combined evaluation JSONL SHA-256:
  `406b9a48036374552d9e819c63d5c3ba25feea764c68f391e21c552f6221af82`;
- official evaluator SHA-256:
  `14d78c936b60bfe365db9d03162e35627e9b00e2bdc17f5946df812d29dff392`;
- paper PDF SHA-256:
  `835509e5d90fc550756c81b0bdbfbaf39ffe44c9a3380c56e690d7da7ffe5533`.

The audited data file is
`ask_eval/data/ask_bench/ask_mind/test.jsonl`.

## Frozen Eligibility Rule

A row is eligible only when all conditions hold:

1. `source_task == "ask_mind_medqade"`;
2. `id`, `ori_question`, `degraded_question`, `degraded_info`,
   `expected_answer`, and `required_points` have the released types;
3. `ori_question != degraded_question`;
4. `degraded_info` does not state that no modification was made;
5. there are between three and eight nonempty required points;
6. the degraded question contains exactly one complete set of answer labels
   `A.`, `B.`, `C.`, and `D.`;
7. `expected_answer` has the exact form `The answer is X.` for one of
   `X in {A,B,C,D}`; and
8. all row IDs are unique.

No semantic quality judgment, answer content, disease category, or anticipated
model behavior enters selection.

## Frozen Split

Sort eligible rows by ID, shuffle with Python `random.Random(37300)`, then
assign:

- first 10: serving/opportunity development;
- next 40: untouched confirmatory holdout;
- all remaining rows: unused.

The public manifest records IDs and cryptographic row hashes but not hidden
original questions, expected answers, or simulator context.

## Source Gates

The route is source-valid only if:

1. repository and file hashes match;
2. at least 60 rows are eligible;
3. all 10 development and 40 holdout IDs are distinct;
4. every selected row has three to eight required points and four answer
   options;
5. the candidate-visible degraded question excludes the hidden original
   question, expected answer, degradation explanation, and checklist;
6. the official simulator receives the complete original question and
   checklist but instructs the user to answer only the immediate clarification;
7. the official standard loop permits at least two clarification turns before
   forcing a final answer; and
8. final correctness is judged against the released expected answer rather
   than the policy's own belief score.

## Decision

A complete source pass authorizes a separately frozen exact 10-call initial
support serving/opportunity screen on the development IDs. That screen must
show noncollapsed answer uncertainty before any branch tree or holdout row is
opened.

Any source-gate failure closes AskBench for this project. There is no row
substitution, eligibility repair, split reseeding, or paid call before the
source result is committed.
