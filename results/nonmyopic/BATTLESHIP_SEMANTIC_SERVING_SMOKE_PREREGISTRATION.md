# Collaborative Battleship Semantic Serving Smoke Preregistration

Date: 2026-07-29

Status: **frozen before any fresh Battleship model response**.

## Authorization

The zero-call opportunity audit passed every frozen gate:

- public result SHA-256:
  `16aff4012fd5992341f161c01a4dfab48d562e04fe2591eb6fd87d85feba0872`;
- official Collaborative Battleship commit:
  `b98a4ba1c55be1bd5aa8038d42ad0070c41cabcb`; and
- released trajectory SHA-256:
  `c39aa87d888fb98d5c6750ce4a3d70da6757adab9a5d4a69a44801f298290e94`.

The audit found stable, disjoint depth-one, depth-two, and depth-three roots
and strictly increasing three-question endpoint hit utility on two independent
4,096-board prior blocks. This is a source-opportunity result, not fresh
policy efficacy.

## Exact Ten Calls

Phase one makes two concurrent calls:

- model: `openai/gpt-5.4-mini`;
- non-thinking, seed `39700`, temperature `.7`;
- each proposes exactly four distinct objective semantic yes/no questions for
  an empty 8x8 board with ships of lengths 2, 3, 4, and 5; and
- select the first two questions from each response without model-dependent
  ranking, for four selected questions total.

Phase two independently translates every selected question with both:

- `openai/gpt-5.4-mini`, non-thinking, seed `39800`, temperature `0`; and
- `google/gemini-2.5-flash`, non-thinking, seed `39900`, temperature `0`.

Each of the eight translation calls returns one schema-enforced pure Python
boolean expression. A frozen AST validator permits only board indexing,
boolean/comparison/arithmetic operations, bounded comprehensions, selected
safe built-ins, and selected NumPy reductions. It rejects statements, imports,
assignments, lambdas, private names/attributes, and arbitrary calls.

Expected accepted requests and HTTP attempts: exactly `10`. There is no retry,
repair, continuation, reissue, response normalization, model substitution, or
reasoning fallback. Projected cost is `$0.10`; the hard run cap is `$0.25`.

## Held-Out Board Test

Execute each safe translation on the same two fresh official-prior blocks:

- seeds `39710` and `39711`;
- 4,096 boards per block;
- fully hidden partial board; and
- official `FastSampler` board encoding.

These samples are separate from opportunity-audit seeds `39600` and `39601`.
No released trajectory question is supplied to either model. No hidden policy
endpoint is opened.

## Frozen Gates

All gates are conjunctive:

- exactly 10 accepted requests and 10 HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all ten strict schemas parse;
- the four deterministically selected questions are globally unique and
  direct grammatical yes/no questions;
- all eight expressions pass the AST validator and return Boolean values on
  every held-out board;
- every expression has `Yes` prevalence in `[.05,.95]` on both blocks;
- GPT-5.4-Mini and Gemini translations agree on at least 97% of boards in
  each block for at least three of four questions;
- the GPT-5.4-Mini reference translations induce at least three distinct
  joint behaviors across both blocks; and
- total cost is at most `$0.25`.

Public evidence contains question text, behavior hashes, prevalence,
agreement, and expression hashes. Generated expressions and raw model
responses remain private.

Failure closes this exact semantic interface. There is no prompt, model,
question selection, seed, validator, prevalence, agreement, or threshold
repair. Passage authorizes only a separately frozen branch-conditioned
ranking/mechanics experiment. It does not authorize a fresh policy or depth
claim.

## Dry Verification

Before any real response:

- the safe-expression and serving tests must pass;
- an exact ten-call deterministic fixture must pass; and
- implementation, tests, and this preregistration must be committed and
  pushed.
