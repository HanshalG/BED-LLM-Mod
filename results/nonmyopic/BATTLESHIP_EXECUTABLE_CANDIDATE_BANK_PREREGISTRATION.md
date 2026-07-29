# Battleship Executable Candidate-Bank Gate Preregistration

Date: 2026-07-29

Status: **frozen before any executable-candidate model response**.

## Motivation And Separation

The released Collaborative Battleship bank has a strong target-blind
non-myopic task-utility opportunity, but the frozen fresh cross-translation
smoke failed. Its public failure SHA-256 is
`27ae82a7dc531a8457a777c68c211be8e92d576b7ff89798ca58c592a2b9a625`.
The saved responses showed both poor question balance and contradictory
translations.

The official implementation is not an appropriate repair template: its
Spotter compiler receives the true hidden board and may retry up to ten times.
That can make a nominally generic measurement function depend on the target
before counterfactual scoring.

This is a distinct operational interface. One target-blind LLM directly
proposes a natural-language question paired with the pure Boolean expression
that defines the experiment. There is no independent semantic translator and
no hidden board in the prompt. The executable predicate, rather than an
unverifiable paraphrase, is the measurement model used by BED.

## Exact Ten Calls

- model: `openai/gpt-5.4`;
- ten concurrent non-thinking calls;
- seed `40000`, temperature `.7`;
- six schema-enforced paired question/expression candidates per call;
- each call has an independent batch index and six fixed diversity slots;
- exactly 60 raw candidates;
- no retry, repair, continuation, reissue, normalization, model substitution,
  or reasoning fallback;
- projected cost `$0.25`; hard cap `$0.50`.

The frozen AST language is inherited unchanged from the failed serving
implementation. It permits pure indexing, comparisons, arithmetic, selected
NumPy reductions, safe built-ins, and at most two bounded comprehensions. It
rejects statements, imports, assignments, lambdas, private names, arbitrary
calls, `np.array`, `reshape`, and `argwhere`.

Invalid, unsafe, constant, extreme, or duplicate candidates are filtered
without a model call. This is part of the proposed production method, not a
response repair.

## Fresh Board And Novelty Test

Use two new official-prior blocks:

- seeds `40010` and `40011`;
- 4,096 boards per block;
- official `FastSampler`;
- fully hidden partial board; and
- BSC answer noise `.1` for planning.

A candidate is valid and nontrivial only if it safely returns Boolean values
on every board and its Yes prevalence lies in `[.05,.95]` on both blocks.
Deduplicate by joint behavior across both blocks.

Compile the 39 released stage-zero programs on the same blocks. A generated
behavior is novel when its exact joint outcome vector differs from every
released behavior. Released question text and the failed smoke responses are
not included in model prompts.

## Fresh Opportunity Test

Take the first 25 unique valid behaviors in deterministic call/candidate order.
For each block, use exact finite-horizon task-utility planning from the source
audit:

- terminal utility: maximum posterior probability that the next shot hits;
- execution budget: three questions;
- receding planning horizons: one, two, and three;
- answer channel: BSC with error `.1`; and
- greedy-EIG control: immediate full-board configuration EIG.

## Frozen Gates

All gates are conjunctive:

- exact 10 accepted requests and 10 HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all ten strict schemas parse;
- every call contributes at least three safe nontrivial candidates;
- at least 20 unique valid joint behaviors;
- at least eight unique behaviors are novel relative to the released bank;
- depth-one and depth-two best-root behavior sets are disjoint on each block;
- depth-two three-question receding endpoint utility beats depth one by at
  least `.01` on each block;
- depth three does not regress from depth two on either block;
- the better of depth two and depth three beats greedy EIG by at least `.02`
  on each block; and
- total cost is at most `$0.50`.

Public evidence includes question text, expression hashes, behavior hashes,
prevalence, novelty flags, planning results, and usage. Expressions and raw
responses remain private.

Failure closes this exact direct-executable bank interface. There is no prompt,
model, seed, slot, validator, filtering, ordering, threshold, or planning
repair. Passage authorizes only a separately preregistered branch-conditioned
candidate-regeneration mechanics test; it does not authorize policy efficacy.

## Dry Verification

Before any real response:

- focused parser/filter/planner tests must pass;
- a deterministic exact-ten-call fixture on real official-prior samples must
  execute end to end; and
- implementation, tests, and this preregistration must be committed and
  pushed.
