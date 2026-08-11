# RegretBench Factorized V2 Source Protocol

Frozen: 2026-08-11 Europe/London, before any factorized-v2 model response or
policy endpoint.

Status: **zero-call source and opportunity protocol**.

## Scientific Motivation

The Bongard OpenWorld Luna interface changed its beliefs after simulated
answers but failed class-balanced answer obedience. It asked one generative
response to regenerate semantic support, assign likelihoods, and absorb the
new observation. The next experiment must change that factorization rather
than tune the failed visual prompt or planner.

RegretBench supplies externally defined hidden intents and a deterministic
semantic response mapper. Factorized v2 assigns the LLM two narrower roles:

1. regenerate an explicit semantic-particle population after dialogue; and
2. in a separate history-free request, predict replies from each fixed child
   particle and fixed clarification question.

Exact code performs Bayesian updating and lookahead. The LLM remains
irreducible because the open-ended interpretations, answers, questions, and
answer-conditioned child particles are not supplied by the benchmark. The
factorization prevents simulated dialogue from entering the child likelihood
twice and makes observation use testable independently of support movement.

## Fresh Source Boundary

Use the official RegretBench OpenDomainQA test release at Git commit
`b2978e1c2e31b7a7c4e1508ee3e1fa1cb98f4aa7`, while continuing to state that
the checksum-listed train files are absent and this is not a pristine official
holdout claim.

Reapply the frozen eligibility rules from the original source audit:

- AmbigDocs only;
- three through six intents;
- two through four semantic facets;
- every intent has answer aliases and every facet value;
- every facet has at least two distinct values.

Exclude all 132 task IDs in the original mechanics/development/confirmation
manifest. Sort the remaining eligible tasks by SHA-256 of
`regretbench-factorized-v2|<cig_id>` and freeze the first 132 as:

- mechanics: 4 tasks;
- development: 64 tasks;
- confirmation: 64 tasks.

All selected prompts must be unique and must not contain an answer alias of
length at least four after normalization. All selected tasks must have exactly
zero fixed-support depth-two gain under the benchmark facets. This is a
structural isolation: any future horizon gain must come from generated support
dynamics, not an already enumerable fixed-support tree.

## Authorization

A passing source audit authorizes only a separate exact-10 mechanics smoke.
It opens no hidden truth, realized trajectory, development endpoint,
confirmation endpoint, model call, or paper claim. The smoke must have its own
strict schema, independent raw-response verifier, dated account-wide budget
wrapper, live catalog check, and write-once result before any request.

The earlier failed RegretBench transport chain and every Bongard artifact stay
immutable. Factorized v2 uses fresh tasks and a new interface; it cannot rescue,
reclassify, or pool either prior result.
