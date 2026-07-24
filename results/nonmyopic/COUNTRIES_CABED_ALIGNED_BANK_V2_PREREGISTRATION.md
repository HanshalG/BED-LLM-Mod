# Countries CA-BED Aligned Global-Bank V2 Preregistration

Date: 2026-07-24

## Decision and Claim

This is the final Countries serving architecture. It is a distinct candidate
process, not a repair or retry of branch-menu V1. V1 produced no depth endpoint.

Each tree uses full GPT-5.4 nonreasoning once to generate one target-blind bank
of 40 natural-language country questions. GPT-5.4 Mini nonreasoning then labels
all 64 countries for every question. The first 32 rows with at least four Yes
and four No labels form the shared planning bank. Failure to obtain all 32
closes the stage without dropping a tree, regenerating, or changing a label.

The same hard binary semantic table defines both the likelihood model and the
realized observation. The hidden target is never supplied to either model.
Thus the LLM is load-bearing for arbitrary semantic actions and 2,560
question-country relations per tree, while exact code performs only Bayesian
planning over the resulting environment.

A pass supports non-myopic planning inside an LLM-defined aligned semantic
environment. It does not establish external factual accuracy or path-dependent
hypothesis regeneration.

## Frozen Planner and Controls

Every one of the 32 bank questions can be the root. On each Yes/No branch, all
31 remaining questions are available and the exact planner chooses the
maximum-EIG follow-up.

- `depth_two` chooses the root minimizing expected entropy after the adaptive
  two-question tree.
- `depth_one` chooses the maximum immediate-EIG root, then receives the same
  branch-optimal follow-up computation.
- `random_root` uses a seeded random root, then receives the same branch-optimal
  follow-up computation.

All 64 countries are evaluated exactly under every selected policy. Final
entropy equals truth negative log posterior under the uniform support and hard
aligned labels. Methods share each generated bank; no policy receives private
questions, labels, targets, or compute.

Frozen seed is `24324`. Question temperature is `.7`; semantic-table
temperature is `0`. Reasoning must remain zero. Content errors are never
retried. Transport backoff may retry only a failed physical request. Private
raw responses are checkpointed after question generation and after semantic
labeling; public artifacts record their hash.

## Serving Smoke

Two fresh style prompts generate two trees. The exact request count is 82:
two full GPT-5.4 bank requests plus 80 GPT-5.4 Mini table requests. Projected
cost is `$0.35`, with a hard run cap of `$1.00`.

All smoke gates are conjunctive:

1. both 32-row banks and all 64-way binary tables complete;
2. exactly 82 physical requests and zero reasoning tokens;
3. depth two is never worse than matched-compute depth one;
4. depth two selects a distinct root in at least one tree;
5. at least one tree has a strictly positive mean depth-two gain; and
6. mean final-entropy gain over depth one is at least `.01` nat.

Failure closes the stage. Smoke responses, trees, and styles cannot enter the
formal endpoint.

## Formal Paired Confirmation

Only smoke passage authorizes twelve untouched style prompts and twelve new
trees. The exact request count is 492. Projected cost is `$2.00`, with a hard
run cap of `$4.00`.

All formal gates are conjunctive:

1. all twelve 32-row banks and tables complete, with exactly 492 requests and
   zero reasoning;
2. depth two is never worse than matched-compute depth one;
3. depth two selects a distinct root on at least 8/12 trees;
4. at least 8/12 tree-level mean gains over depth one are positive;
5. mean gain over depth one is at least `.02` nat and its paired tree-level 90%
   bootstrap lower bound is positive; and
6. mean gain over random-root-with-optimal-follow-up is at least `.02` nat and
   its paired tree-level 90% bootstrap lower bound is positive.

No threshold, target, style, tree, question, table, or model may be changed
after responses are observed. A formal failure is reported as a null and
closes Countries without a V3.
