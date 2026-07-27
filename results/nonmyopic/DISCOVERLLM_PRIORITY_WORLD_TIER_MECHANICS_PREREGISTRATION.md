# DiscoverLLM Priority-World Tier Mechanics Preregistration

Frozen after the realistic tier-serving pass and before loading the three
reserved mechanics artifacts or obtaining any mechanics response.

## Claim Boundary

This is a three-task mechanics and opportunity smoke for an LLM-native
non-myopic BED construction. It is not a DiscoverLLM benchmark result, a powered
policy comparison, or permission to open the 60 opportunity tasks.

The hidden parameter is one of four structurally selected, LLM-generated
priority-criterion subtrees. Two released candidate completions are shared root
actions. GPT-5.4 generates world-conditioned user transitions, a
branch-conditioned continuation policy, and semantic likelihood tiers. Code
only executes Bayes updates, EIG calculations, controls, and exact hidden-world
recovery.

## Frozen Tasks And Model

- Tasks:
  `technical_writing:artifact_333`,
  `creative_writing:artifact_367`,
  `technical_writing:artifact_249`.
- Excluded serving tasks:
  `svg_drawing:artifact_347`,
  `creative_writing:artifact_385`.
- Model: `openai/gpt-5.4`, temperature zero, no reasoning.
- Five stages per task, batched by stage: exactly 15 logical requests.
- At most four logged transport retries total; no semantic retry, repair,
  coercion, partial-task analysis, or replacement.
- Projected cost `$0.25`; hard cap `$0.50`.

## Belief And Policy Scores

Use a uniform four-world prior. Convert the exact `H/M/L` assignments to
likelihood weights `4/2/1`. For each true world's generated observation, update
the posterior by multiplying and normalizing those weights.

- Root EIG is prior entropy minus mean root-posterior entropy.
- Depth-two EIG is prior entropy minus mean terminal-posterior entropy after
  multiplying root and follow-up weights.
- Myopic selects maximum root EIG.
- Non-myopic selects maximum depth-two EIG.
- Random is the equal mixture of the two root actions.
- Ties use action order.

Also recompute every selection with sensitivity weights `3/2/1` and `9/3/1`.
The likelihood scorer and continuation policy never receive the shuffled
observation-to-truth map. After all scores and choices exist, use that map only
to compute mean true-world log posterior and exact MAP accuracy.

## Conjunctive Gates

1. Exact 15 logical requests; HTTP attempts equal requests plus at most four
   transport retries; zero reasoning tokens and forced exits; cost at most
   `$0.50`.
2. The true world is in the top assigned tier in at least 60% of root cells and
   at least 60% of follow-up cells; its mean weight advantage over distractors
   is positive at both stages.
3. At least two tasks have both root-EIG and depth-two action ranges of at least
   `.01` nats.
4. Non-myopic changes the root action on at least one task.
5. At least one changed task sacrifices at least `.005` nats root EIG and has
   strictly higher terminal true-world log posterior.
6. Across tasks, non-myopic mean terminal true-world log posterior is strictly
   above both myopic and random, and mean MAP accuracy is no lower than myopic.
7. Across all six task/action cells, depth-two EIG versus terminal true-world
   log posterior Spearman correlation is at least `.20` and exceeds root-EIG
   correlation by at least `.10`.
8. At least one strict delayed reversal keeps the same myopic and non-myopic
   actions and remains a strict reversal under all three weight maps.

Failure closes this exact task/interface/weight construction without a prompt,
threshold, task, model, or weight repair. Passing authorizes only a separately
frozen opportunity and power plan before opening any of the 60 opportunity
artifacts.

OpenRouter only. OatML jobs: `0`.
