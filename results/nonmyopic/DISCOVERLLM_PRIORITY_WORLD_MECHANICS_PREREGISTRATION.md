# DiscoverLLM Priority-World Mechanics Preregistration

Frozen after the V2 manifest pass and before any mechanics-model response.

## Claim Boundary

This is a three-task mechanics/opportunity smoke for a new LLM-native BED
construction. It is not a DiscoverLLM benchmark result, a policy efficacy
claim, or permission to inspect the 60 opportunity tasks.

The hidden parameter is one of four structurally selected, LLM-generated
priority-criterion subtrees. The two released candidate completions are shared
root actions. The LLM is load-bearing in both the world-conditioned semantic
user transition and the observation likelihood. External code performs only
normalization, Bayes updates, entropy, selection, and exact hidden-world
recovery.

## Frozen Source

- V2 manifest:
  `results/nonmyopic/discoverllm_priority_world_v2/MANIFEST.json`
- Manifest SHA-256:
  `9edfd3b20f762491db78087c95bccb1d345063af3423d4d7ccf6c481aa97ad3a`
- Mechanics split:
  `technical_writing:artifact_240`,
  `creative_writing:artifact_60`,
  `creative_writing:artifact_58`
- Four selected roots per task are reconstructed from seed `24412`. Released
  scores and winner labels remain unread.

## Exact Model Protocol

Model: `openai/gpt-5.4`, non-reasoning, temperature zero. No thinking baseline
is part of this method.

Run five stages with one request per task per stage, exactly 15 physical
requests:

1. Generate concise user feedback for all `2 actions * 4 true worlds`.
2. Independently assign semantic likelihood scores in `[0,100]` for each root
   observation under all four candidate worlds.
3. Generate one target-blind assistant continuation for every observed root
   branch.
4. Generate the next user feedback under that branch's true hidden world.
5. Independently score each second observation under all four worlds.

Before stages 2--5, root observations are deterministically shuffled and
renamed `O1`--`O4` within each action using seed `24413`. Neither the likelihood
scorer nor continuation policy receives the observation-to-truth map.

All responses are strict flat JSON with exact keys. There is no content retry,
repair, coercion, partial-task analysis, or replacement. Provider transport
retries are allowed by the adapter but cause the scientific serving gate to
fail.

Projected cost: `$0.40`. Hard run cap: `$0.75`. OpenRouter only; no OatML.

## Belief And Scores

Start from a uniform four-world prior. Convert each semantic score `s` to a
positive likelihood weight:

```text
exp((s - 50) / 20)
```

Normalize after multiplying by the prior. Root EIG is start entropy minus mean
root-posterior entropy across the four true worlds. Depth-two EIG is start
entropy minus mean terminal-posterior entropy after multiplying the root and
conditional likelihood weights.

For each root action, independently report:

- root EIG;
- depth-two EIG;
- mean terminal log posterior of the exact true world;
- exact terminal MAP world accuracy.

Myopic selects maximum root EIG. Non-myopic selects maximum depth-two EIG.
Ties use action order. Hidden truth is opened only for aggregate endpoint
calculation after all trees and policy scores exist.

## Conjunctive Mechanics Gates

1. Exact 15 requests and 15 HTTP attempts; zero retries, reasoning tokens,
   forced exits, parse errors, or missing cells; cost at most `$0.75`.
2. At least two of three tasks have root-EIG and depth-two-EIG action ranges of
   at least `.02` nats.
3. Non-myopic changes the selected root on at least one task.
4. At least one changed task is a strict delayed-value reversal: non-myopic
   accepts at least `.01` nats lower root EIG and obtains strictly higher mean
   terminal true-world log posterior than myopic.
5. Across the three selected roots, non-myopic mean terminal true-world log
   posterior is strictly higher than myopic and mean MAP accuracy is no lower.
6. Across all six task/action cells, depth-two score versus terminal
   true-world log posterior Spearman correlation is at least `.20` and exceeds
   root-EIG correlation by at least `.10`.

Failure closes this exact fixed-world/two-action construction with no prompt,
temperature, likelihood temperature, task, threshold, or model repair. Passing
authorizes only a separately frozen 60-task zero-cost opportunity/power plan;
it does not itself authorize those calls.
