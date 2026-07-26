# DiscoverLLM Source Audit Result

## Decision

The released DiscoverLLM preference dataset is **not eligible for a direct
non-myopic BED replay**. It contains a genuinely rich, path-dependent semantic
intent state, but its published candidate scores are one-step rewards and the
flattened release omits the counterfactual state transitions needed to compare
myopic and lookahead policies.

Keep the simulator as a promising future construction substrate. Do not present
the public preference scores as non-myopic evidence, and do not spend
OpenRouter budget on a replay of the flattened data.

## Frozen Source

- Code: `https://github.com/tsook/discoverllm`
- Code commit: `a9eb2846f60e3681ac8d325fc57fd4e58e2bdc97`
- Dataset:
  `https://huggingface.co/datasets/kixlab/DiscoverLLM-multiturn-preferences`
- Dataset revision: `c857bbf6265bdd573938eb7eac79a7a3131fa7ca`
- Creative-writing Parquet SHA-256:
  `e8f76a47447b442e59e0d76718228b94bba1a11d8f7f653cdf1498bb0adf7843`
- Technical-writing Parquet SHA-256:
  `4dcc29e2cb9f21d4dabaa7d8eeaad773a433248a2be94498968e908b6858ccd0`
- SVG-drawing Parquet SHA-256:
  `aef93ede39dbbf570bd15bda2b217d4a96dec188af77eeb5d88bcaed869a081d`
- Reproducible audit:
  `results/nonmyopic/discoverllm_source_audit/AUDIT.json`

## What The Release Provides

The three domains contain 9,318 candidate rows from 1,484 artifacts and 4,659
turn groups. Every turn group contains two distinct scored assistant
completions. The initial intent states are substantial:

| Domain | Artifacts | Mean nodes | Median depth | Mean hidden nodes |
| --- | ---: | ---: | ---: | ---: |
| Creative writing | 495 | 29.12 | 5 | 27.16 |
| Technical writing | 495 | 27.32 | 5 | 25.55 |
| SVG drawing | 494 | 23.65 | 5 | 22.00 |

The hierarchy is therefore not the failure. It is LLM-generated, semantic, and
path-dependent: parent discovery controls which descendants can be evaluated
and expressed by the simulated user.

## Why The Published Scores Are Myopic

The official synthesis launcher passes:

```text
--window-size 0
```

The paper likewise defines reward from the immediate change in discovered
intents, less a token penalty capped at one. The audit independently recovers
the committed completion from the next turn's prompt and the corresponding
post-action state from its `criteria_history`.

Across all 2,642 recoverable committed transitions:

- the committed response is exactly the maximum-score candidate;
- every released score equals immediate awareness gain minus a value in the
  documented token-penalty range `[0, 1]`;
- score/immediate-gain correlations are `.99735` for creative writing,
  `.99847` for technical writing, and `.99905` for SVG drawing.

The dataset card calls `score` a multi-turn reward because it was materialized
through the builder's `multiturn` field. With a zero rollout window,
`multiturn` and `singleturn` are the same quantity.

## Missing Causal Evidence

Each flattened row exposes only:

- the hierarchy history before the candidate;
- the candidate completion; and
- its scalar immediate reward.

It omits `updated_criteria_objs`, `full_results`, and `future_trajectory`.
Consequently, only the chosen candidate's next state can be reconstructed from
the following turn. There is no post-action state or future trajectory for the
rejected candidate, and no released prior over mutually exclusive latent user
worlds. A ranking comparison would therefore observe the endpoint only for the
action selected by the source policy.

Using the released score as "lookahead" would be circular. Substituting a
hand-built immediate proxy would no longer test the published simulator's
semantic transition.

## Registered Future Use

A distinct experiment may use the source simulator with a positive rollout
window, but it must first add the pieces required for BED:

1. a frozen prior over alternative latent intent worlds rather than one
   co-active hierarchy;
2. shared candidate actions for myopic and non-myopic policies;
3. candidate-level semantic transition samples for every action;
4. a compute-matched myopic control and a random control;
5. an independent endpoint or evaluator so planner and deployment are not the
   same self-judging model; and
6. a zero-cost opportunity gate demonstrating that a lower-immediate action can
   improve expected terminal truth recovery.

This would be a new LLM-native BED environment built from DiscoverLLM, not a
reanalysis of the released preference benchmark. No such paid construction is
authorized by this source audit alone.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
