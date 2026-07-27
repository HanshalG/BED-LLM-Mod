# DiscoverLLM Priority-World Tier Mechanics Result

## Verdict

The exact three-task tier mechanics smoke **fails the conjunctive scientific
gate**. Serving and semantic calibration are excellent, and the non-myopic
selector has a small positive terminal advantage, but the released two-action
construction contains no strict immediate-information sacrifice.

The two apparent root changes are tie breaks: on
`technical_writing:artifact_333` and `technical_writing:artifact_249`, actions
`A` and `B` have exactly equal root EIG and equal root true-world log posterior.
Depth two selects `B` because its continuation is better. On
`creative_writing:artifact_367`, where root EIG genuinely differs, root and
depth-two policies both select `A`.

This is useful mechanism evidence but not non-myopic value. The exact released
two-action/tier-weight/task construction is closed. Do not lower thresholds,
reinterpret ties as sacrifice, change weights, or open the 60 opportunity
artifacts.

## Run Integrity

- Run: `discoverllm-tier-mechanics-20260727T220000Z`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Logical requests / HTTP attempts / retries: `15 / 15 / 0`
- Prompt / completion / reasoning tokens: `64,960 / 4,942 / 0`
- Cost: `$0.236530`
- Forced exits: `0`
- Private raw SHA-256:
  `84523f4c231517f84f53e08abbe6e24a7c306df91cdf97cf04e5ca911f8baed7`

All responses parsed. No semantic retry, repair, coercion, or partial analysis
occurred.

## Scientific Result

| Metric | Result |
| --- | ---: |
| Dynamic tasks at both depths | **1 / 3** |
| Root changes | 2 / 3 |
| Strict delayed reversals | **0 / 3** |
| Weight-robust delayed reversals | **0 / 3** |
| Non-myopic minus myopic terminal truth log posterior | +0.024371 |
| Non-myopic minus random terminal truth log posterior | +0.021169 |
| Non-myopic minus myopic terminal MAP accuracy | 0 |
| Root EIG / terminal truth-log Spearman | 0.971008 |
| Depth-two EIG / terminal truth-log Spearman | 1.000000 |

Per-task primary-weight values:

| Task | Root EIG A/B | Depth-two EIG A/B | Myopic / d2 |
| --- | --- | --- | --- |
| Technical 333 | .188103 / .188103 | .587480 / .616967 | A / B |
| Creative 367 | .099354 / .000000 | .275114 / .173098 | A / A |
| Technical 249 | .232552 / .232552 | .707680 / .776667 | A / B |

The depth correlation is high, but its advantage over root EIG is only
`.028992`, below the frozen `.10` gate. More importantly, neither changed task
pays even a positive root-information cost.

## Calibration

The failure is not caused by an unusable semantic likelihood interface:

- root truth top-tier rate: `.9167`;
- follow-up truth top-tier rate: `1.0000`;
- root strict-top rate: `.7083`;
- follow-up strict-top rate: `.9583`;
- mean truth-weight advantages: `+2.0417` root and `+2.6250` follow-up.

The LLM is accurately recovering the hidden semantic priorities. The problem
is that both released candidate completions elicit equally diagnostic initial
feedback on two tasks and are myopically aligned on the third.

## Decision

Close the exact released-candidate route. A future DiscoverLLM successor is
admissible only if it is scientifically distinct and frozen on fresh
development content, such as an LLM-generated shared diagnostic action bank
that can contain genuinely enabling first actions. It must pass an
immediate-sacrifice gate before opportunity or holdout data open.

Authenticated post-run balance: `$32.816212594`. No fixed reserve; pace through
Monday, 2026-08-03. OpenRouter only. OatML jobs: `0`.
