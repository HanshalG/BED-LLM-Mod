# VoI Medical Dynamic-Support Recovery Preregistration

## Question

Before building a non-myopic scorer, test whether semantic roots actually cause
different path-dependent recovery of truths omitted from the initial
LLM-generated support.

The frozen serving artifact omitted the external target on exactly two eligible
mechanics rows: row `50` (Gastric ulcer) and row `284` (Cold). These two rows are
fixed before this run; no other task is opened.

## Frozen Tree

- Reuse the exact six initial hypotheses and four roots per task from serving
  artifact SHA `e7effe8977f40fa25e8b8d43f7862d664ea3ee0f991c89a93fc65567d484bb0d`.
- For each of eight task-root pairs, classify the six initial hypotheses into
  `Yes/No/Maybe` using one isolated strict call.
- Use a uniform prior over the six initial hypotheses to calculate exact
  immediate EIG and root outcome masses.
- For every task, root, and outcome, regenerate six free-form hypotheses from
  the self-report, initial support, and complete hypothetical question-answer
  history.
- Do not expose the target, released diagnosis list, stored conversation, or
  any endpoint during generation.
- Load the two external targets only after all maps and 24 branch supports parse
  and freeze.

This requires exactly `8 + 24 = 32` physical requests.

## Opportunity Metrics

For each root, expected target coverage is the frozen outcome mass times the
exact phrase-coverage indicator in each regenerated branch. The myopic root
maximizes immediate EIG. The diagnostic oracle root maximizes expected target
coverage and is used only to establish opportunity, never as an executable
policy.

All gates are conjunctive:

- exact 32 requests and HTTP attempts;
- zero retries, reasoning, forced exits, and forced finalization;
- all eight maps and 24 six-hypothesis branch supports parse;
- every root has at least two positive outcomes;
- both initially missing targets appear in at least one positive-probability
  branch;
- at least one task has root coverage range `>=0.15`;
- at least one task has oracle-minus-myopic expected coverage `>=0.15`;
- mean oracle-minus-myopic coverage across the two tasks is `>=0.10`;
- cost is at most `$0.25`.

## Consequence

Passing establishes only that a delayed truth-coverage opportunity exists and
authorizes a separately frozen target-blind model-aware scorer on these
mechanics trees. Failure closes this MedDG dynamic-support route before the
opportunity split.

- Model: `openai/gpt-5.4`, nonreasoning
- Temperatures: `0` for maps, `0.7` for support regeneration
- Projected cost: `$0.10`
- Hard cap: `$0.25`
- Frozen allowance: `$1.10112155`
- Protected OpenRouter balance: `$25` through Monday
- OatML/cluster use: prohibited
