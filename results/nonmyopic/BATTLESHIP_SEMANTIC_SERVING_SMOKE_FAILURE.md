# Collaborative Battleship Semantic Serving Smoke Failure

Date: 2026-07-29

Status: **failed closed; exact interface closed**.

## Frozen Run

- preregistration commit:
  `c603fd2ab1481a0f514f2b1c75e2a9ef1aecc08f`
- run:
  `battleship-semantic-serving-20260729T033500Z`
- public failure SHA-256:
  `27ae82a7dc531a8457a777c68c211be8e92d576b7ff89798ca58c592a2b9a625`
- private raw SHA-256:
  `be1d19c655d086f6c5d7c23a284d9cbcee8932666e322d6ac20d9744e0a35d3e`
- accepted requests / HTTP attempts: `10 / 10`
- retries / provider-error retries: `0 / 0`
- reasoning tokens / forced exits: `0 / 0`
- prompt / completion tokens: `3,164 / 413`
- cost: `$0.00339205`

All ten strict response schemas parsed. No response was repaired, reissued,
normalized, or substituted.

## Failure

The frozen AST validator rejected Gemini's fourth expression because it called
the forbidden array method `.reshape`. Per preregistration, the expression was
not altered and the run stopped without producing a serving pass.

The four deterministically selected fresh questions were:

1. `Is there a ship segment in row A?`
2. `Does any ship occupy at least one cell in columns 1 through 4?`
3. `Does the board contain a ship segment in row D?`
4. `Is there a ship of length 5 whose cells all lie entirely within columns 1 through 4?`

## Saved-Response Diagnostic

This diagnostic made no model calls and does not change the formal failure.
It executed the saved expressions without the frozen validator on the two
already-frozen 4,096-board blocks.

| Question | GPT Yes prevalence | Gemini Yes prevalence | Agreement |
|---|---:|---:|---:|
| Row A | `.5479 / .5630` | `.5479 / .5630` | `1.000 / 1.000` |
| Columns 1--4 | `.9937 / .9949` | `.9937 / .9949` | `1.000 / 1.000` |
| Row D | `.8811 / .8831` | `.8811 / .8831` | `1.000 / 1.000` |
| Length-5 ship inside columns 1--4 | `.0000 / .0000` | `1.0000 / 1.0000` | `.000 / .000` |

The failure is therefore semantic as well as syntactic. Only the row-A
question satisfies the frozen `[.05,.95]` prevalence interval under both
translations. The left-half question is nearly constant, and the fourth
question is compiled to contradictory constants. The planner also spends two
of four selected questions on single-row occupancy and two on the left side,
despite the frozen request for behaviorally diverse questions.

## Decision

Close the exact GPT-5.4-Mini question plus GPT-5.4-Mini/Gemini expression
translation interface. Do not widen the validator, repair the fourth
expression, change selection order, tune prompts, swap models, relax
prevalence/agreement gates, or rerun these seeds.

The earlier zero-call opportunity result remains valid: released
LLM-generated programs exhibit a strong non-myopic task-utility gap. This
serving result says the newly tested lightweight fresh-generation interface
does not reliably reproduce that quality. No policy endpoint,
branch-conditioned mechanics, or fresh efficacy experiment was accessed.
