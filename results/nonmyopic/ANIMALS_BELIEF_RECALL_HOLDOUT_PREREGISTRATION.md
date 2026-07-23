# Animals Belief-Recall Ranker Holdout Preregistration

Status: **frozen before any response involving the 60 holdout targets**.

## Claim And Design

This gate tests whether a target-blind LLM can rank candidate questions by the
quality of the *future LLM-regenerated belief support*, rather than by immediate
EIG alone.

- Fresh seed `24271`.
- Sixty distinct common animals, none used as targets in the seed-1304/1305
  development traces.
- One ordinary bootstrap question and live answer per target, followed by the
  first three unique production candidate questions.
- Every candidate's Yes/No branch runs through the unchanged production
  hypothesis regeneration, animal-name validation, history filtering, and
  likelihood path.
- The exact frozen non-thinking Gemma 4 26B belief-recall prompt from commit
  `6fab781` scores all three candidates. The only registered parser amendment is
  the single-JSON-fence handling in commit `ada57c3`.
- Prompt construction exposes only history, candidate text, predictive branch
  probabilities, and branch support sizes. Hidden target and all truth-coverage
  measurements are excluded by an explicit allowlist.
- No intermediate coverage artifact is written or inspected before ranker
  completion and gate evaluation.

## Frozen Primary Endpoint

For each state, measure the expected hidden-truth coverage of the ranker's top
candidate minus that of immediate EIG's top candidate. Using 10,000 paired
bootstrap replicates with producer seed `24272`, the gate passes only if:

1. all 60 states complete;
2. at least 15 states have nonzero within-pool coverage spread;
3. the paired mean-gain 95% interval has a strictly positive lower bound;
4. ranker wins exceed losses;
5. mean active-state regret is lower for the ranker than immediate EIG.

Candidate-level Spearman values, selected coverage, oracle candidate coverage,
and support-size behavior are supporting diagnostics only.

## Audit And Stop Rule

An independent replay uses bootstrap seed `24273`. It rebuilds every
model-visible payload, verifies target/truth fields are absent, reparses every
raw score vector, recomputes the summary and producer gates, and requires the
independent bootstrap to pass the same scientific gates.

Any serving, mechanics, producer, or audit failure stops this exact ranker line.
There is no prompt, model, target, seed, threshold, or endpoint repair and no
replacement state. Exactly one attempt is made for each of the 60 targets.

## Spend

- Coverage and answer model: non-thinking `google/gemma-4-26b-a4b-it`.
- OpenRouter concurrency: `256`.
- Coverage run cap: `$1.50`; projected cost: `$0.75`.
- Ranker cap: `$0.10`.
- Project spend before holdout responses: `$40.51745853` of `$110`.
