# Bamboogle No-Search Memory Screen Preregistration

## Purpose

Determine whether non-reasoning GPT-5.4 already answers the five released
Bamboogle mechanics questions from model memory. A saturated benchmark cannot
support a meaningful information-acquisition comparison, so this screen occurs
before any search query, branch generation, semantic belief update, or policy
score.

## Frozen Interface

- Tasks, in manifest order:
  `test_87`, `test_110`, `test_72`, `test_69`, `test_61`.
- Model: `openai/gpt-5.4` through OpenRouter.
- Reasoning: disabled.
- Five independent physical answer calls per task at temperature `0.8`.
- Exact total: `25` logical and physical calls.
- Each call sees only the question and must return exactly
  `{"answer":"short answer"}`.
- Gold answers are loaded only after all responses are checkpointed. They are
  absent from every model message.
- Normalization lowercases, removes ASCII punctuation and articles
  `a`/`an`/`the`, and collapses whitespace.
- Cost cap: `$0.15`; projected cost: `$0.03`.
- No retry, response repair, parser fallback, search request, or OatML job.

## Metrics

For each task, report:

- five exact normalized match flags;
- sample accuracy;
- modal-answer correctness;
- count of unique normalized answers; and
- categorical answer entropy in nats.

The public artifact omits answer strings. Raw responses remain private and
untracked.

## Conjunctive Pass Gate

All conditions must hold:

1. modal answer correct on at most `3/5` tasks;
2. exact sample accuracy at most `.70`; and
3. at least two tasks have gold-answer sample support below `.80`.

A failure closes Bamboogle for the current project before search-tree spend.
A pass authorizes only a separately committed cached-search mechanics
protocol. It does not release opportunity, development, or holdout values.
