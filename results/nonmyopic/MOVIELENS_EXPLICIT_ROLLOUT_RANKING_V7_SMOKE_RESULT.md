# MovieLens Explicit Regeneration-Rollout Ranking v7 Smoke

Date: 2026-07-24

Run: `movielens-explicit-rollout-v7-smoke-20260724T113856Z`

Status: passed; the preregistered formal gate is authorized.

The interface smoke used prospective user 253 and one candidate. It made exactly
12 physical requests: one initial profile call, one initial likelihood call, five
hypothetical profile-regeneration calls, and five history-free downstream likelihood
calls. It used zero reasoning tokens and cost `$0.05171916`.

All five rating paths completed. Their mean downstream predictive entropies were
`1.466171, 1.493936, 1.555546, 1.525054, 1.542157` nats, yielding an
outcome-weighted explicit-rollout score of `-1.534968`. The variation confirms that
the implemented transition is not constant across hypothetical outcomes.

No candidate or held-out outcome was read. Raw model text remains private and
untracked. Its SHA-256 is
`243c964ac99e99117d98a86b8bcb6833349611dc33d67481dd53660f8054c564`.
