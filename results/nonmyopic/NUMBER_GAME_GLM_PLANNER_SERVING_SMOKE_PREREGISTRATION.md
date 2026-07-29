# Number Game GLM Planner Exact-10 Serving Preregistration

Date frozen: 2026-07-29, before any GLM 5.1 response under this protocol.

## Purpose

This is the final planning-family serving screen in the current study. It
tests whether GLM 5.1 can supply valid branch-conditioned executable Number
Game supports under the exact interface that failed for DeepSeek V4 Pro and
Grok 4.3. It computes no efficacy.

Passing authorizes one separately frozen paired replication. Failing closes
this route and ends further model-family screening under this interface.

## Frozen Interface And Gates

- planner `z-ai/glm-5.1`, OpenRouter default routing;
- explicit nonreasoning, temperature `0.7`, requested seed `46000`;
- unchanged strict 24-rule schema, prompt, grammar, and 10 histories;
- exactly two initial, four one-observation, and four two-observation calls.

All of the following must pass:

1. Exactly 10 parsed responses, accepted requests, and HTTP attempts.
2. Zero retries, provider-error retries, reasoning tokens, and forced exits.
3. Both initial supports contain at least 16 valid unique rules.
4. Every conditioned support contains at least eight valid unique rules
   consistent with all observations.
5. Reported cost is at most `$0.10`.

No target concepts, roots, policy scores, or efficacy endpoints are opened.
Raw text remains private; public output contains diagnostics and hashes.
