# Number Game DeepSeek Planner Exact-10 Serving Preregistration

Date frozen: 2026-07-29, before any DeepSeek V4 Pro response under this
protocol.

## Purpose

This is a transport and executable-support gate for a genuinely new planning
model family. It does not measure policy efficacy. Passing authorizes one
separately frozen paired Number Game replication; failing closes this model
route without prompt repair or response replacement.

## Frozen Interface

- planner: `deepseek/deepseek-v4-pro`;
- OpenRouter default routing;
- explicit nonreasoning mode;
- temperature `0.7`;
- strict existing 24-rule JSON schema and safe executable grammar;
- one seeded adapter (`42000`);
- exactly 10 accepted calls in one batch:
  - two initial-support prompts;
  - four one-observation refresh prompts;
  - four two-observation refresh prompts.

The histories are fixed in
`scripts/number_game_deepseek_planner_serving_smoke.py`. No target concepts,
candidate roots, policy scores, or efficacy endpoints are shown.

## Gates

All gates must pass:

1. Exactly 10 responses parse under the unchanged strict schema.
2. Exactly 10 accepted requests and 10 HTTP attempts.
3. Zero retries, provider-error retries, reasoning tokens, and forced exits.
4. Both initial responses yield at least 16 valid unique executable rules.
5. Every conditioned response yields at least eight valid unique rules
   consistent with all stated observations.
6. Total reported cost is at most `$0.10`.

Raw text remains private. The public result contains diagnostics, extension
hashes, accounting, and the raw-response SHA-256.
