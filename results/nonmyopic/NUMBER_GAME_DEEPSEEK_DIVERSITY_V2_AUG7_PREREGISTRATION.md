# Number Game DeepSeek Diversity V2 Aug 7 Preregistration

Date frozen: 2026-08-07 (Europe/London), after both unchanged-interface
reliability128 results were terminal and before any V2 request.

## Development Boundary

The unchanged-interface Luna and DeepSeek results remain gated null. This is a
new prompt-development interface motivated only by public mechanics diagnostics:
DeepSeek returned exact JSON without forced exits, but too many conditioned
draws were extension-duplicate or observation-inconsistent. No policy efficacy,
selected action, latent truth, or unopened endpoint is used.

V2 keeps the executable grammar, strict parser, 24-item schema, posterior code,
and validity floors unchanged. It adds explicit instructions to:

- return four hypotheses from each of six semantic families;
- make all 24 extensions over `0..100` distinct, not merely their strings;
- test every expression against every observed positive and negative label;
- avoid 24 syntactic rewrites or refinements of one salient observed number; and
- use exactly the `name` and `expression` fields required by the schema.

## Fresh Reliability128 Gate

- model: `deepseek/deepseek-v4-flash-0731`, nonreasoning;
- source tree SHA-256:
  `f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0`;
- case-selection seed: `1080900`;
- model seeds: `1080901..1080908`;
- cases: 8 initial, 40 one-observation, 80 two-observation;
- all 120 conditioned histories are disjoint from the Aug 7 unchanged-interface
  reliability histories;
- public case-manifest SHA-256:
  `5252f992f2065b8f2cf983bd5721d7e07b8fdbd7d0a3c16006353ec027df4bc9`;
- exact 128 initial requests, at most three strict-format retries;
- aggregate concurrency 64, temperature 0.7, maximum 4,200 output tokens;
- maximum cost `$0.10`.

Pass requires the unchanged reliability gates: exact accounting, at most three
initial parse failures all retried once, all final responses strict, zero
provider-error retries and reasoning tokens, at most four transport retries and
three forced exits, 24 schema items per parsed response, all initial supports at
least 16 valid, every conditioned support at least 4 valid, conditioned mean at
least 8, and cost within cap.

## Conditional Stress7168

Stress is authorized only if reliability128 passes every gate. It uses no policy
endpoint and cannot establish efficacy.

- model seeds: `1081201..1081232`;
- 32 balanced groups of 224 requests, aggregate concurrency 64;
- exact 7,168 initial requests;
- 32 initial cases plus all 570 source histories disjoint from both the old and
  V2 reliability gates, then 6,566 deterministic repeats across distinct model
  seed groups;
- public case-manifest SHA-256:
  `9ef820703ebc3e046b2323e120248b59cb7aa152443e8d1645e0230b602af613`;
- at most 16 strict-format retries, 8 transport retries, and 16 forced exits;
- every initial support at least 16 valid, every conditioned support at least 4
  valid, conditioned mean at least 8;
- maximum cost `$2.10`.

The same response schema, V2 prompt, parser, and replay apply. Repeated-history
similarity is descriptive only. A stress pass makes V2 eligible for separately
preregistered policy development; it does not reopen the Aug 7 Qwen control,
authorize Aug 8 diversity, or alter any prior result.

## Daily Authorization

Authorize at most `$0.10 + $2.10 = $2.20` from the existing Aug 7 ledger, whose
pre-V2 SHA-256 is
`15022d299a9ba09c7e57ce7e1097347cdf47e4650866beffd25de6ebb6f66ea7`.
Recorded spend is `$2.782034141`, leaving `$2.217965859`. The executor must
reconcile posted and locally measured spend after each stage, never repeat a
banked or partial stage, and close stress with zero calls if reliability is null.
