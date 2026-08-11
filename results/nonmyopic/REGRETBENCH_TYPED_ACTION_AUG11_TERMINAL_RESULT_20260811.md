# RegretBench Typed-Action Aug 11 Terminal Result

Status: **semantic calibration failed closed**.

The prospectively frozen exact-eight wrapper was invoked once from pushed commit `04a96834` after an authenticated read-only preflight passed with zero files and zero model calls. All eight DeepSeek V4 Flash 0731 nonreasoning requests completed cleanly: two typed proposals, two answer-blind likelihood evaluations, and four independently seeded environment-codec responses.

## What Passed

The typed action representation removed the earlier question-syntax failure. Both proposals emitted exact permutations of the four executable action IDs, code joined their canonical questions, both evaluators produced normalized categorical likelihoods, and the selected actions had nondegenerate mutual information of `0.347416` and `0.667465` nats. All privacy, ordering, source-value loading, transport, retry, reasoning, and budget gates passed.

## Terminal Gates

The semantic option calibration failed in two distinct ways:

- Task 0 selected `entity_type`. Its two codec replicates agreed and mapped all three real source values to options 0--2, but the evaluator assigned its largest predictive mass, `0.462500`, to option 3 (`Other`). That top predictive outcome was not realizable under the environment support.
- Task 1 selected `notable_feature`. All four options were realizable and the top predictive outcomes were covered, but the two codec replicates swapped the mappings of source-value indexes 2 and 3. The executable observation model was therefore not reproducible.

The failed frozen gates were `all_top_predictive_options_are_realized` and `all_codec_replicates_agree`. The exact interface, task pair, prompts, seeds, and thresholds are closed. Neither gate will be relaxed and the run will not be retried.

## Scientific Interpretation

This is an upstream semantic-calibration null, not evidence about non-myopic planning or policy efficacy. The model generated a syntactically executable action space and nondegenerate likelihoods, but its categorical partition was not aligned reliably enough with the environment's actual observation support. Mechanics, second actions, development, confirmation, and all endpoint outcomes remained unopened.

The strongest successor should remove the separate stochastic codec rather than tune it. A prospectively new interface can let code own a deterministic mapping from private source values to stable executable value IDs, while the LLM still performs the irreducible work of generating semantic hypotheses and assigning likelihoods over those IDs. It must use untouched tasks, pass adversarial semantic calibration before planning, and retain compute-matched myopic and random controls with sealed endpoints.

## Calls, Cost, And Provenance

- accepted responses: `2` proposals, `2` evaluators, `4` codecs;
- retries, reasoning tokens, and forced exits: `0`;
- prompt/completion tokens: `4,386` / `3,352`;
- locally measured cost: `$0.001219933524`;
- authenticated terminal-audit credits/usage/balance: `$245.000000000` / `$220.134128880` / `$24.865871120`;
- conservative Aug 11 account-wide recorded spend: `$0.004448869524`;
- conservative remaining Aug 11 allowance: `$4.995551130476`.

Artifacts:

- public daily result: `results/nonmyopic/regretbench_typed_action_smoke/DAILY_RESULT_20260811.json` (SHA-256 `657408c603cf542e968777cb5b35c7d1b56294c7d06669484930f225425a30de`);
- smoke result: `results/nonmyopic/regretbench_typed_action_smoke/smoke-20260811/RESULT.json` (SHA-256 `c21312b3e8d752bec8ef0d96ec6d12332c3ae5f9bc55a9574a03b2c11442e28c`);
- independent verification: `results/nonmyopic/regretbench_typed_action_smoke/smoke-20260811/VERIFICATION.json` (SHA-256 `7fdec1765e81096070610a9928830c4ee3d97dc4aaebac0439383805c23cd122`);
- stage ledger: `results/nonmyopic/openrouter_daily_budget/2026-08-11-regretbench-typed-action-smoke.json` (SHA-256 `7a39a6bbafe272f130a5de5d9c0c7d96e37d03b6374ea6086a652489fa2786f9`);
- public audit: `results/nonmyopic/regretbench_typed_action_smoke/TERMINAL_AUDIT.json`.

Private raw responses, prompt audits, ordering record, and run log remain untracked. Their hashes are bound in the public audit; no raw response, private source value, hidden intent, or endpoint appears in the public terminal package.
