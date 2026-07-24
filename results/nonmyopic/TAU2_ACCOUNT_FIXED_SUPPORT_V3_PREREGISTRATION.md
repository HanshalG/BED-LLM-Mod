# Tau2 Account Fixed-Support V3 Capability Preregistration

Date frozen: 2026-07-24

## Hypothesis

Gemma V2 failed because its semantic forward model emitted diagnoses in place
of observable tool responses. V3 asks whether full GPT-5.4 nonreasoning can
preserve observational equivalence on the identical fixed-support BED problem.

This is a model-capability comparison, not a prompt repair. The support,
actions, prompt text, parser, exact simulator, prompt variants, rank metrics,
and thresholds are unchanged. V2 formal prompts remain untouched.

## Model

- `openai/gpt-5.4` through OpenRouter
- reasoning disabled
- temperature zero
- concurrency 128

## Smoke

Two prompt variants, exactly 12 requests, hard cap $0.75. In addition to the V2
mechanics, both variants must satisfy:

1. predicted immediate information for `customer_lookup` is exactly zero;
2. predicted immediate information for `status_bar`, `speed_test`,
   `payment_request`, and `sim_status` is exactly zero;
3. predicted `network_status` immediate information is positive; and
4. every `customer_lookup` branch chooses `line_details`.

These are frozen observational-equivalence checks, not endpoint thresholds.

## Formal

If smoke passes unchanged, run the twelve untouched variants: exactly 72
requests, hard cap $2.00. All smoke equivalence checks must hold on every
variant, plus the V2 frozen conjunction:

- depth-two score/exact-value Spearman at least 0.25;
- depth two selects lookup at least 8/12;
- depth one never selects lookup;
- mean top-one regret improves by at least 0.50 nats;
- depth two beats depth one at least 8/12; and
- lookup uses line details on at least 10/12.

Failure closes the LLM-native Tau2 account line. No third model, prompt change,
or formal retry follows.
