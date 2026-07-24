# Tau2 Account Prerequisite Ranking Preregistration

Date: 2026-07-24

## Distinct Claim

Test whether an LLM-generated semantic belief model can rank a natural
identifier-acquisition action whose immediate EIG is zero. This is distinct
from the closed MMS app-permission V1 interface.

The customer phone number is visible, but the customer and line IDs are not.
`customer_lookup` returns the same identity and line IDs in all worlds. It
therefore has zero immediate information about the hidden account state.
However, it legally enables `line_details`, which exposes carrier status,
roaming configuration, plan, and usage fields.

## Frozen Environment

- Source: official `unimpor/T3` Tau2 telecom environment, commit
  `492f31fa05d2065c750a72d5e798385af282fa5d`.
- Six official hidden worlds form a 2-by-3 factorial:
  data allowance available/exhausted crossed with account roaming
  enabled/device roaming off, account roaming disabled/device roaming on, and
  account roaming disabled/device roaming off.
- Airplane mode is on in every world and cannot be changed during the
  diagnostic window. It is a common nuisance state, not a target.
- Exact zero-call values:
  immediate `customer_lookup` = 0 nats;
  `customer_lookup -> line_details` = 1.329661 nats;
  best direct two-step root = 0.636514 nats;
  setup advantage = 0.693147 nats.

## LLM Interface

- Gemma 4 26B A4B, non-thinking, temperature 0.
- Eight target-blind carrier-account hypotheses are generated per prompt.
- Six fixed root diagnostics are explicitly rolled out for two steps.
- The model predicts outcome partitions and chooses one legal follow-up per
  predicted branch.
- Hidden task IDs, initialization actions, exact states, and simulator
  responses are never shown to the model.
- The official simulator is used only after scoring to measure realized root
  value.

The units are prompt-robustness variants over one physical six-world support
family. They are not represented as independent environment samples.

## Stages

### Serving Smoke

- Two frozen prompt variants, exactly 14 requests.
- Hard cap $0.50.
- Requires complete parsing, exact request accounting, zero reasoning, finite
  scores, mean official-signature coverage at least 4/6, and both setup
  rollouts choosing `line_details`.

### Formal

- Twelve untouched prompt variants, exactly 84 requests.
- Hard cap $2.00.
- Run only if smoke passes unchanged.

Frozen conjunction:

1. Complete exact request accounting and zero reasoning.
2. Mean official-signature coverage at least 5/6.
3. `customer_lookup` uses `line_details` on at least 10/12 variants.
4. Predicted depth-2 score Spearman with exact depth-2 root value at least
   0.25.
5. Depth 2 selects `customer_lookup` on at least 8/12 variants.
6. Depth 1 never selects `customer_lookup`.
7. Mean realized top-1 regret improves by at least 0.50 nats.
8. Depth 2 beats depth 1 on at least 8/12 variants.

Any substantive invalid tree closes the exact interface without repair or
resampling. Passing authorizes a fresh paired sequential policy test.
