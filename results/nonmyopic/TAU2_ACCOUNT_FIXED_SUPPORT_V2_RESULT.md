# Tau2 Account Fixed-Support V2 Result

Date: 2026-07-24

## Outcome

The two-variant Gemma serving smoke completed but failed the frozen mechanism
gate. No formal prompt was used.

- Run ID: `tau2-account-fixed-v2-smoke-20260724T211808Z`
- Requests: 12
- Reasoning tokens: 0
- Cost: $0.00330527
- Complete finite rollouts: 12/12
- Setup trees using `line_details` on every branch: 0/2

## Failure Mode

Gemma simulated hidden diagnoses rather than observable tool responses.

- It assigned 1.0114 nats of immediate information to `status_bar`,
  `speed_test`, `payment_request`, and `sim_status`, although each real tool
  returns one identical outcome over the six worlds.
- It made `customer_lookup` directly return categories such as
  `data_exhaustion`, `account_setting_issue`, or `roaming_mismatch`, despite the
  prompt stating that lookup returns the same customer and line IDs in every
  world.
- Consequently it predicted 1.0114 nats of immediate lookup information
  instead of zero and split the continuation across `line_details`,
  `data_usage`, and phone diagnostics.

The model therefore violated observational equivalence at the first simulated
step. This localizes the LLM-native planning failure upstream of entropy
scoring: the predicted transition model leaks the latent hypothesis into the
observation.

No tree was repaired or rescored under a different interpretation.
