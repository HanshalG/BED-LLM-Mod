# Tau2 MMS Prerequisite Ranking Preregistration

Date: 2026-07-24

## Claim Under Test

An LLM-generated semantic belief model can recognize a genuinely non-myopic
diagnostic action in a released customer-service environment. In the Tau2
telecom MMS task, `check_installed_apps` has zero immediate information about
the hidden fault, but it makes the app name available and legally unlocks
`check_app_permissions("messaging")`. That second observation has four outcomes
over the six hidden fault worlds.

This is a ranking gate, not yet a sequential-policy result.

## Frozen Source And Worlds

- Official environment implementation: `unimpor/T3`, commit
  `492f31fa05d2065c750a72d5e798385af282fa5d`.
- Six remaining fault variants per case:
  `bad_network_preference`, `bad_wifi_calling`, `break_apn_mms_setting`,
  `break_app_sms_permission`, `break_app_storage_permission`, and
  `break_app_both_permissions`.
- Every case adds the same frozen background fault set to all six worlds, so
  the background changes surface context but cannot itself distinguish worlds.
- Smoke and formal background sets are fixed in
  `scripts/tau2_mms_prerequisite_ranking_gate.py` and disjoint.

The exact released simulator gives the setup root 0 immediate EIG and
1.242453 nats of optimal depth-2 information. The best direct two-step root
gives 0.867563 nats, leaving a frozen oracle gap of 0.374890 nats.

## LLM Interface

- Model: Gemma 4 26B A4B, non-thinking, temperature 0.
- The LLM sees the generic MMS ticket, shared background conditions, and
  natural-language tool descriptions.
- It generates eight target-blind semantic hypotheses with structured
  diagnostic signatures.
- For each of eight fixed legal root diagnostics, it explicitly predicts the
  root outcome under every generated hypothesis, chooses one legal follow-up
  per predicted branch, and predicts follow-up outcomes.
- The official task states, initialization actions, true fault IDs, and exact
  simulator responses are hidden from every LLM prompt.
- Exact simulator values are computed only after LLM scoring.

No chain of thought or thinking mode is used.

## Stages And Cost

### Serving Smoke

- 2 frozen cases.
- Exactly 18 physical requests: 2 support generations plus 16 root rollouts.
- Hard run cap: $0.50.
- Must complete all requests with zero reasoning tokens, finite scores, mean
  official-signature coverage at least 4/6, and select
  `messaging_permissions` after every setup observation.

### Formal Ranking Gate

- 12 untouched frozen cases.
- Exactly 108 physical requests: 12 support generations plus 96 root rollouts.
- Hard run cap: $2.00.
- Run only if the smoke passes unchanged.

Frozen conjunction:

1. All cases and exact request accounting complete with zero reasoning tokens.
2. Mean official-signature coverage is at least 5/6.
3. Spearman correlation between predicted and exact depth-2 root value is at
   least 0.25.
4. Predicted depth 2 selects `check_installed_apps` on at least 8/12 cases.
5. Predicted depth 1 never selects the zero-information setup action.
6. Mean realized top-1 regret improves over predicted depth 1 by at least
   0.15 nats.
7. Depth 2 has lower realized regret than depth 1 on at least 8/12 cases.

Failure closes this exact prompt and mechanism. Passing authorizes a fresh
paired sequential policy test; it does not authorize rewriting these gates.

## Pre-Formal Instrumentation Amendment

The first serving-smoke execution completed all 18 calls, but exposed a
coverage-scorer inconsistency. The generation schema explicitly allowed
`unknown` for a status field, while the implementation required exact equality
with an official signature and therefore treated `unknown` as contradicting an
otherwise normal field. Before any formal response, coverage is corrected so
that an official faulty field must be generated as `faulty`, while an official
normal field may be `normal` or `unknown` but not `faulty`. Extra generated
faults still disqualify a match.

The exact raw smoke responses are hash-locked and reused with zero new calls.
No prompt, model, split, action score, threshold, or formal endpoint changes.
