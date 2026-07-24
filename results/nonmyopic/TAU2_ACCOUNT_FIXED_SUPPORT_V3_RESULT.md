# Tau2 Account Fixed-Support V3 Result

Date: 2026-07-24

## Outcome

The full GPT-5.4 nonreasoning smoke completed cleanly but failed one frozen
observational-equivalence gate. Per preregistration, no six-world formal run
follows.

- Run ID:
  `tau2-account-fixed-v3-gpt54-smoke-20260724T212256Z`
- Requests: 12
- Reasoning tokens: 0
- Cost: $0.05791
- Lookup then line-details mechanism: 2/2

GPT-5.4 correctly predicted:

- zero immediate information for lookup, status bar, speed test, payment
  request, and SIM status;
- lookup followed by line details at 1.329661 nats in both variants; and
- zero two-step value for every direct root under its semantic model.

It failed because the official six-world simulator's `network_status` exposes
the one device-roaming-on state, worth 0.636514 nats, while GPT predicted that
airplane mode made network status identical in all worlds.

This is a much more coherent forward model than Gemma V2 and it discovers the
non-myopic setup exactly. It is not a passed six-world gate because it omits a
real observable field.
