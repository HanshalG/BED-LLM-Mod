# Bongard Confirmation64 Execution Amendment

Date frozen: 2026-08-07, before any Bongard mechanics, development, or
confirmation model response.

## Purpose

The interface-v3 confirmation manifest froze tasks, policies, prompts, controls,
model, seeds, dates, budgets, and confirmatory gates before implementation. This
amendment binds the completed execution surface. V3 adds the pre-response
path-dependent fixed-support gate family without changing model calls or data.

## Bound Implementation

- confirmation-owned block replay and 64-task endpoint analysis:
  `scripts/bongard_openworld_luna_confirmation64.py`, SHA-256
  `00995532e8a8e1e7bd4c27e4513c0332e5a9aa16834c79fc7973ee689a0a4a33`;
- exact-date daily budget and predecessor driver:
  `scripts/bongard_openworld_luna_confirmation64_daily_execute.py`, SHA-256
  `0a6355e2239799f29e0fab2bdb0600285051b1217cbe7be97606c2dbdc4e77ee`.

The independent execution verifier must pass before preflight or execution. The
confirmation core owns its result schema and calls the already frozen generic
belief, tree, CRN, prompt-privacy, and endpoint-scoring helpers directly. It does
not mutate or monkeypatch development module constants.

## Runtime Boundary

Every block requires the exact full path-dependent development claim tier, exact confirmation
manifest, exact date, all prior confirmation blocks independently replayed, a
fresh account-wide `$5` ledger, live `$5` balance, Luna model-catalog validity,
and a `$4.75` run cap. A partial or null development tier forbids all calls.

Blocks A--C bank endpoint-blind trees only. Block D loads confirmation endpoint
labels only after all four raw-response trees replay. No result opens the sealed
official test, reserve, model substitution, or a stronger causal claim.
