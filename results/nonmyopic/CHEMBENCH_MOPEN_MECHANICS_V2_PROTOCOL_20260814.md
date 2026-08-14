# ChemBench M-open Mechanics V2 Protocol

Date frozen: 2026-08-14 (Europe/London)

V2 inherits the V1 protocol and development amendment except for the explicit
source-cohort correction below. Require the V1 terminal report. No V1 result or
transition artifact exists, and no `v3` source response was evaluated.

## Mixed-Version Candidate Bank

- The nine simple initial-support structures use their released `v2`
  parameter representatives for likelihood and proxy prediction.
- The other 48 active compound/novel structures use their released `v3`
  parameter representatives.
- The truth cohort is exactly those 48 outside-initial-support structures.
- The nine simple structures are candidates and controls only; they are not V2
  truth cells.

This is scientifically aligned with the M-open question: every evaluated truth
is absent from initial support. It does not impute nonexistent simple `v3`
parameters or spend a future `v4/v5` state.

## Retained Slices

The response-free V1 slices and proxy-query seeds remain:

- `easy/v3`: `2026081601`
- `medium/v3`: `2026081602`
- `hard/v3`: `2026081603`

The suffix names the outside-support truth version. Each result must also store
the complete per-model version map and its SHA256.

## Source Preflight

Before generating proxy queries, allocating response matrices, or evaluating a
rate function, V2 must verify:

1. all nine initial names are active and have `v2` in every difficulty;
2. all 48 truth names are active, outside initial support, and have `v3` in
   every difficulty;
3. the truth cohort has exactly 48 unique entries;
4. every selected parameter mapping is finite and non-empty;
5. the source commit, tree, and file hashes match the parent protocols.

An injected missing-version fixture must fail before a supplied rate function
can increment its call counter.

## Planner and Gate

All V1 mechanics remain exact: categorical likelihoods, initial unknown score,
live/reserve semantics, proposal kernels, six assay groups, 6/3/2 widening,
four-step receding horizons, terminal log-rate MSE, call-matched replay, fixed
and history-blind controls, and practical `1e-6` paired tie tolerance.

Horizon means, comparisons, and truth-cell counts use only the 48 frozen truth
indices. Proposal ranking and posterior inference may use all represented
candidate models. The V1 5% aggregate and paired/root gates are otherwise
unchanged.

## Execution and Replay

- Schema: `chembench-mopen-mechanics-v2`.
- Use new V2 result, transition-bank, and verification paths.
- Run exactly once from a pushed commit that binds this protocol, V1 terminal,
  parent protocols, source adapter, runner, independent verifier, and tests.
- The producer-independent verifier must reconstruct the mixed-version bank and
  48 truth indices from source metadata, not trust producer-provided arrays.
- Model/API calls and cost remain exactly zero.

V2 is terminal after its one exact command. A scientific pass authorizes only
the prospectively frozen LLM semantic/ranking gate.
