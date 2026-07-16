# Rock Diagnosis `5-7` LLM Candidate-Proposal Pilot Registration

Registered: 2026-07-15, after the fresh exact `5-7` gate and before the first LLM
request for this map.

## Purpose

This is the required second external environment-instance replication screen. Figure
4's `5-7` map changes the latent target from three to five rock types, the grid from
6x6 to 7x7, and the sensing geometry, while retaining the paper-defined Rock Diagnosis
contract and exact scorer. The fresh zero-LLM gate passes at K=3; this ≤10-trajectory
pilot asks whether the LLM candidate interface preserves that mechanism.

## Frozen Design

- Seed `9174`; 8 paired trajectories; horizon 8; K=3; Figure 4 `5-7` map and fixed
  start `(0,3)`; exact posterior over all 32 static type vectors; full-vector MAP.
- `google/gemma-4-26b-a4b-it` via OpenRouter, `thinking: false`, temperature `0`,
  128 output tokens, exactly one validation-feedback retry, and no code-side action
  padding/substitution. The prompt includes only the map, current position/history,
  exact posterior, and legal action IDs.
- Exact arms: shared root-cell d1, exact two-step incremental EIG d2, and one-step
  call-matched width. All hidden targets and contextual sensor observations are CRN;
  base cells are cached/shared; width makes one current-state proposal call for every
  d2 root-outcome continuation cell.

## Read And Budget

Primary exploratory readout is paired final exact posterior-entropy reduction,
`H(control)-H(d2)`. Directional promotion requires positive mean reduction against
both controls plus shared-root, legality, width-allocation, and no-terminal-cell-failure
checks. Entropy AUC, MAP, truth log posterior, root moves, selected EIG, raw retries,
candidate diversity, traces, and usage are descriptive diagnostics.

The five-rock posterior prompt is longer than `3-6`, so this screen projects `$0.20`
with a `$0.50` hard adapter cap. It remains below the exploration protocol's `$1` and
10-task limits. A directional result earns exactly one independently seeded, powered,
pre-registered `5-7` confirmation; any other result is ledgered as non-promotion.
