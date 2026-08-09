# Bongard Horizon-Opportunity Paper Handoff Amendment

Frozen: 2026-08-09, before the first paid Bongard response and before any
mechanics, development, confirmation, or endpoint outcome.

## Purpose

The endpoint-blind classical horizon-opportunity stratum is already frozen in
`BONGARD_OPENWORLD_CLASSICAL_HORIZON_OPPORTUNITY_STRATUM_PROTOCOL_20260809.md`.
This amendment makes its direction-agnostic report mandatory in the shared
paper handoff for both Development64 and Confirmation96.

## Mandatory Input

Every replay-valid endpoint paper fragment must receive the saved output of
`scripts/bongard_openworld_classical_horizon_opportunity.py` for the same stage
result. The paper wrapper must independently rerun that analyzer and require
exact equality with the saved report before rendering it.

The report is invalid unless it:

- has status `classical_horizon_opportunity_stratum_complete`;
- names the same stage and exact stage-result hash as the other mandatory
  endpoint audits;
- uses `compute_matched_myopic_ensemble` as its strict control;
- confirms the exact compute contract and zero model calls/cost;
- preserves both the classical-horizon-disagreement and agreement strata;
- changes no primary gate, claim tier, or paid authorization.

Mechanics-failure fragments have no endpoint result and therefore must reject
an opportunity report rather than loading or rendering one.

## Frozen Wording

The compact paper addendum must state:

- the number of tasks in the endpoint-blind disagreement stratum out of the
  full stage;
- dynamic-depth-two minus call-matched-myopic-ensemble paired mean Brier and
  95% interval in the disagreement stratum;
- the same paired Brier result in the agreement complement;
- that negative values favor dynamic depth two and the split is descriptive
  and non-gating.

Both strata are always reported regardless of direction. A favorable subgroup
cannot rescue a failed random-task primary, authorize confirmation, strengthen
a headline, or change a claim tier. An unfavorable complement cannot erase an
otherwise valid primary result. The renderer must preserve those boundaries in
machine-readable metadata.

## Scope

This amendment changes no paid model, prompt, seed, task, action, request,
endpoint, budget, primary gate, claim tier, headline rule, or confirmation
authorization. It makes zero model calls. It only closes the reporting path for
an already-frozen secondary diagnostic.
